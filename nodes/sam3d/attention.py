# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Consolidated attention dispatch for SAM3DObjects.

Dense attention: ComfyUI's optimized_attention_for_device (native).
Sparse/varlen attention: comfy.attention_sparse.dispatch_varlen_attention
    (sage2 > flash > xformers > sdpa).
"""
from typing import *
from enum import Enum
import math
import torch
import torch.nn.functional as F

from comfy.ldm.modules.attention import optimized_attention_for_device
from comfy.attention_sparse import dispatch_varlen_attention

from .sparse import SparseTensor, DEBUG


__all__ = [
    "scaled_dot_product_attention",
    "sparse_scaled_dot_product_attention",
    "sparse_serialized_scaled_dot_product_self_attention",
    "sparse_windowed_scaled_dot_product_self_attention",
    "masked_sdpa",
    "SerializeMode",
    "SerializeModes",
]


# ==========================================================================
# Dense attention
# ==========================================================================

@overload
def scaled_dot_product_attention(qkv: torch.Tensor) -> torch.Tensor: ...

@overload
def scaled_dot_product_attention(q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor: ...

@overload
def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor: ...

def scaled_dot_product_attention(*args, **kwargs):
    arg_names_dict = {1: ["qkv"], 2: ["q", "kv"], 3: ["q", "k", "v"]}
    num_all_args = len(args) + len(kwargs)
    assert (
        num_all_args in arg_names_dict
    ), f"Invalid number of arguments, got {num_all_args}, expected 1, 2, or 3"
    for key in arg_names_dict[num_all_args][len(args):]:
        assert key in kwargs, f"Missing argument {key}"

    if num_all_args == 1:
        qkv = args[0] if len(args) > 0 else kwargs["qkv"]
        q, k, v = qkv.unbind(dim=2)
    elif num_all_args == 2:
        q = args[0] if len(args) > 0 else kwargs["q"]
        kv = args[1] if len(args) > 1 else kwargs["kv"]
        k, v = kv.unbind(dim=2)
    elif num_all_args == 3:
        q = args[0] if len(args) > 0 else kwargs["q"]
        k = args[1] if len(args) > 1 else kwargs["k"]
        v = args[2] if len(args) > 2 else kwargs["v"]

    # Cast to bf16 for sage attention compatibility, preserve original dtype for output
    orig_dtype = q.dtype
    if q.dtype == torch.float32:
        q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()
    elif k.dtype != q.dtype:
        k, v = k.to(q.dtype), v.to(q.dtype)

    # q, k, v: [N, L, H, C] -> [N, H, L, C] for ComfyUI optimized_attention
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)

    attn_fn = optimized_attention_for_device(q.device)
    out = attn_fn(q, k, v, heads=q.shape[1], skip_reshape=True, skip_output_reshape=True)

    # [N, H, L, C] -> [N, L, H, C], restore original dtype
    return out.permute(0, 2, 1, 3).to(orig_dtype)


# ==========================================================================
# Sparse full attention
# ==========================================================================

def sparse_scaled_dot_product_attention(*args, **kwargs):
    arg_names_dict = {1: ["qkv"], 2: ["q", "kv"], 3: ["q", "k", "v"]}
    num_all_args = len(args) + len(kwargs)
    assert (
        num_all_args in arg_names_dict
    ), f"Invalid number of arguments, got {num_all_args}, expected 1, 2, or 3"
    for key in arg_names_dict[num_all_args][len(args):]:
        assert key in kwargs, f"Missing argument {key}"

    if num_all_args == 1:
        qkv = args[0] if len(args) > 0 else kwargs["qkv"]
        assert isinstance(
            qkv, SparseTensor
        ), f"qkv must be a SparseTensor, got {type(qkv)}"
        assert (
            len(qkv.shape) == 4 and qkv.shape[1] == 3
        ), f"Invalid shape for qkv, got {qkv.shape}, expected [N, *, 3, H, C]"
        device = qkv.device

        s = qkv
        q_seqlen = [
            qkv.layout[i].stop - qkv.layout[i].start for i in range(qkv.shape[0])
        ]
        kv_seqlen = q_seqlen
        q, k, v = qkv.feats.unbind(dim=1)

    elif num_all_args == 2:
        q = args[0] if len(args) > 0 else kwargs["q"]
        kv = args[1] if len(args) > 1 else kwargs["kv"]
        assert (
            isinstance(q, SparseTensor)
            and isinstance(kv, (SparseTensor, torch.Tensor))
            or isinstance(q, torch.Tensor)
            and isinstance(kv, SparseTensor)
        ), f"Invalid types, got {type(q)} and {type(kv)}"
        assert (
            q.shape[0] == kv.shape[0]
        ), f"Batch size mismatch, got {q.shape[0]} and {kv.shape[0]}"
        device = q.device

        if isinstance(q, SparseTensor):
            assert (
                len(q.shape) == 3
            ), f"Invalid shape for q, got {q.shape}, expected [N, *, H, C]"
            s = q
            q_seqlen = [q.layout[i].stop - q.layout[i].start for i in range(q.shape[0])]
            q = q.feats
        else:
            assert (
                len(q.shape) == 4
            ), f"Invalid shape for q, got {q.shape}, expected [N, L, H, C]"
            s = None
            N, L, H, C = q.shape
            q_seqlen = [L] * N
            q = q.reshape(N * L, H, C)

        if isinstance(kv, SparseTensor):
            assert (
                len(kv.shape) == 4 and kv.shape[1] == 2
            ), f"Invalid shape for kv, got {kv.shape}, expected [N, *, 2, H, C]"
            kv_seqlen = [
                kv.layout[i].stop - kv.layout[i].start for i in range(kv.shape[0])
            ]
            k, v = kv.feats.unbind(dim=1)
        else:
            assert (
                len(kv.shape) == 5
            ), f"Invalid shape for kv, got {kv.shape}, expected [N, L, 2, H, C]"
            N, L, _, H, C = kv.shape
            kv_seqlen = [L] * N
            kv = kv.reshape(N * L, 2, H, C)
            k, v = kv.unbind(dim=1)

    elif num_all_args == 3:
        q = args[0] if len(args) > 0 else kwargs["q"]
        k = args[1] if len(args) > 1 else kwargs["k"]
        v = args[2] if len(args) > 2 else kwargs["v"]
        assert (
            isinstance(q, SparseTensor)
            and isinstance(k, (SparseTensor, torch.Tensor))
            and type(k) == type(v)
            or isinstance(q, torch.Tensor)
            and isinstance(k, SparseTensor)
            and isinstance(v, SparseTensor)
        ), f"Invalid types, got {type(q)}, {type(k)}, and {type(v)}"
        assert (
            q.shape[0] == k.shape[0] == v.shape[0]
        ), f"Batch size mismatch, got {q.shape[0]}, {k.shape[0]}, and {v.shape[0]}"
        device = q.device

        if isinstance(q, SparseTensor):
            assert (
                len(q.shape) == 3
            ), f"Invalid shape for q, got {q.shape}, expected [N, *, H, Ci]"
            s = q
            q_seqlen = [q.layout[i].stop - q.layout[i].start for i in range(q.shape[0])]
            q = q.feats
        else:
            assert (
                len(q.shape) == 4
            ), f"Invalid shape for q, got {q.shape}, expected [N, L, H, Ci]"
            s = None
            N, L, H, CI = q.shape
            q_seqlen = [L] * N
            q = q.reshape(N * L, H, CI)

        if isinstance(k, SparseTensor):
            assert (
                len(k.shape) == 3
            ), f"Invalid shape for k, got {k.shape}, expected [N, *, H, Ci]"
            assert (
                len(v.shape) == 3
            ), f"Invalid shape for v, got {v.shape}, expected [N, *, H, Co]"
            kv_seqlen = [
                k.layout[i].stop - k.layout[i].start for i in range(k.shape[0])
            ]
            k = k.feats
            v = v.feats
        else:
            assert (
                len(k.shape) == 4
            ), f"Invalid shape for k, got {k.shape}, expected [N, L, H, Ci]"
            assert (
                len(v.shape) == 4
            ), f"Invalid shape for v, got {v.shape}, expected [N, L, H, Co]"
            N, L, H, CI, CO = *k.shape, v.shape[-1]
            kv_seqlen = [L] * N
            k = k.reshape(N * L, H, CI)
            v = v.reshape(N * L, H, CO)

    if DEBUG:
        if s is not None:
            for i in range(s.shape[0]):
                assert (
                    s.coords[s.layout[i]] == i
                ).all(), f"SparseScaledDotProductSelfAttention: batch index mismatch"

    # Build cumulative sequence lengths and dispatch
    cu_seqlens_q = (
        torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(q_seqlen), dim=0)])
        .int()
        .to(device)
    )
    cu_seqlens_kv = (
        torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(kv_seqlen), dim=0)])
        .int()
        .to(device)
    )
    out = dispatch_varlen_attention(
        q, k, v, cu_seqlens_q, cu_seqlens_kv,
        max(q_seqlen), max(kv_seqlen),
    )

    if s is not None:
        return s.replace(out)
    else:
        return out.reshape(N, L, H, -1)


# ==========================================================================
# Serialized attention
# ==========================================================================

class SerializeMode(Enum):
    Z_ORDER = 0
    Z_ORDER_TRANSPOSED = 1
    HILBERT = 2
    HILBERT_TRANSPOSED = 3


SerializeModes = [
    SerializeMode.Z_ORDER,
    SerializeMode.Z_ORDER_TRANSPOSED,
    SerializeMode.HILBERT,
    SerializeMode.HILBERT_TRANSPOSED,
]


def calc_serialization(
    tensor: SparseTensor,
    window_size: int,
    serialize_mode: SerializeMode = SerializeMode.Z_ORDER,
    shift_sequence: int = 0,
    shift_window: Tuple[int, int, int] = (0, 0, 0),
) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[int]]:
    fwd_indices = []
    bwd_indices = []
    seq_lens = []
    seq_batch_indices = []
    offsets = [0]

    if "vox2seq" not in globals():
        import vox2seq

    serialize_coords = tensor.coords[:, 1:].clone()
    serialize_coords += torch.tensor(
        shift_window, dtype=torch.int32, device=tensor.device
    ).reshape(1, 3)
    if serialize_mode == SerializeMode.Z_ORDER:
        code = vox2seq.encode(serialize_coords, mode="z_order", permute=[0, 1, 2])
    elif serialize_mode == SerializeMode.Z_ORDER_TRANSPOSED:
        code = vox2seq.encode(serialize_coords, mode="z_order", permute=[1, 0, 2])
    elif serialize_mode == SerializeMode.HILBERT:
        code = vox2seq.encode(serialize_coords, mode="hilbert", permute=[0, 1, 2])
    elif serialize_mode == SerializeMode.HILBERT_TRANSPOSED:
        code = vox2seq.encode(serialize_coords, mode="hilbert", permute=[1, 0, 2])
    else:
        raise ValueError(f"Unknown serialize mode: {serialize_mode}")

    for bi, s in enumerate(tensor.layout):
        num_points = s.stop - s.start
        num_windows = (num_points + window_size - 1) // window_size
        valid_window_size = num_points / num_windows
        to_ordered = torch.argsort(code[s.start:s.stop])
        if num_windows == 1:
            fwd_indices.append(to_ordered)
            bwd_indices.append(
                torch.zeros_like(to_ordered).scatter_(
                    0, to_ordered, torch.arange(num_points, device=tensor.device)
                )
            )
            fwd_indices[-1] += s.start
            bwd_indices[-1] += offsets[-1]
            seq_lens.append(num_points)
            seq_batch_indices.append(bi)
            offsets.append(offsets[-1] + seq_lens[-1])
        else:
            offset = 0
            mids = [
                (i + 0.5) * valid_window_size + shift_sequence
                for i in range(num_windows)
            ]
            split = [
                math.floor(i * valid_window_size + shift_sequence)
                for i in range(num_windows + 1)
            ]
            bwd_index = torch.zeros(
                (num_points,), dtype=torch.int64, device=tensor.device
            )
            for i in range(num_windows):
                mid = mids[i]
                valid_start = split[i]
                valid_end = split[i + 1]
                padded_start = math.floor(mid - 0.5 * window_size)
                padded_end = padded_start + window_size
                fwd_indices.append(
                    to_ordered[
                        torch.arange(padded_start, padded_end, device=tensor.device)
                        % num_points
                    ]
                )
                offset += valid_start - padded_start
                bwd_index.scatter_(
                    0,
                    fwd_indices[-1][
                        valid_start - padded_start: valid_end - padded_start
                    ],
                    torch.arange(
                        offset, offset + valid_end - valid_start, device=tensor.device
                    ),
                )
                offset += padded_end - valid_start
                fwd_indices[-1] += s.start
            seq_lens.extend([window_size] * num_windows)
            seq_batch_indices.extend([bi] * num_windows)
            bwd_indices.append(bwd_index + offsets[-1])
            offsets.append(offsets[-1] + num_windows * window_size)

    fwd_indices = torch.cat(fwd_indices)
    bwd_indices = torch.cat(bwd_indices)

    return fwd_indices, bwd_indices, seq_lens, seq_batch_indices


def sparse_serialized_scaled_dot_product_self_attention(
    qkv: SparseTensor,
    window_size: int,
    serialize_mode: SerializeMode = SerializeMode.Z_ORDER,
    shift_sequence: int = 0,
    shift_window: Tuple[int, int, int] = (0, 0, 0),
) -> SparseTensor:
    assert (
        len(qkv.shape) == 4 and qkv.shape[1] == 3
    ), f"Invalid shape for qkv, got {qkv.shape}, expected [N, *, 3, H, C]"

    serialization_spatial_cache_name = (
        f"serialization_{serialize_mode}_{window_size}_{shift_sequence}_{shift_window}"
    )
    serialization_spatial_cache = qkv.get_spatial_cache(
        serialization_spatial_cache_name
    )
    if serialization_spatial_cache is None:
        fwd_indices, bwd_indices, seq_lens, seq_batch_indices = calc_serialization(
            qkv, window_size, serialize_mode, shift_sequence, shift_window
        )
        qkv.register_spatial_cache(
            serialization_spatial_cache_name,
            (fwd_indices, bwd_indices, seq_lens, seq_batch_indices),
        )
    else:
        fwd_indices, bwd_indices, seq_lens, seq_batch_indices = (
            serialization_spatial_cache
        )

    M = fwd_indices.shape[0]
    T = qkv.feats.shape[0]
    H = qkv.feats.shape[2]
    C = qkv.feats.shape[3]

    qkv_feats = qkv.feats[fwd_indices]

    if DEBUG:
        start = 0
        qkv_coords = qkv.coords[fwd_indices]
        for i in range(len(seq_lens)):
            assert (
                qkv_coords[start: start + seq_lens[i], 0] == seq_batch_indices[i]
            ).all(), (
                f"SparseWindowedScaledDotProductSelfAttention: batch index mismatch"
            )
            start += seq_lens[i]

    if all([seq_len == window_size for seq_len in seq_lens]):
        B = len(seq_lens)
        N = window_size
        qkv_feats = qkv_feats.reshape(B, N, 3, H, C)
        q, k, v = qkv_feats.unbind(dim=2)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        attn_fn = optimized_attention_for_device(q.device)
        out = attn_fn(q, k, v, heads=H, skip_reshape=True, skip_output_reshape=True)
        out = out.permute(0, 2, 1, 3)
        out = out.reshape(B * N, H, C)
    else:
        q, k, v = qkv_feats.unbind(dim=1)
        cu_seqlens = (
            torch.cat(
                [torch.tensor([0]), torch.cumsum(torch.tensor(seq_lens), dim=0)],
                dim=0,
            )
            .to(qkv.device)
            .int()
        )
        out = dispatch_varlen_attention(
            q, k, v, cu_seqlens, cu_seqlens, max(seq_lens), max(seq_lens),
        )

    out = out[bwd_indices]

    if DEBUG:
        qkv_coords = qkv_coords[bwd_indices]
        assert torch.equal(
            qkv_coords, qkv.coords
        ), "SparseWindowedScaledDotProductSelfAttention: coordinate mismatch"

    return qkv.replace(out)


# ==========================================================================
# Windowed attention
# ==========================================================================

def calc_window_partition(
    tensor: SparseTensor,
    window_size: Union[int, Tuple[int, ...]],
    shift_window: Union[int, Tuple[int, ...]] = 0,
) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[int]]:
    DIM = tensor.coords.shape[1] - 1
    shift_window = (
        (shift_window,) * DIM if isinstance(shift_window, int) else shift_window
    )
    window_size = (window_size,) * DIM if isinstance(window_size, int) else window_size
    shifted_coords = tensor.coords.clone().detach()
    shifted_coords[:, 1:] += torch.tensor(
        shift_window, device=tensor.device, dtype=torch.int32
    ).unsqueeze(0)

    MAX_COORDS = shifted_coords[:, 1:].max(dim=0).values.tolist()
    NUM_WINDOWS = [math.ceil((mc + 1) / ws) for mc, ws in zip(MAX_COORDS, window_size)]
    OFFSET = torch.cumprod(torch.tensor([1] + NUM_WINDOWS[::-1]), dim=0).tolist()[::-1]

    shifted_coords[:, 1:] //= torch.tensor(
        window_size, device=tensor.device, dtype=torch.int32
    ).unsqueeze(0)
    shifted_indices = (
        shifted_coords
        * torch.tensor(OFFSET, device=tensor.device, dtype=torch.int32).unsqueeze(0)
    ).sum(dim=1)
    fwd_indices = torch.argsort(shifted_indices)
    bwd_indices = torch.empty_like(fwd_indices)
    bwd_indices[fwd_indices] = torch.arange(fwd_indices.shape[0], device=tensor.device)
    seq_lens = torch.bincount(shifted_indices)
    seq_batch_indices = (
        torch.arange(seq_lens.shape[0], device=tensor.device, dtype=torch.int32)
        // OFFSET[0]
    )
    mask = seq_lens != 0
    seq_lens = seq_lens[mask].tolist()
    seq_batch_indices = seq_batch_indices[mask].tolist()

    return fwd_indices, bwd_indices, seq_lens, seq_batch_indices


def sparse_windowed_scaled_dot_product_self_attention(
    qkv: SparseTensor, window_size: int, shift_window: Tuple[int, int, int] = (0, 0, 0)
) -> SparseTensor:
    assert (
        len(qkv.shape) == 4 and qkv.shape[1] == 3
    ), f"Invalid shape for qkv, got {qkv.shape}, expected [N, *, 3, H, C]"

    serialization_spatial_cache_name = f"window_partition_{window_size}_{shift_window}"
    serialization_spatial_cache = qkv.get_spatial_cache(
        serialization_spatial_cache_name
    )
    if serialization_spatial_cache is None:
        fwd_indices, bwd_indices, seq_lens, seq_batch_indices = calc_window_partition(
            qkv, window_size, shift_window
        )
        qkv.register_spatial_cache(
            serialization_spatial_cache_name,
            (fwd_indices, bwd_indices, seq_lens, seq_batch_indices),
        )
    else:
        fwd_indices, bwd_indices, seq_lens, seq_batch_indices = (
            serialization_spatial_cache
        )

    M = fwd_indices.shape[0]
    T = qkv.feats.shape[0]
    H = qkv.feats.shape[2]
    C = qkv.feats.shape[3]

    qkv_feats = qkv.feats[fwd_indices]

    if DEBUG:
        start = 0
        qkv_coords = qkv.coords[fwd_indices]
        for i in range(len(seq_lens)):
            seq_coords = qkv_coords[start: start + seq_lens[i]]
            assert (
                seq_coords[:, 0] == seq_batch_indices[i]
            ).all(), (
                f"SparseWindowedScaledDotProductSelfAttention: batch index mismatch"
            )
            assert (
                seq_coords[:, 1:].max(dim=0).values
                - seq_coords[:, 1:].min(dim=0).values
                < window_size
            ).all(), (
                f"SparseWindowedScaledDotProductSelfAttention: window size exceeded"
            )
            start += seq_lens[i]

    if all([seq_len == window_size for seq_len in seq_lens]):
        B = len(seq_lens)
        N = window_size
        qkv_feats = qkv_feats.reshape(B, N, 3, H, C)
        q, k, v = qkv_feats.unbind(dim=2)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        attn_fn = optimized_attention_for_device(q.device)
        out = attn_fn(q, k, v, heads=H, skip_reshape=True, skip_output_reshape=True)
        out = out.permute(0, 2, 1, 3)
        out = out.reshape(B * N, H, C)
    else:
        q, k, v = qkv_feats.unbind(dim=1)
        cu_seqlens = (
            torch.cat(
                [torch.tensor([0]), torch.cumsum(torch.tensor(seq_lens), dim=0)],
                dim=0,
            )
            .to(qkv.device)
            .int()
        )
        out = dispatch_varlen_attention(
            q, k, v, cu_seqlens, cu_seqlens, max(seq_lens), max(seq_lens),
        )

    out = out[bwd_indices]

    if DEBUG:
        qkv_coords = qkv_coords[bwd_indices]
        assert torch.equal(
            qkv_coords, qkv.coords
        ), "SparseWindowedScaledDotProductSelfAttention: coordinate mismatch"

    return qkv.replace(out)


# ==========================================================================
# Masked SDPA (block-diagonal)
# ==========================================================================

def block_diag_attn_mask(q_seqlens, kv_seqlens, device=None, dtype=torch.float32):
    total_q = sum(q_seqlens)
    total_kv = sum(kv_seqlens)
    attn_mask = torch.full(
        (total_q, total_kv), float("-inf"), device=device, dtype=dtype
    )
    q_start = 0
    kv_start = 0
    for q_len, kv_len in zip(q_seqlens, kv_seqlens):
        attn_mask[q_start: q_start + q_len, kv_start: kv_start + kv_len] = 0
        q_start += q_len
        kv_start += kv_len
    return attn_mask


def masked_sdpa(q, k, v, q_seqlen, kv_seqlen):
    attn_mask_2d = block_diag_attn_mask(
        q_seqlen, kv_seqlen, device=q.device, dtype=q.dtype
    )
    attn_mask_4d = attn_mask_2d.unsqueeze(0).unsqueeze(0)
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)
    out = F.scaled_dot_product_attention(
        query=q,
        key=k,
        value=v,
        attn_mask=attn_mask_4d,
        dropout_p=0.0,
        is_causal=False,
    )
    out = out.permute(0, 2, 1, 3)
    return out[0]
