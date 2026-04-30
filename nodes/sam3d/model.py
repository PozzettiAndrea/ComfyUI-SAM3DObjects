# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
All flow models, transformer blocks, attention modules, embedders, norms, and FFN
for SAM3DObjects — converted to ComfyUI-native format with operations= pattern.

Key patterns:
- operations= parameter for configurable Linear/LayerNorm/GroupNorm
- sparse_operations= parameter for SparseLinear/SparseGroupNorm/SparseConv3d etc.
- WrapperExecutor + transformer_options for top-level flow models
- No torch.autocast — dtype handled by operations tier
- No convert_to_fp16/fp32 — manual_cast tier handles dtype automatically
"""
from functools import partial
from typing import *
from collections import namedtuple
from enum import Enum
from torch.utils import _pytree
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.ops
import comfy.patcher_extension

from . import ops_sparse
from .sparse import SparseTensor, sparse_batch_broadcast, sparse_batch_op
from .attention import (
    scaled_dot_product_attention,
    sparse_scaled_dot_product_attention,
    sparse_serialized_scaled_dot_product_self_attention,
    sparse_windowed_scaled_dot_product_self_attention,
    SerializeMode,
)

# Default operations tiers
ops = comfy.ops.manual_cast
sparse_ops = ops_sparse.manual_cast


# ==========================================================================
# Norms
# ==========================================================================

class LayerNorm32(ops.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)


class GroupNorm32(ops.GroupNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)


class ChannelLayerNorm32(LayerNorm32):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        DIM = x.dim()
        x = x.permute(0, *range(2, DIM), 1).contiguous()
        x = super().forward(x)
        x = x.permute(0, DIM - 1, *range(1, DIM - 1)).contiguous()
        return x


# ==========================================================================
# Spatial utilities (patchify/unpatchify)
# ==========================================================================

def pixel_shuffle_3d(x: torch.Tensor, scale_factor: int) -> torch.Tensor:
    B, C, H, W, D = x.shape
    C_ = C // scale_factor**3
    x = x.reshape(B, C_, scale_factor, scale_factor, scale_factor, H, W, D)
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4)
    x = x.reshape(B, C_, H * scale_factor, W * scale_factor, D * scale_factor)
    return x


def patchify(x: torch.Tensor, patch_size: int):
    DIM = x.dim() - 2
    for d in range(2, DIM + 2):
        assert (
            x.shape[d] % patch_size == 0
        ), f"Dimension {d} of input tensor must be divisible by patch size, got {x.shape[d]} and {patch_size}"
    x = x.reshape(
        *x.shape[:2],
        *sum([[x.shape[d] // patch_size, patch_size] for d in range(2, DIM + 2)], []),
    )
    x = x.permute(
        0, 1, *([2 * i + 3 for i in range(DIM)] + [2 * i + 2 for i in range(DIM)])
    )
    x = x.reshape(x.shape[0], x.shape[1] * (patch_size**DIM), *(x.shape[-DIM:]))
    return x


def unpatchify(x: torch.Tensor, patch_size: int):
    DIM = x.dim() - 2
    assert (
        x.shape[1] % (patch_size**DIM) == 0
    ), f"Second dimension of input tensor must be divisible by patch size to unpatchify, got {x.shape[1]} and {patch_size ** DIM}"
    x = x.reshape(
        x.shape[0],
        x.shape[1] // (patch_size**DIM),
        *([patch_size] * DIM),
        *(x.shape[-DIM:]),
    )
    x = x.permute(0, 1, *(sum([[2 + DIM + i, 2 + i] for i in range(DIM)], [])))
    x = x.reshape(
        x.shape[0], x.shape[1], *[x.shape[2 + 2 * i] * patch_size for i in range(DIM)]
    )
    return x


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


# ==========================================================================
# Position embedders (no learnable Linear layers)
# ==========================================================================

class AbsolutePositionEmbedder(nn.Module):
    def __init__(self, channels: int, in_channels: int = 3):
        super().__init__()
        self.channels = channels
        self.in_channels = in_channels
        self.freq_dim = channels // in_channels // 2

    def _sin_cos_embedding(self, x: torch.Tensor) -> torch.Tensor:
        freqs = torch.arange(self.freq_dim, dtype=torch.float32, device=x.device) / self.freq_dim
        freqs = 1.0 / (10000**freqs)
        out = torch.outer(x, freqs)
        out = torch.cat([torch.sin(out), torch.cos(out)], dim=-1)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, D = x.shape
        assert D == self.in_channels, "Input dimension must match number of input channels"
        embed = self._sin_cos_embedding(x.reshape(-1))
        embed = embed.reshape(N, -1)
        if embed.shape[1] < self.channels:
            embed = torch.cat(
                [embed, torch.zeros(N, self.channels - embed.shape[1], device=embed.device)],
                dim=-1,
            )
        return embed


class RotaryPositionEmbedder(nn.Module):
    def __init__(self, hidden_size: int, in_channels: int = 3):
        super().__init__()
        assert hidden_size % 2 == 0, "Hidden size must be divisible by 2"
        self.hidden_size = hidden_size
        self.in_channels = in_channels
        self.freq_dim = hidden_size // in_channels // 2

    def _get_phases(self, indices: torch.Tensor) -> torch.Tensor:
        freqs = torch.arange(self.freq_dim, dtype=torch.float32, device=indices.device) / self.freq_dim
        freqs = 1.0 / (10000**freqs)
        phases = torch.outer(indices, freqs)
        phases = torch.polar(torch.ones_like(phases), phases)
        return phases

    def _rotary_embedding(self, x: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        x_rotated = x_complex * phases
        x_embed = torch.view_as_real(x_rotated).reshape(*x_rotated.shape[:-1], -1).to(x.dtype)
        return x_embed

    def forward(self, q, k, indices=None):
        if indices is None:
            indices = torch.arange(q.shape[-2], device=q.device)
            if len(q.shape) > 2:
                indices = indices.unsqueeze(0).expand(q.shape[:-2] + (-1,))
        phases = self._get_phases(indices.reshape(-1)).reshape(*indices.shape[:-1], -1)
        if phases.shape[1] < self.hidden_size // 2:
            phases = torch.cat(
                [
                    phases,
                    torch.polar(
                        torch.ones(*phases.shape[:-1], self.hidden_size // 2 - phases.shape[1], device=phases.device),
                        torch.zeros(*phases.shape[:-1], self.hidden_size // 2 - phases.shape[1], device=phases.device),
                    ),
                ],
                dim=-1,
            )
        q_embed = self._rotary_embedding(q, phases)
        k_embed = self._rotary_embedding(k, phases)
        return q_embed, k_embed


# ==========================================================================
# Timestep embedder
# ==========================================================================

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256, freeze=False,
                 dtype=None, device=None, operations=ops):
        super().__init__()
        self.mlp = nn.Sequential(
            operations.Linear(frequency_embedding_size, hidden_size, bias=True, dtype=dtype, device=device),
            nn.SiLU(),
            operations.Linear(hidden_size, hidden_size, bias=True, dtype=dtype, device=device),
        )
        self.frequency_embedding_size = frequency_embedding_size
        if freeze:
            self.requires_grad_(False)
            self.eval()

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        orig_dtype = t.dtype
        t = t[:, None].float()
        args = t * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        if torch.is_floating_point(embedding):
            embedding = embedding.to(orig_dtype)
        return embedding

    def forward(self, t):
        if t.ndim == 0:
            t = t.unsqueeze(0)
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


# ==========================================================================
# Dense attention modules
# ==========================================================================

class MultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (F.normalize(x, dim=-1) * self.gamma.to(x.device) * self.scale).to(x.dtype)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int] = None,
        type: Literal["self", "cross"] = "self",
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        qkv_bias: bool = True,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        dtype=None, device=None, operations=ops,
    ):
        super().__init__()
        assert channels % num_heads == 0
        assert type in ["self", "cross"]
        assert attn_mode in ["full", "windowed"]
        assert type == "self" or attn_mode == "full", "Cross-attention only supports full attention"

        if attn_mode == "windowed":
            raise NotImplementedError("Windowed attention is not yet implemented")

        self.channels = channels
        self.head_dim = channels // num_heads
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self.num_heads = num_heads
        self._type = type
        self.attn_mode = attn_mode
        self.window_size = window_size
        self.shift_window = shift_window
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm

        if self._type == "self":
            self.to_qkv = operations.Linear(channels, channels * 3, bias=qkv_bias, dtype=dtype, device=device)
        else:
            self.to_q = operations.Linear(channels, channels, bias=qkv_bias, dtype=dtype, device=device)
            self.to_kv = operations.Linear(self.ctx_channels, channels * 2, bias=qkv_bias, dtype=dtype, device=device)

        if self.qk_rms_norm:
            self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)

        self.to_out = operations.Linear(channels, channels, dtype=dtype, device=device)

        if use_rope:
            self.rope = RotaryPositionEmbedder(channels)

    def forward(self, x, context=None, indices=None, transformer_options={}):
        B, L, C = x.shape
        if self._type == "self":
            qkv = self.to_qkv(x)
            qkv = qkv.reshape(B, L, 3, self.num_heads, -1)
            if self.use_rope:
                q, k, v = qkv.unbind(dim=2)
                q, k = self.rope(q, k, indices)
                qkv = torch.stack([q, k, v], dim=2)
            if self.attn_mode == "full":
                if self.qk_rms_norm:
                    q, k, v = qkv.unbind(dim=2)
                    q = self.q_rms_norm(q)
                    k = self.k_rms_norm(k)
                    h = scaled_dot_product_attention(q, k, v)
                else:
                    h = scaled_dot_product_attention(qkv)
        else:
            Lkv = context.shape[1]
            q = self.to_q(x)
            kv = self.to_kv(context)
            q = q.reshape(B, L, self.num_heads, -1)
            kv = kv.reshape(B, Lkv, 2, self.num_heads, -1)
            if self.qk_rms_norm:
                q = self.q_rms_norm(q)
                k, v = kv.unbind(dim=2)
                k = self.k_rms_norm(k)
                h = scaled_dot_product_attention(q, k, v)
            else:
                h = scaled_dot_product_attention(q, kv)
        h = h.reshape(B, L, -1)
        h = self.to_out(h)
        return h


class MOTMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int] = None,
        type: Literal["self", "cross"] = "self",
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        qkv_bias: bool = True,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        latent_names: List = None,
        protect_modality_list: List = ["shape"],
        dtype=None, device=None, operations=ops,
    ):
        super().__init__()
        assert channels % num_heads == 0
        assert type in ["self", "cross"]
        assert attn_mode in ["full", "windowed"]
        assert type == "self" or attn_mode == "full"

        self.channels = channels
        self.head_dim = channels // num_heads
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self.num_heads = num_heads
        self._type = type
        self.attn_mode = attn_mode
        self.window_size = window_size
        self.shift_window = shift_window
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm
        self.protect_modality_list = protect_modality_list

        if self._type == "self":
            self.to_qkv = torch.nn.ModuleDict(
                {ln: operations.Linear(channels, channels * 3, bias=qkv_bias, dtype=dtype, device=device) for ln in latent_names}
            )
        else:
            self.to_q = torch.nn.ModuleDict(
                {ln: operations.Linear(channels, channels, bias=qkv_bias, dtype=dtype, device=device) for ln in latent_names}
            )
            self.to_kv = torch.nn.ModuleDict(
                {ln: operations.Linear(self.ctx_channels, channels * 2, bias=qkv_bias, dtype=dtype, device=device) for ln in latent_names}
            )

        if self.qk_rms_norm:
            self.q_rms_norm = torch.nn.ModuleDict({ln: MultiHeadRMSNorm(self.head_dim, num_heads) for ln in latent_names})
            self.k_rms_norm = torch.nn.ModuleDict({ln: MultiHeadRMSNorm(self.head_dim, num_heads) for ln in latent_names})

        self.to_out = torch.nn.ModuleDict(
            {ln: operations.Linear(channels, channels, dtype=dtype, device=device) for ln in latent_names}
        )

        if use_rope:
            self.rope = RotaryPositionEmbedder(channels)

    def _reshape(self, qkv, tensor_shape, num_heads):
        B, L, _ = tensor_shape
        return qkv.reshape(B, L, 3, num_heads, -1)

    def _reshape_back(self, qkv, tensor_shape):
        B, L, _ = tensor_shape
        return qkv.reshape(B, L, -1)

    def _apply_module(self, x, module):
        return module(x)

    def _moduledict_to_dict(self, module):
        return {key: module for key, module in module.items()}

    def unbind_qkv(self, qkv):
        q, k, v = {}, {}, {}
        for latent_name, _qkv in qkv.items():
            _q, _k, _v = _qkv.unbind(dim=2)
            q[latent_name] = _q
            k[latent_name] = _k
            v[latent_name] = _v
        return q, k, v

    def _get_shape(self, x):
        return x.shape

    def concatenate_tensor(self, tensor_dict, latent_names):
        merged = []
        indicies_mapping = {}
        total_tokens = 0
        for latent_name in latent_names:
            merged.append(tensor_dict[latent_name])
            cur_token_len = tensor_dict[latent_name].shape[1]
            indicies_mapping[latent_name] = [total_tokens, cur_token_len]
            total_tokens += cur_token_len
        return torch.cat(merged, dim=1), indicies_mapping

    def unpack_tensors(self, h_others, indicies_mapping):
        h = {}
        for latent_name, (start, cur_token_len) in indicies_mapping.items():
            h[latent_name] = h_others[:, start: start + cur_token_len]
        return h

    def mm_scale_dot_product_attention(self, q, k, v):
        h = {}
        latent_names = list(q.keys())
        for protect_modality in self.protect_modality_list:
            _q = q[protect_modality]
            _k = k[protect_modality]
            _v = v[protect_modality]
            h[protect_modality] = scaled_dot_product_attention(_q, _k, _v)

        other_modalities = [n for n in latent_names if n not in self.protect_modality_list]
        _q, indicies_mapping = self.concatenate_tensor(q, other_modalities)
        o_k, _ = self.concatenate_tensor(k, other_modalities)
        o_v, _ = self.concatenate_tensor(v, other_modalities)
        _k, _ = self.concatenate_tensor(k, self.protect_modality_list)
        _v, _ = self.concatenate_tensor(v, self.protect_modality_list)
        _k = _k.detach()
        _v = _v.detach()
        _k = torch.cat([o_k, _k], dim=1)
        _v = torch.cat([o_v, _v], dim=1)
        h_others = scaled_dot_product_attention(_q, _k, _v)
        h.update(self.unpack_tensors(h_others, indicies_mapping))
        return h

    def forward(self, x: Dict, transformer_options={}):
        shapes = _pytree.tree_map(self._get_shape, x)
        if self._type == "self":
            qkv = _pytree.tree_map(self._apply_module, x, self._moduledict_to_dict(self.to_qkv))
            qkv = _pytree.tree_map(partial(self._reshape, num_heads=self.num_heads), qkv, shapes)
            if self.attn_mode == "full":
                if self.qk_rms_norm:
                    q, k, v = self.unbind_qkv(qkv)
                    q = _pytree.tree_map(self._apply_module, q, self._moduledict_to_dict(self.q_rms_norm))
                    k = _pytree.tree_map(self._apply_module, k, self._moduledict_to_dict(self.k_rms_norm))
                    h = self.mm_scale_dot_product_attention(q, k, v)
                else:
                    raise NotImplementedError
        else:
            raise NotImplementedError

        h = _pytree.tree_map(self._reshape_back, h, shapes)
        h = _pytree.tree_map(self._apply_module, h, self._moduledict_to_dict(self.to_out))
        return h


# ==========================================================================
# Sparse attention modules
# ==========================================================================

class SparseMultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x):
        x_type = x.dtype
        x = x.float()
        ref_device = x.feats.device if isinstance(x, SparseTensor) else x.device
        gamma = self.gamma.to(ref_device)
        if isinstance(x, SparseTensor):
            x = x.replace(F.normalize(x.feats, dim=-1))
        else:
            x = F.normalize(x, dim=-1)
        return (x * gamma * self.scale).to(x_type)


class SparseMultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int] = None,
        type: Literal["self", "cross"] = "self",
        attn_mode: Literal["full", "serialized", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_sequence: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        serialize_mode: Optional[SerializeMode] = None,
        qkv_bias: bool = True,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        dtype=None, device=None, operations=ops, sparse_operations=sparse_ops,
    ):
        super().__init__()
        assert channels % num_heads == 0
        assert type in ["self", "cross"]
        assert attn_mode in ["full", "serialized", "windowed"]
        assert type == "self" or attn_mode == "full"
        assert type == "self" or use_rope is False

        self.channels = channels
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self.num_heads = num_heads
        self._type = type
        self.attn_mode = attn_mode
        self.window_size = window_size
        self.shift_sequence = shift_sequence
        self.shift_window = shift_window
        self.serialize_mode = serialize_mode
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm

        # Use sparse_operations for projections that operate on SparseTensor
        if self._type == "self":
            self.to_qkv = sparse_operations.SparseLinear(channels, channels * 3, bias=qkv_bias, dtype=dtype, device=device)
        else:
            self.to_q = sparse_operations.SparseLinear(channels, channels, bias=qkv_bias, dtype=dtype, device=device)
            self.to_kv = sparse_operations.SparseLinear(self.ctx_channels, channels * 2, bias=qkv_bias, dtype=dtype, device=device)

        if self.qk_rms_norm:
            self.q_rms_norm = SparseMultiHeadRMSNorm(channels // num_heads, num_heads)
            self.k_rms_norm = SparseMultiHeadRMSNorm(channels // num_heads, num_heads)

        self.to_out = sparse_operations.SparseLinear(channels, channels, dtype=dtype, device=device)

        if use_rope:
            self.rope = RotaryPositionEmbedder(channels)

    @staticmethod
    def _linear(module, x):
        if isinstance(x, SparseTensor):
            return module(x)
        else:
            return module(x)

    @staticmethod
    def _reshape_chs(x, shape):
        if isinstance(x, SparseTensor):
            return x.reshape(*shape)
        else:
            return x.reshape(*x.shape[:2], *shape)

    def _fused_pre(self, x, num_fused):
        if isinstance(x, SparseTensor):
            x_feats = x.feats.unsqueeze(0)
        else:
            x_feats = x
        x_feats = x_feats.reshape(*x_feats.shape[:2], num_fused, self.num_heads, -1)
        return x.replace(x_feats.squeeze(0)) if isinstance(x, SparseTensor) else x_feats

    def _rope(self, qkv):
        q, k, v = qkv.feats.unbind(dim=1)
        q, k = self.rope(q, k, qkv.coords[:, 1:])
        qkv = qkv.replace(torch.stack([q, k, v], dim=1))
        return qkv

    def forward(self, x, context=None, transformer_options={}):
        if self._type == "self":
            qkv = self._linear(self.to_qkv, x)
            qkv = self._fused_pre(qkv, num_fused=3)
            if self.use_rope:
                qkv = self._rope(qkv)
            if self.qk_rms_norm:
                q, k, v = qkv.unbind(dim=1)
                q = self.q_rms_norm(q)
                k = self.k_rms_norm(k)
                qkv = qkv.replace(torch.stack([q.feats, k.feats, v.feats], dim=1))
            if self.attn_mode == "full":
                h = sparse_scaled_dot_product_attention(qkv)
            elif self.attn_mode == "serialized":
                h = sparse_serialized_scaled_dot_product_self_attention(
                    qkv, self.window_size,
                    serialize_mode=self.serialize_mode,
                    shift_sequence=self.shift_sequence,
                    shift_window=self.shift_window,
                )
            elif self.attn_mode == "windowed":
                h = sparse_windowed_scaled_dot_product_self_attention(
                    qkv, self.window_size, shift_window=self.shift_window
                )
        else:
            q = self._linear(self.to_q, x)
            q = self._reshape_chs(q, (self.num_heads, -1))
            kv = self._linear(self.to_kv, context)
            kv = self._fused_pre(kv, num_fused=2)
            if self.qk_rms_norm:
                q = self.q_rms_norm(q)
                k, v = kv.unbind(dim=1)
                k = self.k_rms_norm(k)
                kv = kv.replace(torch.stack([k.feats, v.feats], dim=1))
            h = sparse_scaled_dot_product_attention(q, kv)
        h = self._reshape_chs(h, (-1,))
        h = self._linear(self.to_out, h)
        return h


# ==========================================================================
# FFN layers
# ==========================================================================

class FeedForwardNet(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 4.0,
                 dtype=None, device=None, operations=ops):
        super().__init__()
        self.mlp = nn.Sequential(
            operations.Linear(channels, int(channels * mlp_ratio), dtype=dtype, device=device),
            nn.GELU(approximate="tanh"),
            operations.Linear(int(channels * mlp_ratio), channels, dtype=dtype, device=device),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class FeedForward(nn.Module):
    """Llama3 FeedForward layer with SiLU gating."""
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int = 256,
                 ffn_dim_multiplier: Optional[float] = None,
                 output_dim: Optional[int] = None, skip_w2: bool = False,
                 dtype=None, device=None, operations=ops):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        if output_dim is None:
            output_dim = dim
        self.skip_w2 = skip_w2
        self.w1 = operations.Linear(dim, hidden_dim, bias=False, dtype=dtype, device=device)
        if not self.skip_w2:
            self.w2 = operations.Linear(hidden_dim, output_dim, bias=False, dtype=dtype, device=device)
        self.w3 = operations.Linear(dim, hidden_dim, bias=False, dtype=dtype, device=device)

    def forward(self, x):
        x = F.silu(self.w1(x)) * self.w3(x)
        if self.skip_w2:
            return x
        return self.w2(x)


class SparseFeedForwardNet(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 4.0,
                 dtype=None, device=None, sparse_operations=sparse_ops):
        super().__init__()
        self.mlp = nn.Sequential(
            sparse_operations.SparseLinear(channels, int(channels * mlp_ratio), dtype=dtype, device=device),
            sparse_operations.SparseGELU(approximate="tanh"),
            sparse_operations.SparseLinear(int(channels * mlp_ratio), channels, dtype=dtype, device=device),
        )

    def forward(self, x: SparseTensor) -> SparseTensor:
        return self.mlp(x)


# ==========================================================================
# Dense transformer blocks
# ==========================================================================

class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, mlp_ratio=4.0,
                 attn_mode="full", window_size=None, shift_window=None,
                 use_checkpoint=False, use_rope=False, qk_rms_norm=False,
                 qkv_bias=True, ln_affine=False,
                 dtype=None, device=None, operations=ops):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6)
        self.attn = MultiHeadAttention(
            channels, num_heads=num_heads, attn_mode=attn_mode,
            window_size=window_size, shift_window=shift_window,
            qkv_bias=qkv_bias, use_rope=use_rope, qk_rms_norm=qk_rms_norm,
            dtype=dtype, device=device, operations=operations,
        )
        self.mlp = FeedForwardNet(channels, mlp_ratio=mlp_ratio,
                                  dtype=dtype, device=device, operations=operations)

    def _forward(self, x, transformer_options={}):
        h = self.norm1(x)
        h = self.attn(h, transformer_options=transformer_options)
        x = x + h
        h = self.norm2(x)
        h = self.mlp(h)
        x = x + h
        return x

    def forward(self, x, transformer_options={}):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, transformer_options, use_reentrant=False)
        return self._forward(x, transformer_options)


class TransformerCrossBlock(nn.Module):
    def __init__(self, channels, ctx_channels, num_heads, mlp_ratio=4.0,
                 attn_mode="full", window_size=None, shift_window=None,
                 use_checkpoint=False, use_rope=False, qk_rms_norm=False,
                 qk_rms_norm_cross=False, qkv_bias=True, ln_affine=False,
                 dtype=None, device=None, operations=ops):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6)
        self.self_attn = MultiHeadAttention(
            channels, num_heads=num_heads, type="self", attn_mode=attn_mode,
            window_size=window_size, shift_window=shift_window,
            qkv_bias=qkv_bias, use_rope=use_rope, qk_rms_norm=qk_rms_norm,
            dtype=dtype, device=device, operations=operations,
        )
        self.cross_attn = MultiHeadAttention(
            channels, ctx_channels=ctx_channels, num_heads=num_heads,
            type="cross", attn_mode="full", qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
            dtype=dtype, device=device, operations=operations,
        )
        self.mlp = FeedForwardNet(channels, mlp_ratio=mlp_ratio,
                                  dtype=dtype, device=device, operations=operations)

    def _forward(self, x, context, transformer_options={}):
        h = self.norm1(x)
        h = self.self_attn(h, transformer_options=transformer_options)
        x = x + h
        h = self.norm2(x)
        h = self.cross_attn(h, context, transformer_options=transformer_options)
        x = x + h
        h = self.norm3(x)
        h = self.mlp(h)
        x = x + h
        return x

    def forward(self, x, context, transformer_options={}):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, context, transformer_options, use_reentrant=False)
        return self._forward(x, context, transformer_options)


class ModulatedTransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, mlp_ratio=4.0,
                 attn_mode="full", window_size=None, shift_window=None,
                 use_checkpoint=False, use_rope=False, qk_rms_norm=False,
                 qkv_bias=True, share_mod=False,
                 dtype=None, device=None, operations=ops):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.attn = MultiHeadAttention(
            channels, num_heads=num_heads, attn_mode=attn_mode,
            window_size=window_size, shift_window=shift_window,
            qkv_bias=qkv_bias, use_rope=use_rope, qk_rms_norm=qk_rms_norm,
            dtype=dtype, device=device, operations=operations,
        )
        self.mlp = FeedForwardNet(channels, mlp_ratio=mlp_ratio,
                                  dtype=dtype, device=device, operations=operations)
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), operations.Linear(channels, 6 * channels, bias=True, dtype=dtype, device=device)
            )

    def _forward(self, x, mod, transformer_options={}):
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h = self.attn(h, transformer_options=transformer_options)
        h = h * gate_msa.unsqueeze(1)
        x = x + h
        h = self.norm2(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.mlp(h)
        h = h * gate_mlp.unsqueeze(1)
        x = x + h
        return x

    def forward(self, x, mod, transformer_options={}):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, transformer_options, use_reentrant=False)
        return self._forward(x, mod, transformer_options)


class ModulatedTransformerCrossBlock(nn.Module):
    def __init__(self, channels, ctx_channels, num_heads, mlp_ratio=4.0,
                 attn_mode="full", window_size=None, shift_window=None,
                 use_checkpoint=False, use_rope=False, qk_rms_norm=False,
                 qk_rms_norm_cross=False, qkv_bias=True, share_mod=False,
                 dtype=None, device=None, operations=ops):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.self_attn = MultiHeadAttention(
            channels, num_heads=num_heads, type="self", attn_mode=attn_mode,
            window_size=window_size, shift_window=shift_window,
            qkv_bias=qkv_bias, use_rope=use_rope, qk_rms_norm=qk_rms_norm,
            dtype=dtype, device=device, operations=operations,
        )
        self.cross_attn = MultiHeadAttention(
            channels, ctx_channels=ctx_channels, num_heads=num_heads,
            type="cross", attn_mode="full", qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
            dtype=dtype, device=device, operations=operations,
        )
        self.mlp = FeedForwardNet(channels, mlp_ratio=mlp_ratio,
                                  dtype=dtype, device=device, operations=operations)
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), operations.Linear(channels, 6 * channels, bias=True, dtype=dtype, device=device)
            )

    def _forward(self, x, mod, context, transformer_options={}):
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h = self.self_attn(h, transformer_options=transformer_options)
        h = h * gate_msa.unsqueeze(1)
        x = x + h
        h = self.norm2(x)
        h = self.cross_attn(h, context, transformer_options=transformer_options)
        x = x + h
        h = self.norm3(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.mlp(h)
        h = h * gate_mlp.unsqueeze(1)
        x = x + h
        return x

    def forward(self, x, mod, context, transformer_options={}):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, context, transformer_options, use_reentrant=False)
        return self._forward(x, mod, context, transformer_options)


class MOTModulatedTransformerCrossBlock(nn.Module):
    def __init__(self, channels, ctx_channels, num_heads, mlp_ratio=4.0,
                 attn_mode="full", window_size=None, shift_window=None,
                 use_checkpoint=False, use_rope=False, qk_rms_norm=False,
                 qk_rms_norm_cross=False, qkv_bias=True, share_mod=False,
                 latent_names=None, freeze_shared_parameters=False,
                 dtype=None, device=None, operations=ops):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.norm1 = torch.nn.ModuleDict({ln: LayerNorm32(channels, elementwise_affine=False, eps=1e-6) for ln in latent_names})
        self.norm2 = torch.nn.ModuleDict({ln: LayerNorm32(channels, elementwise_affine=True, eps=1e-6) for ln in latent_names})
        self.norm3 = torch.nn.ModuleDict({ln: LayerNorm32(channels, elementwise_affine=False, eps=1e-6) for ln in latent_names})
        self.self_attn = MOTMultiHeadSelfAttention(
            channels, num_heads=num_heads, type="self", attn_mode=attn_mode,
            window_size=window_size, shift_window=shift_window,
            qkv_bias=qkv_bias, use_rope=use_rope, qk_rms_norm=qk_rms_norm,
            latent_names=latent_names,
            dtype=dtype, device=device, operations=operations,
        )
        self.cross_attn = torch.nn.ModuleDict({
            ln: MultiHeadAttention(
                channels, ctx_channels=ctx_channels, num_heads=num_heads,
                type="cross", attn_mode="full", qkv_bias=qkv_bias,
                qk_rms_norm=qk_rms_norm_cross,
                dtype=dtype, device=device, operations=operations,
            ) for ln in latent_names
        })
        self.mlp = torch.nn.ModuleDict({
            ln: FeedForwardNet(channels, mlp_ratio=mlp_ratio,
                               dtype=dtype, device=device, operations=operations)
            for ln in latent_names
        })
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), operations.Linear(channels, 6 * channels, bias=True, dtype=dtype, device=device)
            )
            if freeze_shared_parameters:
                self.adaLN_modulation.eval()
                self.adaLN_modulation.requires_grad_(False)

    def _apply_module(self, h, module):
        return module(h)

    def _apply_cross_attn(self, h, cross_attn, context):
        return cross_attn(h, context)

    def _apply_msa(self, h, scale_msa, shift_msa):
        return h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)

    def _apply_mlp(self, h, scale_mlp, shift_mlp):
        return h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)

    def _apply_add(self, x, h):
        return x + h

    def _apply_multiplication(self, h, multiplier):
        return h * multiplier.unsqueeze(1)

    def _moduledict_to_dict(self, module):
        return {key: module for key, module in module.items()}

    def _forward(self, x, mod, context, transformer_options={}):
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        h = _pytree.tree_map(self._apply_module, x, self._moduledict_to_dict(self.norm1))
        h = _pytree.tree_map(partial(self._apply_msa, scale_msa=scale_msa, shift_msa=shift_msa), h)
        h = self.self_attn(h, transformer_options=transformer_options)
        h = _pytree.tree_map(partial(self._apply_multiplication, multiplier=gate_msa), h)
        x = _pytree.tree_map(self._apply_add, x, h)
        h = _pytree.tree_map(self._apply_module, x, self._moduledict_to_dict(self.norm2))
        h = _pytree.tree_map(partial(self._apply_cross_attn, context=context), h, self._moduledict_to_dict(self.cross_attn))
        x = _pytree.tree_map(self._apply_add, x, h)
        h = _pytree.tree_map(self._apply_module, x, self._moduledict_to_dict(self.norm3))
        h = _pytree.tree_map(partial(self._apply_mlp, scale_mlp=scale_mlp, shift_mlp=shift_mlp), h)
        h = _pytree.tree_map(self._apply_module, h, self._moduledict_to_dict(self.mlp))
        h = _pytree.tree_map(partial(self._apply_multiplication, multiplier=gate_mlp), h)
        x = _pytree.tree_map(self._apply_add, x, h)
        return x

    def forward(self, x, mod, context, transformer_options={}):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, context, transformer_options, use_reentrant=False)
        return self._forward(x, mod, context, transformer_options)


# ==========================================================================
# Sparse transformer blocks
# ==========================================================================

class SparseTransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, mlp_ratio=4.0,
                 attn_mode="full", window_size=None, shift_sequence=None,
                 shift_window=None, serialize_mode=None,
                 use_checkpoint=False, use_rope=False, qk_rms_norm=False,
                 qkv_bias=True, ln_affine=False,
                 dtype=None, device=None, operations=ops, sparse_operations=sparse_ops):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6)
        self.attn = SparseMultiHeadAttention(
            channels, num_heads=num_heads, attn_mode=attn_mode,
            window_size=window_size, shift_sequence=shift_sequence,
            shift_window=shift_window, serialize_mode=serialize_mode,
            qkv_bias=qkv_bias, use_rope=use_rope, qk_rms_norm=qk_rms_norm,
            dtype=dtype, device=device, operations=operations, sparse_operations=sparse_operations,
        )
        self.mlp = SparseFeedForwardNet(channels, mlp_ratio=mlp_ratio,
                                        dtype=dtype, device=device, sparse_operations=sparse_operations)

    def _forward(self, x, transformer_options={}):
        h = x.replace(self.norm1(x.feats))
        h = self.attn(h, transformer_options=transformer_options)
        x = x + h
        h = x.replace(self.norm2(x.feats))
        h = self.mlp(h)
        x = x + h
        return x

    def forward(self, x, transformer_options={}):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, transformer_options, use_reentrant=False)
        return self._forward(x, transformer_options)


class SparseTransformerCrossBlock(nn.Module):
    def __init__(self, channels, ctx_channels, num_heads, mlp_ratio=4.0,
                 attn_mode="full", window_size=None, shift_sequence=None,
                 shift_window=None, serialize_mode=None,
                 use_checkpoint=False, use_rope=False, qk_rms_norm=False,
                 qk_rms_norm_cross=False, qkv_bias=True, ln_affine=False,
                 dtype=None, device=None, operations=ops, sparse_operations=sparse_ops):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6)
        self.self_attn = SparseMultiHeadAttention(
            channels, num_heads=num_heads, type="self", attn_mode=attn_mode,
            window_size=window_size, shift_sequence=shift_sequence,
            shift_window=shift_window, serialize_mode=serialize_mode,
            qkv_bias=qkv_bias, use_rope=use_rope, qk_rms_norm=qk_rms_norm,
            dtype=dtype, device=device, operations=operations, sparse_operations=sparse_operations,
        )
        self.cross_attn = SparseMultiHeadAttention(
            channels, ctx_channels=ctx_channels, num_heads=num_heads,
            type="cross", attn_mode="full", qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
            dtype=dtype, device=device, operations=operations, sparse_operations=sparse_operations,
        )
        self.mlp = SparseFeedForwardNet(channels, mlp_ratio=mlp_ratio,
                                        dtype=dtype, device=device, sparse_operations=sparse_operations)

    def _forward(self, x, context, transformer_options={}):
        h = x.replace(self.norm1(x.feats))
        h = self.self_attn(h, transformer_options=transformer_options)
        x = x + h
        h = x.replace(self.norm2(x.feats))
        h = self.cross_attn(h, context, transformer_options=transformer_options)
        x = x + h
        h = x.replace(self.norm3(x.feats))
        h = self.mlp(h)
        x = x + h
        return x

    def forward(self, x, context, transformer_options={}):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, context, transformer_options, use_reentrant=False)
        return self._forward(x, context, transformer_options)


class ModulatedSparseTransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, mlp_ratio=4.0,
                 attn_mode="full", window_size=None, shift_sequence=None,
                 shift_window=None, serialize_mode=None,
                 use_checkpoint=False, use_rope=False, qk_rms_norm=False,
                 qkv_bias=True, share_mod=False,
                 dtype=None, device=None, operations=ops, sparse_operations=sparse_ops):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.attn = SparseMultiHeadAttention(
            channels, num_heads=num_heads, attn_mode=attn_mode,
            window_size=window_size, shift_sequence=shift_sequence,
            shift_window=shift_window, serialize_mode=serialize_mode,
            qkv_bias=qkv_bias, use_rope=use_rope, qk_rms_norm=qk_rms_norm,
            dtype=dtype, device=device, operations=operations, sparse_operations=sparse_operations,
        )
        self.mlp = SparseFeedForwardNet(channels, mlp_ratio=mlp_ratio,
                                        dtype=dtype, device=device, sparse_operations=sparse_operations)
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), operations.Linear(channels, 6 * channels, bias=True, dtype=dtype, device=device)
            )

    def _forward(self, x, mod, transformer_options={}):
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        h = x.replace(self.norm1(x.feats))
        h = h * (1 + scale_msa) + shift_msa
        h = self.attn(h, transformer_options=transformer_options)
        h = h * gate_msa
        x = x + h
        h = x.replace(self.norm2(x.feats))
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        h = h * gate_mlp
        x = x + h
        return x

    def forward(self, x, mod, transformer_options={}):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, transformer_options, use_reentrant=False)
        return self._forward(x, mod, transformer_options)


class ModulatedSparseTransformerCrossBlock(nn.Module):
    def __init__(self, channels, ctx_channels, num_heads, mlp_ratio=4.0,
                 attn_mode="full", window_size=None, shift_sequence=None,
                 shift_window=None, serialize_mode=None,
                 use_checkpoint=False, use_rope=False, qk_rms_norm=False,
                 qk_rms_norm_cross=False, qkv_bias=True, share_mod=False,
                 dtype=None, device=None, operations=ops, sparse_operations=sparse_ops):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.self_attn = SparseMultiHeadAttention(
            channels, num_heads=num_heads, type="self", attn_mode=attn_mode,
            window_size=window_size, shift_sequence=shift_sequence,
            shift_window=shift_window, serialize_mode=serialize_mode,
            qkv_bias=qkv_bias, use_rope=use_rope, qk_rms_norm=qk_rms_norm,
            dtype=dtype, device=device, operations=operations, sparse_operations=sparse_operations,
        )
        self.cross_attn = SparseMultiHeadAttention(
            channels, ctx_channels=ctx_channels, num_heads=num_heads,
            type="cross", attn_mode="full", qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
            dtype=dtype, device=device, operations=operations, sparse_operations=sparse_operations,
        )
        self.mlp = SparseFeedForwardNet(channels, mlp_ratio=mlp_ratio,
                                        dtype=dtype, device=device, sparse_operations=sparse_operations)
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), operations.Linear(channels, 6 * channels, bias=True, dtype=dtype, device=device)
            )

    def _forward(self, x, mod, context, transformer_options={}):
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        h = x.replace(self.norm1(x.feats))
        h = h * (1 + scale_msa) + shift_msa
        h = self.self_attn(h, transformer_options=transformer_options)
        h = h * gate_msa
        x = x + h
        h = x.replace(self.norm2(x.feats))
        h = self.cross_attn(h, context, transformer_options=transformer_options)
        x = x + h
        h = x.replace(self.norm3(x.feats))
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        h = h * gate_mlp
        x = x + h
        return x

    def forward(self, x, mod, context, transformer_options={}):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, context, transformer_options, use_reentrant=False)
        return self._forward(x, mod, context, transformer_options)


# ==========================================================================
# Sparse ResBlock for SLatFlowModel
# ==========================================================================

class SparseResBlock3d(nn.Module):
    def __init__(self, channels, emb_channels, out_channels=None,
                 downsample=False, upsample=False,
                 dtype=None, device=None, operations=ops, sparse_operations=sparse_ops):
        super().__init__()
        from .sparse import SparseDownsample, SparseUpsample
        self.channels = channels
        self.emb_channels = emb_channels
        self.out_channels = out_channels or channels
        self.downsample = downsample
        self.upsample = upsample
        assert not (downsample and upsample)

        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6)
        self.conv1 = sparse_operations.SparseConv3d(channels, self.out_channels, 3, dtype=dtype, device=device)
        self.conv2 = zero_module(sparse_operations.SparseConv3d(self.out_channels, self.out_channels, 3, dtype=dtype, device=device))
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            operations.Linear(emb_channels, 2 * self.out_channels, bias=True, dtype=dtype, device=device),
        )
        self.skip_connection = (
            sparse_operations.SparseLinear(channels, self.out_channels, dtype=dtype, device=device)
            if channels != self.out_channels
            else nn.Identity()
        )
        self.updown = None
        if self.downsample:
            self.updown = SparseDownsample(2)
        elif self.upsample:
            self.updown = SparseUpsample(2)

    def _updown(self, x):
        if self.updown is not None:
            x = self.updown(x)
        return x

    def forward(self, x, emb):
        emb_out = self.emb_layers(emb).type(x.dtype)
        scale, shift = torch.chunk(emb_out, 2, dim=1)

        x = self._updown(x)
        h = x.replace(self.norm1(x.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv1(h)
        h = h.replace(self.norm2(h.feats)) * (1 + scale) + shift
        h = h.replace(F.silu(h.feats))
        h = self.conv2(h)
        h = h + self.skip_connection(x)
        return h


# ==========================================================================
# Top-level flow models with WrapperExecutor + transformer_options
# ==========================================================================

DataType = namedtuple("DataType", ["shape", "pose"])


class SparseStructureFlowModel(nn.Module):
    """Dense-space sparse structure flow model with WrapperExecutor support."""

    def __init__(self, resolution, in_channels, model_channels, cond_channels,
                 out_channels, num_blocks, num_heads=None, num_head_channels=64,
                 mlp_ratio=4, patch_size=2, pe_mode="ape",
                 use_fp16=False, use_checkpoint=False, share_mod=False,
                 qk_rms_norm=False, qk_rms_norm_cross=False,
                 include_pose=False, pose_weight=1.0,
                 dtype=None, device=None, operations=ops):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size
        self.pe_mode = pe_mode
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.include_pose = include_pose
        self.pose_weight = pose_weight

        self.t_embedder = TimestepEmbedder(model_channels, dtype=dtype, device=device, operations=operations)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), operations.Linear(model_channels, 6 * model_channels, bias=True, dtype=dtype, device=device)
            )

        if pe_mode == "ape":
            pos_embedder = AbsolutePositionEmbedder(model_channels, 3)
            coords = torch.meshgrid(
                *[torch.arange(resolution // patch_size) for _ in range(3)],
                indexing="ij",
            )
            coords = torch.stack(coords, dim=-1).reshape(-1, 3)
            pos_emb = pos_embedder(coords)
            if include_pose:
                pose_pos_emb = torch.ones(1, model_channels) * 0.5
                pos_emb = torch.cat([pos_emb, pose_pos_emb], dim=0)
            self.register_buffer("pos_emb", pos_emb)

        self.input_layer = operations.Linear(in_channels * patch_size**3, model_channels, dtype=dtype, device=device)

        self.blocks = nn.ModuleList([
            ModulatedTransformerCrossBlock(
                model_channels, cond_channels,
                num_heads=self.num_heads, mlp_ratio=self.mlp_ratio,
                attn_mode="full", use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"), share_mod=share_mod,
                qk_rms_norm=self.qk_rms_norm, qk_rms_norm_cross=self.qk_rms_norm_cross,
                dtype=dtype, device=device, operations=operations,
            )
            for _ in range(num_blocks)
        ])

        self.out_layer = operations.Linear(model_channels, out_channels * patch_size**3, dtype=dtype, device=device)

    def forward(self, x, t, cond, transformer_options={}, **kwargs):
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward, self,
            comfy.patcher_extension.get_all_wrappers(
                comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                transformer_options,
            ),
        ).execute(x, t, cond, transformer_options, **kwargs)

    def _forward(self, x, t, cond, transformer_options={}, **kwargs):
        transformer_options = transformer_options.copy()
        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})

        pose = x.pose
        x = x.shape

        h = patchify(x, self.patch_size)
        h = h.view(*h.shape[:2], -1).permute(0, 2, 1).contiguous()
        if self.include_pose:
            h = torch.cat([h, pose], dim=1)

        h = self.input_layer(h)
        h = h + self.pos_emb.to(h.device)[None]
        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)

        transformer_options["block_type"] = "double"
        transformer_options["total_blocks"] = len(self.blocks)

        for i, block in enumerate(self.blocks):
            transformer_options["block_index"] = i
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["x"] = block(args["x"], args["mod"], args["context"],
                                     transformer_options=args.get("transformer_options", {}))
                    return out
                out = blocks_replace[("double_block", i)](
                    {"x": h, "mod": t_emb, "context": cond, "transformer_options": transformer_options},
                    {"original_block": block_wrap},
                )
                h = out["x"]
            else:
                h = block(h, t_emb, cond, transformer_options=transformer_options)

        h = h.type(x.dtype)
        h = F.layer_norm(h, h.shape[-1:])
        h = self.out_layer(h)
        if self.include_pose:
            pose = h[:, -1:]
            h = h[:, :-1]

        h = h.permute(0, 2, 1).view(h.shape[0], h.shape[2], *[self.resolution // self.patch_size] * 3)
        h = unpatchify(h, self.patch_size).contiguous()
        return DataType(h, pose)


class SparseStructureFlowTdfyWrapper(SparseStructureFlowModel):
    def __init__(self, *args, **kwargs):
        condition_embedder = kwargs.pop("condition_embedder", None)
        force_zeros_cond = kwargs.pop("force_zeros_cond", False)
        kwargs.pop("shape_attend_pose", None)
        kwargs.pop("freeze_d_time_embedder", None)
        super().__init__(*args, **kwargs)
        if condition_embedder is not None:
            self.condition_embedder = condition_embedder
        else:
            self.condition_embedder = lambda x: x
        self.force_zeros_cond = force_zeros_cond

    def forward(self, x, t, *condition_args, transformer_options={}, **condition_kwargs):
        cfg_activate = condition_kwargs.pop("cfg", False)
        if self.force_zeros_cond and cfg_activate:
            cond = self.condition_embedder(*condition_args, **condition_kwargs)
            cond = cond * 0
        else:
            cond = self.condition_embedder(*condition_args, **condition_kwargs)
        if self.include_pose:
            pose = x[:, -1:]
            x = x[:, :-1]
        else:
            pose = None
        x = x.permute(0, 2, 1).contiguous()
        n_voxels_cubed = x.shape[-1]
        cube_root = n_voxels_cubed ** (1 / 3)
        n_voxels = round(cube_root)
        x = x.view(x.shape[0], x.shape[1], n_voxels, n_voxels, n_voxels)
        input_data = DataType(x, pose)
        output = super().forward(input_data, t, cond, transformer_options=transformer_options)
        h = output.shape
        pose = output.pose
        h = h.view(h.shape[0], h.shape[1], -1).permute(0, 2, 1).contiguous()
        if self.include_pose:
            h = torch.cat([h, pose], dim=1)
        return h


# ==========================================================================
# MOT (Multi-Object Tracking) Sparse Structure Flow Model
# ==========================================================================

class MOTSparseStructureFlowModel(nn.Module):
    """MOT variant of the sparse structure flow model with WrapperExecutor support."""

    def __init__(self, in_channels, model_channels, cond_channels, out_channels,
                 num_blocks, num_heads=None, num_head_channels=64, mlp_ratio=4,
                 pe_mode="ape", use_fp16=False, use_checkpoint=False,
                 share_mod=False, qk_rms_norm=False, qk_rms_norm_cross=False,
                 freeze_shared_parameters=False, is_shortcut_model=False,
                 latent_names=None,
                 dtype=None, device=None, operations=ops, *args, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.pe_mode = pe_mode
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.is_shortcut_model = is_shortcut_model
        self.latent_names = latent_names

        if is_shortcut_model:
            self.d_embedder = TimestepEmbedder(model_channels, dtype=dtype, device=device, operations=operations)

        self.t_embedder = TimestepEmbedder(model_channels, freeze=freeze_shared_parameters,
                                           dtype=dtype, device=device, operations=operations)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), operations.Linear(model_channels, 6 * model_channels, bias=True, dtype=dtype, device=device)
            )

        self.blocks = nn.ModuleList([
            MOTModulatedTransformerCrossBlock(
                model_channels, cond_channels,
                num_heads=self.num_heads, mlp_ratio=self.mlp_ratio,
                attn_mode="full", use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"), share_mod=share_mod,
                qk_rms_norm=self.qk_rms_norm, qk_rms_norm_cross=self.qk_rms_norm_cross,
                latent_names=self.latent_names,
                freeze_shared_parameters=freeze_shared_parameters,
                dtype=dtype, device=device, operations=operations,
            )
            for _ in range(num_blocks)
        ])

    def forward(self, h, t, cond, d=None, transformer_options={}, **kwargs):
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward, self,
            comfy.patcher_extension.get_all_wrappers(
                comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                transformer_options,
            ),
        ).execute(h, t, cond, d, transformer_options, **kwargs)

    def _forward(self, h, t, cond, d=None, transformer_options={}, **kwargs):
        transformer_options = transformer_options.copy()
        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})

        t_emb = self.t_embedder(t)
        if d is not None:
            d_emb = self.d_embedder(d)
            t_emb = t_emb + d_emb
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)

        transformer_options["block_type"] = "double"
        transformer_options["total_blocks"] = len(self.blocks)

        for i, block in enumerate(self.blocks):
            transformer_options["block_index"] = i
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["x"] = block(args["x"], args["mod"], args["context"],
                                     transformer_options=args.get("transformer_options", {}))
                    return out
                out = blocks_replace[("double_block", i)](
                    {"x": h, "mod": t_emb, "context": cond, "transformer_options": transformer_options},
                    {"original_block": block_wrap},
                )
                h = out["x"]
            else:
                h = block(h, t_emb, cond, transformer_options=transformer_options)

        return h


class MOTSparseStructureFlowTdfyWrapper(MOTSparseStructureFlowModel):
    def __init__(self, latent_mapping, latent_share_transformer=None, *args, **kwargs):
        condition_embedder = kwargs.pop("condition_embedder", None)
        force_zeros_cond = kwargs.pop("force_zeros_cond", False)
        kwargs.pop("shape_attend_pose", None)
        if latent_share_transformer is None:
            latent_share_transformer = {}

        merge_latent_names = [i for _, v in latent_share_transformer.items() for i in v]
        latent_names = [
            ln for ln in list(latent_mapping.keys()) if ln not in merge_latent_names
        ] + list(latent_share_transformer.keys())
        kwargs["latent_names"] = latent_names

        super().__init__(*args, **kwargs)

        if condition_embedder is not None:
            self.condition_embedder = condition_embedder
        else:
            self.condition_embedder = lambda x: x
        self.force_zeros_cond = force_zeros_cond
        self.latent_mapping = nn.ModuleDict(latent_mapping)
        self.latent_share_transformer = latent_share_transformer if isinstance(latent_share_transformer, dict) else dict(latent_share_transformer)
        self.input_latent_mappings = list(self.latent_mapping.keys())

    def forward(self, latents_dict, t, *condition_args, transformer_options={}, **condition_kwargs):
        d = condition_kwargs.pop("d", None)
        cfg_activate = condition_kwargs.pop("cfg", False)
        if self.force_zeros_cond and cfg_activate:
            cond = self.condition_embedder(*condition_args, **condition_kwargs)
            cond = cond * 0
        else:
            cond = self.condition_embedder(*condition_args, **condition_kwargs)

        latent_dict = self.project_input(latents_dict)
        output = super().forward(latent_dict, t, cond, d, transformer_options=transformer_options)
        output_latents = self.project_output(output)
        return output_latents

    def project_input(self, latents_dict):
        latent_dict = {}
        for latent_name in self.input_latent_mappings:
            assert latent_name in latents_dict, f"'{latent_name}' not found in latents_dict"
            latent_input = latents_dict[latent_name]
            x = self.latent_mapping[latent_name].to_input(latent_input)
            latent_dict[latent_name] = x
        latent_dict = self._merge_latent_share_transformer(latent_dict)
        return latent_dict

    def project_output(self, output):
        output = self._split_latent_share_transformer(output)
        output_latents = {}
        for latent_name in self.input_latent_mappings:
            latent = self.latent_mapping[latent_name].to_output(output[latent_name])
            output_latents[latent_name] = latent
        return output_latents

    def _merge_latent_share_transformer(self, latent_dict):
        visited = set()
        return_dict = {}
        for merged_name, latent_names in self.latent_share_transformer.items():
            tensors = []
            for ln in latent_names:
                visited.add(ln)
                tensors.append(latent_dict[ln])
            return_dict[merged_name] = torch.cat(tensors, dim=1)
        for ln in latent_dict:
            if ln not in visited:
                return_dict[ln] = latent_dict[ln]
        return return_dict

    def _split_latent_share_transformer(self, output_latents):
        return_dict = {}
        visited = set()
        for merged_name, latent_names in self.latent_share_transformer.items():
            start = 0
            visited.add(merged_name)
            tensors = output_latents[merged_name]
            for ln in latent_names:
                token_len = self.latent_mapping[ln].pos_emb.shape[0]
                return_dict[ln] = tensors[:, start: start + token_len]
                start += token_len
        for ln in output_latents:
            if ln not in visited:
                return_dict[ln] = output_latents[ln]
        return return_dict


# ==========================================================================
# SLatFlowModel — Structured Latent Flow with WrapperExecutor
# ==========================================================================

class SLatFlowModel(nn.Module):
    def __init__(self, resolution, in_channels, model_channels, cond_channels,
                 out_channels, num_blocks, num_heads=None, num_head_channels=64,
                 mlp_ratio=4, patch_size=2, num_io_res_blocks=2,
                 io_block_channels=None, pe_mode="ape",
                 use_fp16=False, use_checkpoint=False, use_skip_connection=True,
                 share_mod=False, qk_rms_norm=False, qk_rms_norm_cross=False,
                 is_shortcut_model=False,
                 dtype=None, device=None, operations=ops, sparse_operations=sparse_ops):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size
        self.num_io_res_blocks = num_io_res_blocks
        self.io_block_channels = io_block_channels
        self.pe_mode = pe_mode
        self.use_checkpoint = use_checkpoint
        self.use_skip_connection = use_skip_connection
        self.share_mod = share_mod
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.is_shortcut_model = is_shortcut_model

        if is_shortcut_model:
            self.d_embedder = TimestepEmbedder(model_channels, dtype=dtype, device=device, operations=operations)

        assert int(np.log2(patch_size)) == np.log2(patch_size), "Patch size must be a power of 2"
        assert np.log2(patch_size) == len(io_block_channels), "Number of IO stages must match log2(patch_size)"

        self.t_embedder = TimestepEmbedder(model_channels, dtype=dtype, device=device, operations=operations)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), operations.Linear(model_channels, 6 * model_channels, bias=True, dtype=dtype, device=device)
            )

        if pe_mode == "ape":
            self.pos_embedder = AbsolutePositionEmbedder(model_channels)

        self.input_layer = sparse_operations.SparseLinear(in_channels, io_block_channels[0], dtype=dtype, device=device)
        self.input_blocks = nn.ModuleList([])
        for chs, next_chs in zip(io_block_channels, io_block_channels[1:] + [model_channels]):
            self.input_blocks.extend([
                SparseResBlock3d(chs, model_channels, out_channels=chs,
                                 dtype=dtype, device=device, operations=operations, sparse_operations=sparse_operations)
                for _ in range(num_io_res_blocks - 1)
            ])
            self.input_blocks.append(
                SparseResBlock3d(chs, model_channels, out_channels=next_chs, downsample=True,
                                 dtype=dtype, device=device, operations=operations, sparse_operations=sparse_operations)
            )

        self.blocks = nn.ModuleList([
            ModulatedSparseTransformerCrossBlock(
                model_channels, cond_channels,
                num_heads=self.num_heads, mlp_ratio=self.mlp_ratio,
                attn_mode="full", use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"), share_mod=self.share_mod,
                qk_rms_norm=self.qk_rms_norm, qk_rms_norm_cross=self.qk_rms_norm_cross,
                dtype=dtype, device=device, operations=operations, sparse_operations=sparse_operations,
            )
            for _ in range(num_blocks)
        ])

        self.out_blocks = nn.ModuleList([])
        for chs, prev_chs in zip(
            reversed(io_block_channels),
            [model_channels] + list(reversed(io_block_channels[1:])),
        ):
            self.out_blocks.append(
                SparseResBlock3d(
                    prev_chs * 2 if self.use_skip_connection else prev_chs,
                    model_channels, out_channels=chs, upsample=True,
                    dtype=dtype, device=device, operations=operations, sparse_operations=sparse_operations,
                )
            )
            self.out_blocks.extend([
                SparseResBlock3d(
                    chs * 2 if self.use_skip_connection else chs,
                    model_channels, out_channels=chs,
                    dtype=dtype, device=device, operations=operations, sparse_operations=sparse_operations,
                )
                for _ in range(num_io_res_blocks - 1)
            ])
        self.out_layer = sparse_operations.SparseLinear(io_block_channels[0], out_channels, dtype=dtype, device=device)

    def forward(self, x, t, cond, d=None, transformer_options={}, **kwargs):
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward, self,
            comfy.patcher_extension.get_all_wrappers(
                comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                transformer_options,
            ),
        ).execute(x, t, cond, d, transformer_options, **kwargs)

    def _forward(self, x, t, cond, d=None, transformer_options={}, **kwargs):
        transformer_options = transformer_options.copy()
        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})

        h = self.input_layer(x)
        t_emb = self.t_embedder(t)
        if d is not None:
            d_emb = self.d_embedder(d)
            t_emb = t_emb + d_emb
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)

        skips = []
        for block in self.input_blocks:
            h = block(h, t_emb)
            skips.append(h.feats)

        if self.pe_mode == "ape":
            h = h + self.pos_embedder(h.coords[:, 1:])

        transformer_options["block_type"] = "double"
        transformer_options["total_blocks"] = len(self.blocks)

        for i, block in enumerate(self.blocks):
            transformer_options["block_index"] = i
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["x"] = block(args["x"], args["mod"], args["context"],
                                     transformer_options=args.get("transformer_options", {}))
                    return out
                out = blocks_replace[("double_block", i)](
                    {"x": h, "mod": t_emb, "context": cond, "transformer_options": transformer_options},
                    {"original_block": block_wrap},
                )
                h = out["x"]
            else:
                h = block(h, t_emb, cond, transformer_options=transformer_options)

        for block, skip in zip(self.out_blocks, reversed(skips)):
            if self.use_skip_connection:
                h = block(h.replace(torch.cat([h.feats, skip], dim=1)), t_emb)
            else:
                h = block(h, t_emb)

        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.out_layer(h.type(x.dtype))
        return h


class SLatFlowModelTdfyWrapper(SLatFlowModel):
    def __init__(self, *args, **kwargs):
        condition_embedder = kwargs.pop("condition_embedder", None)
        force_zeros_cond = kwargs.pop("force_zeros_cond", False)
        super().__init__(*args, **kwargs)
        if condition_embedder is not None:
            self.condition_embedder = condition_embedder
        else:
            self.condition_embedder = lambda x: x
        self.force_zeros_cond = force_zeros_cond

    def forward(self, x, t, *condition_args, transformer_options={}, **condition_kwargs):
        if not torch.compiler.is_compiling():
            d = condition_kwargs.pop("d", None)
            if "coords" in condition_kwargs:
                coords = condition_kwargs["coords"]
                del condition_kwargs["coords"]
            else:
                coords = condition_args[-1]
                condition_args = condition_args[:-1]
        else:
            coords = condition_args[-1]
            condition_args = condition_args[:-1]
            d = condition_kwargs.pop("d", None)

        coords = torch.tensor(coords).to(x.device)
        x = SparseTensor(feats=x[0], coords=coords)
        cfg_activate = condition_kwargs.pop("cfg", False)
        if self.force_zeros_cond and cfg_activate:
            cond = self.condition_embedder(*condition_args, **condition_kwargs)
            cond = cond * 0
        else:
            cond = self.condition_embedder(*condition_args, **condition_kwargs)
        h = super().forward(x, t, cond, d, transformer_options=transformer_options)
        h = h.feats[None]
        return h


# ==========================================================================
# Latent mapping modules (for MOT flow)
# ==========================================================================

class Latent(nn.Module):
    def __init__(self, in_channels, model_channels, pos_embedder,
                 dtype=None, device=None, operations=ops):
        super().__init__()
        self.input_layer = operations.Linear(in_channels, model_channels, dtype=dtype, device=device)
        self.out_layer = operations.Linear(model_channels, in_channels, dtype=dtype, device=device)

        pos_emb = pos_embedder()
        if isinstance(pos_emb, torch.nn.Parameter):
            self.register_parameter("pos_emb", pos_emb)
        elif isinstance(pos_emb, torch.Tensor):
            self.register_buffer("pos_emb", pos_emb)
        else:
            raise NotImplementedError

    def to_input(self, x):
        x = self.input_layer(x)
        x = x + self.pos_emb.to(x.device)[None]
        return x

    def to_output(self, h):
        h = F.layer_norm(h, h.shape[-1:])
        h = self.out_layer(h)
        return h


def ShapePositionEmbedder(model_channels, resolution, patch_size):
    def embedder():
        pos_embedder = AbsolutePositionEmbedder(model_channels, 3)
        coords = torch.meshgrid(
            *[torch.arange(res) for res in [resolution // patch_size] * 3],
            indexing="ij",
        )
        coords = torch.stack(coords, dim=-1).reshape(-1, 3)
        pos_emb = pos_embedder(coords)
        return pos_emb
    return embedder


def RandomPositionEmbedder(model_channels, token_len):
    def embedder():
        return torch.randn(token_len, model_channels)
    return embedder


def LearntPositionEmbedder(model_channels, token_len):
    def embedder():
        return torch.nn.Parameter(torch.randn(token_len, model_channels))
    return embedder


# ==========================================================================
# Embedders (Dino, EmbedderFuser, PointPatchEmbed)
# ==========================================================================

class PointRemapper(nn.Module):
    """Handles remapping of 3D point coordinates."""
    VALID_TYPES = ["linear", "sinh", "exp", "sinh_exp", "exp_disparity"]

    def __init__(self, remap_type="exp"):
        super().__init__()
        self.remap_type = remap_type
        if remap_type not in self.VALID_TYPES:
            raise ValueError(f"Invalid remap type: {remap_type}. Must be one of {self.VALID_TYPES}")

    def forward(self, points):
        if self.remap_type == "linear":
            return points
        elif self.remap_type == "sinh":
            return torch.asinh(points)
        elif self.remap_type == "exp":
            xy_scaled, z_exp = points.split([2, 1], dim=-1)
            z = torch.log1p(z_exp)
            xy = xy_scaled / (1 + z_exp)
            return torch.cat([xy, z], dim=-1)
        elif self.remap_type == "exp_disparity":
            xy_scaled, z_exp = points.split([2, 1], dim=-1)
            xy = xy_scaled / z_exp
            z = torch.log(z_exp)
            return torch.cat([xy, z], dim=-1)
        elif self.remap_type == "sinh_exp":
            xy_sinh, z_exp = points.split([2, 1], dim=-1)
            xy = torch.asinh(xy_sinh)
            z = torch.log(z_exp.clamp(min=1e-8))
            return torch.cat([xy, z], dim=-1)
        else:
            raise ValueError(f"Unknown remap type: {self.remap_type}")

    def inverse(self, points):
        if self.remap_type == "linear":
            return points
        elif self.remap_type == "sinh":
            return torch.sinh(points)
        elif self.remap_type == "exp":
            xy, z = points.split([2, 1], dim=-1)
            z_exp = torch.expm1(z)
            return torch.cat([xy * (1 + z_exp), z_exp], dim=-1)
        elif self.remap_type == "exp_disparity":
            xy, z = points.split([2, 1], dim=-1)
            z_exp = torch.exp(z)
            return torch.cat([xy * z_exp, z_exp], dim=-1)
        elif self.remap_type == "sinh_exp":
            xy, z = points.split([2, 1], dim=-1)
            return torch.cat([torch.sinh(xy), torch.exp(z)], dim=-1)
        else:
            raise ValueError(f"Unknown remap type: {self.remap_type}")


class PointPatchEmbed(nn.Module):
    """Projects (x,y,z) -> D, splits into patches, runs self-attention inside each window."""

    def __init__(self, input_size=256, patch_size=8, embed_dim=768,
                 remap_output="exp", dropout_prob=0.0, force_dropout_always=False,
                 dtype=None, device=None, operations=ops):
        super().__init__()
        from timm.models.vision_transformer import Block
        self.input_size = input_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.dropout_prob = dropout_prob
        self.force_dropout_always = force_dropout_always

        self.point_remapper = PointRemapper(remap_output)
        self.point_proj = operations.Linear(3, embed_dim, dtype=dtype, device=device)
        self.invalid_xyz_token = nn.Parameter(torch.zeros(embed_dim))

        if dropout_prob > 0:
            self.dropped_xyz_token = nn.Parameter(torch.zeros(embed_dim))

        num_patches = input_size // patch_size
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, num_patches, num_patches))
        self.pos_embed_window = nn.Parameter(torch.zeros(1, 1 + patch_size * patch_size, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # timm Block — keep as-is (third-party, not in weight state dict)
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads=16, mlp_ratio=2.0, qkv_bias=True,
                  norm_layer=partial(nn.LayerNorm, eps=1e-6))
        ])

    def _get_pos_embed(self, hw, device=None):
        h, w = hw
        pos_embed = self.pos_embed
        if device is not None:
            pos_embed = pos_embed.to(device)
        pos_embed = F.interpolate(pos_embed, size=(h, w), mode="bilinear", align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1)
        return pos_embed

    def resize_input(self, xyz):
        resized_xyz = F.interpolate(xyz, size=self.input_size, mode="nearest")
        resized_xyz = resized_xyz.permute(0, 2, 3, 1)
        return resized_xyz

    def apply_pointmap_dropout(self, embeddings):
        should_apply = (self.training or self.force_dropout_always) and self.dropout_prob > 0
        if not should_apply:
            return embeddings
        if not hasattr(self, 'dropped_xyz_token'):
            return embeddings
        batch_size, n_windows, embed_dim = embeddings.shape
        if self.force_dropout_always and not self.training:
            drop_mask = torch.ones(batch_size, device=embeddings.device, dtype=torch.bool)
        else:
            drop_mask = torch.rand(batch_size, device=embeddings.device) < self.dropout_prob
        dropped_embedding = self.dropped_xyz_token.to(embeddings.device).view(1, 1, embed_dim).expand(batch_size, n_windows, embed_dim)
        n_windows_h = n_windows_w = int(n_windows ** 0.5)
        pos_embed_patch = self._get_pos_embed((n_windows_h, n_windows_w), device=embeddings.device).reshape(1, n_windows, embed_dim)
        dropped_embedding = dropped_embedding + pos_embed_patch
        drop_mask_expanded = drop_mask.view(batch_size, 1, 1).expand_as(embeddings)
        embeddings = torch.where(drop_mask_expanded, dropped_embedding, embeddings)
        return embeddings

    @torch._dynamo.disable()
    def embed_pointmap_windows(self, xyz, valid_mask=None):
        with torch.no_grad():
            xyz = self.resize_input(xyz)
            if valid_mask is None:
                valid_mask = xyz.isfinite().all(dim=-1)
            B, H, W, _ = xyz.shape
            assert H % self.patch_size == 0 and W % self.patch_size == 0
            xyz_safe = xyz.clone()
            xyz_safe[~valid_mask] = 0.0
            xyz_remapped = self.point_remapper(xyz_safe)
        x = self.point_proj(xyz_remapped)
        x[~valid_mask] = 0.0
        x[~valid_mask] += self.invalid_xyz_token.to(x.device)
        return x, B, H, W

    def inner_forward(self, x, B, H, W):
        x = x.view(B, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size, self.embed_dim)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(-1, self.patch_size * self.patch_size, self.embed_dim)
        cls_tok = self.cls_token.to(x.device).expand(x.shape[0], -1, -1)
        toks = torch.cat([cls_tok, x], dim=1)
        toks = toks + self.pos_embed_window.to(toks.device)
        for blk in self.blocks:
            toks = blk(toks)
        n_windows_h = H // self.patch_size
        n_windows_w = W // self.patch_size
        window_embeddings = toks[:, 0].view(B, n_windows_h * n_windows_w, self.embed_dim)
        pos_embed_patch = self._get_pos_embed((n_windows_h, n_windows_w), device=window_embeddings.device).reshape(
            1, n_windows_h * n_windows_w, self.embed_dim
        )
        out = window_embeddings + pos_embed_patch
        if (self.training or self.force_dropout_always) and self.dropout_prob > 0:
            out = self.apply_pointmap_dropout(out)
        return out

    def forward(self, xyz, valid_mask=None):
        x, B, H, W = self.embed_pointmap_windows(xyz, valid_mask)
        return self.inner_forward(x, B, H, W)


def _wrap_attn_comfy(module):
    """Class-swap a DINOv2 Attention module to use ComfyUI native optimized attention."""
    from comfy.ldm.modules.attention import optimized_attention_for_device

    class _W(module.__class__):
        def forward(self, x, **kwargs):
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
            q, k, v = torch.unbind(qkv, 2)
            # Cast to bf16 for sage attention compatibility
            orig_dtype = q.dtype
            if q.dtype == torch.float32:
                q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            attn_fn = optimized_attention_for_device(q.device)
            out = attn_fn(q, k, v, heads=self.num_heads, skip_reshape=True, skip_output_reshape=True)
            out = out.transpose(1, 2).contiguous().view(B, N, C).to(orig_dtype)
            return self.proj_drop(self.proj(out))

    module.__class__ = _W


class Dino(torch.nn.Module):
    def __init__(self, input_size=224, repo_or_dir="facebookresearch/dinov2",
                 dino_model="dinov2_vitb14", source="github",
                 backbone_kwargs=None, normalize_images=True,
                 prenorm_features=False, freeze_backbone=True,
                 prune_network=False):
        super().__init__()
        from .moge import get_dinov2_backbone

        # Structure only — weights loaded externally via load_state_dict(assign=True),
        # attention wrapping done by _wrap_dino_attention post-load.
        self.backbone = get_dinov2_backbone(dino_model, pretrained=False)

        self.resize_input_size = (input_size, input_size)
        self.embed_dim = self.backbone.embed_dim
        self.input_size = input_size
        self.input_channels = 3
        self.normalize_images = normalize_images
        self.prenorm_features = prenorm_features
        self.register_buffer('mean', torch.as_tensor([[0.485, 0.456, 0.406]]).view(-1, 1, 1), persistent=False)
        self.register_buffer('std', torch.as_tensor([[0.229, 0.224, 0.225]]).view(-1, 1, 1), persistent=False)

        if freeze_backbone:
            self.requires_grad_(False)
            self.eval()

        if prune_network:
            self._prune_network()

    def _preprocess_input(self, x):
        _resized_images = torch.nn.functional.interpolate(x, size=self.resize_input_size, mode="bilinear", align_corners=False)
        if x.shape[1] == 1:
            _resized_images = _resized_images.repeat(1, 3, 1, 1)
        if self.normalize_images:
            _resized_images = _resized_images.sub_(self.mean.to(_resized_images.device)).div_(self.std.to(_resized_images.device))
        return _resized_images

    def _forward_last_layer(self, input_img):
        device = input_img.device
        for p in self.backbone.parameters(recurse=False):
            if p.data.device != device:
                p.data = p.data.to(device)
        for b in self.backbone.buffers(recurse=False):
            if b.device != device:
                b.data = b.data.to(device)
        output = self.backbone.forward_features(input_img)
        if self.prenorm_features:
            features = output["x_prenorm"]
            tokens = F.layer_norm(features, features.shape[-1:])
        else:
            tokens = torch.cat([output["x_norm_clstoken"].unsqueeze(1), output["x_norm_patchtokens"]], dim=1)
        return tokens

    def forward(self, x, **kwargs):
        _resized_images = self._preprocess_input(x)
        tokens = self._forward_last_layer(_resized_images)
        return tokens.to(x.dtype)

    def _prune_network(self):
        self.backbone.mask_token = None
        if self.prenorm_features:
            self.backbone.norm = torch.nn.Identity()


class DinoForMasks(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = self.backbone.embed_dim

    def forward(self, image, mask):
        return self.backbone.forward(mask)


class EmbedderFuser(torch.nn.Module):
    def __init__(self, embedder_list, use_pos_embedding="learned",
                 projection_pre_norm=True, projection_net_hidden_dim_multiplier=4.0,
                 compression_projection_multiplier=0, freeze=False,
                 drop_modalities_weight=None, dropout_prob=0.0,
                 force_drop_modalities=None,
                 dtype=None, device=None, operations=ops):
        super().__init__()
        from loguru import logger

        if not isinstance(embedder_list, list):
            self.embedder_list = list(embedder_list)
        else:
            self.embedder_list = embedder_list

        self.embed_dims = 0
        self.compression_projection_multiplier = compression_projection_multiplier
        self.concate_embed_dims = 0
        self.module_list = []
        max_positional_embed_idx = 0
        self.positional_embed_map = {}
        for condition_embedder, kwargs_info in self.embedder_list:
            self.embed_dims = max(self.embed_dims, condition_embedder.embed_dim)
            self.module_list.append(condition_embedder)
            for _, pos_group in kwargs_info:
                self.concate_embed_dims += condition_embedder.embed_dim
                if pos_group is not None:
                    if pos_group not in self.positional_embed_map:
                        self.positional_embed_map[pos_group] = max_positional_embed_idx
                        max_positional_embed_idx += 1
        self.module_list = nn.ModuleList(self.module_list)
        self.use_pos_embedding = use_pos_embedding
        if self.use_pos_embedding == "random":
            idx_emb = torch.randn(max_positional_embed_idx + 1, 1, self.embed_dims)
            self.register_buffer("idx_emb", idx_emb)
        elif self.use_pos_embedding == "learned":
            self.idx_emb = nn.Parameter(torch.empty(max_positional_embed_idx + 1, self.embed_dims))
            nn.init.normal_(self.idx_emb, mean=0.0, std=1.0 / math.sqrt(self.embed_dims))
        else:
            raise NotImplementedError(f"Unknown pos embedding {self.use_pos_embedding}")

        self.projection_pre_norm = projection_pre_norm
        self.projection_net_hidden_dim_multiplier = projection_net_hidden_dim_multiplier
        if projection_net_hidden_dim_multiplier > 0:
            self.projection_nets = []
            for condition_embedder, _ in self.embedder_list:
                self.projection_nets.append(
                    self._make_projection_net(condition_embedder.embed_dim, self.embed_dims,
                                              self.projection_net_hidden_dim_multiplier,
                                              dtype=dtype, device=device, operations=operations)
                )
            self.projection_nets = nn.ModuleList(self.projection_nets)

        if compression_projection_multiplier > 0:
            self.compression_projector = self._make_projection_net(
                self.concate_embed_dims, self.embed_dims,
                self.compression_projection_multiplier,
                dtype=dtype, device=device, operations=operations,
            )

        self.drop_modalities_weight = drop_modalities_weight if drop_modalities_weight is not None else []
        self.dropout_prob = dropout_prob
        self.force_drop_modalities = force_drop_modalities

        if freeze:
            self.requires_grad_(False)
            self.eval()

    def _make_projection_net(self, input_embed_dim, output_embed_dim, multiplier,
                             dtype=None, device=None, operations=ops):
        if self.projection_pre_norm:
            pre_norm = nn.LayerNorm(input_embed_dim)
        else:
            pre_norm = nn.Identity()
        ff_net = FeedForward(
            dim=input_embed_dim,
            hidden_dim=int(multiplier * output_embed_dim),
            output_dim=output_embed_dim,
            dtype=dtype, device=device, operations=operations,
        )
        return nn.Sequential(pre_norm, ff_net)

    def _build_dropout_distribution(self, device):
        dropout_configs = []
        weights = []
        dropout_configs.append(set())
        weights.append(1.0 - self.dropout_prob)
        total_dropout_weight = sum(w for _, w in self.drop_modalities_weight)
        assert total_dropout_weight > 0
        for modality_list, weight in self.drop_modalities_weight:
            dropout_configs.append(set(modality_list))
            weights.append(self.dropout_prob * weight / total_dropout_weight)
        weights_tensor = torch.tensor(weights, device=device)
        was_deterministic = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(False)
        cumsum_weights = torch.cumsum(weights_tensor, dim=0)
        torch.use_deterministic_algorithms(was_deterministic)
        return dropout_configs, cumsum_weights

    def _apply_force_drop(self, kwarg_names, tokens):
        if not self.force_drop_modalities:
            return tokens
        force_drop_set = set(self.force_drop_modalities)
        result_tokens = []
        for kwarg_name, token_tensor in zip(kwarg_names, tokens):
            mask = 0.0 if kwarg_name in force_drop_set else 1.0
            result_tokens.append(token_tensor * mask)
        return result_tokens

    def _dropout_modalities(self, kwarg_names, tokens):
        tokens = self._apply_force_drop(kwarg_names, tokens)
        if not self.training or self.dropout_prob <= 0 or not self.drop_modalities_weight:
            return tokens
        batch_size = tokens[0].shape[0]
        device = tokens[0].device
        dropout_configs, cumsum_weights = self._build_dropout_distribution(device)
        rand_vals = torch.rand(batch_size, device=device)
        config_indices = torch.searchsorted(cumsum_weights, rand_vals).clamp(max=len(dropout_configs) - 1)
        result_tokens = []
        for kwarg_name, token_tensor in zip(kwarg_names, tokens):
            mask = torch.ones(batch_size, dtype=token_tensor.dtype, device=device)
            for config_idx, modalities_to_drop in enumerate(dropout_configs):
                if kwarg_name in modalities_to_drop:
                    mask[config_indices == config_idx] = 0.0
            mask = mask.view([batch_size] + [1] * (token_tensor.ndim - 1))
            result_tokens.append(token_tensor * mask)
        return result_tokens

    def forward(self, *args, **kwargs):
        from loguru import logger
        tokens = []
        kwarg_names = []
        for i, (condition_embedder, kwargs_info) in enumerate(self.embedder_list):
            for kwarg_name, pos_group in kwargs_info:
                if kwarg_name not in kwargs:
                    logger.warning(f"{kwarg_name} not in kwargs to condition embedder!")
                input_cond = kwargs[kwarg_name]
                cond_token = condition_embedder(input_cond)
                if self.projection_net_hidden_dim_multiplier > 0:
                    cond_token = self.projection_nets[i](cond_token)
                if pos_group is not None:
                    pos_idx = self.positional_embed_map[pos_group]
                    idx_emb = self.idx_emb.to(cond_token.device)
                    if self.use_pos_embedding == "random":
                        cond_token += idx_emb[pos_idx: pos_idx + 1]
                    elif self.use_pos_embedding == "learned":
                        cond_token += idx_emb[pos_idx: pos_idx + 1, None]
                tokens.append(cond_token)
                kwarg_names.append(kwarg_name)

        tokens = self._dropout_modalities(kwarg_names, tokens)

        if self.compression_projection_multiplier > 0:
            tokens = torch.cat(tokens, dim=-1)
            tokens = self.compression_projector(tokens)
        else:
            tokens = torch.cat(tokens, dim=1)
        return tokens
