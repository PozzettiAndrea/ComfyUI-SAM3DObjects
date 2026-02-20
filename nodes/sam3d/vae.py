# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
All encoder/decoder VAE classes for SAM3DObjects — converted to ComfyUI-native
format with operations= pattern.

Classes:
- ResBlock3d, DownsampleBlock3d, UpsampleBlock3d (dense 3D blocks)
- SparseStructureEncoder, SparseStructureDecoder (sparse structure VAE)
- SparseTransformerBase (base for structured latent VAE)
- SLatEncoder (structured latent encoder)
- SLatGaussianDecoder (gaussian splatting decoder)
- SLatMeshDecoder + SparseSubdivideBlock3d (mesh decoder)
- SLatRadianceFieldDecoder (radiance field decoder)
- TdfyWrapper variants for all of the above
"""
import os
import math
from typing import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.ops
import comfy.utils

from . import ops_sparse
from .model import (
    GroupNorm32,
    ChannelLayerNorm32,
    AbsolutePositionEmbedder,
    SparseTransformerBlock,
    pixel_shuffle_3d,
    zero_module,
)
from .sparse import SparseTensor, SparseSubdivide
from .attention import SerializeMode, SerializeModes

# Default operations tiers
ops = comfy.ops.manual_cast
sparse_ops = ops_sparse.manual_cast


# ==========================================================================
# Helpers
# ==========================================================================

def norm_layer(norm_type: str, *args, dtype=None, device=None, operations=ops, **kwargs):
    """Return a normalization layer."""
    if norm_type == "group":
        return GroupNorm32(32, *args, dtype=dtype, device=device, **kwargs)
    elif norm_type == "layer":
        return ChannelLayerNorm32(*args, dtype=dtype, device=device, **kwargs)
    else:
        raise ValueError(f"Invalid norm type {norm_type}")


# ==========================================================================
# Dense 3D Blocks (for SparseStructure VAE)
# ==========================================================================

class ResBlock3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        norm_type: Literal["group", "layer"] = "layer",
        dtype=None, device=None, operations=ops,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        self.norm1 = norm_layer(norm_type, channels, dtype=dtype, device=device, operations=operations)
        self.norm2 = norm_layer(norm_type, self.out_channels, dtype=dtype, device=device, operations=operations)
        self.conv1 = operations.Conv3d(channels, self.out_channels, 3, padding=1, dtype=dtype, device=device)
        self.conv2 = zero_module(
            operations.Conv3d(self.out_channels, self.out_channels, 3, padding=1, dtype=dtype, device=device)
        )
        self.skip_connection = (
            operations.Conv3d(channels, self.out_channels, 1, dtype=dtype, device=device)
            if channels != self.out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        h = h + self.skip_connection(x)
        return h


class DownsampleBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mode: Literal["conv", "avgpool"] = "conv",
        dtype=None, device=None, operations=ops,
    ):
        assert mode in ["conv", "avgpool"], f"Invalid mode {mode}"
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if mode == "conv":
            self.conv = operations.Conv3d(in_channels, out_channels, 2, stride=2, dtype=dtype, device=device)
        elif mode == "avgpool":
            assert in_channels == out_channels, \
                "Pooling mode requires in_channels to be equal to out_channels"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "conv"):
            return self.conv(x)
        else:
            return F.avg_pool3d(x, 2)


class UpsampleBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mode: Literal["conv", "nearest"] = "conv",
        dtype=None, device=None, operations=ops,
    ):
        assert mode in ["conv", "nearest"], f"Invalid mode {mode}"
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if mode == "conv":
            self.conv = operations.Conv3d(in_channels, out_channels * 8, 3, padding=1, dtype=dtype, device=device)
        elif mode == "nearest":
            assert in_channels == out_channels, \
                "Nearest mode requires in_channels to be equal to out_channels"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "conv"):
            x = self.conv(x)
            return pixel_shuffle_3d(x, 2)
        else:
            return F.interpolate(x, scale_factor=2, mode="nearest")


# ==========================================================================
# Sparse Structure VAE (Encoder / Decoder)
# ==========================================================================

class SparseStructureEncoder(nn.Module):
    """
    Encoder for Sparse Structure (E_S in the paper Sec. 3.3).

    Args:
        in_channels: Channels of the input.
        latent_channels: Channels of the latent representation.
        num_res_blocks: Number of residual blocks at each resolution.
        channels: Channels of the encoder blocks.
        num_res_blocks_middle: Number of residual blocks in the middle.
        norm_type: Type of normalization layer.
        use_fp16: Accepted for config compat, ignored (manual_cast handles dtype).
    """

    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        num_res_blocks: int,
        channels: List[int],
        num_res_blocks_middle: int = 2,
        norm_type: Literal["group", "layer"] = "layer",
        use_fp16: bool = False,
        dtype=None, device=None, operations=ops,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.num_res_blocks = num_res_blocks
        self.channels = channels
        self.num_res_blocks_middle = num_res_blocks_middle
        self.norm_type = norm_type

        self.input_layer = operations.Conv3d(in_channels, channels[0], 3, padding=1, dtype=dtype, device=device)

        self.blocks = nn.ModuleList([])
        for i, ch in enumerate(channels):
            self.blocks.extend([
                ResBlock3d(ch, ch, dtype=dtype, device=device, operations=operations)
                for _ in range(num_res_blocks)
            ])
            if i < len(channels) - 1:
                self.blocks.append(
                    DownsampleBlock3d(ch, channels[i + 1], dtype=dtype, device=device, operations=operations)
                )

        self.middle_block = nn.Sequential(*[
            ResBlock3d(channels[-1], channels[-1], dtype=dtype, device=device, operations=operations)
            for _ in range(num_res_blocks_middle)
        ])

        self.out_layer = nn.Sequential(
            norm_layer(norm_type, channels[-1], dtype=dtype, device=device, operations=operations),
            nn.SiLU(),
            operations.Conv3d(channels[-1], latent_channels * 2, 3, padding=1, dtype=dtype, device=device),
        )

    def forward(
        self, x: torch.Tensor, sample_posterior: bool = False, return_raw: bool = False
    ) -> torch.Tensor:
        h = self.input_layer(x)

        for block in self.blocks:
            h = block(h)
        h = self.middle_block(h)

        h = self.out_layer(h)

        mean, logvar = h.chunk(2, dim=1)

        if sample_posterior:
            std = torch.exp(0.5 * logvar)
            z = mean + std * torch.randn_like(std)
        else:
            z = mean

        if return_raw:
            return z, mean, logvar
        return z


class SparseStructureDecoder(nn.Module):
    """
    Decoder for Sparse Structure (D_S in the paper Sec. 3.3).

    Args:
        out_channels: Channels of the output.
        latent_channels: Channels of the latent representation.
        num_res_blocks: Number of residual blocks at each resolution.
        channels: Channels of the decoder blocks.
        num_res_blocks_middle: Number of residual blocks in the middle.
        norm_type: Type of normalization layer.
        reshape_input_to_cube: Whether to reshape flat input to cube.
        use_fp16: Accepted for config compat, ignored (manual_cast handles dtype).
    """

    def __init__(
        self,
        out_channels: int,
        latent_channels: int,
        num_res_blocks: int,
        channels: List[int],
        num_res_blocks_middle: int = 2,
        norm_type: Literal["group", "layer"] = "layer",
        reshape_input_to_cube: bool = False,
        use_fp16: bool = False,
        dtype=None, device=None, operations=ops,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.num_res_blocks = num_res_blocks
        self.channels = channels
        self.num_res_blocks_middle = num_res_blocks_middle
        self.norm_type = norm_type
        self.reshape_input_to_cube = reshape_input_to_cube

        self.input_layer = operations.Conv3d(latent_channels, channels[0], 3, padding=1, dtype=dtype, device=device)

        self.middle_block = nn.Sequential(*[
            ResBlock3d(channels[0], channels[0], dtype=dtype, device=device, operations=operations)
            for _ in range(num_res_blocks_middle)
        ])

        self.blocks = nn.ModuleList([])
        for i, ch in enumerate(channels):
            self.blocks.extend([
                ResBlock3d(ch, ch, dtype=dtype, device=device, operations=operations)
                for _ in range(num_res_blocks)
            ])
            if i < len(channels) - 1:
                self.blocks.append(
                    UpsampleBlock3d(ch, channels[i + 1], dtype=dtype, device=device, operations=operations)
                )

        self.out_layer = nn.Sequential(
            norm_layer(norm_type, channels[-1], dtype=dtype, device=device, operations=operations),
            nn.SiLU(),
            operations.Conv3d(channels[-1], out_channels, 3, padding=1, dtype=dtype, device=device),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.reshape_input_to_cube:
            x = self.flat_to_cube(x)

        h = self.input_layer(x)
        h = self.middle_block(h)
        for block in self.blocks:
            h = block(h)
        h = self.out_layer(h)
        return h

    @staticmethod
    def flat_to_cube(flat_latent: torch.Tensor) -> torch.Tensor:
        """
        Convert latent tokens from generator to cube.

        Args:
            flat_latent: (B, T, C)
        Returns:
            cube: (B, C, D, H, W)
        """
        k = round(math.pow(flat_latent.shape[1], 1 / 3))
        assert (
            k ** 3 == flat_latent.shape[1]
        ), f"Flat latent must be a cube {k ** 3} != {flat_latent.shape[1]}"
        latent = flat_latent.view(
            flat_latent.shape[0], k, k, k, flat_latent.shape[2]
        ).permute(0, 4, 1, 2, 3)
        return latent


class SparseStructureDecoderTdfyWrapper(SparseStructureDecoder):
    def __init__(self, *args, **kwargs):
        pretrained_ckpt_path = kwargs.pop("pretrained_ckpt_path", None)
        super().__init__(*args, **kwargs)
        if pretrained_ckpt_path is not None:
            if os.path.exists(pretrained_ckpt_path):
                self.load_state_dict(
                    comfy.utils.load_torch_file(pretrained_ckpt_path, safe_load=True)
                )
            else:
                raise FileNotFoundError(
                    f"The path for the SS decoder does not exist: {pretrained_ckpt_path}"
                )


class SparseStructureEncoderTdfyWrapper(SparseStructureEncoder):
    def __init__(self, sample_posterior=True, return_raw=True, *args, **kwargs):
        pretrained_ckpt_path = kwargs.pop("pretrained_ckpt_path", None)
        super().__init__(*args, **kwargs)
        if pretrained_ckpt_path is not None:
            if os.path.exists(pretrained_ckpt_path):
                self.load_state_dict(
                    comfy.utils.load_torch_file(pretrained_ckpt_path, safe_load=True)
                )
            else:
                raise FileNotFoundError(
                    f"The path for the SS encoder does not exist: {pretrained_ckpt_path}"
                )
        self.sample_posterior = sample_posterior
        self.return_raw = return_raw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, mean, logvar = super().forward(
            x, sample_posterior=self.sample_posterior, return_raw=True
        )
        if self.return_raw:
            return {"z": z, "mean": mean, "logvar": logvar}
        else:
            return {"z": z}


# ==========================================================================
# Structured Latent VAE Base
# ==========================================================================

def block_attn_config(self):
    """Return the attention configuration of the model."""
    for i in range(self.num_blocks):
        if self.attn_mode == "shift_window":
            yield "serialized", self.window_size, 0, (16 * (i % 2),) * 3, SerializeMode.Z_ORDER
        elif self.attn_mode == "shift_sequence":
            yield "serialized", self.window_size, self.window_size // 2 * (i % 2), (0, 0, 0), SerializeMode.Z_ORDER
        elif self.attn_mode == "shift_order":
            yield "serialized", self.window_size, 0, (0, 0, 0), SerializeModes[i % 4]
        elif self.attn_mode == "full":
            yield "full", None, None, None, None
        elif self.attn_mode == "swin":
            yield "windowed", self.window_size, None, self.window_size // 2 * (i % 2), None


class SparseTransformerBase(nn.Module):
    """
    Sparse Transformer without output layers.
    Serves as the base class for structured latent encoder and decoders.
    """

    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "full",
        window_size: Optional[int] = None,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
        dtype=None, device=None, operations=ops, sparse_operations=sparse_ops,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_blocks = num_blocks
        self.window_size = window_size
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.attn_mode = attn_mode
        self.pe_mode = pe_mode
        self.use_checkpoint = use_checkpoint
        self.qk_rms_norm = qk_rms_norm

        if pe_mode == "ape":
            self.pos_embedder = AbsolutePositionEmbedder(model_channels)

        self.input_layer = sparse_operations.SparseLinear(in_channels, model_channels, dtype=dtype, device=device)
        self.blocks = nn.ModuleList([
            SparseTransformerBlock(
                model_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_mode=attn_mode_i,
                window_size=window_size_i,
                shift_sequence=shift_sequence,
                shift_window=shift_window,
                serialize_mode=serialize_mode,
                use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"),
                qk_rms_norm=self.qk_rms_norm,
                dtype=dtype, device=device,
                operations=operations,
                sparse_operations=sparse_operations,
            )
            for attn_mode_i, window_size_i, shift_sequence, shift_window, serialize_mode
            in block_attn_config(self)
        ])

    def forward(self, x: SparseTensor) -> SparseTensor:
        h = self.input_layer(x)
        if self.pe_mode == "ape":
            h = h + self.pos_embedder(x.coords[:, 1:])
        for block in self.blocks:
            h = block(h)
        return h


# ==========================================================================
# Structured Latent Encoder
# ==========================================================================

class SLatEncoder(SparseTransformerBase):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        latent_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "swin",
        window_size: int = 8,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
        dtype=None, device=None, operations=ops, sparse_operations=sparse_ops,
    ):
        super().__init__(
            in_channels=in_channels,
            model_channels=model_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            mlp_ratio=mlp_ratio,
            attn_mode=attn_mode,
            window_size=window_size,
            pe_mode=pe_mode,
            use_fp16=use_fp16,
            use_checkpoint=use_checkpoint,
            qk_rms_norm=qk_rms_norm,
            dtype=dtype, device=device,
            operations=operations,
            sparse_operations=sparse_operations,
        )
        self.resolution = resolution
        self.out_layer = sparse_operations.SparseLinear(model_channels, 2 * latent_channels, dtype=dtype, device=device)

    def forward(self, x: SparseTensor, sample_posterior=True, return_raw=False):
        h = super().forward(x)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.out_layer(h)

        mean, logvar = h.feats.chunk(2, dim=-1)
        if sample_posterior:
            std = torch.exp(0.5 * logvar)
            z = mean + std * torch.randn_like(std)
        else:
            z = mean
        z = h.replace(z)

        if return_raw:
            return z, mean, logvar
        else:
            return z


class SLatEncoderTdfyWrapper(SLatEncoder):
    def __init__(self, *args, **kwargs):
        pretrained_ckpt_path = kwargs.pop("pretrained_ckpt_path", None)
        super().__init__(*args, **kwargs)
        if pretrained_ckpt_path is not None and os.path.exists(pretrained_ckpt_path):
            self.load_state_dict(
                comfy.utils.load_torch_file(pretrained_ckpt_path, safe_load=True)
            )


# ==========================================================================
# Structured Latent Gaussian Decoder
# ==========================================================================

class SLatGaussianDecoder(SparseTransformerBase):
    def __init__(
        self,
        resolution: int,
        model_channels: int,
        latent_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "swin",
        window_size: int = 8,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
        representation_config: dict = None,
        dtype=None, device=None, operations=ops, sparse_operations=sparse_ops,
    ):
        super().__init__(
            in_channels=latent_channels,
            model_channels=model_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            mlp_ratio=mlp_ratio,
            attn_mode=attn_mode,
            window_size=window_size,
            pe_mode=pe_mode,
            use_fp16=use_fp16,
            use_checkpoint=use_checkpoint,
            qk_rms_norm=qk_rms_norm,
            dtype=dtype, device=device,
            operations=operations,
            sparse_operations=sparse_operations,
        )
        self.resolution = resolution
        self.rep_config = representation_config
        self._calc_layout()
        self.out_layer = sparse_operations.SparseLinear(model_channels, self.out_channels, dtype=dtype, device=device)
        self._build_perturbation()

    def _build_perturbation(self) -> None:
        from .postprocessing import hammersley_sequence
        perturbation = [
            hammersley_sequence(3, i, self.rep_config["num_gaussians"])
            for i in range(self.rep_config["num_gaussians"])
        ]
        perturbation = torch.tensor(perturbation).float() * 2 - 1
        perturbation = perturbation / self.rep_config["voxel_size"]
        perturbation = torch.atanh(perturbation)
        self.register_buffer("offset_perturbation", perturbation)

    def _calc_layout(self) -> None:
        self.layout = {
            "_xyz": {
                "shape": (self.rep_config["num_gaussians"], 3),
                "size": self.rep_config["num_gaussians"] * 3,
            },
            "_features_dc": {
                "shape": (self.rep_config["num_gaussians"], 1, 3),
                "size": self.rep_config["num_gaussians"] * 3,
            },
            "_scaling": {
                "shape": (self.rep_config["num_gaussians"], 3),
                "size": self.rep_config["num_gaussians"] * 3,
            },
            "_rotation": {
                "shape": (self.rep_config["num_gaussians"], 4),
                "size": self.rep_config["num_gaussians"] * 4,
            },
            "_opacity": {
                "shape": (self.rep_config["num_gaussians"], 1),
                "size": self.rep_config["num_gaussians"],
            },
        }
        start = 0
        for k, v in self.layout.items():
            v["range"] = (start, start + v["size"])
            start += v["size"]
        self.out_channels = start

    def to_representation(self, x: SparseTensor) -> List:
        """Convert a batch of network outputs to Gaussian representations."""
        from .representations import Gaussian
        ret = []
        for i in range(x.shape[0]):
            representation = Gaussian(
                sh_degree=0,
                aabb=[-0.5, -0.5, -0.5, 1.0, 1.0, 1.0],
                mininum_kernel_size=self.rep_config["3d_filter_kernel_size"],
                scaling_bias=self.rep_config["scaling_bias"],
                opacity_bias=self.rep_config["opacity_bias"],
                scaling_activation=self.rep_config["scaling_activation"],
            )
            xyz = (x.coords[x.layout[i]][:, 1:].float() + 0.5) / self.resolution
            for k, v in self.layout.items():
                if k == "_xyz":
                    offset = x.feats[x.layout[i]][
                        :, v["range"][0] : v["range"][1]
                    ].reshape(-1, *v["shape"])
                    offset = offset * self.rep_config["lr"][k]
                    if self.rep_config["perturb_offset"]:
                        offset = offset + self.offset_perturbation.to(offset.device)
                    offset = (
                        torch.tanh(offset)
                        / self.resolution
                        * 0.5
                        * self.rep_config["voxel_size"]
                    )
                    _xyz = xyz.unsqueeze(1) + offset
                    setattr(representation, k, _xyz.flatten(0, 1))
                else:
                    feats = (
                        x.feats[x.layout[i]][:, v["range"][0] : v["range"][1]]
                        .reshape(-1, *v["shape"])
                        .flatten(0, 1)
                    )
                    feats = feats * self.rep_config["lr"][k]
                    setattr(representation, k, feats)
            ret.append(representation)
        return ret

    def forward(self, x: SparseTensor) -> List:
        h = super().forward(x)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.out_layer(h)
        return self.to_representation(h)


class SLatGaussianDecoderTdfyWrapper(SLatGaussianDecoder):
    def __init__(self, *args, **kwargs):
        pretrained_ckpt_path = kwargs.pop("pretrained_ckpt_path", None)
        super().__init__(*args, **kwargs)
        if pretrained_ckpt_path is not None:
            if os.path.exists(pretrained_ckpt_path):
                self.load_state_dict(
                    comfy.utils.load_torch_file(pretrained_ckpt_path, safe_load=True)
                )
            else:
                raise FileNotFoundError(
                    f"The path for slat decoder gs does not exist: {pretrained_ckpt_path}"
                )


# ==========================================================================
# Structured Latent Mesh Decoder
# ==========================================================================

class SparseSubdivideBlock3d(nn.Module):
    """
    A 3D subdivide block that can subdivide the sparse tensor.

    Args:
        channels: channels in the inputs and outputs.
        resolution: current resolution level.
        out_channels: if specified, the number of output channels.
        num_groups: the number of groups for the group norm.
    """

    def __init__(
        self,
        channels: int,
        resolution: int,
        out_channels: Optional[int] = None,
        num_groups: int = 32,
        dtype=None, device=None, operations=ops, sparse_operations=sparse_ops,
    ):
        super().__init__()
        self.channels = channels
        self.resolution = resolution
        self.out_resolution = resolution * 2
        self.out_channels = out_channels or channels

        self.act_layers = nn.Sequential(
            sparse_operations.SparseGroupNorm32(num_groups, channels, dtype=dtype, device=device),
            sparse_operations.SparseSiLU(),
        )

        self.sub = SparseSubdivide()

        self.out_layers = nn.Sequential(
            sparse_operations.SparseConv3d(
                channels, self.out_channels, 3,
                indice_key=f"res_{self.out_resolution}",
                dtype=dtype, device=device,
            ),
            sparse_operations.SparseGroupNorm32(num_groups, self.out_channels, dtype=dtype, device=device),
            sparse_operations.SparseSiLU(),
            zero_module(
                sparse_operations.SparseConv3d(
                    self.out_channels, self.out_channels, 3,
                    indice_key=f"res_{self.out_resolution}",
                    dtype=dtype, device=device,
                )
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = sparse_operations.SparseConv3d(
                channels, self.out_channels, 1,
                indice_key=f"res_{self.out_resolution}",
                dtype=dtype, device=device,
            )

    def forward(self, x: SparseTensor) -> SparseTensor:
        h = self.act_layers(x)
        h = self.sub(h)
        x = self.sub(x)
        h = self.out_layers(h)
        h = h + self.skip_connection(x)
        return h


class SLatMeshDecoder(SparseTransformerBase):
    def __init__(
        self,
        resolution: int,
        model_channels: int,
        latent_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "swin",
        window_size: int = 8,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
        representation_config: dict = None,
        device="cuda",
        dtype=None, operations=ops, sparse_operations=sparse_ops,
    ):
        super().__init__(
            in_channels=latent_channels,
            model_channels=model_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            mlp_ratio=mlp_ratio,
            attn_mode=attn_mode,
            window_size=window_size,
            pe_mode=pe_mode,
            use_fp16=use_fp16,
            use_checkpoint=use_checkpoint,
            qk_rms_norm=qk_rms_norm,
            dtype=dtype, device=device,
            operations=operations,
            sparse_operations=sparse_operations,
        )
        self.resolution = resolution
        self.rep_config = representation_config

        from .representations import SparseFeatures2Mesh
        self.mesh_extractor = SparseFeatures2Mesh(
            res=self.resolution * 4,
            use_color=self.rep_config.get("use_color", False),
            device=device,
        )
        self.out_channels = self.mesh_extractor.feats_channels
        self.upsample = nn.ModuleList([
            SparseSubdivideBlock3d(
                channels=model_channels,
                resolution=resolution,
                out_channels=model_channels // 4,
                dtype=dtype, device=device,
                operations=operations,
                sparse_operations=sparse_operations,
            ),
            SparseSubdivideBlock3d(
                channels=model_channels // 4,
                resolution=resolution * 2,
                out_channels=model_channels // 8,
                dtype=dtype, device=device,
                operations=operations,
                sparse_operations=sparse_operations,
            ),
        ])
        self.out_layer = sparse_operations.SparseLinear(model_channels // 8, self.out_channels, dtype=dtype, device=device)

    def to_representation(self, x: SparseTensor) -> List:
        """Convert a batch of network outputs to mesh representations."""
        from .representations import MeshExtractResult
        ret = []
        for i in range(x.shape[0]):
            mesh = self.mesh_extractor(x[i], training=self.training)
            ret.append(mesh)
        return ret

    def forward(self, x: SparseTensor) -> List:
        h = super().forward(x)
        for block in self.upsample:
            h = block(h)
        h = self.out_layer(h)
        return self.to_representation(h)


class SLatMeshDecoderTdfyWrapper(SLatMeshDecoder):
    def __init__(self, *args, **kwargs):
        pretrained_ckpt_path = kwargs.pop("pretrained_ckpt_path", None)
        super().__init__(*args, **kwargs)
        if pretrained_ckpt_path is not None and os.path.exists(pretrained_ckpt_path):
            self.load_state_dict(
                comfy.utils.load_torch_file(pretrained_ckpt_path, safe_load=True)
            )


# ==========================================================================
# Structured Latent Radiance Field Decoder
# ==========================================================================

class SLatRadianceFieldDecoder(SparseTransformerBase):
    def __init__(
        self,
        resolution: int,
        model_channels: int,
        latent_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "swin",
        window_size: int = 8,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
        representation_config: dict = None,
        dtype=None, device=None, operations=ops, sparse_operations=sparse_ops,
    ):
        super().__init__(
            in_channels=latent_channels,
            model_channels=model_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            mlp_ratio=mlp_ratio,
            attn_mode=attn_mode,
            window_size=window_size,
            pe_mode=pe_mode,
            use_fp16=use_fp16,
            use_checkpoint=use_checkpoint,
            qk_rms_norm=qk_rms_norm,
            dtype=dtype, device=device,
            operations=operations,
            sparse_operations=sparse_operations,
        )
        self.resolution = resolution
        self.rep_config = representation_config
        self._calc_layout()
        self.out_layer = sparse_operations.SparseLinear(model_channels, self.out_channels, dtype=dtype, device=device)

    def _calc_layout(self) -> None:
        self.layout = {
            "trivec": {
                "shape": (self.rep_config["rank"], 3, self.rep_config["dim"]),
                "size": self.rep_config["rank"] * 3 * self.rep_config["dim"],
            },
            "density": {
                "shape": (self.rep_config["rank"],),
                "size": self.rep_config["rank"],
            },
            "features_dc": {
                "shape": (self.rep_config["rank"], 1, 3),
                "size": self.rep_config["rank"] * 3,
            },
        }
        start = 0
        for k, v in self.layout.items():
            v["range"] = (start, start + v["size"])
            start += v["size"]
        self.out_channels = start

    def to_representation(self, x: SparseTensor) -> List:
        """Convert a batch of network outputs to radiance field representations."""
        from .representations import Strivec
        ret = []
        for i in range(x.shape[0]):
            representation = Strivec(
                sh_degree=0,
                resolution=self.resolution,
                aabb=[-0.5, -0.5, -0.5, 1, 1, 1],
                rank=self.rep_config["rank"],
                dim=self.rep_config["dim"],
                device="cuda",
            )
            representation.density_shift = 0.0
            representation.position = (
                x.coords[x.layout[i]][:, 1:].float() + 0.5
            ) / self.resolution
            representation.depth = torch.full(
                (representation.position.shape[0], 1),
                int(np.log2(self.resolution)),
                dtype=torch.uint8,
                device="cuda",
            )
            for k, v in self.layout.items():
                setattr(
                    representation,
                    k,
                    x.feats[x.layout[i]][:, v["range"][0] : v["range"][1]].reshape(
                        -1, *v["shape"]
                    ),
                )
            representation.trivec = representation.trivec + 1
            ret.append(representation)
        return ret

    def forward(self, x: SparseTensor) -> List:
        h = super().forward(x)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.out_layer(h)
        return self.to_representation(h)
