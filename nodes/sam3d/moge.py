# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
MoGe depth model + DINOv2 backbone — consolidated into ComfyUI-native format.

Contains:
- DINOv2 backbone layers: Attention, Mlp, SwiGLUFFN, PatchEmbed, Block, etc.
- DinoVisionTransformer + factory functions (vit_small/base/large/giant2)
- Hub backbone loaders (dinov2_vits14, dinov2_vitb14, dinov2_vitl14, etc.)
- MoGe v1: ResidualConvBlock, Head, MoGeModel (safetensors-based)

Key patterns:
- operations= parameter for configurable Linear/Conv2d/LayerNorm etc.
- dtype=None, device=None constructor parameters
- No torch.autocast — explicit dtype casting at boundaries
"""
import logging
import math
import os
import json
import warnings
import importlib
import itertools
import functools
from functools import partial
from typing import *
from numbers import Number
from pathlib import Path
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor
from torch.nn.init import trunc_normal_

import comfy.ops
import comfy.utils

log = logging.getLogger("sam3dobjects")

# Default operations — used only during meta-device construction.
# Real operations are resolved via pick_operations() at load time.
ops = comfy.ops.disable_weight_init


# ==========================================================================
# Utility functions
# ==========================================================================

def wrap_module_with_gradient_checkpointing(module: nn.Module):
    from torch.utils.checkpoint import checkpoint
    class _CheckpointingWrapper(module.__class__):
        _restore_cls = module.__class__
        def forward(self, *args, **kwargs):
            return checkpoint(super().forward, *args, use_reentrant=False, **kwargs)
    module.__class__ = _CheckpointingWrapper
    return module


def unwrap_module_with_gradient_checkpointing(module: nn.Module):
    module.__class__ = module.__class__._restore_cls


def wrap_dinov2_attention_with_comfy_attn(module: nn.Module):
    """Replace DINOv2 attention forward with ComfyUI native optimized attention."""
    from comfy.ldm.modules.attention import optimized_attention_for_device

    class _ComfyAttnWrapper(module.__class__):
        def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
            q, k, v = torch.unbind(qkv, 2)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            attn_fn = optimized_attention_for_device(q.device)
            out = attn_fn(q, k, v, heads=self.num_heads, skip_reshape=True, skip_output_reshape=True)
            out = out.transpose(1, 2).contiguous().view(B, N, C)
            return self.proj_drop(self.proj(out))

    module.__class__ = _ComfyAttnWrapper
    return module


def wrap_dinov2_attention_with_sdpa(module: nn.Module):
    """Replace DINOv2 attention forward with PyTorch native scaled_dot_product_attention."""
    class _SDPAWrapper(module.__class__):
        def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
            q, k, v = torch.unbind(qkv, 2)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            out = F.scaled_dot_product_attention(q, k, v)
            out = out.transpose(1, 2).contiguous().view(B, N, C)
            return self.proj_drop(self.proj(out))

    module.__class__ = _SDPAWrapper
    return module


# ==========================================================================
# DINOv2 Layers
# ==========================================================================

def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: Union[float, Tensor] = 1e-5, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        dtype=None, device=None, operations=ops,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = operations.Linear(dim, dim * 3, bias=qkv_bias, dtype=dtype, device=device)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = operations.Linear(dim, dim, bias=proj_bias, dtype=dtype, device=device)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        # Try xformers, fall back to base Attention
        try:
            from xformers.ops import memory_efficient_attention, unbind as xf_unbind
        except ImportError:
            return super().forward(x, attn_bias=attn_bias)

        if attn_bias is not None:
            # xformers path with nested tensors
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
            q, k, v = xf_unbind(qkv, 2)
            x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
            x = x.reshape([B, N, C])
            x = self.proj(x)
            x = self.proj_drop(x)
            return x

        return super().forward(x)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
        dtype=None, device=None, operations=ops,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = operations.Linear(in_features, hidden_features, bias=bias, dtype=dtype, device=device)
        self.act = act_layer()
        self.fc2 = operations.Linear(hidden_features, out_features, bias=bias, dtype=dtype, device=device)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
        dtype=None, device=None, operations=ops,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = operations.Linear(in_features, 2 * hidden_features, bias=bias, dtype=dtype, device=device)
        self.w3 = operations.Linear(hidden_features, out_features, bias=bias, dtype=dtype, device=device)

    def forward(self, x: Tensor) -> Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class SwiGLUFFNFused(SwiGLUFFN):
    """SwiGLU with fused hidden dim calculation (always uses our SwiGLUFFN, not xformers)."""
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
        dtype=None, device=None, operations=ops,
    ) -> None:
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            bias=bias,
            dtype=dtype, device=device, operations=operations,
        )


def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x
    assert isinstance(x, int)
    return (x, x)


class PatchEmbed(nn.Module):
    """2D image to patch embedding: (B,C,H,W) -> (B,N,D)"""

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten_embedding: bool = True,
        dtype=None, device=None, operations=ops,
    ) -> None:
        super().__init__()
        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (image_HW[0] // patch_HW[0], image_HW[1] // patch_HW[1])

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.flatten_embedding = flatten_embedding

        self.proj = operations.Conv2d(in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW, dtype=dtype, device=device)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        _, _, H, W = x.shape
        patch_H, patch_W = self.patch_size
        assert H % patch_H == 0 and W % patch_W == 0
        x = self.proj(x)
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)
        return x


# ==========================================================================
# DINOv2 Blocks
# ==========================================================================

class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        dtype=None, device=None, operations=ops,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias,
            attn_drop=attn_drop, proj_drop=drop,
            dtype=dtype, device=device, operations=operations,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim, hidden_features=mlp_hidden_dim,
            act_layer=act_layer, drop=drop, bias=ffn_bias,
            dtype=dtype, device=device, operations=operations,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.sample_drop_ratio = drop_path

    def forward(self, x: Tensor) -> Tensor:
        def attn_residual_func(x):
            return self.ls1(self.attn(self.norm1(x)))
        def ffn_residual_func(x):
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            x = _drop_add_residual_stochastic_depth(x, attn_residual_func, self.sample_drop_ratio)
            x = _drop_add_residual_stochastic_depth(x, ffn_residual_func, self.sample_drop_ratio)
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x))
            x = x + self.drop_path1(ffn_residual_func(x))
        else:
            x = x + attn_residual_func(x)
            x = x + ffn_residual_func(x)
        return x


def _drop_add_residual_stochastic_depth(x, residual_func, sample_drop_ratio=0.0):
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]
    residual = residual_func(x_subset)
    x_flat = x.flatten(1)
    residual = residual.flatten(1)
    residual_scale_factor = b / sample_subset_size
    x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    return x_plus_residual.view_as(x)


class NestedTensorBlock(Block):
    """Block with nested tensor support for xformers."""
    def forward(self, x_or_x_list):
        if isinstance(x_or_x_list, Tensor):
            return super().forward(x_or_x_list)
        elif isinstance(x_or_x_list, list):
            return self._forward_nested(x_or_x_list)
        else:
            raise AssertionError

    def _forward_nested(self, x_list):
        try:
            from xformers.ops import fmha, index_select_cat
        except ImportError:
            raise AssertionError("xFormers is required for nested tensors")

        def attn_residual_func(x, attn_bias=None):
            return self.ls1(self.attn(self.norm1(x), attn_bias=attn_bias))
        def ffn_residual_func(x, attn_bias=None):
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.0:
            # Stochastic depth with nested tensors
            from xformers.ops import scaled_index_add
            def get_branges_scales(x_):
                b_, n_, d_ = x_.shape
                subset = max(int(b_ * (1 - self.sample_drop_ratio)), 1)
                br = torch.randperm(b_, device=x_.device)[:subset]
                return br, b_ / subset

            branges_scales = [get_branges_scales(x_) for x_ in x_list]
            branges = [s[0] for s in branges_scales]
            res_scales = [s[1] for s in branges_scales]

            batch_sizes = [br.shape[0] for br in branges]
            all_shapes = tuple((b_, x_.shape[1]) for b_, x_ in zip(batch_sizes, x_list))
            seqlens = []
            for b_, x_ in zip(batch_sizes, x_list):
                for _ in range(b_):
                    seqlens.append(x_.shape[1])
            attn_bias = fmha.BlockDiagonalMask.from_seqlens(seqlens)
            attn_bias._batch_sizes = batch_sizes
            cat_tensors = index_select_cat([x_.flatten(1) for x_ in x_list], branges).view(1, -1, x_list[0].shape[-1])

            residual_list = attn_bias.split(attn_residual_func(cat_tensors, attn_bias=attn_bias))
            outputs = []
            for x_, br, res, scale in zip(x_list, branges, residual_list, res_scales):
                sv = self.ls1.gamma if isinstance(self.ls1, LayerScale) else None
                if sv is None:
                    x_flat = x_.flatten(1)
                    res_flat = res.flatten(1)
                    out = torch.index_add(x_flat, 0, br, res_flat.to(dtype=x_.dtype), alpha=scale)
                else:
                    out = scaled_index_add(x_, br, res.to(dtype=x_.dtype), scaling=sv, alpha=scale)
                outputs.append(out.view_as(x_))
            x_list = outputs

            # Repeat for FFN
            branges_scales = [get_branges_scales(x_) for x_ in x_list]
            branges = [s[0] for s in branges_scales]
            res_scales = [s[1] for s in branges_scales]
            batch_sizes = [br.shape[0] for br in branges]
            all_shapes2 = tuple((b_, x_.shape[1]) for b_, x_ in zip(batch_sizes, x_list))
            seqlens2 = []
            for b_, x_ in zip(batch_sizes, x_list):
                for _ in range(b_):
                    seqlens2.append(x_.shape[1])
            attn_bias2 = fmha.BlockDiagonalMask.from_seqlens(seqlens2)
            attn_bias2._batch_sizes = batch_sizes
            cat_tensors2 = index_select_cat([x_.flatten(1) for x_ in x_list], branges).view(1, -1, x_list[0].shape[-1])

            residual_list2 = attn_bias2.split(ffn_residual_func(cat_tensors2))
            outputs2 = []
            for x_, br, res, scale in zip(x_list, branges, residual_list2, res_scales):
                sv = self.ls2.gamma if isinstance(self.ls2, LayerScale) else None
                if sv is None:
                    x_flat = x_.flatten(1)
                    res_flat = res.flatten(1)
                    out = torch.index_add(x_flat, 0, br, res_flat.to(dtype=x_.dtype), alpha=scale)
                else:
                    out = scaled_index_add(x_, br, res.to(dtype=x_.dtype), scaling=sv, alpha=scale)
                outputs2.append(out.view_as(x_))
            return outputs2
        else:
            batch_sizes = [x_.shape[0] for x_ in x_list]
            all_shapes = tuple((b_, x_.shape[1]) for b_, x_ in zip(batch_sizes, x_list))
            seqlens = []
            for b_, x_ in zip(batch_sizes, x_list):
                for _ in range(b_):
                    seqlens.append(x_.shape[1])
            attn_bias = fmha.BlockDiagonalMask.from_seqlens(seqlens)
            attn_bias._batch_sizes = batch_sizes
            tensors_bs1 = tuple(x_.reshape([1, -1, *x_.shape[2:]]) for x_ in x_list)
            x = torch.cat(tensors_bs1, dim=1)
            x = x + attn_residual_func(x, attn_bias=attn_bias)
            x = x + ffn_residual_func(x)
            return attn_bias.split(x)


class BlockChunk(nn.ModuleList):
    def forward(self, x):
        for b in self:
            x = b(x)
        return x


# ==========================================================================
# DINOv2 Vision Transformer
# ==========================================================================

def named_apply(fn, module, name="", depth_first=True, include_root=False):
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


def init_weights_vit_timm(module, name=""):
    """ViT weight initialization, original timm impl."""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=None,
        ffn_layer="mlp",
        block_chunks=1,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        dtype=None, device=None, operations=ops,
    ):
        super().__init__()
        if block_fn is None:
            block_fn = partial(NestedTensorBlock, attn_class=MemEffAttention)

        norm_layer = partial(operations.LayerNorm, eps=1e-6, dtype=dtype, device=device)

        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            dtype=dtype, device=device, operations=operations,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None
        )

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]

        if ffn_layer == "mlp":
            ffn_layer_cls = Mlp
        elif ffn_layer in ("swiglufused", "swiglu"):
            ffn_layer_cls = SwiGLUFFNFused
        elif ffn_layer == "identity":
            ffn_layer_cls = lambda *args, **kwargs: nn.Identity()
        else:
            raise NotImplementedError(f"Unknown ffn_layer: {ffn_layer}")

        blocks_list = [
            block_fn(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, proj_bias=proj_bias, ffn_bias=ffn_bias,
                drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                ffn_layer=ffn_layer_cls, init_values=init_values,
                dtype=dtype, device=device, operations=operations,
            )
            for i in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                chunked_blocks.append([nn.Identity()] * i + blocks_list[i : i + chunksize])
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        self.init_weights()

    def init_weights(self):
        if self.pos_embed.device.type == 'meta':
            return
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))
        assert N == M * M
        kwargs = {}
        if self.interpolate_offset:
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sx, sy)
        else:
            kwargs["size"] = (w0, h0)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic", antialias=self.interpolate_antialias, **kwargs,
        )
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        if self.register_tokens is not None:
            x = torch.cat((x[:, :1], self.register_tokens.expand(x.shape[0], -1, -1), x[:, 1:]), dim=1)
        return x

    def forward_features_list(self, x_list, masks_list):
        x = [self.prepare_tokens_with_masks(x_, masks) for x_, masks in zip(x_list, masks_list)]
        for blk in self.blocks:
            x = blk(x)
        output = []
        for x_, masks in zip(x, masks_list):
            x_norm = self.norm(x_)
            output.append({
                "x_norm_clstoken": x_norm[:, 0],
                "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
                "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
                "x_prenorm": x_,
                "masks": masks,
            })
        return output

    def forward_features(self, x, masks=None):
        if isinstance(x, list):
            return self.forward_features_list(x, masks)
        x = self.prepare_tokens_with_masks(x, masks)
        for blk in self.blocks:
            x = blk(x)
        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
            "x_prenorm": x,
            "masks": masks,
        }

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take)
        return output

    def _get_intermediate_layers_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output, i, total_block_len = [], 0, len(self.blocks[-1])
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:
                x = blk(x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
        assert len(output) == len(blocks_to_take)
        return output

    def get_intermediate_layers(self, x, n=1, reshape=False, return_class_token=False, norm=True):
        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(x, n)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1 + self.num_register_tokens :] for out in outputs]
        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward(self, *args, is_training=False, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        return self.head(ret["x_norm_clstoken"])


# ==========================================================================
# DINOv2 Factory Functions
# ==========================================================================

def vit_small(patch_size=16, num_register_tokens=0, dtype=None, device=None, operations=ops, **kwargs):
    return DinoVisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        block_fn=partial(NestedTensorBlock, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        dtype=dtype, device=device, operations=operations, **kwargs,
    )


def vit_base(patch_size=16, num_register_tokens=0, dtype=None, device=None, operations=ops, **kwargs):
    return DinoVisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        block_fn=partial(NestedTensorBlock, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        dtype=dtype, device=device, operations=operations, **kwargs,
    )


def vit_large(patch_size=16, num_register_tokens=0, dtype=None, device=None, operations=ops, **kwargs):
    return DinoVisionTransformer(
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
        block_fn=partial(NestedTensorBlock, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        dtype=dtype, device=device, operations=operations, **kwargs,
    )


def vit_giant2(patch_size=16, num_register_tokens=0, dtype=None, device=None, operations=ops, **kwargs):
    return DinoVisionTransformer(
        patch_size=patch_size, embed_dim=1536, depth=40, num_heads=24, mlp_ratio=4,
        block_fn=partial(NestedTensorBlock, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        dtype=dtype, device=device, operations=operations, **kwargs,
    )


_ARCH_REGISTRY = {
    "vit_small": vit_small,
    "vit_base": vit_base,
    "vit_large": vit_large,
    "vit_giant2": vit_giant2,
}


# ==========================================================================
# Hub Backbone Loaders
# ==========================================================================

def _make_dinov2_model(
    *,
    arch_name="vit_large",
    img_size=518,
    patch_size=14,
    init_values=1.0,
    ffn_layer="mlp",
    block_chunks=0,
    num_register_tokens=0,
    interpolate_antialias=False,
    interpolate_offset=0.1,
    dtype=None, device=None, operations=ops,
    **kwargs,
):
    vit_kwargs = dict(
        img_size=img_size, patch_size=patch_size, init_values=init_values,
        ffn_layer=ffn_layer, block_chunks=block_chunks,
        num_register_tokens=num_register_tokens,
        interpolate_antialias=interpolate_antialias, interpolate_offset=interpolate_offset,
    )
    vit_kwargs.update(**kwargs)
    # Strip hub-style kwargs that the underlying ViT class doesn't accept.
    # Weights are loaded separately via safetensors, so `pretrained` is a no-op here.
    vit_kwargs.pop("pretrained", None)
    return _ARCH_REGISTRY[arch_name](dtype=dtype, device=device, operations=operations, **vit_kwargs)


def dinov2_vits14(*, dtype=None, device=None, operations=ops, **kwargs):
    return _make_dinov2_model(arch_name="vit_small", dtype=dtype, device=device, operations=operations, **kwargs)

def dinov2_vitb14(*, dtype=None, device=None, operations=ops, **kwargs):
    return _make_dinov2_model(arch_name="vit_base", dtype=dtype, device=device, operations=operations, **kwargs)

def dinov2_vitl14(*, dtype=None, device=None, operations=ops, **kwargs):
    return _make_dinov2_model(arch_name="vit_large", dtype=dtype, device=device, operations=operations, **kwargs)

def dinov2_vitl14_reg(*, dtype=None, device=None, operations=ops, **kwargs):
    return _make_dinov2_model(arch_name="vit_large", num_register_tokens=4, dtype=dtype, device=device, operations=operations, **kwargs)

def dinov2_vitg14(*, dtype=None, device=None, operations=ops, **kwargs):
    return _make_dinov2_model(arch_name="vit_giant2", ffn_layer="swiglufused", dtype=dtype, device=device, operations=operations, **kwargs)

# Lookup table for hub-style loading (used by MoGeModelV1)
_HUB_REGISTRY = {
    "dinov2_vits14": dinov2_vits14,
    "dinov2_vitb14": dinov2_vitb14,
    "dinov2_vitl14": dinov2_vitl14,
    "dinov2_vitl14_reg": dinov2_vitl14_reg,
    "dinov2_vitg14": dinov2_vitg14,
}


def get_dinov2_backbone(name, dtype=None, device=None, operations=ops, **kwargs):
    """Create a DINOv2 backbone architecture by name (no pretrained weights)."""
    if name in _HUB_REGISTRY:
        return _HUB_REGISTRY[name](dtype=dtype, device=device, operations=operations, **kwargs)
    raise ValueError(f"Unknown DINOv2 backbone: {name}. Available: {list(_HUB_REGISTRY.keys())}")


# ==========================================================================
# MoGe v1 — Depth estimation model (safetensors-based)
# ==========================================================================

class ResidualConvBlockV1(nn.Module):
    """Residual conv block used in MoGe v1 Head."""
    def __init__(
        self, in_channels, out_channels=None, hidden_channels=None,
        padding_mode='replicate',
        activation: Literal['relu', 'leaky_relu', 'silu', 'elu'] = 'relu',
        norm: Literal['group_norm', 'layer_norm'] = 'group_norm',
        dtype=None, device=None, operations=ops,
    ):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        if hidden_channels is None:
            hidden_channels = in_channels

        act_map = {
            'relu': lambda: nn.ReLU(inplace=True),
            'leaky_relu': lambda: nn.LeakyReLU(negative_slope=0.2, inplace=True),
            'silu': lambda: nn.SiLU(inplace=True),
            'elu': lambda: nn.ELU(inplace=True),
        }
        activation_cls = act_map.get(activation)
        if activation_cls is None:
            raise ValueError(f'Unsupported activation: {activation}')

        self.layers = nn.Sequential(
            operations.GroupNorm(1, in_channels, dtype=dtype, device=device),
            activation_cls(),
            operations.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, padding_mode=padding_mode, dtype=dtype, device=device),
            operations.GroupNorm(hidden_channels // 32 if norm == 'group_norm' else 1, hidden_channels, dtype=dtype, device=device),
            activation_cls(),
            operations.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1, padding_mode=padding_mode, dtype=dtype, device=device),
        )
        self.skip_connection = (
            operations.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, dtype=dtype, device=device)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        return self.layers(x) + self.skip_connection(x)


class Head(nn.Module):
    """MoGe v1 decoder head."""
    def __init__(
        self,
        num_features, dim_in, dim_out, dim_proj=512,
        dim_upsample=(256, 128, 128),
        dim_times_res_block_hidden=1, num_res_blocks=1,
        res_block_norm='group_norm',
        last_res_blocks=0, last_conv_channels=32, last_conv_size=1,
        dtype=None, device=None, operations=ops,
    ):
        super().__init__()
        self.projects = nn.ModuleList([
            operations.Conv2d(in_channels=dim_in, out_channels=dim_proj, kernel_size=1, stride=1, padding=0, dtype=dtype, device=device)
            for _ in range(num_features)
        ])

        self.upsample_blocks = nn.ModuleList([
            nn.Sequential(
                self._make_upsampler(in_ch + 2, out_ch, dtype=dtype, device=device, operations=operations),
                *(ResidualConvBlockV1(out_ch, out_ch, dim_times_res_block_hidden * out_ch, activation="relu", norm=res_block_norm, dtype=dtype, device=device, operations=operations) for _ in range(num_res_blocks))
            ) for in_ch, out_ch in zip([dim_proj] + list(dim_upsample[:-1]), dim_upsample)
        ])

        self.output_block = nn.ModuleList([
            self._make_output_block(
                dim_upsample[-1] + 2, dim_out_, dim_times_res_block_hidden,
                last_res_blocks, last_conv_channels, last_conv_size, res_block_norm,
                dtype=dtype, device=device, operations=operations,
            ) for dim_out_ in dim_out
        ])

    def _make_upsampler(self, in_channels, out_channels, dtype=None, device=None, operations=ops):
        upsampler = nn.Sequential(
            operations.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, dtype=dtype, device=device),
            operations.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='replicate', dtype=dtype, device=device),
        )
        if upsampler[0].weight.device.type != 'meta':
            upsampler[0].weight.data[:] = upsampler[0].weight.data[:, :, :1, :1]
        return upsampler

    def _make_output_block(self, dim_in, dim_out, dim_times_res_block_hidden, last_res_blocks, last_conv_channels, last_conv_size, res_block_norm, dtype=None, device=None, operations=ops):
        return nn.Sequential(
            operations.Conv2d(dim_in, last_conv_channels, kernel_size=3, stride=1, padding=1, padding_mode='replicate', dtype=dtype, device=device),
            *(ResidualConvBlockV1(last_conv_channels, last_conv_channels, dim_times_res_block_hidden * last_conv_channels, activation='relu', norm=res_block_norm, dtype=dtype, device=device, operations=operations) for _ in range(last_res_blocks)),
            nn.ReLU(inplace=True),
            operations.Conv2d(last_conv_channels, dim_out, kernel_size=last_conv_size, stride=1, padding=last_conv_size // 2, padding_mode='replicate', dtype=dtype, device=device),
        )

    def forward(self, hidden_states, image):
        from .geometry import normalized_view_plane_uv
        img_h, img_w = image.shape[-2:]
        patch_h, patch_w = img_h // 14, img_w // 14

        x = torch.stack([
            proj(feat.permute(0, 2, 1).unflatten(2, (patch_h, patch_w)).contiguous())
            for proj, (feat, clstoken) in zip(self.projects, hidden_states)
        ], dim=1).sum(dim=1)

        for i, block in enumerate(self.upsample_blocks):
            uv = normalized_view_plane_uv(width=x.shape[-1], height=x.shape[-2], aspect_ratio=img_w / img_h, dtype=x.dtype, device=x.device)
            uv = uv.permute(2, 0, 1).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
            x = torch.cat([x, uv], dim=1)
            for layer in block:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)

        x = F.interpolate(x, (img_h, img_w), mode="bilinear", align_corners=False)
        uv = normalized_view_plane_uv(width=x.shape[-1], height=x.shape[-2], aspect_ratio=img_w / img_h, dtype=x.dtype, device=x.device)
        uv = uv.permute(2, 0, 1).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
        x = torch.cat([x, uv], dim=1)

        if isinstance(self.output_block, nn.ModuleList):
            output = [torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False) for block in self.output_block]
        else:
            output = torch.utils.checkpoint.checkpoint(self.output_block, x, use_reentrant=False)
        return output


class MoGeModelV1(nn.Module):
    """MoGe v1 depth estimation model (safetensors-based, used by SAM3DObjects)."""
    image_mean: torch.Tensor
    image_std: torch.Tensor

    def __init__(
        self,
        encoder='dinov2_vitb14',
        intermediate_layers=4,
        dim_proj=512,
        dim_upsample=(256, 128, 128),
        dim_times_res_block_hidden=1,
        num_res_blocks=1,
        remap_output='linear',
        res_block_norm='group_norm',
        num_tokens_range=(1200, 2500),
        last_res_blocks=0,
        last_conv_channels=32,
        last_conv_size=1,
        mask_threshold=0.5,
        dtype=None, device=None, operations=ops,
        **deprecated_kwargs
    ):
        super().__init__()
        if deprecated_kwargs:
            if 'trained_area_range' in deprecated_kwargs:
                num_tokens_range = [deprecated_kwargs['trained_area_range'][0] // 14 ** 2, deprecated_kwargs['trained_area_range'][1] // 14 ** 2]
                del deprecated_kwargs['trained_area_range']
            warnings.warn(f"Deprecated arguments ignored: {deprecated_kwargs}")

        self.encoder = encoder
        self.remap_output = remap_output
        self.intermediate_layers = intermediate_layers
        self.num_tokens_range = num_tokens_range
        self.mask_threshold = mask_threshold

        self.backbone = get_dinov2_backbone(encoder, dtype=dtype, device=device, operations=operations)
        dim_feature = self.backbone.embed_dim

        self.head = Head(
            num_features=intermediate_layers if isinstance(intermediate_layers, int) else len(intermediate_layers),
            dim_in=dim_feature, dim_out=[3, 1],
            dim_proj=dim_proj, dim_upsample=list(dim_upsample),
            dim_times_res_block_hidden=dim_times_res_block_hidden,
            num_res_blocks=num_res_blocks, res_block_norm=res_block_norm,
            last_res_blocks=last_res_blocks, last_conv_channels=last_conv_channels,
            last_conv_size=last_conv_size,
            dtype=dtype, device=device, operations=operations,
        )

        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406], dtype=dtype).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225], dtype=dtype).view(1, 3, 1, 1))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, model_kwargs=None, dtype=None, **hf_kwargs):
        ckpt_path = Path(pretrained_model_name_or_path)
        config_path = ckpt_path.with_name(ckpt_path.stem + "_config.json")
        with open(config_path) as f:
            model_config = json.load(f)
        if model_kwargs is not None:
            model_config.update(model_kwargs)

        # Fast loading: build on meta device (zero allocation), then assign weights
        with torch.device("meta"):
            model = cls(**model_config)

        state_dict = comfy.utils.load_torch_file(str(ckpt_path))
        model.load_state_dict(state_dict, strict=False, assign=True)

        # Fix buffers left on meta (image_mean/std, etc.)
        for name, buf in list(model.named_buffers()):
            if buf.device.type == "meta":
                parts = name.split(".")
                parent = model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                parent._buffers[parts[-1]] = torch.zeros_like(buf, device="cpu")

        # Cast to target dtype if specified
        if dtype is not None:
            model.to(dtype=dtype)

        # Enable optimized attention dispatch
        model.enable_comfy_attn()

        log.info("MoGe v1 loaded from %s (%d tensors, meta-device fast load)", ckpt_path.name, len(state_dict))
        return model

    def enable_comfy_attn(self):
        blocks = self.backbone.blocks
        if isinstance(blocks[0], BlockChunk):
            # chunked blocks
            for chunk in blocks:
                for blk in chunk:
                    if hasattr(blk, 'attn'):
                        wrap_dinov2_attention_with_comfy_attn(blk.attn)
        else:
            for blk in blocks:
                if hasattr(blk, 'attn'):
                    wrap_dinov2_attention_with_comfy_attn(blk.attn)

    def _remap_points(self, points):
        if self.remap_output == 'linear':
            pass
        elif self.remap_output == 'sinh':
            points = torch.sinh(points)
        elif self.remap_output == 'exp':
            xy, z = points.split([2, 1], dim=-1)
            z = torch.exp(z)
            points = torch.cat([xy * z, z], dim=-1)
        elif self.remap_output == 'sinh_exp':
            xy, z = points.split([2, 1], dim=-1)
            points = torch.cat([torch.sinh(xy), torch.exp(z)], dim=-1)
        else:
            raise ValueError(f"Invalid remap output type: {self.remap_output}")
        return points

    def forward(self, image, num_tokens):
        original_height, original_width = image.shape[-2:]

        resize_factor = ((num_tokens * 14 ** 2) / (original_height * original_width)) ** 0.5
        resized_width = int(original_width * resize_factor)
        resized_height = int(original_height * resize_factor)
        image = F.interpolate(image, (resized_height, resized_width), mode="bicubic", align_corners=False, antialias=True)

        image = (image - self.image_mean.to(image.device, dtype=image.dtype)) / self.image_std.to(image.device, dtype=image.dtype)
        image_14 = F.interpolate(image, (resized_height // 14 * 14, resized_width // 14 * 14), mode="bilinear", align_corners=False, antialias=True)

        features = self.backbone.get_intermediate_layers(image_14, self.intermediate_layers, return_class_token=True)

        output = self.head(features, image)
        points, mask = output

        # Explicit fp32 for output interpolation (replaces torch.autocast)
        points = F.interpolate(points.float(), (original_height, original_width), mode='bilinear', align_corners=False, antialias=False)
        mask = F.interpolate(mask.float(), (original_height, original_width), mode='bilinear', align_corners=False, antialias=False)

        points, mask = points.permute(0, 2, 3, 1), mask.squeeze(1)
        points = self._remap_points(points)

        return {'points': points, 'mask': mask}

    @torch.inference_mode()
    def infer(self, image, fov_x=None, resolution_level=9, num_tokens=None, apply_mask=True, force_projection=True, use_fp16=True):
        import utils3d
        from .geometry import recover_focal_shift

        if image.dim() == 3:
            omit_batch_dim = True
            image = image.unsqueeze(0)
        else:
            omit_batch_dim = False

        original_height, original_width = image.shape[-2:]
        aspect_ratio = original_width / original_height

        if num_tokens is None:
            min_tokens, max_tokens = self.num_tokens_range
            num_tokens = int(min_tokens + (resolution_level / 9) * (max_tokens - min_tokens))

        # Run forward (no autocast — caller controls dtype)
        output = self.forward(image, num_tokens)
        points, mask = output['points'], output['mask']

        mask_binary = mask > self.mask_threshold

        if fov_x is None:
            focal, shift = recover_focal_shift(points, mask_binary)
        else:
            focal = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5 / torch.tan(torch.deg2rad(torch.as_tensor(fov_x, device=points.device, dtype=points.dtype) / 2))
            if focal.ndim == 0:
                focal = focal[None].expand(points.shape[0])
            _, shift = recover_focal_shift(points, mask_binary, focal=focal)

        fx = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio
        fy = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5
        cx = torch.tensor(0.5, device=fx.device, dtype=fx.dtype)
        cy = torch.tensor(0.5, device=fx.device, dtype=fx.dtype)
        intrinsics = utils3d.torch.intrinsics_from_focal_center(fx, fy, cx, cy)
        depth = points[..., 2] + shift[..., None, None]

        if force_projection:
            points = utils3d.torch.depth_to_points(depth, intrinsics=intrinsics)
        else:
            points = points + torch.stack([torch.zeros_like(shift), torch.zeros_like(shift), shift], dim=-1)[..., None, None, :]

        if apply_mask:
            points = torch.where(mask_binary[..., None], points, torch.inf)
            depth = torch.where(mask_binary, depth, torch.inf)

        if omit_batch_dim:
            points, intrinsics, depth = points.squeeze(0), intrinsics.squeeze(0), depth.squeeze(0)
            mask_binary, mask = mask_binary.squeeze(0), mask.squeeze(0)

        return {
            'points': points, 'intrinsics': intrinsics, 'depth': depth,
            'mask': mask_binary, 'mask_prob': torch.sigmoid(mask),
        }


MoGeModel = MoGeModelV1
