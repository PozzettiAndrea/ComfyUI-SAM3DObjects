# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch
from comfy.ops import disable_weight_init


class LayerNorm32(disable_weight_init.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)


class GroupNorm32(disable_weight_init.GroupNorm):
    """
    A GroupNorm layer that converts to float32 before the forward pass.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)


class ChannelLayerNorm32(LayerNorm32):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        DIM = x.dim()
        x = x.permute(0, *range(2, DIM), 1).contiguous()
        x = super().forward(x)
        x = x.permute(0, DIM - 1, *range(1, DIM - 1)).contiguous()
        return x
