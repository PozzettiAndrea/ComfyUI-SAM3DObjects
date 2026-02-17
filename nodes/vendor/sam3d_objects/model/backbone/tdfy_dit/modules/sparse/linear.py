# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch
import torch.nn.functional as F
from comfy.ops import disable_weight_init, cast_bias_weight, uncast_bias_weight
from . import SparseTensor

__all__ = ["SparseLinear"]


class SparseLinear(disable_weight_init.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(SparseLinear, self).__init__(in_features, out_features, bias)

    def forward(self, input: SparseTensor) -> SparseTensor:
        if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
            weight, bias, offload_stream = cast_bias_weight(self, input.feats, offloadable=True)
            result = F.linear(input.feats, weight, bias)
            uncast_bias_weight(self, weight, bias, offload_stream)
            return input.replace(result)
        return input.replace(F.linear(input.feats, self.weight, self.bias))
