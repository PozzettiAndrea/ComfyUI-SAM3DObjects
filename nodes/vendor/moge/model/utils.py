from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    """Replace DINOv2 attention forward with comfy-attn dispatch (sage/flash/sdpa)."""
    from comfy_attn import dispatch_attention

    class _ComfyAttnWrapper(module.__class__):
        def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
            q, k, v = torch.unbind(qkv, 2)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            out = dispatch_attention(q, k, v)
            out = out.transpose(1, 2).contiguous().view(B, N, C)
            return self.proj_drop(self.proj(out))

    module.__class__ = _ComfyAttnWrapper
    return module


def sync_ddp_hook(state, bucket: torch.distributed.GradBucket) -> torch.futures.Future[torch.Tensor]:
    group_to_use = torch.distributed.group.WORLD
    world_size = group_to_use.size()
    grad = bucket.buffer()
    grad.div_(world_size)
    torch.distributed.all_reduce(grad, group=group_to_use)
    fut = torch.futures.Future()
    fut.set_result(grad)
    return fut
