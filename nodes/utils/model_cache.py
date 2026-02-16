"""
Model cache for SAM3D inference.

Supports three memory strategies:
- cache_gpu: keep models on GPU between runs (fastest, most VRAM)
- cpu_offload: move models to CPU RAM after use (moderate speed, frees VRAM)
- delete: delete models after use (slowest, least memory)
"""

import gc
import torch
import comfy.model_management

# Module-level cache: key -> model (on CPU or GPU)
_CACHE = {}


def offload(key, model, mode="cpu_offload"):
    """Offload a model according to memory strategy."""
    if mode == "cache_gpu":
        _CACHE[key] = model
    elif mode == "cpu_offload":
        model.cpu()
        _CACHE[key] = model
        comfy.model_management.soft_empty_cache()
    else:
        del model
        gc.collect()
        comfy.model_management.soft_empty_cache()


def try_load(key):
    """Try to load a cached model. Returns model on GPU, or None."""
    if key not in _CACHE:
        return None
    model = _CACHE[key]
    device = comfy.model_management.get_torch_device()
    if next(model.parameters()).device != device:
        model = model.to(device)
        _CACHE[key] = model
    return model


def clear():
    """Clear all cached models and free memory."""
    global _CACHE
    for model in _CACHE.values():
        del model
    _CACHE.clear()
    gc.collect()
    comfy.model_management.soft_empty_cache()
