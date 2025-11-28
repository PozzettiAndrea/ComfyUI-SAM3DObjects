# SAM3D Inference Worker Environment

This folder contains configuration files for the **isolated micromamba environment** used by the SAM3D inference worker.

## Architecture

```
ComfyUI (main environment)
    ↓ spawns subprocess
SAM3D Inference Worker (isolated micromamba environment in _env/)
```

The SAM3D custom node uses **two separate Python environments**:

1. **ComfyUI Environment** (where the node runs)
   - Dependencies in `../requirements.txt`
   - Handles ComfyUI integration, node UI, subprocess management

2. **Inference Worker Environment** (isolated micromamba env in `../_env/`)
   - Dependencies in this folder
   - Handles model inference with PyTorch, PyTorch3D, CUDA
   - **This is what these files configure!**

## Files

### `environment.yml`
Minimal conda packages (only what MUST be from conda):
- python, pytorch, torchvision (CUDA-optimized builds)
- pytorch3d (prebuilt binary, saves 15+ min compilation)
- pytorch-cuda (CUDA runtime)

Solving time: ~2-3 minutes (only 7 packages)

### `requirements_env.txt`
All other packages installed via UV (10-100x faster than conda):
- Scientific: numpy, scipy, opencv, matplotlib
- ML: transformers, diffusers, timm
- 3D: open3d, trimesh, pyvista
- CUDA: xformers, gsplat, spconv-cu121
- Vendored: MoGe (in vendor/moge/), nvdiffrast (prebuilt wheel)

Install time: ~2-3 minutes

## Why This Separation?

**Performance:** Minimal conda environment solves dependencies **10x faster**:
- Before: 40+ conda packages = 12+ minutes solving
- After: 7 conda packages = 2-3 minutes solving

**Isolation:** The inference worker has different requirements than ComfyUI:
- Needs specific PyTorch/CUDA versions
- Requires heavy 3D processing libraries
- Runs in subprocess, doesn't affect main ComfyUI

## Installation

The `install.py` script automatically:
1. Creates isolated micromamba environment in `../_env/`
2. Installs minimal conda packages from `environment.yml`
3. Uses UV to install remaining packages from `requirements_env.txt`

Total time: ~4-6 minutes (down from 15+ minutes!)
