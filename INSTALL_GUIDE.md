# Installation Guide for ComfyUI-SAM3DObjects

## Quick Start (Recommended Method)

The install script is **completely automatic** and makes the node **100% standalone**!

### One-Command Installation

```bash
cd ComfyUI/custom_nodes/ComfyUI-SAM3DObjects
python install.py
```

**That's it!** No manual repository cloning, no directory setup, no complicated steps.

### What the installer does:

1. Auto-detects conda/mamba/micromamba (uses whichever is available)
2. Installs pytorch3d via conda (fastest method)
3. **sam3d_objects code is VENDORED (already included!)**
4. Installs all other dependencies
5. Falls back to pip if conda is not available

### The node is now 100% STANDALONE!

- ✅ sam3d_objects code is vendored (included in `vendor/` directory)
- ✅ No GitHub cloning during installation
- ✅ No external repository dependencies
- ✅ Works completely offline (after initial dependency install)
- ✅ All code included in this package!

## Alternative: Install Everything with pip

If conda is not available or you prefer pip:

```bash
cd ComfyUI/custom_nodes/ComfyUI-SAM3DObjects

# This will build pytorch3d from source (takes 10-20 minutes)
pip install -r requirements.txt
```

**Note:** Building pytorch3d from source requires:
- CUDA toolkit installed
- C++ compiler (gcc/g++)
- ninja build tool
- PyTorch with CUDA support

## Troubleshooting

### pytorch3d installation fails

If you get errors installing pytorch3d:

**Option 1: Install via conda/mamba/micromamba (easiest)**
```bash
# Use whichever you have installed:
conda install -c facebookresearch pytorch3d
# OR
mamba install -c facebookresearch pytorch3d
# OR
micromamba install -c facebookresearch pytorch3d
```

**Option 2: Install from conda-forge**
```bash
# Try conda-forge channel as alternative:
conda install -c conda-forge pytorch3d
# OR
mamba install -c conda-forge pytorch3d
```

**Option 3: Build from source with build tools**
```bash
# Install build dependencies
conda install cuda-toolkit -c nvidia
conda install gcc_linux-64 gxx_linux-64 ninja

# Then install pytorch3d
pip install "git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47"
```

### spconv-cu121 installation fails

spconv requires CUDA 12.1. If you have a different CUDA version:

```bash
# For CUDA 11.8
pip install spconv-cu118

# For CUDA 12.0
pip install spconv-cu120

# Update requirements.txt accordingly
```

### MoGe installation fails

MoGe is installed from GitHub. If it fails:

```bash
pip install "git+https://github.com/microsoft/MoGe.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b"
```

### "sam3d_objects module not found"

The sam3d_objects code is vendored in the `vendor/` directory. If you see this error, ensure:

1. The `vendor/sam3d_objects/` directory exists in your custom node folder
2. The vendor directory is being added to Python path correctly

This should not happen with normal installation.

## Verification

Test the installation:

```python
# Check that vendor directory exists
import os
vendor_path = "ComfyUI/custom_nodes/ComfyUI-SAM3DObjects/vendor/sam3d_objects"
print(f"sam3d_objects vendored: {os.path.exists(vendor_path)}")

# Check pytorch3d
python -c "import pytorch3d; print(f'pytorch3d {pytorch3d.__version__}')"

# Check other key dependencies
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import spconv; print('spconv OK')"
python -c "import open3d; print('open3d OK')"
```

## Full Manual Installation

If the automatic installation doesn't work, install manually:

```bash
# Core dependencies
pip install hydra-core==1.3.2
pip install einops-exts==0.0.4
pip install roma==1.5.1
pip install easydict==1.13
pip install loguru==0.7.2

# 3D processing (skip pytorch3d if installed via conda)
conda install -c facebookresearch pytorch3d
pip install open3d==0.18.0
pip install trimesh
pip install point-cloud-utils==0.29.5
pip install xatlas==0.0.9

# Sparse convolutions (match your CUDA version)
pip install spconv-cu121==2.3.8

# Attention optimization
pip install xformers==0.0.28.post3

# Image processing
pip install opencv-python==4.9.0.80
pip install scikit-image==0.23.1

# Model dependencies
pip install timm==0.9.16
pip install transformers
pip install diffusers

# Depth model
pip install "git+https://github.com/microsoft/MoGe.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b"

# Utilities
pip install gdown==5.2.0
pip install huggingface-hub
```

## System Requirements

- **GPU:** NVIDIA GPU with 32GB+ VRAM
- **CUDA:** 12.1 or compatible
- **Python:** 3.10 or 3.11
- **OS:** Linux (recommended), macOS, Windows with WSL2

## Getting Help

If you encounter issues:
1. Check this guide for common solutions
2. Verify your CUDA version matches dependencies
3. Ensure all system requirements are met
4. Open an issue on GitHub with error details
