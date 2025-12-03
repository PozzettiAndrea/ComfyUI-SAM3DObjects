# Compiler Detection - Future Activation Guide

## Overview

The installation system now includes **cross-platform compiler detection** functions that are currently **commented out** in `nodes/env_manager.py`. These functions can detect system compilers (g++, clang++, cl.exe, nvcc) to avoid downloading them, making installation faster and smaller.

## Current Status

**Status**: DISABLED (commented out)
**Reason**: Need to verify fallback download path works first
**Location**: `nodes/env_manager.py` lines 676-880

## What's Implemented (Commented Out)

### 1. C++ Compiler Detection (`detect_cxx_compiler`)
- **Linux**: Detects `g++` or `clang++`
- **macOS**: Detects `clang++` or `g++`
- **Windows**: Detects `cl.exe` (MSVC), `g++` (MinGW), or `clang++`
- **Verifies CUDA 12.1 compatibility**:
  - g++ 7.x-12.x supported
  - clang 11.x-15.x supported
  - MSVC 2019+ supported

### 2. NVCC Compiler Detection (`detect_nvcc`)
- Checks `PATH` for `nvcc`
- Checks common CUDA installation directories:
  - Linux: `/usr/local/cuda*/bin/nvcc`
  - Windows: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.*\bin\nvcc.exe`
  - macOS: `/Developer/NVIDIA/CUDA-*/bin/nvcc`
- **Verifies CUDA 12.x version** (compatible with PyTorch 2.5.1)

### 3. Environment Setup (`_setup_compiler_environment`)
- Sets `CUDAHOSTCXX` to detected C++ compiler
- Sets `CUDA_HOME` and `CUDA_PATH` to detected CUDA toolkit
- Updates `PATH` to include compiler directories

## How to Activate (Future)

### Step 1: Uncomment Functions

In `nodes/env_manager.py`, uncomment lines 684-880:
- `detect_cxx_compiler()`
- `detect_nvcc()`
- `_verify_cxx_cuda_compatibility()`
- `_verify_nvcc_version()`
- `_setup_compiler_environment()`

### Step 2: Integrate Detection into Installation

Modify `install_dependencies()` method (around line 260):

```python
def install_dependencies(self) -> None:
    """Install all dependencies using pip/uv."""
    print("[SAM3DObjects] Installing dependencies...")

    # ... existing steps 1-5 (pip upgrade, uv, PyTorch, requirements) ...

    # Step 6: Install kaolin
    # ... existing kaolin installation ...

    # Step 7: Install pytorch3d
    # ... existing pytorch3d installation ...

    # NEW: Step 8 - Detect system compilers
    print("[SAM3DObjects] Checking for system compilers...")
    detected_cxx = self.detect_cxx_compiler()
    detected_nvcc = self.detect_nvcc()

    # Step 9: Install CUDA toolkit (only if not detected)
    if detected_nvcc:
        print("[SAM3DObjects] Using system nvcc, skipping CUDA toolkit download")
    else:
        print("[SAM3DObjects] Installing CUDA toolkit for JIT compilation...")
        if not self._install_cuda_toolkit_pypi(python_exe):
            self._install_cuda_toolkit_from_conda(python_exe)

    # Step 10: Install g++ (only if not detected)
    if detected_cxx:
        print("[SAM3DObjects] Using system C++ compiler, skipping g++ download")
    else:
        print("[SAM3DObjects] Installing g++ compiler for CUDA JIT compilation...")
        self._install_gcc_from_conda(python_exe)

    # Step 11: Setup environment variables
    compiler_env = self._setup_compiler_environment(detected_cxx, detected_nvcc)
    # TODO: Save compiler_env to a file that the inference worker can read
```

### Step 3: Pass Environment to Inference Worker

The inference worker needs to use these environment variables when running CUDA JIT compilation. Options:

1. **Save to config file**: Write `compiler_env` dict to `_env/compiler_config.json`
2. **Update inference launcher**: Modify subprocess launcher to load and apply env vars
3. **Shell script**: Generate `_env/activate_compilers.sh` that sets env vars

### Step 4: Test All Scenarios

Test the following installation scenarios:

- ✅ **Scenario A**: System has both g++ and nvcc (fastest, no downloads)
- ⬜ **Scenario B**: System has g++ but no nvcc (download CUDA only)
- ⬜ **Scenario C**: System has nvcc but no g++ (download g++ only)
- ⬜ **Scenario D**: System has neither (full download, current behavior)

Test on all platforms:
- ⬜ Ubuntu 20.04/22.04 (with/without build-essential)
- ⬜ Windows 10/11 (with/without Visual Studio)
- ⬜ macOS Monterey+ (with/without Xcode)

## Benefits When Activated

### Speed
- **No compiler downloads**: Saves 2-3 minutes
- **Smaller installation**: Saves ~500MB of compiler downloads

### Compatibility
- **Uses tested system compilers**: More likely to work with existing CUDA
- **Respects system configuration**: Doesn't override user's compiler setup

### User Experience
- **Faster for developers**: Most developers already have compilers
- **Still works for everyone**: Falls back to download if needed

## Migration Plan

### Phase 1: Current (Testing Fallback)
- All compiler detection commented out
- Always downloads compilers from conda-forge
- Verify this works on all platforms

### Phase 2: Gradual Rollout
- Uncomment detection functions
- Add environment variable `SAM3D_USE_SYSTEM_COMPILERS=1` to enable
- Default: disabled (download compilers)
- Power users can enable to test

### Phase 3: Default Enable
- Switch default to use detection
- Add `SAM3D_FORCE_DOWNLOAD_COMPILERS=1` to disable detection
- Most users benefit from faster installation

## Known Limitations

1. **MSVC on Windows**: Detection works, but MSVC setup is complex (vcvarsall.bat)
   - May need special handling for environment setup
   - Consider MinGW as primary Windows compiler

2. **macOS CUDA Support**: Apple Silicon doesn't support CUDA
   - Need CPU-only PyTorch3D for M1/M2 Macs
   - Detection should skip nvcc on macOS ARM

3. **Version Compatibility**: Strict version checks may be too conservative
   - g++ 13.x might work but is marked incompatible
   - May need to relax version checks based on testing

## Questions to Answer Before Activation

1. **Environment persistence**: How to ensure detected compilers are available during inference?
2. **Multi-user systems**: What if compiler is in user PATH but not system PATH?
3. **Container environments**: Docker containers may not have compilers, should we detect and warn?
4. **Version conflicts**: What if system has g++ 13.x (newer than conda package)?

## References

- CUDA 12.1 Compatibility: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#system-requirements
- PyTorch3D Build Requirements: https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md
- Conda Compiler Packages: https://anaconda.org/conda-forge/gxx_linux-64
