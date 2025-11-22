# ComfyUI-SAM3DObjects: Installation & Debug History

This document tracks all the problems encountered during the development and installation of ComfyUI-SAM3DObjects, and how they were resolved.

---

## Problem 1: JSON Communication Errors

**Date**: 2025-11-22

**Error**:
```
json.decoder.JSONDecodeError: Expecting value: line 1 column 2 (char 1)
```

**Root Cause**:
The inference worker uses JSON over stdin/stdout for IPC (inter-process communication) between the main ComfyUI process and the isolated SAM3D environment. However, Python libraries like OmegaConf, Hydra, PyTorch, and CUDA were printing debug messages and warnings directly to stdout, polluting the JSON response stream.

**Impact**:
- Worker subprocess couldn't communicate with main process
- All inference requests failed with JSON parsing errors

**Solution**:
1. **In `inference_worker.py`** (lines 275-292): Added comprehensive output suppression:
   ```python
   import warnings
   import logging
   warnings.filterwarnings("ignore")
   os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow
   os.environ['HYDRA_FULL_ERROR'] = '0'       # Suppress Hydra traces
   logging.disable(logging.CRITICAL)           # Disable all logging
   ```

2. **In `nodes/subprocess_bridge.py`** (lines 156-193): Added defensive JSON parsing that skips non-JSON lines:
   ```python
   # Skip lines that don't start with { or [
   if not (response_line.startswith('{') or response_line.startswith('[')):
       continue
   ```

**Status**: ✅ Resolved

---

## Problem 2: UTF-8 Encoding Errors During JIT Compilation

**Date**: 2025-11-22

**Error**:
```
UnicodeDecodeError: 'ascii' codec can't decode byte 0xe2 in position 916: ordinal not in range(128)
```

**Root Cause**:
PyTorch's JIT compiler reads CUDA extension source files to generate version hashes. Some source files (particularly gsplat) contain UTF-8 characters (like em-dashes or special symbols in comments). Python was defaulting to ASCII encoding when reading these files, causing decoding errors.

**Impact**:
- gsplat and other CUDA extensions failed to compile
- JIT compilation crashed before even attempting to build

**Solution**:
**In `inference_worker.py`** (lines 46-50): Force UTF-8 encoding globally:
```python
os.environ['PYTHONIOENCODING'] = 'utf-8'  # Force UTF-8 for file I/O
os.environ['PYTHONUTF8'] = '1'            # PEP 540: Force UTF-8 mode
```

**Status**: ✅ Resolved

---

## Problem 3: G++ Compiler Not Found by nvcc

**Date**: 2025-11-22

**Error**:
```
gcc: fatal error: cannot execute 'cc1plus': No such file or directory
```

**Root Cause**:
We installed g++ via micromamba, which provides cross-compilation wrappers named `x86_64-conda-linux-gnu-g++` and `x86_64-conda-linux-gnu-gcc`. However, nvcc (NVIDIA's CUDA compiler) expects standard compiler names (`g++` and `gcc`) and doesn't understand conda's wrapper naming convention.

**Impact**:
- nvcc couldn't find the C++ compiler to compile CUDA host code
- JIT compilation failed at the linking stage

**Solution**:
**In `inference_worker.py`** (lines 67-94): Create symlinks with standard names:
```python
wrapper_gxx = gcc_bin / "x86_64-conda-linux-gnu-g++"
wrapper_gcc = gcc_bin / "x86_64-conda-linux-gnu-gcc"
symlink_gxx = gcc_bin / "g++"
symlink_gcc = gcc_bin / "gcc"

if wrapper_gxx.exists() and not symlink_gxx.exists():
    symlink_gxx.symlink_to(wrapper_gxx.name)
if wrapper_gcc.exists() and not symlink_gcc.exists():
    symlink_gcc.symlink_to(wrapper_gcc.name)

os.environ['CXX'] = 'g++'
os.environ['CC'] = 'gcc'
```

**Status**: ✅ Resolved

---

## Problem 4: PyTorch Version Incompatibility

**Date**: 2025-11-22

**Error**:
PyTorch3D prebuilt wheels only available for PyTorch 2.4.x, not 2.5.x

**Root Cause**:
We initially used PyTorch 2.5.1 (latest version), but PyTorch3D only provides prebuilt conda packages for PyTorch 2.4.x. Binary incompatibility between PyTorch versions means we can't mix PyTorch 2.5 with PyTorch3D built for 2.4.

**Impact**:
- Could not install PyTorch3D without building from source
- Building PyTorch3D from source is extremely slow (30+ minutes)

**Solution**:
**In `nodes/env_manager.py`**: Downgraded PyTorch to 2.4.1:
```python
pytorch_version = "2.4.1"  # Changed from 2.5.1
# Using PyTorch 2.4.1 - latest version with PyTorch3D prebuilt support
```

**Status**: ✅ Resolved

---

## Problem 5: gsplat JIT Compilation Failures - Incomplete PyPI CUDA Toolkit

**Date**: 2025-11-22

**Error**:
```
fatal error: nv/target: No such file or directory
#include <nv/target>
         ^~~~~~~~~~~
```

**Root Cause**:
We attempted to install CUDA toolkit from PyPI (`nvidia-cuda-nvcc-cu13`) to enable JIT compilation of gsplat. However, the PyPI CUDA toolkit is **incomplete** - it only provides the nvcc compiler binary, but is missing critical CUDA SDK header files like:
- `nv/target`
- `cuda_fp16.h`
- `cuda_bf16.h`
- And many others

The PyPI package is intended for basic compilation tasks, not full CUDA extension builds.

**Impact**:
- gsplat JIT compilation failed
- No way to compile gsplat from source without full CUDA toolkit

**Initial Attempted Solutions (Failed)**:
1. ❌ Install CUDA toolkit from PyPI - incomplete headers
2. ❌ Try different nvcc paths - still missing headers
3. ❌ Set CUDA_HOME correctly - toolkit itself is incomplete

**Why JIT Compilation is Problematic**:
- Requires full NVIDIA CUDA toolkit (~3GB download)
- Requires matching g++ compiler version
- Compilation takes 5-10 minutes per extension
- Error-prone with many compatibility issues
- Not portable across different environments

**Status**: ⚠️ Attempted but abandoned JIT approach

---

## Problem 6: gsplat Prebuilt Wheels Only for Python 3.10

**Date**: 2025-11-22

**Discovery**:
gsplat provides prebuilt wheels at https://docs.gsplat.studio/whl/pt24cu121/ to avoid JIT compilation, but these wheels are only built for Python 3.10 (cp310), not Python 3.11 (cp311).

Available wheels:
```
gsplat-1.4.0+pt24cu121-cp310-cp310-linux_x86_64.whl  ✅ (Python 3.10)
```

Missing wheels:
```
gsplat-1.4.0+pt24cu121-cp311-cp311-linux_x86_64.whl  ❌ (Python 3.11)
```

**Root Cause**:
Python wheel compatibility tags indicate the Python version they're built for:
- `cp310` = CPython 3.10
- `cp311` = CPython 3.11

Python won't install a cp310 wheel when running on Python 3.11 due to ABI (Application Binary Interface) incompatibility.

**Impact**:
- Can't use prebuilt gsplat wheels with Python 3.11
- Must either compile from source or downgrade Python

**User Requirement**:
"I dont want to compile anything mate" - User explicitly wants to avoid all compilation.

**Solution**:
**Decision**: Switch from Python 3.11 → Python 3.10 to use prebuilt wheels

**Implementation**: Updated `nodes/env_manager.py` to create Python 3.10 environment (7 occurrences changed)

**Verification**:
```bash
$ _env/bin/python --version
Python 3.10.19

$ _env/bin/python -c "import gsplat; print(gsplat.__version__)"
1.5.3+pt24cu121
```

The `+pt24cu121` suffix confirms this is the prebuilt wheel for PyTorch 2.4 + CUDA 12.1 (no compilation needed!)

**Status**: ✅ Resolved

---

## Summary of Changes Made

### Files Modified:

1. **`inference_worker.py`**:
   - Added library output suppression (warnings, logging)
   - Added UTF-8 encoding environment variables
   - Added g++/gcc symlink creation for nvcc compatibility
   - Added CUDA_HOME detection and PATH configuration

2. **`nodes/subprocess_bridge.py`**:
   - Added defensive JSON parsing to skip non-JSON lines
   - Added timeout and retry logic for robustness

3. **`nodes/env_manager.py`**:
   - Downgraded PyTorch from 2.5.1 → 2.4.1 (PyTorch3D compatibility)
   - Added separate gsplat installation step with prebuilt wheel support
   - Removed gsplat from requirements_env.txt (now installed separately)

4. **`local_env_settings/requirements_env.txt`**:
   - Removed gsplat (line 64) - now installed via env_manager.py
   - Added documentation comments

5. **`NODE_DESIGN_RECOMMENDATIONS.md`** (new file):
   - Documented decision to keep all-in-one node design
   - Explained SAM3D pipeline stages
   - Provided recommendations for future enhancements

---

## Lessons Learned

### 1. **IPC Design**
When using JSON over stdin/stdout for subprocess communication:
- Suppress ALL library output to stdout
- Use stderr for debugging/logging
- Implement defensive JSON parsing
- Never trust that libraries won't print to stdout

### 2. **Python Encoding**
Always set UTF-8 encoding explicitly when dealing with:
- File I/O operations
- Subprocess communication
- JIT compilation (source code parsing)

### 3. **Compiler Compatibility**
When using conda/micromamba compilers:
- Be aware of cross-compilation wrapper names
- Create symlinks for tools that expect standard names
- Set CXX and CC environment variables

### 4. **PyTorch Ecosystem**
- PyTorch3D lags behind PyTorch releases
- Use PyTorch 2.4.x for best compatibility
- Check prebuilt wheel availability before choosing versions

### 5. **Avoid JIT Compilation**
Prebuilt wheels are always preferable because:
- No compiler dependencies
- Faster installation (seconds vs minutes)
- More reliable across environments
- Better user experience

### 6. **Python Version Selection**
When choosing Python versions for ML projects:
- Check prebuilt wheel availability for all dependencies
- Python 3.10 has better ML library support than 3.11
- Newer Python versions may lack prebuilt wheels

---

## Current Status

**Environment Configuration**:
- Python: **3.10.19** ✅
- PyTorch: **2.4.1** ✅
- PyTorch3D: **0.7.8** ✅ (via micromamba)
- Kaolin: **0.17.0** ✅
- CUDA: **12.1** ✅
- gsplat: **1.5.3+pt24cu121** ✅ (prebuilt wheel, no compilation!)

**Installation Method**:
- Isolated environment via micromamba
- PyTorch + PyTorch3D via conda-forge
- gsplat via prebuilt wheels from https://docs.gsplat.studio/whl/pt24cu121/
- Other packages via pip/uv

**Completed Steps**:
1. ✅ Documented all problems (this file)
2. ✅ Switched to Python 3.10 in env_manager.py
3. ✅ Tested full installation with Python 3.10
4. ✅ Verified gsplat prebuilt wheel works (version 1.5.3+pt24cu121)
5. ⏳ Test SAM3D inference end-to-end (ready to test)

---

*Last Updated: 2025-11-22*
