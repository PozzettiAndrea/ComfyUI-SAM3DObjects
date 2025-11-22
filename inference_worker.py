"""
Inference worker for SAM3D that runs in the isolated environment.

This worker loads the SAM3D model and handles inference requests
via IPC (stdin/stdout communication).
"""

import sys
import json
import pickle
import base64
import traceback
from pathlib import Path
from typing import Any, Dict
import numpy as np
from PIL import Image
import io


# Global model cache
_MODEL = None
_CURRENT_CONFIG = None


def load_model(config_path: str, compile: bool = False):
    """Load the SAM3D model."""
    global _MODEL, _CURRENT_CONFIG

    config_key = f"{config_path}_{compile}"

    if _MODEL is not None and _CURRENT_CONFIG == config_key:
        return _MODEL

    print(f"[Worker] Loading model from {config_path}", file=sys.stderr)

    # Add vendor directory to path for sam3d_objects
    vendor_path = Path(__file__).parent / "vendor"
    if str(vendor_path) not in sys.path:
        sys.path.insert(0, str(vendor_path))
        print(f"[Worker] Added vendor path: {vendor_path}", file=sys.stderr)

    # Skip sam3d_objects initialization (LIDRA_SKIP_INIT)
    import os
    os.environ['LIDRA_SKIP_INIT'] = '1'

    # Force UTF-8 encoding for all file I/O operations
    # This prevents UnicodeDecodeError during JIT compilation of CUDA extensions (gsplat, nvdiffrast)
    # PyTorch's cpp_extension_versioner reads source files to hash them, and some contain UTF-8 chars
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONUTF8'] = '1'  # PEP 540: Force UTF-8 mode in Python 3.7+

    # Add venv's bin directory to PATH for ninja (required by nvdiffrast JIT compilation)
    # Note: Even though we use pytorch3d as rendering_engine, postprocessing (_fill_holes)
    # still uses nvdiffrast through utils3d.torch.RastContext
    venv_bin = (Path(__file__).parent / "_env" / "bin").resolve()
    if venv_bin.exists():
        os.environ['PATH'] = f"{venv_bin}:{os.environ.get('PATH', '')}"
        print(f"[Worker] Added {venv_bin} to PATH for ninja", file=sys.stderr)

    # Add g++ compiler to PATH for CUDA JIT compilation
    # nvcc needs g++ as the host compiler to compile C++ code
    # The compilers are installed by micromamba in _env/bin/
    gcc_bin = (Path(__file__).parent / "_env" / "bin").resolve()
    if gcc_bin.exists():
        os.environ['PATH'] = f"{gcc_bin}:{os.environ['PATH']}"

        # Fix for gsplat JIT: nvcc needs standard g++/gcc names, not conda wrappers
        # The conda compilers use cross-compilation wrapper names that nvcc doesn't understand
        # We create symlinks so nvcc can find them by standard names
        wrapper_gxx = gcc_bin / "x86_64-conda-linux-gnu-g++"
        wrapper_gcc = gcc_bin / "x86_64-conda-linux-gnu-gcc"
        symlink_gxx = gcc_bin / "g++"
        symlink_gcc = gcc_bin / "gcc"

        # Create g++ symlink if it doesn't exist
        if wrapper_gxx.exists() and not symlink_gxx.exists():
            try:
                symlink_gxx.symlink_to(wrapper_gxx.name)  # Relative symlink
                print(f"[Worker] Created g++ symlink for nvcc", file=sys.stderr)
            except (FileExistsError, OSError) as e:
                print(f"[Worker] Could not create g++ symlink: {e}", file=sys.stderr)

        # Create gcc symlink if it doesn't exist
        if wrapper_gcc.exists() and not symlink_gcc.exists():
            try:
                symlink_gcc.symlink_to(wrapper_gcc.name)  # Relative symlink
                print(f"[Worker] Created gcc symlink for nvcc", file=sys.stderr)
            except (FileExistsError, OSError) as e:
                print(f"[Worker] Could not create gcc symlink: {e}", file=sys.stderr)

        # Set environment to use standard names (nvcc will find them in PATH)
        os.environ['CXX'] = 'g++'
        os.environ['CC'] = 'gcc'
        print(f"[Worker] Set CXX=g++, CC=gcc (via symlinks in {gcc_bin})", file=sys.stderr)

    # Setup CUDA_HOME for JIT compilation (gsplat, nvdiffrast, etc.)
    # Try to find CUDA toolkit installed by env_manager.py
    venv_root = (Path(__file__).parent / "_env").resolve()
    cuda_home = None

    # Option 1: CUDA toolkit from conda-forge (installed to _env/cuda/)
    conda_cuda = venv_root / "cuda"
    if (conda_cuda / "bin" / "nvcc").exists():
        cuda_home = conda_cuda
        print(f"[Worker] Found CUDA toolkit from conda-forge: {cuda_home}", file=sys.stderr)

    # Option 2: CUDA toolkit from PyPI (scattered in site-packages)
    if not cuda_home:
        # Try to find nvcc in venv
        import subprocess
        try:
            result = subprocess.run(
                ["find", str(venv_root), "-name", "nvcc", "-type", "f"],
                capture_output=True, text=True, timeout=10
            )
            if result.stdout.strip():
                nvcc_path = Path(result.stdout.strip().split('\n')[0])
                # CUDA_HOME should be parent of bin/
                if nvcc_path.parent.name == "bin":
                    cuda_home = nvcc_path.parent.parent
                    print(f"[Worker] Found CUDA toolkit from PyPI: {cuda_home}", file=sys.stderr)
        except Exception as e:
            print(f"[Worker] Could not search for nvcc: {e}", file=sys.stderr)

    # Set CUDA_HOME and update PATH
    if cuda_home:
        os.environ['CUDA_HOME'] = str(cuda_home)
        cuda_bin = cuda_home / "bin"
        if cuda_bin.exists():
            os.environ['PATH'] = f"{cuda_bin}:{os.environ.get('PATH', '')}"
            print(f"[Worker] Set CUDA_HOME={cuda_home}", file=sys.stderr)
            print(f"[Worker] Added {cuda_bin} to PATH", file=sys.stderr)
    else:
        print("[Worker] Warning: CUDA toolkit not found in venv", file=sys.stderr)
        print("[Worker] JIT compilation may fail for gsplat and other CUDA extensions", file=sys.stderr)

    # Redirect all model downloads to ComfyUI/models/sam3d/
    # This includes torch.hub (DINO), huggingface, transformers, etc.
    config_dir = Path(config_path).parent.parent  # Go up from checkpoints/ to model tag dir
    models_cache_dir = config_dir / "_models_cache"
    models_cache_dir.mkdir(exist_ok=True)

    os.environ['TORCH_HOME'] = str(models_cache_dir / "torch")
    os.environ['HF_HOME'] = str(models_cache_dir / "huggingface")
    os.environ['TRANSFORMERS_CACHE'] = str(models_cache_dir / "transformers")
    print(f"[Worker] Model cache directory: {models_cache_dir}", file=sys.stderr)

    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    from sam3d_objects.pipeline.inference_pipeline_pointmap import InferencePipelinePointMap

    # Load config and instantiate model using Hydra (like original SAM3D does)
    config = OmegaConf.load(config_path)
    config.rendering_engine = "pytorch3d"  # overwrite to disable nvdiffrast
    config.compile_model = compile
    config.workspace_dir = os.path.dirname(config_path)

    # Instantiate the pipeline with all config parameters (including depth_model)
    _MODEL = instantiate(config)
    _CURRENT_CONFIG = config_key

    print(f"[Worker] Model loaded successfully", file=sys.stderr)
    return _MODEL


def deserialize_image(image_b64: str) -> Image.Image:
    """Deserialize base64-encoded image."""
    image_bytes = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(image_bytes))


def deserialize_mask(mask_b64: str) -> np.ndarray:
    """Deserialize base64-encoded mask."""
    mask_bytes = base64.b64decode(mask_b64)
    return pickle.loads(mask_bytes)


def serialize_output(output: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize output for IPC transfer."""
    # We need to serialize complex objects
    serialized = {}

    for key, value in output.items():
        if value is None:
            serialized[key] = None
        elif isinstance(value, (int, float, str, bool)):
            serialized[key] = value
        elif isinstance(value, np.ndarray):
            # Serialize numpy arrays as base64 pickles
            serialized[key] = {
                "_type": "numpy",
                "_data": base64.b64encode(pickle.dumps(value)).decode('utf-8')
            }
        elif isinstance(value, dict):
            # Recursively serialize dicts
            serialized[key] = serialize_output(value)
        elif isinstance(value, (list, tuple)):
            # Serialize lists/tuples
            serialized[key] = {
                "_type": "list" if isinstance(value, list) else "tuple",
                "_data": [serialize_output({"v": v})["v"] for v in value]
            }
        else:
            # For complex objects (gaussian splats, etc), pickle them
            try:
                serialized[key] = {
                    "_type": "pickle",
                    "_data": base64.b64encode(pickle.dumps(value)).decode('utf-8'),
                    "_class": type(value).__name__
                }
            except Exception as e:
                print(f"[Worker] Warning: Could not serialize {key}: {e}", file=sys.stderr)
                serialized[key] = None

    return serialized


def run_inference(request: Dict[str, Any]) -> Dict[str, Any]:
    """Run inference on the given request."""
    try:
        # Extract request parameters
        config_path = request["config_path"]
        compile_model = request.get("compile", False)
        image_b64 = request["image"]
        mask_b64 = request["mask"]
        seed = request.get("seed", 42)
        with_mesh_postprocess = request.get("with_mesh_postprocess", True)

        # Load model
        model = load_model(config_path, compile_model)

        # Deserialize inputs
        image = deserialize_image(image_b64)
        mask = deserialize_mask(mask_b64)

        print(f"[Worker] Running inference (seed={seed}, with_mesh_postprocess={with_mesh_postprocess})", file=sys.stderr)
        print(f"[Worker] Image: mode={image.mode}, size={image.size}", file=sys.stderr)
        print(f"[Worker] Mask: shape={mask.shape}, dtype={mask.dtype}, range=[{mask.min()}, {mask.max()}]", file=sys.stderr)

        # Ensure mask is uint8 in [0, 255] range to match image
        if mask.dtype != np.uint8:
            # Convert from float [0, 1] to uint8 [0, 255]
            if mask.max() <= 1.0:
                mask = (mask * 255).astype(np.uint8)
            else:
                mask = mask.astype(np.uint8)
            print(f"[Worker] Converted mask to uint8: shape={mask.shape}, range=[{mask.min()}, {mask.max()}]", file=sys.stderr)

        # Run inference using the run() method
        output = model.run(image, mask, seed=seed, with_mesh_postprocess=with_mesh_postprocess)

        print(f"[Worker] Inference completed", file=sys.stderr)

        # Serialize output
        serialized_output = serialize_output(output)

        return {
            "status": "success",
            "output": serialized_output
        }

    except Exception as e:
        print(f"[Worker] Error during inference: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def main():
    """Main worker loop - reads requests from stdin, writes responses to stdout."""

    # CRITICAL: Suppress all library output to prevent stdout pollution
    # Libraries like OmegaConf, Hydra, PyTorch, CUDA can print to stdout,
    # which interferes with our JSON-based IPC protocol
    import warnings
    import logging
    import os

    # Suppress Python warnings from all libraries
    warnings.filterwarnings("ignore")

    # Suppress TensorFlow logs (if used by any dependency)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Suppress Hydra full error traces
    os.environ['HYDRA_FULL_ERROR'] = '0'

    # Disable all Python logging from libraries
    logging.disable(logging.CRITICAL)

    print("[Worker] SAM3D inference worker started", file=sys.stderr)
    print(f"[Worker] Python: {sys.executable}", file=sys.stderr)
    print(f"[Worker] Working directory: {Path.cwd()}", file=sys.stderr)

    # Verify critical imports
    try:
        import torch
        import pytorch3d
        print(f"[Worker] PyTorch version: {torch.__version__}", file=sys.stderr)
        print(f"[Worker] PyTorch3D version: {pytorch3d.__version__}", file=sys.stderr)
        print(f"[Worker] CUDA available: {torch.cuda.is_available()}", file=sys.stderr)
    except Exception as e:
        print(f"[Worker] Warning: Could not verify dependencies: {e}", file=sys.stderr)

    print("[Worker] Ready for requests", file=sys.stderr)

    # Read requests from stdin
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)

            # Handle special commands
            if request.get("command") == "ping":
                response = {"status": "pong"}
            elif request.get("command") == "shutdown":
                print("[Worker] Shutdown requested", file=sys.stderr)
                response = {"status": "shutdown"}
                print(json.dumps(response), flush=True)
                break
            else:
                # Run inference
                response = run_inference(request)

            # Send response
            print(json.dumps(response), flush=True)

        except Exception as e:
            print(f"[Worker] Error processing request: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            error_response = {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            print(json.dumps(error_response), flush=True)

    print("[Worker] Worker shutting down", file=sys.stderr)


if __name__ == "__main__":
    main()
