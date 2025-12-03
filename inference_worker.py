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
import torch
import os


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


def save_output_to_disk(output: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    """
    Save output to disk and return file paths.

    This is much more robust than trying to serialize complex objects through IPC.
    Following ComfyUI's standard pattern of saving outputs to disk.
    """
    import json

    # Create sequentially numbered output directory (inference_1, inference_2, etc.)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find next available number
    existing = [d.name for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("inference_")]
    numbers = []
    for dirname in existing:
        try:
            num = int(dirname.split("_")[1])
            numbers.append(num)
        except (IndexError, ValueError):
            pass

    next_num = max(numbers) + 1 if numbers else 1
    save_dir = output_dir / f"inference_{next_num}"
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Worker] Saving outputs to: {save_dir}", file=sys.stderr)

    result = {
        "output_dir": str(save_dir),
        "files": {},
        "metadata": {}
    }

    # Save sparse structure (Stage 1 output)
    # Identify sparse structure by keys present in stage 1 but not others
    if "coords" in output and "pointmap" in output and "slat" not in output:
        sparse_path = save_dir / "sparse_structure.pt"
        torch.save(output, sparse_path)
        result["files"]["sparse_structure"] = str(sparse_path)
        print(f"[Worker] Saved sparse structure: {sparse_path}", file=sys.stderr)

    # Save SLAT (Stage 2 intermediate output)
    if "slat" in output:
        slat_path = save_dir / "slat.pt"
        torch.save(output, slat_path)
        result["files"]["slat"] = str(slat_path)
        print(f"[Worker] Saved SLAT: {slat_path}", file=sys.stderr)

    # Save GLB file (textured mesh)
    if "glb" in output and output["glb"] is not None:
        glb_path = save_dir / "mesh.glb"

        # Check if it's a Trimesh object that needs to be exported
        import trimesh
        if isinstance(output["glb"], trimesh.Trimesh):
            # Export Trimesh to GLB format
            glb_bytes = output["glb"].export(file_type="glb")
            with open(glb_path, 'wb') as f:
                f.write(glb_bytes)
            print(f"[Worker] Saved GLB: {glb_path} ({len(glb_bytes)} bytes)", file=sys.stderr)
        else:
            # Already bytes
            with open(glb_path, 'wb') as f:
                f.write(output["glb"])
            print(f"[Worker] Saved GLB: {glb_path} ({len(output['glb'])} bytes)", file=sys.stderr)

        result["files"]["glb"] = str(glb_path)

    # Save Gaussian Splat PLY file (colored point cloud)
    if "gs" in output and output["gs"] is not None:
        ply_path = save_dir / "gaussian.ply"
        try:
            output["gs"].save_ply(str(ply_path))
            result["files"]["ply"] = str(ply_path)
            print(f"[Worker] Saved Gaussian PLY: {ply_path}", file=sys.stderr)
        except Exception as e:
            print(f"[Worker] Warning: Failed to save Gaussian PLY: {e}", file=sys.stderr)

    # Save metadata (simple types only)
    metadata = {}
    for key, value in output.items():
        if isinstance(value, (int, float, str, bool)):
            metadata[key] = value
        elif isinstance(value, torch.Tensor):
            # Convert torch tensors to lists for JSON serialization
            metadata[key] = value.cpu().tolist()
        elif isinstance(value, np.ndarray):
            # Convert numpy arrays to lists for JSON serialization
            metadata[key] = value.tolist()
        elif isinstance(value, dict) and key not in ["glb", "gaussian_splat", "mesh"]:
            # Save simple dict metadata
            try:
                json.dumps(value)  # Test if it's JSON-serializable
                metadata[key] = value
            except:
                pass

    if metadata:
        metadata_path = save_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        result["files"]["metadata"] = str(metadata_path)
        result["metadata"] = metadata

    return result


def run_inference(request: Dict[str, Any]) -> Dict[str, Any]:
    """Run inference on the given request."""
    try:
        # Extract request parameters
        config_path = request["config_path"]
        compile_model = request.get("compile", False)
        use_cache = request.get("use_cache", False)
        image_b64 = request["image"]
        mask_b64 = request["mask"]
        seed = request.get("seed", 42)
        stage1_inference_steps = request.get("stage1_inference_steps", 25)
        stage2_inference_steps = request.get("stage2_inference_steps", 25)
        stage1_cfg_strength = request.get("stage1_cfg_strength", 7.0)
        stage2_cfg_strength = request.get("stage2_cfg_strength", 5.0)
        texture_size = request.get("texture_size", 1024)
        simplify = request.get("simplify", 0.95)
        output_dir = request.get("output_dir", "/tmp/sam3d_output")  # Default fallback
        stage1_only = request.get("stage1_only", False)
        stage1_output_b64 = request.get("stage1_output", None)
        stage2_only = request.get("stage2_only", False)
        stage2_output_b64 = request.get("stage2_output", None)
        slat_only = request.get("slat_only", False)
        slat_output_b64 = request.get("slat_output", None)
        gaussian_only = request.get("gaussian_only", False)
        mesh_only = request.get("mesh_only", False)
        save_files = request.get("save_files", False)
        with_mesh_postprocess = request.get("with_mesh_postprocess", False)
        with_texture_baking = request.get("with_texture_baking", True)
        use_vertex_color = request.get("use_vertex_color", False)

        # Load model
        model = load_model(config_path, compile_model)

        # Deserialize inputs
        image = deserialize_image(image_b64)
        mask = deserialize_mask(mask_b64)

        # Load stage1_output if provided (from path or base64 for backward compat)
        stage1_output = None
        if request.get("stage1_output_path") is not None and os.path.exists(request.get("stage1_output_path")):
            print(f"[Worker] Loading Stage 1 output from: {request.get('stage1_output_path')}", file=sys.stderr)
            stage1_output = torch.load(request.get("stage1_output_path"))
        elif request.get("stage1_output") is not None:
            stage1_output = pickle.loads(base64.b64decode(request.get("stage1_output")))

        # Deserialize stage2_output if provided
        stage2_output = None
        if stage2_output_b64 is not None:
            stage2_output = pickle.loads(base64.b64decode(stage2_output_b64))

            # Check if this needs combining separate Gaussian and Mesh data
            if isinstance(stage2_output, dict) and stage2_output.get("_needs_combination"):
                print(f"[Worker] Combining separate Gaussian and Mesh data", file=sys.stderr)
                gaussian_b64 = stage2_output["_gaussian_serialized"]
                mesh_b64 = stage2_output["_mesh_serialized"]

                # Deserialize in worker context where sam3d_objects is available
                gaussian_dict = pickle.loads(base64.b64decode(gaussian_b64))
                mesh_dict = pickle.loads(base64.b64decode(mesh_b64))

                print(f"[Worker] Gaussian dict keys: {list(gaussian_dict.keys())}", file=sys.stderr)
                print(f"[Worker] Mesh dict keys: {list(mesh_dict.keys())}", file=sys.stderr)

                # Combine into single dict for stage2_output
                stage2_output = {
                    "gaussian": gaussian_dict.get("gaussian"),
                    "mesh": mesh_dict.get("mesh"),
                    "stage1_data": mesh_dict.get("stage1_data", gaussian_dict.get("stage1_data", {}))
                }
                print(f"[Worker] Combined stage2_output keys: {list(stage2_output.keys())}", file=sys.stderr)

        # Load slat_output if provided (from path or base64)
        slat_output = None
        if request.get("slat_output_path") is not None and os.path.exists(request.get("slat_output_path")):
            print(f"[Worker] Loading SLAT output from: {request.get('slat_output_path')}", file=sys.stderr)
            slat_output = torch.load(request.get("slat_output_path"))
        elif request.get("slat_output") is not None:
            slat_output = pickle.loads(base64.b64decode(request.get("slat_output")))

        print(f"[Worker] Running inference (seed={seed})", file=sys.stderr)
        print(f"[Worker] Stage 1: steps={stage1_inference_steps}, cfg={stage1_cfg_strength}", file=sys.stderr)
        print(f"[Worker] Stage 2: steps={stage2_inference_steps}, cfg={stage2_cfg_strength}", file=sys.stderr)
        print(f"[Worker] Postprocess: texture_size={texture_size}, simplify={simplify}", file=sys.stderr)
        if use_cache:
            print(f"[Worker] use_cache=True: Models will be offloaded to CPU after each stage (~50% VRAM reduction)", file=sys.stderr)
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
        # Using TRELLIS nvdiffrast 0.3.3 built for PyTorch 2.4.0 (compatible with our 2.4.1)
        output = model.run(
            image, mask,
            seed=seed,
            stage1_inference_steps=stage1_inference_steps,
            stage2_inference_steps=stage2_inference_steps,
            stage1_cfg_strength=stage1_cfg_strength,
            stage2_cfg_strength=stage2_cfg_strength,
            simplify=simplify,
            texture_size=texture_size,
            with_mesh_postprocess=with_mesh_postprocess,
            with_texture_baking=with_texture_baking,
            use_vertex_color=use_vertex_color,
            stage1_only=stage1_only,
            stage1_output=stage1_output,
            stage2_only=stage2_only,
            stage2_output=stage2_output,
            slat_only=slat_only,
            slat_output=slat_output,
            gaussian_only=gaussian_only,
            mesh_only=mesh_only,
            save_files=save_files,
            use_cache=use_cache,
        )

        print(f"[Worker] Inference completed", file=sys.stderr)

        # Special handling for stage1_only mode - save to disk and return path
        if stage1_only:
            print(f"[Worker] Stage 1 only - saving to disk", file=sys.stderr)
            saved_output = save_output_to_disk(output, Path(output_dir))
            return {
                "status": "success",
                "stage1_mode": True,
                "output": saved_output  # Contains file paths
            }

        # Special handling for stage2_only mode - return serialized Gaussian + Mesh data
        if stage2_only:
            print(f"[Worker] Stage 2 only - serializing Gaussian + Mesh output for caching", file=sys.stderr)
            print(f"[Worker] Output keys: {list(output.keys())}", file=sys.stderr)

            # Serialize the entire output dict (including Gaussian and Mesh objects)
            serialized_output = base64.b64encode(pickle.dumps(output)).decode('utf-8')

            return {
                "status": "success",
                "stage2_mode": True,
                "output": serialized_output
            }

        # Special handling for slat_only mode - save to disk and return path
        if slat_only:
            print(f"[Worker] SLAT only - saving to disk", file=sys.stderr)
            saved_output = save_output_to_disk(output, Path(output_dir))
            return {
                "status": "success",
                "stage2_mode": True,  # Use same mode as stage2 for compatibility
                "output": saved_output
            }

        # Special handling for gaussian_only and mesh_only modes - return both files and serialized data
        # This allows the texture baking node to access raw data while still saving files
        if gaussian_only or mesh_only:
            mode_name = "Gaussian" if gaussian_only else "Mesh"
            print(f"[Worker] {mode_name} only - saving files and serializing output for downstream use", file=sys.stderr)
            print(f"[Worker] Output keys: {list(output.keys())}", file=sys.stderr)

            # Save files to disk first
            saved_output = save_output_to_disk(output, Path(output_dir))

            # Also serialize the raw data for texture baking node
            serialized_output = base64.b64encode(pickle.dumps(output)).decode('utf-8')

            return {
                "status": "success",
                "stage2_mode": True,  # Use same mode as stage2 for serialized data
                "output": serialized_output,
                "file_output": saved_output  # Include file paths too
            }

        # Normal mode: Save final output to disk and return file paths
        # This avoids complex pickle serialization and module dependency issues
        saved_output = save_output_to_disk(output, Path(output_dir))

        return {
            "status": "success",
            "output": saved_output
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
