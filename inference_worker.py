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
import platform
import shutil
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
        os.environ['PATH'] = f"{venv_bin}{os.pathsep}{os.environ.get('PATH', '')}"
        print(f"[Worker] Added {venv_bin} to PATH for ninja", file=sys.stderr)

    # Add compiler to PATH for CUDA JIT compilation
    # nvcc needs a host compiler (g++/gcc on Linux/Windows, clang on macOS)
    # The compilers are installed by micromamba in _env/bin/
    compiler_bin = (Path(__file__).parent / "_env" / "bin").resolve()
    if compiler_bin.exists():
        os.environ['PATH'] = f"{compiler_bin}{os.pathsep}{os.environ['PATH']}"

        # Fix for gsplat JIT: nvcc needs standard compiler names, not conda wrappers
        # The conda compilers use cross-compilation wrapper names that nvcc doesn't understand
        # We create symlinks (Unix) or copies (Windows) so nvcc can find them by standard names

        system = platform.system()

        if system == "Linux":
            # Linux: x86_64-conda-linux-gnu-g++ -> g++
            wrapper_gxx = compiler_bin / "x86_64-conda-linux-gnu-g++"
            wrapper_gcc = compiler_bin / "x86_64-conda-linux-gnu-gcc"
            target_gxx = compiler_bin / "g++"
            target_gcc = compiler_bin / "gcc"
            cxx_name = "g++"
            cc_name = "gcc"
            use_symlink = True

        elif system == "Windows":
            # Windows: x86_64-w64-mingw32-g++.exe -> g++.exe (or use MSVC cl.exe)
            # First check for MSVC
            cl_exe = compiler_bin / "cl.exe"
            if cl_exe.exists():
                # MSVC is available, use it directly
                os.environ['CXX'] = str(cl_exe)
                os.environ['CC'] = str(cl_exe)
                print(f"[Worker] Using MSVC compiler: {cl_exe}", file=sys.stderr)
                wrapper_gxx = None
                wrapper_gcc = None
            else:
                # Use m2w64 MinGW compiler
                wrapper_gxx = compiler_bin / "x86_64-w64-mingw32-g++.exe"
                wrapper_gcc = compiler_bin / "x86_64-w64-mingw32-gcc.exe"
                target_gxx = compiler_bin / "g++.exe"
                target_gcc = compiler_bin / "gcc.exe"
                cxx_name = "g++.exe"
                cc_name = "gcc.exe"
                use_symlink = False  # Windows: use copies instead of symlinks

        elif system == "Darwin":
            # macOS: x86_64-apple-darwin*-clang++ -> clang++
            # Find the darwin wrapper (version number varies)
            darwin_wrappers_gxx = list(compiler_bin.glob("*-apple-darwin*-clang++"))
            darwin_wrappers_gcc = list(compiler_bin.glob("*-apple-darwin*-clang"))

            if darwin_wrappers_gxx and darwin_wrappers_gcc:
                wrapper_gxx = darwin_wrappers_gxx[0]
                # Filter out clang++ to get just clang
                wrapper_gcc = [w for w in darwin_wrappers_gcc if not w.name.endswith("++")][0]
                target_gxx = compiler_bin / "clang++"
                target_gcc = compiler_bin / "clang"
                cxx_name = "clang++"
                cc_name = "clang"
                use_symlink = True
            else:
                wrapper_gxx = None
                wrapper_gcc = None
        else:
            print(f"[Worker] Warning: Unsupported platform {system}, skipping compiler wrapper setup", file=sys.stderr)
            wrapper_gxx = None
            wrapper_gcc = None

        # Create symlinks or copies if wrappers were found
        if wrapper_gxx is not None and wrapper_gcc is not None:
            # Create CXX (g++ or clang++) link/copy
            if wrapper_gxx.exists() and not target_gxx.exists():
                try:
                    if use_symlink:
                        target_gxx.symlink_to(wrapper_gxx.name)  # Relative symlink
                        print(f"[Worker] Created {cxx_name} symlink for nvcc", file=sys.stderr)
                    else:
                        shutil.copy2(wrapper_gxx, target_gxx)  # Copy for Windows
                        print(f"[Worker] Created {cxx_name} copy for nvcc", file=sys.stderr)
                except (FileExistsError, OSError) as e:
                    print(f"[Worker] Could not create {cxx_name}: {e}", file=sys.stderr)

            # Create CC (gcc or clang) link/copy
            if wrapper_gcc.exists() and not target_gcc.exists():
                try:
                    if use_symlink:
                        target_gcc.symlink_to(wrapper_gcc.name)  # Relative symlink
                        print(f"[Worker] Created {cc_name} symlink for nvcc", file=sys.stderr)
                    else:
                        shutil.copy2(wrapper_gcc, target_gcc)  # Copy for Windows
                        print(f"[Worker] Created {cc_name} copy for nvcc", file=sys.stderr)
                except (FileExistsError, OSError) as e:
                    print(f"[Worker] Could not create {cc_name}: {e}", file=sys.stderr)

            # Set environment to use standard names (nvcc will find them in PATH)
            os.environ['CXX'] = cxx_name
            os.environ['CC'] = cc_name
            print(f"[Worker] Set CXX={cxx_name}, CC={cc_name} in {compiler_bin}", file=sys.stderr)

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
        # Try to find nvcc in venv (cross-platform using Path.glob)
        try:
            # Search for nvcc executable (nvcc on Unix, nvcc.exe on Windows)
            nvcc_pattern = "nvcc.exe" if os.name == "nt" else "nvcc"
            nvcc_paths = list(venv_root.glob(f"**/{nvcc_pattern}"))
            if nvcc_paths:
                nvcc_path = nvcc_paths[0]  # Take first match
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
            os.environ['PATH'] = f"{cuda_bin}{os.pathsep}{os.environ.get('PATH', '')}"
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


def transform_to_global_coordinates(output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform Gaussian splat and mesh to global (scene) coordinates using pose data.

    This applies rotation, translation, and scale from depth estimation so that
    multiple objects from the same image can be correctly positioned when combined.
    """
    rotation = output.get("rotation")
    translation = output.get("translation")
    scale = output.get("scale")

    # If no pose data, nothing to transform
    if rotation is None or translation is None or scale is None:
        print("[Worker] No pose data available, skipping global coordinate transform", file=sys.stderr)
        return output

    print(f"[Worker] Transforming to global coordinates", file=sys.stderr)
    print(f"[Worker] Pose: rotation shape={rotation.shape if hasattr(rotation, 'shape') else 'N/A'}, translation shape={translation.shape if hasattr(translation, 'shape') else 'N/A'}, scale={scale}", file=sys.stderr)

    # Transform Gaussian if present
    if "gs" in output and output["gs"] is not None:
        try:
            gs = output["gs"]

            # Import required utilities
            from sam3d_objects.utils.visualization.scene_visualizer import SceneVisualizer
            from pytorch3d.transforms import quaternion_multiply, quaternion_invert

            # Get Gaussian positions in local coordinates
            xyz_local = gs.get_xyz  # (N, 3)

            # Transform to global coordinates using pose
            # SceneVisualizer.object_pointcloud expects batched input
            PC = SceneVisualizer.object_pointcloud(
                points_local=xyz_local.unsqueeze(0),  # (1, N, 3)
                quat_l2c=rotation,
                trans_l2c=translation,
                scale_l2c=scale,
            )
            # Set transformed positions
            gs.from_xyz(PC.points_list()[0])

            # Rotate the Gaussian rotation parameters
            gs.from_rotation(
                quaternion_multiply(
                    quaternion_invert(rotation),
                    gs.get_rotation,
                )
            )

            # Scale the Gaussian scaling parameters
            adjusted_scale = gs.get_scaling * scale
            # Ensure minimum kernel size is maintained
            if hasattr(gs, 'mininum_kernel_size'):
                gs.mininum_kernel_size *= scale[0, 0].item()
                adjusted_scale = torch.maximum(
                    adjusted_scale,
                    torch.tensor(
                        gs.mininum_kernel_size * 1.1,
                        device=adjusted_scale.device,
                    ),
                )
            gs.from_scaling(adjusted_scale)

            print(f"[Worker] Transformed Gaussian to global coordinates", file=sys.stderr)

        except Exception as e:
            print(f"[Worker] Warning: Failed to transform Gaussian: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    # Transform mesh vertices if present
    if "mesh" in output and output["mesh"] is not None:
        try:
            mesh_list = output["mesh"]
            if isinstance(mesh_list, list) and len(mesh_list) > 0:
                mesh = mesh_list[0]

                # Import required utilities
                from sam3d_objects.utils.visualization.scene_visualizer import SceneVisualizer

                # Get mesh vertices
                if hasattr(mesh, 'vertices'):
                    vertices = mesh.vertices  # (N, 3)
                    if hasattr(vertices, 'unsqueeze'):
                        # Transform using pose
                        PC = SceneVisualizer.object_pointcloud(
                            points_local=vertices.unsqueeze(0),
                            quat_l2c=rotation,
                            trans_l2c=translation,
                            scale_l2c=scale,
                        )
                        mesh.vertices = PC.points_list()[0]
                        print(f"[Worker] Transformed mesh vertices to global coordinates", file=sys.stderr)

        except Exception as e:
            print(f"[Worker] Warning: Failed to transform mesh: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    # Transform trimesh GLB if it's already been converted
    if "glb" in output and output["glb"] is not None:
        try:
            import trimesh
            if isinstance(output["glb"], trimesh.Trimesh):
                glb_mesh = output["glb"]

                # Import required utilities
                from sam3d_objects.utils.visualization.scene_visualizer import SceneVisualizer

                # Transform vertices
                vertices = torch.from_numpy(glb_mesh.vertices).float()
                if torch.cuda.is_available():
                    vertices = vertices.cuda()

                PC = SceneVisualizer.object_pointcloud(
                    points_local=vertices.unsqueeze(0),
                    quat_l2c=rotation,
                    trans_l2c=translation,
                    scale_l2c=scale,
                )
                glb_mesh.vertices = PC.points_list()[0].cpu().numpy()
                print(f"[Worker] Transformed GLB mesh to global coordinates", file=sys.stderr)

        except Exception as e:
            print(f"[Worker] Warning: Failed to transform GLB mesh: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    return output


def save_output_to_disk(output: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    """
    Save output to disk and return file paths.

    This is much more robust than trying to serialize complex objects through IPC.
    Following ComfyUI's standard pattern of saving outputs to disk.

    The output_dir should be a sam3d_inference_N directory created by depth_estimate.
    Files are saved directly to this directory (no subdirectory creation).
    """
    import json

    # Use provided directory directly (created by depth_estimate node)
    save_dir = Path(output_dir)
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


def _unload_model(model, model_type: str) -> Dict[str, Any]:
    """
    Unload specific model component to free VRAM.

    Args:
        model: InferencePipelinePointMap instance
        model_type: One of 'depth', 'sparse', 'slat', 'decoders', 'all'

    Returns:
        Status dict
    """
    import gc

    try:
        if model_type == "depth" or model_type == "all":
            if hasattr(model, 'depth_model') and model.depth_model is not None:
                model.depth_model.cpu()
                print("[Worker] Moved depth_model to CPU", file=sys.stderr)

        if model_type == "sparse" or model_type == "all":
            if hasattr(model, 'models') and 'ss_generator' in model.models:
                model.models['ss_generator'].cpu()
                print("[Worker] Moved ss_generator to CPU", file=sys.stderr)

        if model_type == "slat" or model_type == "all":
            if hasattr(model, 'models') and 'slat_generator' in model.models:
                model.models['slat_generator'].cpu()
                print("[Worker] Moved slat_generator to CPU", file=sys.stderr)

        if model_type == "decoders" or model_type == "all":
            if hasattr(model, 'models'):
                if 'slat_decoder_gs' in model.models:
                    model.models['slat_decoder_gs'].cpu()
                    print("[Worker] Moved slat_decoder_gs to CPU", file=sys.stderr)
                if 'slat_decoder_mesh' in model.models:
                    model.models['slat_decoder_mesh'].cpu()
                    print("[Worker] Moved slat_decoder_mesh to CPU", file=sys.stderr)

        # Force garbage collection and clear CUDA cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("[Worker] Cleared CUDA cache", file=sys.stderr)

        return {
            "status": "success",
            "unloaded": model_type
        }

    except Exception as e:
        print(f"[Worker] Warning during unload: {e}", file=sys.stderr)
        return {
            "status": "partial",
            "unloaded": model_type,
            "warning": str(e)
        }


def _load_pointmap_from_file(pointmap_path: str) -> torch.Tensor:
    """
    Load pointmap from a .pt tensor file.

    Args:
        pointmap_path: Path to .pt file

    Returns:
        Pointmap tensor in HWC format (H, W, 3)
    """
    pointmap = torch.load(pointmap_path)
    print(f"[Worker] Loaded pointmap tensor: shape={pointmap.shape}", file=sys.stderr)

    if torch.cuda.is_available():
        pointmap = pointmap.cuda()

    return pointmap


def _run_depth_only(model, image) -> Dict[str, Any]:
    """
    Run only depth estimation (MoGe) and return pointmap + intrinsics.

    Args:
        model: InferencePipelinePointMap instance
        image: PIL Image

    Returns:
        Dict with pointmap, intrinsics, depth
    """
    import numpy as np

    # Convert image to numpy array with alpha channel (as expected by compute_pointmap)
    image_np = np.array(image)
    if image_np.ndim == 2:
        # Grayscale - convert to RGBA
        image_np = np.stack([image_np] * 3 + [np.full_like(image_np, 255)], axis=-1)
    elif image_np.shape[-1] == 3:
        # RGB - add alpha channel
        alpha = np.full((image_np.shape[0], image_np.shape[1], 1), 255, dtype=np.uint8)
        image_np = np.concatenate([image_np, alpha], axis=-1)

    print(f"[Worker] Image shape for depth estimation: {image_np.shape}", file=sys.stderr)

    # Run compute_pointmap from the model
    pointmap_dict = model.compute_pointmap(image_np)

    pointmap = pointmap_dict["pointmap"]
    intrinsics = pointmap_dict.get("intrinsics")

    print(f"[Worker] Pointmap computed: shape={pointmap.shape if hasattr(pointmap, 'shape') else 'unknown'}", file=sys.stderr)
    print(f"[Worker] Intrinsics: {intrinsics is not None}", file=sys.stderr)

    # Serialize pointmap and intrinsics for transfer
    # Convert tensors to CPU numpy for pickle serialization
    # IMPORTANT: compute_pointmap returns CHW format (3, H, W), but model.run() expects HWC (H, W, 3)
    # So we transpose here for downstream compatibility
    if hasattr(pointmap, 'cpu'):
        # Transpose from CHW (3, H, W) to HWC (H, W, 3)
        pointmap_hwc = pointmap.permute(1, 2, 0).contiguous()
        pointmap_np = pointmap_hwc.cpu().numpy()
        print(f"[Worker] Pointmap transposed to HWC: {pointmap_np.shape}", file=sys.stderr)
    else:
        pointmap_np = pointmap

    if intrinsics is not None and hasattr(intrinsics, 'cpu'):
        intrinsics_np = intrinsics.cpu().numpy()
    else:
        intrinsics_np = intrinsics

    # Serialize for transfer back to main process
    pointmap_b64 = base64.b64encode(pickle.dumps(pointmap_np)).decode('utf-8')
    intrinsics_b64 = base64.b64encode(pickle.dumps(intrinsics_np)).decode('utf-8') if intrinsics_np is not None else None

    return {
        "status": "success",
        "depth_only": True,
        "pointmap": pointmap_b64,
        "intrinsics": intrinsics_b64,
    }


def run_pose_optimization(request: Dict[str, Any]) -> Dict[str, Any]:
    """Run pose optimization using layout_post_optimization."""
    try:
        import trimesh
        from pytorch3d.transforms import quaternion_to_matrix
        from sam3d_objects.pipeline.inference_utils import layout_post_optimization

        print("[Worker] Running pose optimization", file=sys.stderr)

        # Extract parameters
        glb_path = request["glb_path"]
        pointmap_path = request["pointmap_path"]
        enable_icp = request.get("enable_icp", True)
        enable_render_opt = request.get("enable_render_opt", True)

        # Deserialize intrinsics
        intrinsics_np = pickle.loads(base64.b64decode(request["intrinsics_b64"]))
        intrinsics = torch.from_numpy(intrinsics_np).float()

        # Deserialize pose
        pose_b64 = request["pose_b64"]
        rotation_np = pickle.loads(base64.b64decode(pose_b64["rotation"]))
        translation_np = pickle.loads(base64.b64decode(pose_b64["translation"]))
        scale_np = pickle.loads(base64.b64decode(pose_b64["scale"]))

        # Deserialize mask
        mask_np = pickle.loads(base64.b64decode(request["mask_b64"]))

        # Use GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Convert to tensors
        rotation = torch.from_numpy(rotation_np).float().to(device)
        translation = torch.from_numpy(translation_np).float().to(device)
        scale = torch.from_numpy(scale_np).float().to(device)
        mask_tensor = torch.from_numpy(mask_np).float().to(device)
        intrinsics = intrinsics.to(device)

        # Ensure correct shapes
        if rotation.dim() == 1:
            rotation = rotation.unsqueeze(0)
        if translation.dim() == 1:
            translation = translation.unsqueeze(0)
        if scale.dim() == 0:
            scale = scale.unsqueeze(0).unsqueeze(0).expand(1, 3)
        elif scale.dim() == 1:
            scale = scale.unsqueeze(0)

        # Load mesh
        mesh = trimesh.load(glb_path)
        if isinstance(mesh, trimesh.Scene):
            meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if not meshes:
                raise ValueError("No mesh found in GLB file")
            mesh = meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)

        # Load pointmap
        pointmap_data = torch.load(pointmap_path, weights_only=False)
        if isinstance(pointmap_data, dict):
            pointmap_tensor = pointmap_data.get("pointmap") or pointmap_data.get("data")
        else:
            pointmap_tensor = pointmap_data
        pointmap_tensor = pointmap_tensor.float().to(device)

        # Run optimization
        (
            refined_quat,
            refined_trans,
            refined_scale,
            final_iou,
            used_icp,
            used_render_opt,
        ) = layout_post_optimization(
            Mesh=mesh,
            Quaternion=rotation,
            Translation=translation,
            Scale=scale,
            Mask=mask_tensor,
            Point_Map=pointmap_tensor,
            Intrinsics=intrinsics,
            Enable_shape_ICP=enable_icp,
            Enable_rendering_optimization=enable_render_opt,
            device=device,
        )

        print(f"[Worker] Optimization complete: IoU={final_iou:.3f}", file=sys.stderr)

        # Convert results to numpy
        quat_np = refined_quat.cpu().numpy() if hasattr(refined_quat, 'cpu') else np.array(refined_quat)
        trans_np = refined_trans.cpu().numpy() if hasattr(refined_trans, 'cpu') else np.array(refined_trans)
        scale_np_out = refined_scale.cpu().numpy() if hasattr(refined_scale, 'cpu') else np.array(refined_scale)

        # Ensure correct shapes
        if quat_np.ndim > 1:
            quat_np = quat_np.squeeze()
        if trans_np.ndim > 1:
            trans_np = trans_np.squeeze()
        if scale_np_out.ndim > 1:
            scale_np_out = scale_np_out.squeeze()

        # Convert quaternion to rotation matrix using pytorch3d (already in wxyz format)
        quat_tensor = torch.from_numpy(quat_np).float().unsqueeze(0)
        rot_matrix = quaternion_to_matrix(quat_tensor).squeeze(0).numpy()

        # Apply transformation to mesh
        scale_val = scale_np_out.mean() if scale_np_out.ndim > 0 else float(scale_np_out)
        vertices = mesh.vertices.copy()
        vertices_transformed = (vertices @ (rot_matrix.T * scale_val)) + trans_np
        mesh.vertices = vertices_transformed

        # Save reposed GLB
        input_dir = os.path.dirname(glb_path)
        input_name = os.path.splitext(os.path.basename(glb_path))[0]
        output_glb_path = os.path.join(input_dir, f"{input_name}_reposed.glb")
        mesh.export(output_glb_path)

        print(f"[Worker] Saved reposed GLB: {output_glb_path}", file=sys.stderr)

        # Serialize refined pose for response
        refined_pose_b64 = {
            "rotation": base64.b64encode(pickle.dumps(quat_np)).decode('utf-8'),
            "translation": base64.b64encode(pickle.dumps(trans_np)).decode('utf-8'),
            "scale": base64.b64encode(pickle.dumps(scale_np_out)).decode('utf-8'),
        }

        return {
            "status": "success",
            "output_glb_path": output_glb_path,
            "refined_pose_b64": refined_pose_b64,
            "iou": float(final_iou),
            "used_icp": bool(used_icp),
            "used_render_opt": bool(used_render_opt),
        }

    except Exception as e:
        print(f"[Worker] Pose optimization error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def run_inference(request: Dict[str, Any]) -> Dict[str, Any]:
    """Run inference on the given request."""
    try:
        # Check for special modes that don't need full inference setup
        config_path = request.get("config_path")
        compile_model = request.get("compile", False)

        # Handle unload_model command
        if request.get("unload_model"):
            model_type = request.get("unload_model")
            print(f"[Worker] Unloading model: {model_type}", file=sys.stderr)
            model = load_model(config_path, compile_model)
            return _unload_model(model, model_type)

        # Handle depth_only mode (MoGe depth estimation only)
        if request.get("depth_only", False):
            print("[Worker] Running depth-only mode (MoGe)", file=sys.stderr)
            model = load_model(config_path, compile_model)
            image_b64 = request["image"]
            image = deserialize_image(image_b64)
            return _run_depth_only(model, image)

        # Extract request parameters
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
        use_stage1_distillation = request.get("use_stage1_distillation", False)
        use_stage2_distillation = request.get("use_stage2_distillation", False)
        texture_mode = request.get("texture_mode", "opt")
        rendering_engine = request.get("rendering_engine", "pytorch3d")
        merge_mask = request.get("merge_mask", True)
        auto_resize_mask = request.get("auto_resize_mask", True)

        # Load pointmap from .pt file if provided
        pointmap = None
        intrinsics = None
        if request.get("pointmap_path") is not None:
            pointmap_path = request.get("pointmap_path")
            print(f"[Worker] Loading pointmap from: {pointmap_path}", file=sys.stderr)
            pointmap = _load_pointmap_from_file(pointmap_path)
            print(f"[Worker] Pointmap shape: {pointmap.shape}", file=sys.stderr)
        if request.get("intrinsics") is not None:
            intrinsics_np = pickle.loads(base64.b64decode(request.get("intrinsics")))
            intrinsics = torch.from_numpy(intrinsics_np).cuda() if torch.cuda.is_available() else torch.from_numpy(intrinsics_np)
            print(f"[Worker] Intrinsics shape: {intrinsics.shape}", file=sys.stderr)

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

            # Check if this needs loading files from paths
            elif isinstance(stage2_output, dict) and stage2_output.get("_needs_file_loading"):
                print(f"[Worker] Loading Gaussian and Mesh from file paths", file=sys.stderr)
                glb_path = stage2_output["_glb_path"]
                ply_path = stage2_output["_ply_path"]

                print(f"[Worker] GLB path: {glb_path}", file=sys.stderr)
                print(f"[Worker] PLY path: {ply_path}", file=sys.stderr)

                # Check file sizes
                import os as os_module
                glb_size = os_module.path.getsize(glb_path) if os_module.path.exists(glb_path) else -1
                ply_size = os_module.path.getsize(ply_path) if os_module.path.exists(ply_path) else -1
                print(f"[Worker] GLB file size: {glb_size:,} bytes", file=sys.stderr)
                print(f"[Worker] PLY file size: {ply_size:,} bytes", file=sys.stderr)

                # Import required modules for loading
                print(f"[Worker] Importing Gaussian and trimesh modules...", file=sys.stderr)
                from sam3d_objects.model.backbone.tdfy_dit.representations.gaussian import Gaussian
                import trimesh
                print(f"[Worker] Imports successful", file=sys.stderr)

                # Load PLY as Gaussian
                print(f"[Worker] Creating Gaussian object...", file=sys.stderr)
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                print(f"[Worker] Using device: {device}", file=sys.stderr)
                # TODO: Need to determine AABB for Gaussian initialization
                # For now, use a default AABB - this may need adjustment
                gaussian = Gaussian(
                    aabb=[-1, -1, -1, 2, 2, 2],  # Default AABB
                    sh_degree=0,
                    device=device
                )
                print(f"[Worker] Gaussian object created, loading PLY: {ply_path}", file=sys.stderr)
                gaussian.load_ply(ply_path)
                print(f"[Worker] Loaded Gaussian with {gaussian._xyz.shape[0]} points", file=sys.stderr)
                print(f"[Worker] Gaussian xyz shape: {gaussian._xyz.shape}, dtype: {gaussian._xyz.dtype}", file=sys.stderr)

                # Load GLB as Mesh
                print(f"[Worker] Loading Mesh from GLB: {glb_path}", file=sys.stderr)
                loaded = trimesh.load(glb_path)
                print(f"[Worker] Loaded object type: {type(loaded)}", file=sys.stderr)

                # Handle Scene vs Mesh - GLB files often load as Scene
                if isinstance(loaded, trimesh.Scene):
                    print(f"[Worker] GLB loaded as Scene with {len(loaded.geometry)} geometries", file=sys.stderr)
                    # Combine all geometries into a single mesh
                    meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
                    print(f"[Worker] Found {len(meshes)} Trimesh objects in scene", file=sys.stderr)
                    if len(meshes) == 0:
                        raise RuntimeError("No mesh geometries found in GLB scene")
                    elif len(meshes) == 1:
                        trimesh_mesh = meshes[0]
                    else:
                        # Concatenate all meshes
                        trimesh_mesh = trimesh.util.concatenate(meshes)
                    print(f"[Worker] Extracted mesh from scene", file=sys.stderr)
                else:
                    trimesh_mesh = loaded

                print(f"[Worker] Loaded trimesh with {len(trimesh_mesh.vertices)} vertices, {len(trimesh_mesh.faces)} faces", file=sys.stderr)

                # Convert trimesh to MeshExtractResult (expected by postprocessing_utils.to_glb)
                from sam3d_objects.model.backbone.tdfy_dit.representations.mesh.cube2mesh import MeshExtractResult

                # Get vertices from loaded GLB
                vertices_np = np.array(trimesh_mesh.vertices)

                # IMPORTANT: The saved GLB has Y-up coordinates (transformed during save).
                # The PLY Gaussians are in Z-up coordinates (no transformation on save).
                # The downstream to_glb() expects Z-up and will transform again.
                # So we need to reverse the Y-up back to Z-up to match Gaussians.
                # Y-up to Z-up transformation (inverse of [[1,0,0], [0,0,-1], [0,1,0]])
                Y_TO_Z_UP = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
                vertices_np = vertices_np @ Y_TO_Z_UP
                print(f"[Worker] Transformed mesh vertices from Y-up to Z-up to match Gaussians", file=sys.stderr)

                vertices_tensor = torch.tensor(vertices_np, dtype=torch.float32, device=device)
                faces_tensor = torch.tensor(np.array(trimesh_mesh.faces), dtype=torch.long, device=device)

                # Get vertex colors if available, otherwise use white
                if trimesh_mesh.visual is not None and hasattr(trimesh_mesh.visual, 'vertex_colors') and trimesh_mesh.visual.vertex_colors is not None:
                    vertex_colors = np.array(trimesh_mesh.visual.vertex_colors)[:, :3] / 255.0  # RGB, normalized
                    print(f"[Worker] Got vertex colors from mesh: shape={vertex_colors.shape}", file=sys.stderr)
                else:
                    vertex_colors = np.ones((len(trimesh_mesh.vertices), 3), dtype=np.float32)
                    print(f"[Worker] No vertex colors, using white: shape={vertex_colors.shape}", file=sys.stderr)

                vertex_attrs_tensor = torch.tensor(vertex_colors, dtype=torch.float32, device=device)

                mesh = MeshExtractResult(
                    vertices=vertices_tensor,
                    faces=faces_tensor,
                    vertex_attrs=vertex_attrs_tensor,
                    res=64
                )
                print(f"[Worker] Converted to MeshExtractResult: vertices={mesh.vertices.shape}, faces={mesh.faces.shape}", file=sys.stderr)

                # Combine into single dict for stage2_output
                # Note: gaussian and mesh must be in lists to match expected format from pipeline
                stage2_output = {
                    "gaussian": [gaussian],
                    "mesh": [mesh],
                    "stage1_data": {}  # No stage1 data when loading from files
                }
                print(f"[Worker] Assembled stage2_output dict with keys: {list(stage2_output.keys())}", file=sys.stderr)
                print(f"[Worker] gaussian list length: {len(stage2_output['gaussian'])}, mesh list length: {len(stage2_output['mesh'])}", file=sys.stderr)

        # Load slat_output if provided (from path or base64)
        slat_output = None
        if request.get("slat_output_path") is not None and os.path.exists(request.get("slat_output_path")):
            print(f"[Worker] Loading SLAT output from: {request.get('slat_output_path')}", file=sys.stderr)
            slat_output = torch.load(request.get("slat_output_path"))
        elif request.get("slat_output") is not None:
            slat_output = pickle.loads(base64.b64decode(request.get("slat_output")))

        print(f"[Worker] Running inference (seed={seed})", file=sys.stderr)
        print(f"[Worker] Stage 1: steps={stage1_inference_steps}, cfg={stage1_cfg_strength}, distillation={use_stage1_distillation}", file=sys.stderr)
        print(f"[Worker] Stage 2: steps={stage2_inference_steps}, cfg={stage2_cfg_strength}, distillation={use_stage2_distillation}", file=sys.stderr)
        print(f"[Worker] Postprocess: texture_size={texture_size}, simplify={simplify}, texture_mode={texture_mode}, rendering_engine={rendering_engine}", file=sys.stderr)
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
            pointmap=pointmap,  # Pass pre-computed pointmap if available
            use_stage1_distillation=use_stage1_distillation,
            use_stage2_distillation=use_stage2_distillation,
            texture_mode=texture_mode,
            rendering_engine=rendering_engine,
            merge_mask=merge_mask,
            auto_resize_mask=auto_resize_mask,
        )

        print(f"[Worker] Inference completed", file=sys.stderr)

        # Transform to global coordinates for final outputs (not intermediate stages)
        # This ensures multiple objects from the same image can be correctly combined
        if not stage1_only and not stage2_only and not slat_only:
            output = transform_to_global_coordinates(output)

        # Special handling for stage1_only mode - save to disk and return path + pose
        if stage1_only:
            print(f"[Worker] Stage 1 only - saving to disk", file=sys.stderr)
            saved_output = save_output_to_disk(output, Path(output_dir))

            # Extract pose data for direct access (not just in saved file)
            rotation = output.get("rotation")
            translation = output.get("translation")
            scale = output.get("scale")

            # Convert tensors to lists for JSON serialization
            if rotation is not None and hasattr(rotation, 'tolist'):
                rotation = rotation.cpu().tolist() if hasattr(rotation, 'cpu') else rotation.tolist()
            if translation is not None and hasattr(translation, 'tolist'):
                translation = translation.cpu().tolist() if hasattr(translation, 'cpu') else translation.tolist()
            if scale is not None and hasattr(scale, 'tolist'):
                scale = scale.cpu().tolist() if hasattr(scale, 'cpu') else scale.tolist()

            print(f"[Worker] Pose data: rotation={rotation is not None}, translation={translation is not None}, scale={scale is not None}", file=sys.stderr)

            return {
                "status": "success",
                "stage1_mode": True,
                "output": saved_output,  # Contains file paths
                "rotation": rotation,
                "translation": translation,
                "scale": scale,
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

    # Configure loguru to only show errors (suppress INFO/WARNING spam from vendor code)
    try:
        from loguru import logger
        logger.remove()  # Remove default handler
        logger.add(sys.stderr, level="ERROR", format="{message}")
    except ImportError:
        pass  # loguru not available yet

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
            elif request.get("command") == "pose_optimization":
                response = run_pose_optimization(request)
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
