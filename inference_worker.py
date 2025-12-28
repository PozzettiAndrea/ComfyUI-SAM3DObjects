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
_LAZY_MANAGER = None


class LazyModelManager:
    """
    Lazy loading manager for SAM3D models.

    Loads models only when needed and unloads them after use when use_gpu_cache=False.
    This allows running on GPUs with less than 32GB VRAM.
    """

    def __init__(self, config_path: str, compile: bool = False):
        self.config_path = config_path
        self.compile = compile
        self.checkpoint_dir = Path(config_path).parent

        # Track loaded models
        self.loaded_models = {}
        self.condition_embedders = {}
        self.preprocessors = {}
        self.depth_model = None
        self.pose_decoder = None

        # Store config for later use
        self._config = None
        self._full_pipeline = None  # For backward compat if needed

        # Setup environment (same as load_model does)
        self._setup_environment()

        # Parse config to get model paths
        self._parse_config()

    def _setup_environment(self):
        """Setup environment variables and paths (extracted from load_model)."""
        import os
        import platform
        import shutil

        # Skip sam3d_objects initialization BEFORE any imports
        os.environ['LIDRA_SKIP_INIT'] = '1'

        # Add vendor directory to path
        vendor_path = Path(__file__).parent / "vendor"
        if str(vendor_path) not in sys.path:
            sys.path.insert(0, str(vendor_path))
            print(f"[LazyManager] Added vendor path: {vendor_path}", file=sys.stderr)

        # Preload modules to avoid circular import issues
        # These imports need to happen in a specific order
        try:
            # First load base modules without circular deps
            import sam3d_objects.model.backbone.tdfy_dit.modules.sparse
            import sam3d_objects.model.backbone.tdfy_dit.modules.attention
            # Load sh_utils before gaussian_model (breaks circular dep)
            import sam3d_objects.model.backbone.tdfy_dit.renderers.sh_utils
            # Now we can safely load gaussian_model
            import sam3d_objects.model.backbone.tdfy_dit.representations.gaussian.gaussian_model
            # Then load renderers which depend on gaussian
            import sam3d_objects.model.backbone.tdfy_dit.renderers.gaussian_render
            # Finally load models which use everything
            import sam3d_objects.model.backbone.tdfy_dit.models
            print(f"[LazyManager] Preloaded modules successfully", file=sys.stderr)
        except ImportError as e:
            print(f"[LazyManager] Warning: Could not preload modules: {e}", file=sys.stderr)
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        os.environ['PYTHONUTF8'] = '1'

        # Setup venv bin path
        venv_bin = (Path(__file__).parent / "_env" / "bin").resolve()
        if venv_bin.exists():
            os.environ['PATH'] = f"{venv_bin}{os.pathsep}{os.environ.get('PATH', '')}"

        # Setup model cache directory
        config_dir = self.checkpoint_dir.parent
        models_cache_dir = config_dir / "_models_cache"
        models_cache_dir.mkdir(exist_ok=True)

        os.environ['TORCH_HOME'] = str(models_cache_dir / "torch")
        os.environ['HF_HOME'] = str(models_cache_dir / "huggingface")
        os.environ['TRANSFORMERS_CACHE'] = str(models_cache_dir / "transformers")

    def _parse_config(self):
        """Parse pipeline.yaml to extract model paths."""
        from omegaconf import OmegaConf

        self._config = OmegaConf.load(self.config_path)
        print(f"[LazyManager] Parsed config from {self.config_path}", file=sys.stderr)

        # Store paths for each model component
        self.model_paths = {
            'ss_generator': {
                'config': self.checkpoint_dir / self._config.ss_generator_config_path,
                'ckpt': self.checkpoint_dir / self._config.ss_generator_ckpt_path,
            },
            'slat_generator': {
                'config': self.checkpoint_dir / self._config.slat_generator_config_path,
                'ckpt': self.checkpoint_dir / self._config.slat_generator_ckpt_path,
            },
            'ss_decoder': {
                'config': self.checkpoint_dir / self._config.ss_decoder_config_path,
                'ckpt': self.checkpoint_dir / self._config.ss_decoder_ckpt_path,
            },
            'slat_decoder_gs': {
                'config': self.checkpoint_dir / self._config.slat_decoder_gs_config_path,
                'ckpt': self.checkpoint_dir / self._config.slat_decoder_gs_ckpt_path,
            },
            'slat_decoder_mesh': {
                'config': self.checkpoint_dir / self._config.slat_decoder_mesh_config_path,
                'ckpt': self.checkpoint_dir / self._config.slat_decoder_mesh_ckpt_path,
            },
        }

        # Optional models
        if hasattr(self._config, 'slat_decoder_gs_4_config_path') and self._config.slat_decoder_gs_4_config_path:
            self.model_paths['slat_decoder_gs_4'] = {
                'config': self.checkpoint_dir / self._config.slat_decoder_gs_4_config_path,
                'ckpt': self.checkpoint_dir / self._config.slat_decoder_gs_4_ckpt_path,
            }

    def _get_dtype(self):
        """Get torch dtype from config."""
        dtype_str = getattr(self._config, 'dtype', 'float16')
        if dtype_str == 'bfloat16':
            return torch.bfloat16
        elif dtype_str == 'float16':
            return torch.float16
        else:
            return torch.float32

    def load_depth_model(self):
        """Load only the depth model (MoGe)."""
        if self.depth_model is not None:
            return self.depth_model

        print(f"[LazyManager] Loading depth model (MoGe)...", file=sys.stderr)

        from hydra.utils import instantiate

        # Instantiate depth model from config
        depth_config = self._config.get('depth_model')
        if depth_config:
            self.depth_model = instantiate(depth_config)
            if hasattr(self.depth_model, 'cuda'):
                self.depth_model = self.depth_model.cuda()
            print(f"[LazyManager] Depth model loaded", file=sys.stderr)
        else:
            raise RuntimeError("No depth_model config found in pipeline.yaml")

        return self.depth_model

    def unload_depth_model(self):
        """Unload depth model from GPU."""
        if self.depth_model is not None:
            print(f"[LazyManager] Unloading depth model...", file=sys.stderr)
            if hasattr(self.depth_model, 'cpu'):
                self.depth_model.cpu()
            del self.depth_model
            self.depth_model = None
            self._clear_cache()

    def load_condition_embedder(self, embedder_type='ss'):
        """Load condition embedder (shared between stages)."""
        key = f'{embedder_type}_condition_embedder'
        if key in self.condition_embedders:
            return self.condition_embedders[key]

        print(f"[LazyManager] Loading {key}...", file=sys.stderr)

        from omegaconf import OmegaConf
        from hydra.utils import instantiate
        from sam3d_objects.model.io import load_model_from_checkpoint, filter_and_remove_prefix_state_dict_fn

        # Get the generator config path
        if embedder_type == 'ss':
            config_path = self.model_paths['ss_generator']['config']
            ckpt_path = self.model_paths['ss_generator']['ckpt']
        else:
            config_path = self.model_paths['slat_generator']['config']
            ckpt_path = self.model_paths['slat_generator']['ckpt']

        # Load config and extract embedder backbone
        gen_config = OmegaConf.load(config_path)
        embedder_config = gen_config.module.condition_embedder.backbone

        # Instantiate embedder
        embedder = instantiate(embedder_config)

        # Load weights from checkpoint
        # The generator checkpoint has state_dict wrapper and prefix "_base_models.condition_embedder."
        embedder = load_model_from_checkpoint(
            embedder,
            str(ckpt_path),
            strict=False,
            device="cpu",
            freeze=True,
            eval=True,
            state_dict_key="state_dict",
            state_dict_fn=filter_and_remove_prefix_state_dict_fn("_base_models.condition_embedder."),
        )

        embedder = embedder.cuda()
        embedder = embedder.to(self._get_dtype())

        self.condition_embedders[key] = embedder
        print(f"[LazyManager] {key} loaded", file=sys.stderr)

        return embedder

    def load_model(self, model_name):
        """Load a specific model component."""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]

        print(f"[LazyManager] Loading {model_name}...", file=sys.stderr)

        from omegaconf import OmegaConf
        from hydra.utils import instantiate
        from sam3d_objects.model.io import (
            load_model_from_checkpoint,
            filter_and_remove_prefix_state_dict_fn,
            remove_prefix_state_dict_fn,
        )

        paths = self.model_paths.get(model_name)
        if not paths:
            raise ValueError(f"Unknown model: {model_name}")

        # Load config
        config = OmegaConf.load(paths['config'])

        # Generators use nested config and different state_dict handling
        is_generator = model_name in ['ss_generator', 'slat_generator']

        if is_generator:
            # Generators: use nested config path and filter prefix
            config = config["module"]["generator"]["backbone"]
            state_dict_key = "state_dict"
            state_dict_fn = filter_and_remove_prefix_state_dict_fn("_base_models.generator.")
        else:
            # Decoders: use full config and raw state dict
            state_dict_key = None
            state_dict_fn = remove_prefix_state_dict_fn("module.")

        # Instantiate model
        model = instantiate(config)

        # Load checkpoint weights
        model = load_model_from_checkpoint(
            model,
            str(paths['ckpt']),
            strict=False,
            device="cpu",
            freeze=True,
            eval=True,
            state_dict_key=state_dict_key,
            state_dict_fn=state_dict_fn,
        )

        # Move to GPU and set dtype
        model = model.cuda()
        if is_generator:
            model = model.to(self._get_dtype())

        self.loaded_models[model_name] = model
        print(f"[LazyManager] {model_name} loaded", file=sys.stderr)

        return model

    def unload_model(self, model_name):
        """Unload a specific model from GPU."""
        if model_name in self.loaded_models:
            print(f"[LazyManager] Unloading {model_name}...", file=sys.stderr)
            model = self.loaded_models[model_name]
            if hasattr(model, 'cpu'):
                model.cpu()
            del self.loaded_models[model_name]
            self._clear_cache()

    def unload_all(self):
        """Unload all models."""
        print(f"[LazyManager] Unloading all models...", file=sys.stderr)

        # Unload depth model
        self.unload_depth_model()

        # Unload all loaded models
        for name in list(self.loaded_models.keys()):
            self.unload_model(name)

        # Unload condition embedders
        for name in list(self.condition_embedders.keys()):
            if hasattr(self.condition_embedders[name], 'cpu'):
                self.condition_embedders[name].cpu()
            del self.condition_embedders[name]

        self.condition_embedders = {}
        self._clear_cache()

    def _clear_cache(self):
        """Clear CUDA cache and run garbage collection."""
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_full_pipeline(self):
        """
        Get the full pipeline (loads everything).
        For backward compatibility with existing code that needs all models.
        """
        if self._full_pipeline is not None:
            return self._full_pipeline

        print(f"[LazyManager] Loading full pipeline (all models)...", file=sys.stderr)

        from omegaconf import OmegaConf
        from hydra.utils import instantiate

        config = OmegaConf.load(self.config_path)
        config.compile_model = self.compile
        config.workspace_dir = str(self.checkpoint_dir)

        self._full_pipeline = instantiate(config)

        return self._full_pipeline

    def get_preprocessor(self, preprocessor_type='ss'):
        """Get preprocessor (instantiated lazily)."""
        key = f'{preprocessor_type}_preprocessor'
        if key in self.preprocessors:
            return self.preprocessors[key]

        from hydra.utils import instantiate

        if preprocessor_type == 'ss':
            preprocessor_config = self._config.get('ss_preprocessor')
        else:
            preprocessor_config = self._config.get('slat_preprocessor')

        if preprocessor_config:
            self.preprocessors[key] = instantiate(preprocessor_config)
        else:
            from sam3d_objects.pipeline import preprocess_utils
            self.preprocessors[key] = preprocess_utils.get_default_preprocessor()

        return self.preprocessors[key]

    def get_pose_decoder(self):
        """Get pose decoder (instantiated lazily)."""
        if self.pose_decoder is not None:
            return self.pose_decoder

        from sam3d_objects.pipeline.inference_utils import get_pose_decoder

        # Get pose decoder name from config
        pose_decoder_name = getattr(self._config, 'pose_decoder_name', 'default')

        self.pose_decoder = get_pose_decoder(pose_decoder_name)
        return self.pose_decoder

    def get_config_value(self, key, default=None):
        """Get a value from the config."""
        return getattr(self._config, key, default)

    def unload_condition_embedder(self, embedder_type='ss'):
        """Unload a condition embedder to free VRAM."""
        key = f'{embedder_type}_condition_embedder'
        if key in self.condition_embedders:
            print(f"[LazyManager] Unloading {key}...", file=sys.stderr)
            if hasattr(self.condition_embedders[key], 'cpu'):
                self.condition_embedders[key].cpu()
            del self.condition_embedders[key]
            self._clear_cache()

    def get_slat_stats(self):
        """Get SLAT mean and std from config."""
        slat_mean = self._config.get('slat_mean', [0.0] * 8)
        slat_std = self._config.get('slat_std', [1.0] * 8)
        return torch.tensor(slat_mean), torch.tensor(slat_std)


def get_lazy_manager(config_path: str, compile: bool = False) -> LazyModelManager:
    """Get or create a LazyModelManager instance."""
    global _LAZY_MANAGER

    if _LAZY_MANAGER is not None and _LAZY_MANAGER.config_path == config_path:
        return _LAZY_MANAGER

    print(f"[Worker] Creating LazyModelManager for {config_path}", file=sys.stderr)
    _LAZY_MANAGER = LazyModelManager(config_path, compile)
    return _LAZY_MANAGER


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
    # nvdiffrast is the default rendering engine for better quality
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
    # rendering_engine comes from user's node parameter, not hardcoded here
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
    pointmap = torch.load(pointmap_path, weights_only=False)
    print(f"[Worker] Loaded pointmap tensor: shape={pointmap.shape}", file=sys.stderr)

    if torch.cuda.is_available():
        pointmap = pointmap.cuda()

    return pointmap


def _preprocess_image_lazy(image_np, mask_np, preprocessor, pointmap=None, device="cuda"):
    """
    Preprocess image using a preprocessor (extracted from InferencePipeline).

    Args:
        image_np: Image as numpy array (H, W, 4) RGBA format
        mask_np: Mask as numpy array (unused - mask comes from alpha channel)
        preprocessor: Preprocessor instance
        pointmap: Optional pointmap tensor (H, W, 3) or numpy array

    Returns:
        dict with preprocessed image, mask, pointmap etc.
    """
    import numpy as np
    from sam3d_objects.data.dataset.tdfy.img_and_mask_transforms import get_mask

    # Ensure RGBA format
    assert image_np.ndim == 3, f"Expected 3D image, got {image_np.ndim}D"
    assert image_np.shape[-1] == 4, f"Expected RGBA (4 channels), got {image_np.shape[-1]}"
    assert image_np.dtype == np.uint8, f"Expected uint8, got {image_np.dtype}"

    # Convert to float [0, 1] and tensor (CHW format)
    image_float = (image_np / 255.0).astype(np.float32)
    rgba_image = torch.from_numpy(image_float).permute(2, 0, 1).contiguous()
    rgb_image = rgba_image[:3]
    rgb_image_mask = get_mask(rgba_image, None, "ALPHA_CHANNEL")

    # Convert pointmap to tensor if needed (should be CHW format for preprocessor)
    if pointmap is not None:
        if isinstance(pointmap, np.ndarray):
            pointmap = torch.from_numpy(pointmap).float()
        # If HWC, convert to CHW
        if pointmap.dim() == 3 and pointmap.shape[-1] == 3:
            pointmap = pointmap.permute(2, 0, 1).contiguous()

    # Use the preprocessor's internal method
    preprocessor_return_dict = preprocessor._process_image_mask_pointmap_mess(
        rgb_image, rgb_image_mask, pointmap
    )

    # Build result dict with batch dimension and move to device
    _item = preprocessor_return_dict
    item = {
        "mask": _item["mask"][None].to(device),
        "image": _item["image"][None].to(device),
        "rgb_image": _item["rgb_image"][None].to(device),
        "rgb_image_mask": _item["rgb_image_mask"][None].to(device),
    }

    # Add pointmap-related fields if available
    if pointmap is not None and hasattr(preprocessor, 'pointmap_transform') and preprocessor.pointmap_transform != (None,):
        if "pointmap" in _item:
            item["pointmap"] = _item["pointmap"][None].to(device)
        if "rgb_pointmap" in _item:
            item["rgb_pointmap"] = _item["rgb_pointmap"][None].to(device)
        if "pointmap_scale" in _item:
            item["pointmap_scale"] = _item["pointmap_scale"][None].to(device)
        if "pointmap_shift" in _item:
            item["pointmap_shift"] = _item["pointmap_shift"][None].to(device)
        if "rgb_pointmap_scale" in _item:
            item["rgb_pointmap_scale"] = _item["rgb_pointmap_scale"][None].to(device)
        if "rgb_pointmap_shift" in _item:
            item["rgb_pointmap_shift"] = _item["rgb_pointmap_shift"][None].to(device)

    return item


def _run_stage1_lazy(
    lazy_manager: LazyModelManager,
    image,
    mask,
    pointmap,
    seed: int = 42,
    inference_steps: int = 25,
    cfg_strength: float = 7.0,
    unload_after: bool = True,
    output_dir: str = None
) -> Dict[str, Any]:
    """
    Run Stage 1 (sparse structure generation) using lazy loading.

    Models loaded: ss_generator (~6.7 GB), ss_condition_embedder (~1.2 GB), ss_decoder (~150 MB)
    Peak VRAM: ~8-9 GB

    Args:
        lazy_manager: LazyModelManager instance
        image: PIL Image
        mask: PIL Image (mask)
        pointmap: numpy array (H, W, 3) from depth estimation
        seed: Random seed
        inference_steps: Number of inference steps
        cfg_strength: CFG strength
        unload_after: Whether to unload models after use
        output_dir: Directory to save output files

    Returns:
        Dict with stage1 output file paths and pose data
    """
    import numpy as np
    from sam3d_objects.pipeline.inference_utils import (
        downsample_sparse_structure,
        prune_sparse_structure,
    )

    print(f"[Worker] Running Stage 1 (sparse gen) with lazy loading", file=sys.stderr)

    # Set seed
    torch.manual_seed(seed)

    # Convert image/mask to numpy
    image_np = np.array(image)
    if image_np.ndim == 2:
        image_np = np.stack([image_np] * 3 + [np.full_like(image_np, 255)], axis=-1)
    elif image_np.shape[-1] == 3:
        alpha = np.full((image_np.shape[0], image_np.shape[1], 1), 255, dtype=np.uint8)
        image_np = np.concatenate([image_np, alpha], axis=-1)

    mask_np = np.array(mask) if mask is not None else None

    # Get preprocessor and preprocess image
    print(f"[Worker] Preprocessing image...", file=sys.stderr)
    ss_preprocessor = lazy_manager.get_preprocessor('ss')

    # Convert pointmap to tensor for preprocessing
    if isinstance(pointmap, np.ndarray):
        pointmap_tensor = torch.from_numpy(pointmap).float()
    else:
        pointmap_tensor = pointmap

    # Preprocess
    ss_input_dict = _preprocess_image_lazy(image_np, mask_np, ss_preprocessor, pointmap=pointmap_tensor)

    # Store pointmap scale/shift for pose decoding
    pointmap_scale = ss_input_dict.get("pointmap_scale", None)
    pointmap_shift = ss_input_dict.get("pointmap_shift", None)

    # Load models
    print(f"[Worker] Loading Stage 1 models...", file=sys.stderr)
    ss_generator = lazy_manager.load_model('ss_generator')
    ss_decoder = lazy_manager.load_model('ss_decoder')
    ss_embedder = lazy_manager.load_condition_embedder('ss')

    # Configure generator
    ss_generator.no_shortcut = True
    ss_generator.reverse_fn.strength = cfg_strength
    ss_generator.reverse_fn.strength_pm = lazy_manager.get_config_value('ss_cfg_strength_pm', 0.0)
    ss_generator.inference_steps = inference_steps

    print(f"[Worker] Running sparse structure generation...", file=sys.stderr)

    dtype = lazy_manager._get_dtype()
    downsample_ss_dist = lazy_manager.get_config_value('downsample_ss_dist', 0)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=dtype):
            # Get condition embeddings
            image_tensor = ss_input_dict["image"]
            bs = image_tensor.shape[0]

            # Check if MM-DiT architecture
            has_latent_mapping = hasattr(ss_generator.reverse_fn.backbone, "latent_mapping")

            if has_latent_mapping:
                latent_shape_dict = {
                    k: (bs,) + (v.pos_emb.shape[0], v.input_layer.in_features)
                    for k, v in ss_generator.reverse_fn.backbone.latent_mapping.items()
                }
            else:
                latent_shape_dict = (bs,) + (4096, 8)

            # Get condition input mapping from config
            ss_condition_input_mapping = lazy_manager.get_config_value('ss_condition_input_mapping', ['image'])

            # Get condition embeddings
            condition_args = [ss_input_dict[k] for k in ss_condition_input_mapping if k in ss_input_dict]
            condition_kwargs = {k: v for k, v in ss_input_dict.items() if k not in ss_condition_input_mapping}

            # Embed conditions
            if ss_embedder is not None and len(condition_args) > 0:
                tokens = ss_embedder(*condition_args, **condition_kwargs)
                condition_args = (tokens,)
                condition_kwargs = {}
            elif ss_embedder is not None:
                tokens = ss_embedder(**condition_kwargs)
                condition_args = (tokens,)
                condition_kwargs = {}

            # Run generator
            return_dict = ss_generator(
                latent_shape_dict,
                image_tensor.device,
                *condition_args,
                **condition_kwargs,
            )

            if not has_latent_mapping:
                return_dict = {"shape": return_dict}

            shape_latent = return_dict["shape"]

            # Decode sparse structure
            ss = ss_decoder(
                shape_latent.permute(0, 2, 1)
                .contiguous()
                .view(shape_latent.shape[0], 8, 16, 16, 16)
            )
            coords = torch.argwhere(ss > 0)[:, [0, 2, 3, 4]].int()

            # Downsample output
            return_dict["coords_original"] = coords
            original_shape = coords.shape

            if downsample_ss_dist > 0:
                coords = prune_sparse_structure(
                    coords,
                    max_neighbor_axes_dist=downsample_ss_dist,
                )

            coords, downsample_factor = downsample_sparse_structure(coords)
            print(f"[Worker] Downsampled coords from {original_shape[0]} to {coords.shape[0]}", file=sys.stderr)

            return_dict["coords"] = coords
            return_dict["downsample_factor"] = downsample_factor

    # Apply pose decoding
    print(f"[Worker] Decoding pose...", file=sys.stderr)
    pose_decoder = lazy_manager.get_pose_decoder()
    pose_result = pose_decoder(
        return_dict,
        scene_scale=pointmap_scale,
        scene_shift=pointmap_shift,
    )
    return_dict.update(pose_result)

    # Rescale after downsampling
    return_dict["scale"] = return_dict["scale"] * return_dict["downsample_factor"]

    # Add voxel coordinates
    return_dict["voxel"] = return_dict["coords"][:, 1:] / 64 - 0.5

    # Unload models if requested
    if unload_after:
        print(f"[Worker] Unloading Stage 1 models...", file=sys.stderr)
        lazy_manager.unload_model('ss_generator')
        lazy_manager.unload_model('ss_decoder')
        lazy_manager.unload_condition_embedder('ss')

    print(f"[Worker] Stage 1 complete", file=sys.stderr)

    # Save sparse structure directly (don't use save_output_to_disk which has detection logic)
    if output_dir:
        save_dir = Path(output_dir)
    else:
        import tempfile
        save_dir = Path(tempfile.mkdtemp())
    save_dir.mkdir(parents=True, exist_ok=True)

    sparse_path = save_dir / "sparse_structure.pt"
    torch.save(return_dict, sparse_path)
    print(f"[Worker] Saved sparse structure: {sparse_path}", file=sys.stderr)

    saved_output = {
        "output_dir": str(save_dir),
        "files": {"sparse_structure": str(sparse_path)},
        "metadata": {}
    }

    # Extract pose data for direct access
    rotation = return_dict.get("rotation")
    translation = return_dict.get("translation")
    scale = return_dict.get("scale")

    # Convert tensors to lists for JSON serialization
    if rotation is not None and hasattr(rotation, 'tolist'):
        rotation = rotation.cpu().tolist() if hasattr(rotation, 'cpu') else rotation.tolist()
    if translation is not None and hasattr(translation, 'tolist'):
        translation = translation.cpu().tolist() if hasattr(translation, 'cpu') else translation.tolist()
    if scale is not None and hasattr(scale, 'tolist'):
        scale = scale.cpu().tolist() if hasattr(scale, 'cpu') else scale.tolist()

    return {
        "status": "success",
        "stage1_mode": True,
        "output": saved_output,
        "rotation": rotation,
        "translation": translation,
        "scale": scale,
    }


def _run_stage2_lazy(
    lazy_manager: LazyModelManager,
    image,
    mask,
    stage1_output: Dict,
    seed: int = 42,
    inference_steps: int = 25,
    cfg_strength: float = 5.0,
    unload_after: bool = True,
    output_dir: str = None
) -> Dict[str, Any]:
    """
    Run Stage 2 (SLAT generation) using lazy loading.

    Models loaded: slat_generator (~4.9 GB), slat_condition_embedder (~1.2 GB)
    Peak VRAM: ~6-7 GB

    Args:
        lazy_manager: LazyModelManager instance
        image: PIL Image
        mask: PIL Image (mask)
        stage1_output: Dict with coords from Stage 1
        seed: Random seed
        inference_steps: Number of inference steps
        cfg_strength: CFG strength
        unload_after: Whether to unload models after use
        output_dir: Directory to save output files

    Returns:
        Dict with SLAT file path
    """
    import numpy as np
    import base64
    import pickle
    from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp

    print(f"[Worker] Running Stage 2 (SLAT gen) with lazy loading", file=sys.stderr)

    # Set seed
    torch.manual_seed(seed)

    # Convert image to numpy
    image_np = np.array(image)
    if image_np.ndim == 2:
        image_np = np.stack([image_np] * 3 + [np.full_like(image_np, 255)], axis=-1)
    elif image_np.shape[-1] == 3:
        alpha = np.full((image_np.shape[0], image_np.shape[1], 1), 255, dtype=np.uint8)
        image_np = np.concatenate([image_np, alpha], axis=-1)

    mask_np = np.array(mask) if mask is not None else None

    # Get preprocessor and preprocess image
    print(f"[Worker] Preprocessing image for SLAT...", file=sys.stderr)
    slat_preprocessor = lazy_manager.get_preprocessor('slat')
    slat_input_dict = _preprocess_image_lazy(image_np, mask_np, slat_preprocessor)

    # Get coords from stage1_output (may be base64-encoded from lazy stage1)
    coords = stage1_output.get("coords")
    if isinstance(coords, str):
        coords = pickle.loads(base64.b64decode(coords))
    if isinstance(coords, np.ndarray):
        coords = torch.from_numpy(coords).int()
    coords = coords.cuda()

    # Load models
    print(f"[Worker] Loading Stage 2 models...", file=sys.stderr)
    slat_generator = lazy_manager.load_model('slat_generator')
    slat_embedder = lazy_manager.load_condition_embedder('slat')

    # Configure generator
    slat_generator.no_shortcut = True
    slat_generator.reverse_fn.strength = cfg_strength
    slat_generator.inference_steps = inference_steps

    print(f"[Worker] Running SLAT generation...", file=sys.stderr)

    dtype = lazy_manager._get_dtype()
    slat_mean, slat_std = lazy_manager.get_slat_stats()
    slat_mean = slat_mean.cuda()
    slat_std = slat_std.cuda()

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=dtype):
            image_tensor = slat_input_dict["image"]
            DEVICE = image_tensor.device
            latent_shape = (image_tensor.shape[0],) + (coords.shape[0], 8)

            # Get condition input mapping from config
            slat_condition_input_mapping = lazy_manager.get_config_value('slat_condition_input_mapping', ['image'])

            # Get condition embeddings
            condition_args = [slat_input_dict[k] for k in slat_condition_input_mapping if k in slat_input_dict]
            condition_kwargs = {k: v for k, v in slat_input_dict.items() if k not in slat_condition_input_mapping}

            # Embed conditions
            if slat_embedder is not None and len(condition_args) > 0:
                tokens = slat_embedder(*condition_args, **condition_kwargs)
                condition_args = (tokens,)
                condition_kwargs = {}
            elif slat_embedder is not None:
                tokens = slat_embedder(**condition_kwargs)
                condition_args = (tokens,)
                condition_kwargs = {}

            # Add coords to condition args
            condition_args = condition_args + (coords.cpu().numpy(),)

            # Run generator
            slat_feats = slat_generator(
                latent_shape, DEVICE, *condition_args, **condition_kwargs
            )

            # Create SparseTensor
            slat = sp.SparseTensor(
                coords=coords,
                feats=slat_feats[0],
            ).to(DEVICE)

            # Apply mean/std normalization
            slat = slat * slat_std + slat_mean

    # Unload models if requested
    if unload_after:
        print(f"[Worker] Unloading Stage 2 models...", file=sys.stderr)
        lazy_manager.unload_model('slat_generator')
        lazy_manager.unload_condition_embedder('slat')

    print(f"[Worker] Stage 2 complete", file=sys.stderr)

    # Build output dict with SLAT for saving
    output_dict = {
        "slat": slat,
        "stage1_data": stage1_output,
    }

    # Save output to disk (consistent with non-lazy path)
    if output_dir:
        saved_output = save_output_to_disk(output_dict, Path(output_dir))
    else:
        # Fallback to temp directory
        import tempfile
        temp_dir = Path(tempfile.mkdtemp())
        slat_path = temp_dir / "slat.pt"
        torch.save(output_dict, slat_path)
        saved_output = {
            "output_dir": str(temp_dir),
            "files": {"slat": str(slat_path)},
            "metadata": {}
        }

    return {
        "status": "success",
        "stage2_mode": True,
        "output": saved_output,
    }


def _run_decode_lazy(
    lazy_manager: LazyModelManager,
    slat_data: Dict,
    decode_format: str = "gaussian",
    unload_after: bool = True,
    output_dir: str = None,
    simplify: float = 0.95
) -> Dict[str, Any]:
    """
    Run Stage 3 (Gaussian or Mesh decoding) using lazy loading.

    Models loaded: slat_decoder_gs (~170 MB) or slat_decoder_mesh (~364 MB)
    Peak VRAM: ~1-2 GB

    Args:
        lazy_manager: LazyModelManager instance
        slat_data: Dict with slat from Stage 2 (may be base64-encoded)
        decode_format: "gaussian" or "mesh"
        unload_after: Whether to unload models after use
        output_dir: Directory to save output files
        simplify: Mesh simplification ratio

    Returns:
        Dict with file paths (ply_path for gaussian, glb_path for mesh)
    """
    import base64
    import pickle
    import numpy as np
    from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp

    print(f"[Worker] Running decode ({decode_format}) with lazy loading", file=sys.stderr)

    # Extract slat - handle different input formats
    # 1. Dict with "slat" key containing SparseTensor (from disk load)
    # 2. Dict with "slat" key containing base64-encoded data
    # 3. SparseTensor directly
    # 4. Dict with coords/feats keys

    slat = None

    if isinstance(slat_data, sp.SparseTensor):
        # Already a SparseTensor
        slat = slat_data
    elif isinstance(slat_data, dict) and "slat" in slat_data:
        slat_inner = slat_data["slat"]
        if isinstance(slat_inner, sp.SparseTensor):
            # SparseTensor from disk load
            slat = slat_inner
        elif isinstance(slat_inner, str):
            # Base64-encoded
            slat_inner = pickle.loads(base64.b64decode(slat_inner))
            if isinstance(slat_inner, sp.SparseTensor):
                slat = slat_inner
            else:
                # Dict with coords/feats
                coords = slat_inner["coords"]
                feats = slat_inner["feats"]
        elif isinstance(slat_inner, dict):
            coords = slat_inner["coords"]
            feats = slat_inner["feats"]
    elif isinstance(slat_data, dict):
        coords = slat_data.get("coords")
        feats = slat_data.get("feats")

    # If we don't have slat yet, reconstruct from coords/feats
    if slat is None:
        # Decode if base64-encoded
        if isinstance(coords, str):
            coords = pickle.loads(base64.b64decode(coords))
        if isinstance(feats, str):
            feats = pickle.loads(base64.b64decode(feats))

        # Convert to tensors
        coords = torch.from_numpy(coords).int().cuda() if isinstance(coords, np.ndarray) else coords.int().cuda()
        feats = torch.from_numpy(feats).cuda() if isinstance(feats, np.ndarray) else feats.cuda()
        slat = sp.SparseTensor(coords=coords, feats=feats)
    else:
        # Ensure slat is on GPU
        slat = slat.cuda()

    # Load decoder
    if decode_format == "gaussian":
        decoder_name = 'slat_decoder_gs'
    else:
        decoder_name = 'slat_decoder_mesh'

    print(f"[Worker] Loading decoder: {decoder_name}...", file=sys.stderr)
    decoder = lazy_manager.load_model(decoder_name)

    print(f"[Worker] Running decoder...", file=sys.stderr)

    with torch.no_grad():
        output = decoder(slat)

    # Unload decoder if requested
    if unload_after:
        print(f"[Worker] Unloading decoder...", file=sys.stderr)
        lazy_manager.unload_model(decoder_name)

    print(f"[Worker] Decode complete", file=sys.stderr)

    # Determine output directory
    if output_dir:
        save_dir = Path(output_dir)
    else:
        import tempfile
        save_dir = Path(tempfile.mkdtemp())
    save_dir.mkdir(parents=True, exist_ok=True)

    saved_files = {"output_dir": str(save_dir), "files": {}}

    if decode_format == "gaussian":
        # Gaussian output is a list of Gaussian objects
        gaussian = output[0] if isinstance(output, (list, tuple)) else output

        # Save PLY file
        ply_path = save_dir / "gaussian.ply"
        try:
            gaussian.save_ply(str(ply_path))
            saved_files["files"]["ply"] = str(ply_path)
            print(f"[Worker] Saved Gaussian PLY: {ply_path}", file=sys.stderr)
        except Exception as e:
            print(f"[Worker] Warning: Failed to save Gaussian PLY: {e}", file=sys.stderr)

        return {
            "status": "success",
            "stage2_mode": True,
            "output": saved_files,
            "file_output": saved_files,
        }
    else:
        # Mesh output
        mesh = output[0] if isinstance(output, (list, tuple)) else output
        import trimesh

        try:
            # Extract mesh data
            vertices = mesh.vertices.cpu().numpy() if hasattr(mesh.vertices, 'cpu') else mesh.vertices
            faces = mesh.faces.cpu().numpy() if hasattr(mesh.faces, 'cpu') else mesh.faces

            # Get vertex colors if available
            vertex_colors = None
            if hasattr(mesh, 'vertex_attrs') and mesh.vertex_attrs is not None:
                if isinstance(mesh.vertex_attrs, torch.Tensor):
                    vc = mesh.vertex_attrs.cpu().numpy()
                elif isinstance(mesh.vertex_attrs, dict) and 'color' in mesh.vertex_attrs:
                    vc = mesh.vertex_attrs['color']
                    if hasattr(vc, 'cpu'):
                        vc = vc.cpu().numpy()
                else:
                    vc = None

                if vc is not None:
                    # Ensure colors are in [0, 255] uint8 format with alpha
                    if vc.max() <= 1.0:
                        vc = (vc * 255).astype(np.uint8)
                    if vc.shape[-1] == 3:
                        # Add alpha channel
                        alpha = np.full((vc.shape[0], 1), 255, dtype=np.uint8)
                        vc = np.concatenate([vc, alpha], axis=-1)
                    vertex_colors = vc

            # Create trimesh with vertex colors
            trimesh_mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                vertex_colors=vertex_colors,
                process=False
            )

            # Save GLB
            glb_path = save_dir / "mesh.glb"
            glb_bytes = trimesh_mesh.export(file_type="glb")
            with open(glb_path, 'wb') as f:
                f.write(glb_bytes)
            saved_files["files"]["glb"] = str(glb_path)
            print(f"[Worker] Saved Mesh GLB: {glb_path}", file=sys.stderr)

        except Exception as e:
            print(f"[Worker] Warning: Failed to save Mesh GLB: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            # Fallback: save raw mesh data
            mesh_path = save_dir / "mesh.pt"
            mesh_data = {
                "vertices": mesh.vertices.cpu() if hasattr(mesh.vertices, 'cpu') else mesh.vertices,
                "faces": mesh.faces.cpu() if hasattr(mesh.faces, 'cpu') else mesh.faces,
            }
            torch.save(mesh_data, mesh_path)
            saved_files["files"]["mesh_pt"] = str(mesh_path)

        return {
            "status": "success",
            "stage2_mode": True,
            "output": saved_files,
            "file_output": saved_files,
        }


def _run_depth_only_lazy(lazy_manager: LazyModelManager, image, unload_after: bool = True) -> Dict[str, Any]:
    """
    Run depth estimation using lazy loading (loads only MoGe, not entire pipeline).

    This uses ~2GB VRAM instead of 15GB+ for the full pipeline.

    Args:
        lazy_manager: LazyModelManager instance
        image: PIL Image
        unload_after: Whether to unload depth model after use

    Returns:
        Dict with pointmap, intrinsics, depth
    """
    import numpy as np
    from pytorch3d.renderer import look_at_view_transform
    from pytorch3d.transforms import Transform3d

    print(f"[Worker] Running depth estimation with lazy loading", file=sys.stderr)

    # Load only the depth model
    depth_model = lazy_manager.load_depth_model()

    # Convert image to tensor format expected by depth model
    image_np = np.array(image)
    if image_np.ndim == 2:
        image_np = np.stack([image_np] * 3 + [np.full_like(image_np, 255)], axis=-1)
    elif image_np.shape[-1] == 3:
        alpha = np.full((image_np.shape[0], image_np.shape[1], 1), 255, dtype=np.uint8)
        image_np = np.concatenate([image_np, alpha], axis=-1)

    print(f"[Worker] Image shape: {image_np.shape}", file=sys.stderr)

    # Convert to float and tensor
    loaded_image = image_np.astype(np.float32) / 255.0
    loaded_image = torch.from_numpy(loaded_image)
    loaded_image = loaded_image.permute(2, 0, 1).contiguous()[:3]  # CHW, RGB only

    # Run depth model
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=lazy_manager._get_dtype()):
            output = depth_model(loaded_image)

    pointmaps = output["pointmaps"]

    # Apply camera convention transform (R3 -> PyTorch3D camera space)
    device = pointmaps.device
    r3_to_p3d_R, r3_to_p3d_T = look_at_view_transform(
        eye=np.array([[0, 0, -1]]),
        at=np.array([[0, 0, 0]]),
        up=np.array([[0, -1, 0]]),
        device=device,
    )
    camera_transform = Transform3d().rotate(r3_to_p3d_R).to(device)
    points_tensor = camera_transform.transform_points(pointmaps)

    intrinsics = output.get("intrinsics", None)

    # Convert to CHW format
    points_tensor = points_tensor.permute(2, 0, 1)

    # Infer intrinsics if not provided
    if intrinsics is None:
        from sam3d_objects.pipeline.utils.pointmap import infer_intrinsics_from_pointmap
        intrinsics_result = infer_intrinsics_from_pointmap(
            points_tensor.permute(1, 2, 0), device=device
        )
        intrinsics = intrinsics_result["intrinsics"]

    print(f"[Worker] Pointmap computed: shape={points_tensor.shape}", file=sys.stderr)
    print(f"[Worker] Intrinsics available: {intrinsics is not None}", file=sys.stderr)

    # Unload depth model if requested (frees ~2GB VRAM)
    if unload_after:
        lazy_manager.unload_depth_model()
        print(f"[Worker] Depth model unloaded", file=sys.stderr)

    # Serialize for transfer
    # Transpose from CHW (3, H, W) to HWC (H, W, 3)
    pointmap_hwc = points_tensor.permute(1, 2, 0).contiguous()
    pointmap_np = pointmap_hwc.cpu().numpy()
    print(f"[Worker] Pointmap transposed to HWC: {pointmap_np.shape}", file=sys.stderr)

    if intrinsics is not None and hasattr(intrinsics, 'cpu'):
        intrinsics_np = intrinsics.cpu().numpy()
    else:
        intrinsics_np = intrinsics

    pointmap_b64 = base64.b64encode(pickle.dumps(pointmap_np)).decode('utf-8')
    intrinsics_b64 = base64.b64encode(pickle.dumps(intrinsics_np)).decode('utf-8') if intrinsics_np is not None else None

    return {
        "status": "success",
        "depth_only": True,
        "lazy_loading": True,
        "pointmap": pointmap_b64,
        "intrinsics": intrinsics_b64,
    }


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


def run_texture_bake_direct(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run texture baking directly without loading any models.

    Loads Gaussian from PLY and Mesh from GLB, then calls to_glb() for texture baking.
    No embedders, generators, or other models are needed.

    Args:
        request: Dict with ply_path, glb_path, output_dir, texture_mode, etc.

    Returns:
        Dict with status and output (glb_path)
    """
    import trimesh
    import numpy as np
    from pathlib import Path

    print("[Worker] Running direct texture baking (no models)", file=sys.stderr)

    # Extract parameters
    ply_path = request["ply_path"]
    glb_path = request["glb_path"]
    output_dir = request["output_dir"]
    texture_mode = request.get("texture_mode", "opt")
    texture_size = request.get("texture_size", 1024)
    simplify = request.get("simplify", 0.95)
    with_mesh_postprocess = request.get("with_mesh_postprocess", False)
    rendering_engine = request.get("rendering_engine", "nvdiffrast")

    print(f"[Worker] PLY: {ply_path}", file=sys.stderr)
    print(f"[Worker] GLB: {glb_path}", file=sys.stderr)
    print(f"[Worker] Mode: {texture_mode}, Size: {texture_size}", file=sys.stderr)

    # Setup environment for sam3d_objects imports
    vendor_path = Path(__file__).parent / "vendor"
    if str(vendor_path) not in sys.path:
        sys.path.insert(0, str(vendor_path))

    # Import required modules
    from sam3d_objects.model.backbone.tdfy_dit.representations.gaussian import Gaussian
    from sam3d_objects.model.backbone.tdfy_dit.representations.mesh.cube2mesh import MeshExtractResult
    from sam3d_objects.model.backbone.tdfy_dit.utils.postprocessing_utils import to_glb

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load Gaussian from PLY
    print(f"[Worker] Loading Gaussian from PLY...", file=sys.stderr)
    gaussian = Gaussian(
        aabb=[-1, -1, -1, 2, 2, 2],  # Default AABB
        sh_degree=0,
        device=device
    )
    gaussian.load_ply(ply_path)
    print(f"[Worker] Loaded Gaussian with {gaussian._xyz.shape[0]} points", file=sys.stderr)

    # Load Mesh from GLB
    print(f"[Worker] Loading Mesh from GLB...", file=sys.stderr)
    loaded = trimesh.load(glb_path)

    # Handle Scene vs Mesh
    if isinstance(loaded, trimesh.Scene):
        meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise RuntimeError("No mesh geometries found in GLB")
        trimesh_mesh = meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)
    else:
        trimesh_mesh = loaded

    print(f"[Worker] Loaded mesh with {len(trimesh_mesh.vertices)} vertices", file=sys.stderr)

    # Convert to MeshExtractResult format expected by to_glb
    # NOTE: The GLB from _run_decode_lazy is saved in Z-up (raw decoder output)
    # The Gaussian PLY is also in Z-up. Both are aligned - no transformation needed.
    # to_glb() will apply the final Z-up→Y-up transform when saving.
    vertices_np = np.array(trimesh_mesh.vertices)
    vertices_tensor = torch.tensor(vertices_np, dtype=torch.float32, device=device)
    faces_tensor = torch.tensor(np.array(trimesh_mesh.faces), dtype=torch.long, device=device)

    # Get vertex colors if available
    if trimesh_mesh.visual is not None and hasattr(trimesh_mesh.visual, 'vertex_colors') and trimesh_mesh.visual.vertex_colors is not None:
        vertex_colors = np.array(trimesh_mesh.visual.vertex_colors)[:, :3] / 255.0
    else:
        vertex_colors = np.ones((len(trimesh_mesh.vertices), 3), dtype=np.float32)

    vertex_attrs_tensor = torch.tensor(vertex_colors, dtype=torch.float32, device=device)

    mesh = MeshExtractResult(
        vertices=vertices_tensor,
        faces=faces_tensor,
        vertex_attrs=vertex_attrs_tensor,
        res=64
    )

    # Run texture baking using to_glb
    print(f"[Worker] Running texture baking...", file=sys.stderr)
    result_mesh = to_glb(
        gaussian,
        mesh,
        simplify=simplify,
        fill_holes=with_mesh_postprocess,
        texture_size=texture_size,
        verbose=True,
        with_mesh_postprocess=with_mesh_postprocess,
        with_texture_baking=True,
        rendering_engine=rendering_engine,
        texture_mode=texture_mode,
    )

    # Save textured GLB
    output_path = Path(output_dir) / "mesh_textured.glb"
    glb_bytes = result_mesh.export(file_type="glb")
    with open(output_path, 'wb') as f:
        f.write(glb_bytes)

    print(f"[Worker] Saved textured GLB: {output_path}", file=sys.stderr)

    # Cleanup GPU memory
    del gaussian, mesh
    torch.cuda.empty_cache()

    return {
        "status": "success",
        "output": {
            "glb_path": str(output_path),
        }
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
            image_b64 = request["image"]
            image = deserialize_image(image_b64)
            use_lazy_loading = request.get("use_lazy_loading", True)

            if use_lazy_loading:
                # Lazy loading: loads only depth model (~2GB VRAM)
                print("[Worker] Running depth-only mode with LAZY LOADING (low VRAM)", file=sys.stderr)
                lazy_manager = get_lazy_manager(config_path, compile_model)
                return _run_depth_only_lazy(lazy_manager, image, unload_after=True)
            else:
                # Full pipeline loading (15GB+ VRAM)
                print("[Worker] Running depth-only mode (full pipeline)", file=sys.stderr)
                model = load_model(config_path, compile_model)
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
        rendering_engine = request.get("rendering_engine", "nvdiffrast")
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

        # Determine lazy loading mode
        # use_lazy_loading = True when we want to minimize VRAM (use_cache = True = CPU offload mode)
        use_lazy_loading = request.get("use_lazy_loading", use_cache)

        # Check if we're running an individual stage that supports lazy loading
        # In this case, we DON'T load the full pipeline upfront
        # Note: Check request params since stage1_output/slat_output aren't loaded yet
        has_stage1_input = request.get("stage1_output_path") is not None or request.get("stage1_output") is not None
        has_slat_input = request.get("slat_output_path") is not None or request.get("slat_output") is not None
        can_use_lazy_stage = use_lazy_loading and (
            (stage1_only and pointmap is not None) or
            (slat_only and has_stage1_input) or
            (gaussian_only and has_slat_input) or
            (mesh_only and has_slat_input)
        )

        # Only load full pipeline if we're NOT using lazy stage loading
        model = None
        if not can_use_lazy_stage:
            if use_lazy_loading:
                # Try lazy loading approach - load full pipeline but with better error handling
                try:
                    lazy_manager = get_lazy_manager(config_path, compile_model)
                    model = lazy_manager.get_full_pipeline()
                except torch.cuda.OutOfMemoryError as e:
                    # Provide helpful error message
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()

                    error_msg = (
                        f"Out of memory loading SAM3D models. "
                        f"SAM3D requires 32GB+ VRAM for full pipeline. "
                        f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB. "
                        f"Tip: Use individual stage nodes (DepthEstimate, SparseGen, etc.) with lazy loading "
                        f"to run on GPUs with less VRAM."
                    )
                    raise RuntimeError(error_msg) from e
            else:
                model = load_model(config_path, compile_model)

        # Deserialize inputs
        image = deserialize_image(image_b64)
        mask = deserialize_mask(mask_b64)

        # Load stage1_output if provided (from path or base64 for backward compat)
        stage1_output = None
        if request.get("stage1_output_path") is not None and os.path.exists(request.get("stage1_output_path")):
            print(f"[Worker] Loading Stage 1 output from: {request.get('stage1_output_path')}", file=sys.stderr)
            stage1_output = torch.load(request.get("stage1_output_path"), weights_only=False)
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

                # NOTE: The GLB from _run_decode_lazy is saved in Z-up (raw decoder output).
                # The Gaussian PLY is also in Z-up. Both are aligned - no transformation needed.
                # to_glb() will apply the final Z-up→Y-up transform when saving.

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
            slat_output = torch.load(request.get("slat_output_path"), weights_only=False)
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

        # =====================================================================
        # LAZY LOADING BRANCHES
        # When use_lazy_loading=True, run individual stages with minimal VRAM
        # =====================================================================
        if use_lazy_loading and (stage1_only or slat_only or gaussian_only or mesh_only):
            print(f"[Worker] *** LAZY LOADING MODE *** (low VRAM, ~6GB per stage)", file=sys.stderr)
            lazy_manager = get_lazy_manager(config_path, compile_model)

            # Stage 1 only (sparse structure generation)
            if stage1_only and pointmap is not None:
                print(f"[Worker] Running Stage 1 with lazy loading", file=sys.stderr)
                from PIL import Image as PILImage
                mask_pil = PILImage.fromarray(mask)
                return _run_stage1_lazy(
                    lazy_manager, image, mask_pil, pointmap,
                    seed=seed,
                    inference_steps=stage1_inference_steps,
                    cfg_strength=stage1_cfg_strength,
                    unload_after=True,
                    output_dir=output_dir
                )

            # Stage 2 only (SLAT generation) - requires stage1_output
            if slat_only and stage1_output is not None:
                print(f"[Worker] Running Stage 2 (SLAT) with lazy loading", file=sys.stderr)
                from PIL import Image as PILImage
                mask_pil = PILImage.fromarray(mask)
                return _run_stage2_lazy(
                    lazy_manager, image, mask_pil, stage1_output,
                    seed=seed,
                    inference_steps=stage2_inference_steps,
                    cfg_strength=stage2_cfg_strength,
                    unload_after=True,
                    output_dir=output_dir
                )

            # Gaussian decode only - requires slat_output
            if gaussian_only and slat_output is not None:
                print(f"[Worker] Running Gaussian decode with lazy loading", file=sys.stderr)
                slat_data = slat_output.get("slat") if isinstance(slat_output, dict) else slat_output
                if isinstance(slat_data, str):
                    slat_data = pickle.loads(base64.b64decode(slat_data))
                return _run_decode_lazy(
                    lazy_manager, slat_data,
                    decode_format="gaussian",
                    unload_after=True,
                    output_dir=output_dir
                )

            # Mesh decode only - requires slat_output
            if mesh_only and slat_output is not None:
                print(f"[Worker] Running Mesh decode with lazy loading", file=sys.stderr)
                slat_data = slat_output.get("slat") if isinstance(slat_output, dict) else slat_output
                if isinstance(slat_data, str):
                    slat_data = pickle.loads(base64.b64decode(slat_data))
                return _run_decode_lazy(
                    lazy_manager, slat_data,
                    decode_format="mesh",
                    unload_after=True,
                    output_dir=output_dir,
                    simplify=simplify
                )

            # If we reach here, lazy loading was requested but prerequisites not met
            print(f"[Worker] Lazy loading requested but falling back to full pipeline", file=sys.stderr)
            print(f"[Worker] stage1_only={stage1_only}, slat_only={slat_only}, gaussian_only={gaussian_only}, mesh_only={mesh_only}", file=sys.stderr)
            print(f"[Worker] pointmap={pointmap is not None}, stage1_output={stage1_output is not None}, slat_output={slat_output is not None}", file=sys.stderr)

        # =====================================================================
        # END LAZY LOADING BRANCHES
        # =====================================================================

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
            elif request.get("command") == "texture_bake_direct":
                response = run_texture_bake_direct(request)
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
