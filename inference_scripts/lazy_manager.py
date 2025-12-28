"""
Lazy model loading manager for SAM3D.

Loads models only when needed and unloads them after use when use_gpu_cache=False.
This allows running on GPUs with less than 32GB VRAM.
"""

import sys
import os
import platform
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
import torch


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
        self._full_pipeline = None

        # Setup environment
        self._setup_environment()

        # Parse config to get model paths
        self._parse_config()

    def _setup_environment(self):
        """Setup environment variables and paths."""
        # Skip sam3d_objects initialization BEFORE any imports
        os.environ['LIDRA_SKIP_INIT'] = '1'

        # Add vendor directory to path
        vendor_path = Path(__file__).parent.parent / "vendor"
        if str(vendor_path) not in sys.path:
            sys.path.insert(0, str(vendor_path))
            print(f"[LazyManager] Added vendor path: {vendor_path}", file=sys.stderr)

        # Preload modules to avoid circular import issues
        try:
            import sam3d_objects.model.backbone.tdfy_dit.modules.sparse
            import sam3d_objects.model.backbone.tdfy_dit.modules.attention
            import sam3d_objects.model.backbone.tdfy_dit.renderers.sh_utils
            import sam3d_objects.model.backbone.tdfy_dit.representations.gaussian.gaussian_model
            import sam3d_objects.model.backbone.tdfy_dit.renderers.gaussian_render
            import sam3d_objects.model.backbone.tdfy_dit.models
            print(f"[LazyManager] Preloaded modules successfully", file=sys.stderr)
        except ImportError as e:
            print(f"[LazyManager] Warning: Could not preload modules: {e}", file=sys.stderr)

        os.environ['PYTHONIOENCODING'] = 'utf-8'
        os.environ['PYTHONUTF8'] = '1'

        # Setup venv bin path
        venv_bin = (Path(__file__).parent.parent / "_env" / "bin").resolve()
        if venv_bin.exists():
            os.environ['PATH'] = f"{venv_bin}{os.pathsep}{os.environ.get('PATH', '')}"

        # Setup model cache directory
        config_dir = self.checkpoint_dir.parent
        models_cache_dir = config_dir / "_models_cache"
        models_cache_dir.mkdir(parents=True, exist_ok=True)

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

    def load_depth_model(self, backend: str = "moge2"):
        """
        Load only the depth model (MoGe).

        Args:
            backend: "moge2" (newer, metric scale) or "moge" (original v1)
        """
        # Check if we already have the right backend loaded
        if self.depth_model is not None:
            current_backend = getattr(self, '_depth_backend', 'moge2')
            if current_backend == backend:
                return self.depth_model
            else:
                # Different backend requested, unload current
                print(f"[LazyManager] Switching depth backend from {current_backend} to {backend}", file=sys.stderr)
                self.unload_depth_model()

        print(f"[LazyManager] Loading depth model (backend={backend})...", file=sys.stderr)

        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        # Build depth model config based on backend selection
        if backend == "moge2":
            # MoGe v2 - newer model with metric scale
            depth_config = OmegaConf.create({
                "_target_": "sam3d_objects.pipeline.depth_models.moge.MoGe",
                "model": {
                    "_target_": "moge.model.v2.MoGeModel.from_pretrained",
                    "pretrained_model_name_or_path": "Ruicheng/moge-2-vitl"
                }
            })
        else:
            # MoGe v1 - original model
            depth_config = OmegaConf.create({
                "_target_": "sam3d_objects.pipeline.depth_models.moge.MoGe",
                "model": {
                    "_target_": "moge.model.v1.MoGeModel.from_pretrained",
                    "pretrained_model_name_or_path": "Ruicheng/moge-vitl"
                }
            })

        self.depth_model = instantiate(depth_config)
        if hasattr(self.depth_model, 'cuda'):
            self.depth_model = self.depth_model.cuda()

        self._depth_backend = backend
        print(f"[LazyManager] Depth model loaded ({backend})", file=sys.stderr)

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

        if embedder_type == 'ss':
            config_path = self.model_paths['ss_generator']['config']
            ckpt_path = self.model_paths['ss_generator']['ckpt']
        else:
            config_path = self.model_paths['slat_generator']['config']
            ckpt_path = self.model_paths['slat_generator']['ckpt']

        gen_config = OmegaConf.load(config_path)
        embedder_config = gen_config.module.condition_embedder.backbone

        embedder = instantiate(embedder_config)

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

        config = OmegaConf.load(paths['config'])

        is_generator = model_name in ['ss_generator', 'slat_generator']

        if is_generator:
            config = config["module"]["generator"]["backbone"]
            state_dict_key = "state_dict"
            state_dict_fn = filter_and_remove_prefix_state_dict_fn("_base_models.generator.")
        else:
            state_dict_key = None
            state_dict_fn = remove_prefix_state_dict_fn("module.")

        model = instantiate(config)

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

        self.unload_depth_model()

        for name in list(self.loaded_models.keys()):
            self.unload_model(name)

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
        """Get the full pipeline (loads everything)."""
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
    """Load the SAM3D model (full pipeline)."""
    global _MODEL, _CURRENT_CONFIG

    config_key = f"{config_path}_{compile}"

    if _MODEL is not None and _CURRENT_CONFIG == config_key:
        return _MODEL

    print(f"[Worker] Loading model from {config_path}", file=sys.stderr)

    # Add vendor directory to path
    vendor_path = Path(__file__).parent.parent / "vendor"
    if str(vendor_path) not in sys.path:
        sys.path.insert(0, str(vendor_path))
        print(f"[Worker] Added vendor path: {vendor_path}", file=sys.stderr)

    os.environ['LIDRA_SKIP_INIT'] = '1'
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONUTF8'] = '1'

    # Setup venv paths
    venv_bin = (Path(__file__).parent.parent / "_env" / "bin").resolve()
    if venv_bin.exists():
        os.environ['PATH'] = f"{venv_bin}{os.pathsep}{os.environ.get('PATH', '')}"

    # Setup compiler for CUDA JIT
    _setup_compiler(venv_bin)

    # Setup CUDA_HOME
    _setup_cuda_home(Path(__file__).parent.parent / "_env")

    # Setup model cache
    config_dir = Path(config_path).parent.parent
    models_cache_dir = config_dir / "_models_cache"
    models_cache_dir.mkdir(exist_ok=True)

    os.environ['TORCH_HOME'] = str(models_cache_dir / "torch")
    os.environ['HF_HOME'] = str(models_cache_dir / "huggingface")
    os.environ['TRANSFORMERS_CACHE'] = str(models_cache_dir / "transformers")

    from omegaconf import OmegaConf
    from hydra.utils import instantiate

    config = OmegaConf.load(config_path)
    config.compile_model = compile
    config.workspace_dir = os.path.dirname(config_path)

    _MODEL = instantiate(config)
    _CURRENT_CONFIG = config_key

    print(f"[Worker] Model loaded successfully", file=sys.stderr)
    return _MODEL


def _setup_compiler(compiler_bin: Path):
    """Setup compiler symlinks for CUDA JIT compilation."""
    if not compiler_bin.exists():
        return

    os.environ['PATH'] = f"{compiler_bin}{os.pathsep}{os.environ['PATH']}"

    system = platform.system()

    if system == "Linux":
        wrapper_gxx = compiler_bin / "x86_64-conda-linux-gnu-g++"
        wrapper_gcc = compiler_bin / "x86_64-conda-linux-gnu-gcc"
        target_gxx = compiler_bin / "g++"
        target_gcc = compiler_bin / "gcc"
        cxx_name, cc_name = "g++", "gcc"
        use_symlink = True
    elif system == "Windows":
        cl_exe = compiler_bin / "cl.exe"
        if cl_exe.exists():
            os.environ['CXX'] = str(cl_exe)
            os.environ['CC'] = str(cl_exe)
            return
        wrapper_gxx = compiler_bin / "x86_64-w64-mingw32-g++.exe"
        wrapper_gcc = compiler_bin / "x86_64-w64-mingw32-gcc.exe"
        target_gxx = compiler_bin / "g++.exe"
        target_gcc = compiler_bin / "gcc.exe"
        cxx_name, cc_name = "g++.exe", "gcc.exe"
        use_symlink = False
    elif system == "Darwin":
        darwin_gxx = list(compiler_bin.glob("*-apple-darwin*-clang++"))
        darwin_gcc = list(compiler_bin.glob("*-apple-darwin*-clang"))
        if darwin_gxx and darwin_gcc:
            wrapper_gxx = darwin_gxx[0]
            wrapper_gcc = [w for w in darwin_gcc if not w.name.endswith("++")][0]
            target_gxx = compiler_bin / "clang++"
            target_gcc = compiler_bin / "clang"
            cxx_name, cc_name = "clang++", "clang"
            use_symlink = True
        else:
            return
    else:
        return

    # Create symlinks/copies
    for wrapper, target in [(wrapper_gxx, target_gxx), (wrapper_gcc, target_gcc)]:
        if wrapper.exists() and not target.exists():
            try:
                if use_symlink:
                    target.symlink_to(wrapper.name)
                else:
                    shutil.copy2(wrapper, target)
            except (FileExistsError, OSError):
                pass

    os.environ['CXX'] = cxx_name
    os.environ['CC'] = cc_name


def _setup_cuda_home(venv_root: Path):
    """Setup CUDA_HOME for JIT compilation."""
    cuda_home = None

    # Option 1: conda-forge CUDA
    conda_cuda = venv_root / "cuda"
    if (conda_cuda / "bin" / "nvcc").exists():
        cuda_home = conda_cuda

    # Option 2: PyPI CUDA
    if not cuda_home:
        try:
            nvcc_pattern = "nvcc.exe" if os.name == "nt" else "nvcc"
            nvcc_paths = list(venv_root.glob(f"**/{nvcc_pattern}"))
            if nvcc_paths:
                nvcc_path = nvcc_paths[0]
                if nvcc_path.parent.name == "bin":
                    cuda_home = nvcc_path.parent.parent
        except Exception:
            pass

    if cuda_home:
        os.environ['CUDA_HOME'] = str(cuda_home)
        cuda_bin = cuda_home / "bin"
        if cuda_bin.exists():
            os.environ['PATH'] = f"{cuda_bin}{os.pathsep}{os.environ.get('PATH', '')}"
