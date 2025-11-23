"""LoadSAM3DModel node for loading SAM 3D Objects inference pipeline."""

import os
import torch
from pathlib import Path
from typing import Any

from .utils import (
    _MODEL_CACHE,
    get_sam3d_models_path,
    get_device,
)


class LoadSAM3DModel:
    """
    Load SAM 3D Objects model for generating 3D objects from images.

    This node loads the inference pipeline and downloads checkpoints if needed.
    Models are cached globally to avoid reloading.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_tag": (["hf"], {
                    "default": "hf",
                    "tooltip": "Model variant to load (hf = facebook/sam-3d-objects from HuggingFace)"
                }),
                "compile": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable PyTorch model compilation for faster inference (requires more VRAM)"
                }),
                "use_cache": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Offload models to CPU after use to reduce VRAM (slower but ~50% less VRAM)"
                }),
                "force_reload": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Force reload model even if already cached in memory"
                }),
            },
            "optional": {
                "hf_token": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "HuggingFace token for private/gated repositories (leave empty for public models)"
                }),
                "dtype": (["bfloat16", "float16", "float32", "auto"], {
                    "default": "bfloat16",
                    "tooltip": "Model precision: bfloat16 (RTX 30xx+, fastest), float16 (older GPUs), float32 (slowest, most compatible), auto (detect based on GPU)"
                }),
                "keep_model_loaded": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep model loaded in GPU memory between inferences. Faster but uses VRAM. Disable to free memory after each inference."
                }),
            }
        }

    RETURN_TYPES = ("SAM3D_MODEL", "SAM3D_MODEL", "SAM3D_MODEL", "SAM3D_MODEL", "SAM3D_MODEL")
    RETURN_NAMES = ("ss_generator", "slat_generator", "slat_decoder_gs", "slat_decoder_mesh", "embedders")
    OUTPUT_TOOLTIPS = (
        "Sparse structure generator (Stage 1)",
        "SLAT generator (Stage 2)",
        "Gaussian decoder (Stage 3)",
        "Mesh decoder (Stage 3)",
        "Condition embedders (shared across stages)"
    )
    FUNCTION = "load_model"
    CATEGORY = "SAM3DObjects"
    DESCRIPTION = "Load SAM 3D Objects model for generating 3D objects from images."

    def load_model(self, model_tag: str, compile: bool, use_cache: bool, force_reload: bool, hf_token: str = "", dtype: str = "bfloat16", keep_model_loaded: bool = True):
        """
        Load the SAM3D model.

        Args:
            model_tag: Model variant to load
            compile: Whether to compile the model
            use_cache: Offload models to CPU after use for VRAM savings
            force_reload: Force reload even if cached
            hf_token: HuggingFace token for private/gated repos (optional)
            dtype: Model precision (bfloat16/float16/float32/auto)
            keep_model_loaded: Keep model in GPU memory between inferences

        Returns:
            5 model outputs (all point to same model wrapper, selective loading handled by worker)
        """
        print(f"[SAM3DObjects] Loading SAM3D model (tag: {model_tag}, compile: {compile}, use_cache: {use_cache})")

        # Check CUDA availability
        device = get_device()
        if device.type == "cpu":
            print("[SAM3DObjects] WARNING: CUDA not available, running on CPU will be extremely slow!")
        else:
            gpu_props = torch.cuda.get_device_properties(0)
            vram_gb = gpu_props.total_memory / (1024**3)
            print(f"[SAM3DObjects] Using GPU: {gpu_props.name} ({vram_gb:.1f} GB VRAM)")

            if vram_gb < 32:
                print(
                    f"[SAM3DObjects] WARNING: GPU has {vram_gb:.1f} GB VRAM. "
                    "SAM3D officially requires 32GB+ VRAM. May run out of memory!"
                )

        # Create cache key
        cache_key = f"{model_tag}_{compile}_{use_cache}"

        # Return cached model if available and not forcing reload
        if not force_reload and cache_key in _MODEL_CACHE:
            print(f"[SAM3DObjects] Using cached model: {cache_key}")
            model = _MODEL_CACHE[cache_key]
            # Return same model 5 times (one for each output)
            return (model, model, model, model, model)

        # Get checkpoint path
        checkpoint_path = self._get_or_download_checkpoint(model_tag, hf_token)

        # Get config path
        config_path = checkpoint_path / "checkpoints" / "pipeline.yaml"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {config_path}\n"
                "Please ensure the checkpoint contains checkpoints/pipeline.yaml"
            )

        print(f"[SAM3DObjects] Creating isolated model wrapper for config: {config_path}")

        # Import isolated model wrapper
        try:
            from .isolated_model import IsolatedSAM3DModel
        except ImportError as e:
            raise ImportError(
                f"Failed to import IsolatedSAM3DModel: {e}\n"
                "Please ensure the isolated environment is set up:\n"
                "  python install.py"
            ) from e

        # Create isolated model wrapper
        # This doesn't actually load the model yet - that happens in the subprocess
        try:
            inference_pipeline = IsolatedSAM3DModel(
                str(config_path),
                compile=compile,
                use_cache=use_cache
            )
            print("[SAM3DObjects] Isolated model wrapper created successfully!")
            print(f"[SAM3DObjects] Note: dtype={dtype}, keep_model_loaded={keep_model_loaded} parameters not yet implemented for isolated mode")
            print("[SAM3DObjects] Inference will run in isolated subprocess")
            if use_cache:
                print("[SAM3DObjects] use_cache=True: Models will be offloaded to CPU after each stage (~50% VRAM reduction)")

        except Exception as e:
            raise RuntimeError(f"Failed to create isolated model wrapper: {e}") from e

        # Cache the model wrapper
        _MODEL_CACHE[cache_key] = inference_pipeline
        print(f"[SAM3DObjects] Model cached as: {cache_key}")

        # Return same model 5 times (one for each output)
        return (inference_pipeline, inference_pipeline, inference_pipeline, inference_pipeline, inference_pipeline)

    @classmethod
    def _get_or_download_checkpoint(cls, model_tag: str, hf_token: str = "") -> Path:
        """
        Get checkpoint path, downloading if necessary.

        Args:
            model_tag: Model variant tag
            hf_token: HuggingFace token for authentication (optional)

        Returns:
            Path to checkpoint directory
        """
        models_dir = get_sam3d_models_path()
        checkpoint_dir = models_dir / model_tag

        # Check if checkpoint already exists
        if checkpoint_dir.exists() and (checkpoint_dir / "checkpoints" / "pipeline.yaml").exists():
            print(f"[SAM3DObjects] Found existing checkpoint at: {checkpoint_dir}")
            return checkpoint_dir

        # Download checkpoint
        print(f"[SAM3DObjects] Checkpoint not found. Downloading model '{model_tag}'...")
        print(f"[SAM3DObjects] Download location: {checkpoint_dir}")

        try:
            cls._download_checkpoint(model_tag, checkpoint_dir, hf_token)
        except Exception as e:
            raise RuntimeError(
                f"Failed to download checkpoint: {e}\n"
                "Please check your internet connection and try again."
            ) from e

        # Verify download
        if not (checkpoint_dir / "checkpoints" / "pipeline.yaml").exists():
            raise RuntimeError(
                f"Download completed but checkpoints/pipeline.yaml not found in {checkpoint_dir}"
            )

        print("[SAM3DObjects] Checkpoint downloaded successfully!")
        return checkpoint_dir

    @classmethod
    def _download_checkpoint(cls, model_tag: str, target_dir: Path, hf_token: str = ""):
        """
        Download checkpoint from HuggingFace.

        Args:
            model_tag: Model variant tag
            target_dir: Target directory for download
            hf_token: HuggingFace token for authentication (optional)
        """
        target_dir.mkdir(parents=True, exist_ok=True)

        # Map model tags to HuggingFace repo IDs
        repo_mapping = {
            "hf": "facebook/sam-3d-objects",
        }

        if model_tag not in repo_mapping:
            raise ValueError(f"Unknown model tag: {model_tag}")

        repo_id = repo_mapping[model_tag]

        try:
            from huggingface_hub import snapshot_download

            print(f"[SAM3DObjects] Downloading from HuggingFace: {repo_id}")
            print("[SAM3DObjects] This may take a while (several GB)...")

            # Download all files from the repo
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(target_dir),
                local_dir_use_symlinks=False,
                token=hf_token or None,  # Use token if provided, None otherwise
            )

        except ImportError:
            raise ImportError(
                "huggingface_hub is required for downloading checkpoints. "
                "Please install it: pip install huggingface-hub"
            )
        except Exception as e:
            # Clean up partial download
            import shutil
            if target_dir.exists():
                shutil.rmtree(target_dir)
            raise RuntimeError(f"Download failed: {e}") from e
