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
                "compile": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable PyTorch model compilation for faster inference (requires more VRAM)"
                }),
                "use_gpu_cache": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep models on GPU between stages for faster inference (uses more VRAM)"
                }),
            },
            "optional": {
                "dtype": (["bfloat16", "float16", "float32", "auto"], {
                    "default": "bfloat16",
                    "tooltip": "Model precision: bfloat16 (RTX 30xx+, fastest), float16 (older GPUs), float32 (slowest, most compatible), auto (detect based on GPU)"
                }),
            }
        }

    RETURN_TYPES = ("SAM3D_MODEL", "SAM3D_MODEL", "SAM3D_MODEL", "SAM3D_MODEL")
    RETURN_NAMES = ("depth_model", "generator", "slat_decoder_gs", "slat_decoder_mesh")
    OUTPUT_TOOLTIPS = (
        "Depth estimation model (MoGe) - use with SAM3D_DepthEstimate",
        "SLAT generator (Stage 1 + 2) - use with SAM3DGenerateSLAT",
        "Gaussian decoder (Stage 3)",
        "Mesh decoder (Stage 3)",
    )
    FUNCTION = "load_model"
    CATEGORY = "SAM3DObjects"
    DESCRIPTION = "Load SAM 3D Objects model for generating 3D objects from images."

    def load_model(self, compile: bool, use_gpu_cache: bool, dtype: str = "bfloat16"):
        """
        Load the SAM3D model.

        Args:
            compile: Whether to compile the model
            use_gpu_cache: Keep models on GPU between stages (higher VRAM, faster)
            dtype: Model precision (bfloat16/float16/float32/auto)

        Returns:
            5 model outputs (all point to same model wrapper, selective loading handled by worker)
        """
        print(f"[SAM3DObjects] Loading SAM3D model...")

        # Check CUDA availability
        device = get_device()
        if device.type == "cpu":
            print("[SAM3DObjects] WARNING: CUDA not available, running on CPU will be extremely slow!")
        else:
            gpu_props = torch.cuda.get_device_properties(0)
            vram_gb = gpu_props.total_memory / (1024**3)

            if vram_gb < 32:
                print(
                    f"[SAM3DObjects] WARNING: GPU has {vram_gb:.1f} GB VRAM. "
                    "SAM3D officially requires 32GB+ VRAM. May run out of memory!"
                )

        # Create cache key
        cache_key = f"{compile}_{use_gpu_cache}"

        # Return cached model if available
        if cache_key in _MODEL_CACHE:
            print(f"[SAM3DObjects] Using cached model")
            model = _MODEL_CACHE[cache_key]
            # Return same model 4 times (one for each output)
            return (model, model, model, model)

        # Get checkpoint path
        checkpoint_path = self._get_or_download_checkpoint()

        # Get config path
        config_path = checkpoint_path / "checkpoints" / "pipeline.yaml"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {config_path}\n"
                "Please ensure the checkpoint contains checkpoints/pipeline.yaml"
            )


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
                use_gpu_cache=use_gpu_cache
            )

        except Exception as e:
            raise RuntimeError(f"Failed to create isolated model wrapper: {e}") from e

        # Cache the model wrapper
        _MODEL_CACHE[cache_key] = inference_pipeline
        print(f"[SAM3DObjects] Model loaded successfully")

        # Return same model 4 times (one for each output)
        return (inference_pipeline, inference_pipeline, inference_pipeline, inference_pipeline)

    @classmethod
    def _get_or_download_checkpoint(cls) -> Path:
        """
        Get checkpoint path, downloading if necessary.

        Returns:
            Path to checkpoint directory
        """
        models_dir = get_sam3d_models_path()
        checkpoint_dir = models_dir / "sam-3d-objects"

        # Check if checkpoint already exists
        if checkpoint_dir.exists() and (checkpoint_dir / "checkpoints" / "pipeline.yaml").exists():
            return checkpoint_dir

        # Download checkpoint
        print(f"[SAM3DObjects] Downloading model...")

        try:
            cls._download_checkpoint(checkpoint_dir)
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

        return checkpoint_dir

    @classmethod
    def _download_checkpoint(cls, target_dir: Path):
        """
        Download checkpoint from HuggingFace.

        Args:
            target_dir: Target directory for download
        """
        target_dir.mkdir(parents=True, exist_ok=True)

        repo_id = "jetjodh/sam-3d-objects"

        try:
            from huggingface_hub import snapshot_download

            print(f"[SAM3DObjects] Downloading from HuggingFace: {repo_id} (this may take a while)")

            # Download all files from the repo
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(target_dir),
                local_dir_use_symlinks=False,
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
