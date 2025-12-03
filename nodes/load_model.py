"""LoadSAM3DModel node for loading SAM 3D Objects inference pipeline."""

import os
import torch
from pathlib import Path
from typing import Any
from comfy_api.latest import io

from .utils import (
    _MODEL_CACHE,
    get_sam3d_models_path,
    get_device,
)


class LoadSAM3DModel(io.ComfyNode):
    """
    Load SAM 3D Objects model for generating 3D objects from images.

    This node loads the inference pipeline and downloads checkpoints if needed.
    Models are cached globally to avoid reloading.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LoadSAM3DModel",
            display_name="Load SAM3D Model",
            category="SAM3DObjects",
            inputs=[
                io.Combo.Input(
                    "model_tag",
                    options=["hf"],
                    default="hf",
                    tooltip="Model variant to load. 'hf' is the HuggingFace released model."
                ),
                io.Bool.Input(
                    "compile",
                    default=False,
                    tooltip="Enable torch.compile for faster inference (requires PyTorch 2.0+). First run will be slower."
                ),
                io.Bool.Input(
                    "force_reload",
                    default=False,
                    tooltip="Force reload the model even if cached."
                ),
            ],
            outputs=[
                io.Any.Output(
                    "model",
                    tooltip="SAM3D inference pipeline model. Connect to SAM3DGenerate node."
                ),
            ],
        )

    @classmethod
    def execute(cls, model_tag: str, compile: bool, force_reload: bool) -> io.NodeOutput:
        """
        Load the SAM3D model.

        Args:
            model_tag: Model variant to load
            compile: Whether to compile the model
            force_reload: Force reload even if cached

        Returns:
            Loaded inference pipeline
        """
        print(f"[SAM3DObjects] Loading SAM3D model (tag: {model_tag}, compile: {compile})")

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
        cache_key = f"{model_tag}_{compile}"

        # Return cached model if available and not forcing reload
        if not force_reload and cache_key in _MODEL_CACHE:
            print(f"[SAM3DObjects] Using cached model: {cache_key}")
            return io.NodeOutput(_MODEL_CACHE[cache_key])

        # Get checkpoint path
        checkpoint_path = cls._get_or_download_checkpoint(model_tag)

        # Import Inference class from our vendored copy
        try:
            from .sam3d_inference import Inference
        except ImportError as e:
            raise ImportError(
                f"Failed to import Inference class: {e}\n"
                "Please ensure sam3d_objects package is properly installed:\n"
                "  pip install git+https://github.com/facebookresearch/sam-3d-objects.git"
            ) from e

        # Load model
        config_path = checkpoint_path / "pipeline.yaml"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {config_path}\n"
                "Please ensure the checkpoint contains pipeline.yaml"
            )

        print(f"[SAM3DObjects] Loading model from config: {config_path}")

        try:
            inference_pipeline = Inference(
                str(config_path),
                compile=compile
            )
            print("[SAM3DObjects] Model loaded successfully!")

        except Exception as e:
            raise RuntimeError(f"Failed to load SAM3D model: {e}") from e

        # Cache the model
        _MODEL_CACHE[cache_key] = inference_pipeline
        print(f"[SAM3DObjects] Model cached as: {cache_key}")

        return io.NodeOutput(inference_pipeline)

    @classmethod
    def _get_or_download_checkpoint(cls, model_tag: str) -> Path:
        """
        Get checkpoint path, downloading if necessary.

        Args:
            model_tag: Model variant tag

        Returns:
            Path to checkpoint directory
        """
        models_dir = get_sam3d_models_path()
        checkpoint_dir = models_dir / model_tag

        # Check if checkpoint already exists
        if checkpoint_dir.exists() and (checkpoint_dir / "pipeline.yaml").exists():
            print(f"[SAM3DObjects] Found existing checkpoint at: {checkpoint_dir}")
            return checkpoint_dir

        # Download checkpoint
        print(f"[SAM3DObjects] Checkpoint not found. Downloading model '{model_tag}'...")
        print(f"[SAM3DObjects] Download location: {checkpoint_dir}")

        try:
            cls._download_checkpoint(model_tag, checkpoint_dir)
        except Exception as e:
            raise RuntimeError(
                f"Failed to download checkpoint: {e}\n"
                "Please check your internet connection and try again."
            ) from e

        # Verify download
        if not (checkpoint_dir / "pipeline.yaml").exists():
            raise RuntimeError(
                f"Download completed but pipeline.yaml not found in {checkpoint_dir}"
            )

        print("[SAM3DObjects] Checkpoint downloaded successfully!")
        return checkpoint_dir

    @classmethod
    def _download_checkpoint(cls, model_tag: str, target_dir: Path):
        """
        Download checkpoint from HuggingFace.

        Args:
            model_tag: Model variant tag
            target_dir: Target directory for download
        """
        target_dir.mkdir(parents=True, exist_ok=True)

        # Map model tags to HuggingFace repo IDs
        repo_mapping = {
            "hf": "bennyguo/sam-3d-objects-hf",  # This is a placeholder - update with actual repo
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
