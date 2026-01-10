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


# HuggingFace repo for SAM3D checkpoints
REPO_ID = "jetjodh/sam-3d-objects"

# Output names for detecting which are connected
OUTPUT_NAMES = ("depth_model", "generator", "slat_decoder_gs", "slat_decoder_mesh")

# Map outputs to required checkpoint files
REQUIRED_FILES = {
    "depth_model": [],  # Depth uses separate moge2 model from HuggingFace
    "generator": [
        "ss_generator.ckpt",
        "ss_generator.yaml",
        "ss_decoder.ckpt",
        "ss_decoder.yaml",
        "slat_generator.ckpt",
        "slat_generator.yaml",
    ],
    "slat_decoder_gs": [
        "slat_decoder_gs.ckpt",
        "slat_decoder_gs.yaml",
        "slat_decoder_gs_4.ckpt",
        "slat_decoder_gs_4.yaml",
    ],
    "slat_decoder_mesh": [
        "slat_decoder_mesh.ckpt",
        "slat_decoder_mesh.yaml",
    ],
}

# Expected file sizes for verification (within 10% tolerance)
EXPECTED_SIZES = {
    "ss_generator.ckpt": 6_690_000_000,   # ~6.69GB
    "slat_generator.ckpt": 4_910_000_000,  # ~4.91GB
    "ss_decoder.ckpt": 147_600_000,        # ~141MB
    "slat_decoder_gs.ckpt": 171_000_000,   # ~171MB
    "slat_decoder_gs_4.ckpt": 170_000_000, # ~170MB
    "slat_decoder_mesh.ckpt": 364_000_000, # ~364MB
}

# Config files to always download (tiny, needed for all operations)
ALWAYS_DOWNLOAD = [
    "pipeline.yaml",
]


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
                "depth_backend": (["moge2", "moge"], {
                    "default": "moge2",
                    "tooltip": "Depth model backend: moge2 (newer, metric scale) or moge (original)"
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
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

    def load_model(
        self,
        compile: bool,
        use_gpu_cache: bool,
        dtype: str = "bfloat16",
        depth_backend: str = "moge2",
        unique_id: str = None,
        prompt: dict = None,
    ):
        """
        Load the SAM3D model.

        Args:
            compile: Whether to compile the model
            use_gpu_cache: Keep models on GPU between stages (higher VRAM, faster)
            dtype: Model precision (bfloat16/float16/float32/auto)
            depth_backend: Depth model backend (moge2/moge)
            unique_id: Node ID (hidden input for graph analysis)
            prompt: Workflow prompt (hidden input for graph analysis)

        Returns:
            4 model outputs (all point to same model wrapper, selective loading handled by worker)
        """
        print(f"[SAM3DObjects] Loading SAM3D model...")

        # Detect which outputs are connected downstream
        used_outputs = self._detect_used_outputs(prompt, unique_id)
        if used_outputs:
            print(f"[SAM3DObjects] Connected outputs: {', '.join(used_outputs)}")
        else:
            print(f"[SAM3DObjects] Could not detect connections, will download all models")

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

        # Get checkpoint path, downloading required files if needed
        # (Do this BEFORE cache check so new outputs trigger downloads)
        checkpoint_path = self._get_or_download_checkpoint(used_outputs)

        # Create cache key
        cache_key = f"{compile}_{use_gpu_cache}_{depth_backend}"

        # Return cached model if available
        if cache_key in _MODEL_CACHE:
            print(f"[SAM3DObjects] Using cached model")
            model = _MODEL_CACHE[cache_key]
            # Return same model 4 times (one for each output)
            return (model, model, model, model)

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
                use_gpu_cache=use_gpu_cache,
                depth_backend=depth_backend,
            )

        except Exception as e:
            raise RuntimeError(f"Failed to create isolated model wrapper: {e}") from e

        # Cache the model wrapper
        _MODEL_CACHE[cache_key] = inference_pipeline
        print(f"[SAM3DObjects] Model loaded successfully")

        # Return same model 4 times (one for each output)
        return (inference_pipeline, inference_pipeline, inference_pipeline, inference_pipeline)

    def _detect_used_outputs(self, prompt: dict, unique_id: str) -> set:
        """
        Analyze the workflow to find which outputs are connected downstream.

        Args:
            prompt: Workflow prompt dictionary
            unique_id: This node's ID

        Returns:
            Set of output names that are connected (e.g., {"depth_model", "generator"})
        """
        if not prompt or not unique_id:
            return set()  # Can't detect, return empty (will download all)

        used = set()

        for node_id, node_info in prompt.items():
            if not isinstance(node_info, dict):
                continue

            for input_name, input_value in node_info.get("inputs", {}).items():
                # Check if this input links to one of our outputs
                # Links are represented as [node_id, output_index]
                if isinstance(input_value, list) and len(input_value) == 2:
                    linked_node_id, output_index = input_value
                    if str(linked_node_id) == str(unique_id) and output_index < len(OUTPUT_NAMES):
                        used.add(OUTPUT_NAMES[output_index])

        return used

    @classmethod
    def _get_or_download_checkpoint(cls, used_outputs: set = None) -> Path:
        """
        Get checkpoint path, downloading required files if necessary.

        Args:
            used_outputs: Set of output names that are connected (for selective download)

        Returns:
            Path to checkpoint directory
        """
        models_dir = get_sam3d_models_path()
        checkpoint_dir = models_dir / "sam-3d-objects"
        checkpoints_path = checkpoint_dir / "checkpoints"

        # Determine which files are required
        required_files = set(ALWAYS_DOWNLOAD)
        if used_outputs:
            for output in used_outputs:
                required_files.update(REQUIRED_FILES.get(output, []))
        else:
            # No detection available - download everything
            for files in REQUIRED_FILES.values():
                required_files.update(files)

        # Check which files are missing or invalid
        missing_files = []
        for filename in required_files:
            filepath = checkpoints_path / filename
            if not cls._verify_checkpoint(filepath, filename):
                missing_files.append(filename)

        if missing_files:
            print(f"[SAM3DObjects] Need to download {len(missing_files)} file(s)...")
            cls._download_files(checkpoint_dir, missing_files)

            # Verify downloads succeeded
            still_missing = []
            for filename in missing_files:
                filepath = checkpoints_path / filename
                if not cls._verify_checkpoint(filepath, filename):
                    still_missing.append(filename)

            if still_missing:
                raise RuntimeError(
                    f"Download verification failed for: {', '.join(still_missing)}\n"
                    "Please check your internet connection and try again."
                )
        else:
            print(f"[SAM3DObjects] All required checkpoints present")

        return checkpoint_dir

    @classmethod
    def _verify_checkpoint(cls, filepath: Path, filename: str) -> bool:
        """
        Verify a checkpoint file exists and has expected size.

        Args:
            filepath: Path to the file
            filename: Filename for size lookup

        Returns:
            True if file exists and size is valid
        """
        if not filepath.exists():
            return False

        # YAML config files just need to exist and not be empty
        if filename.endswith('.yaml'):
            return filepath.stat().st_size > 0

        # For checkpoint files, verify size is within 10% of expected
        expected = EXPECTED_SIZES.get(filename)
        if expected:
            actual = filepath.stat().st_size
            tolerance = expected * 0.1
            if abs(actual - expected) > tolerance:
                print(f"[SAM3DObjects] Warning: {filename} size mismatch (got {actual}, expected ~{expected})")
                return False

        return True

    @classmethod
    def _download_files(cls, target_dir: Path, files: list):
        """
        Download specific files from HuggingFace.

        Args:
            target_dir: Target directory for download
            files: List of filenames to download (relative to checkpoints/)
        """
        target_dir.mkdir(parents=True, exist_ok=True)

        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "huggingface_hub is required for downloading checkpoints. "
                "Please install it: pip install huggingface-hub"
            )

        print(f"[SAM3DObjects] Downloading from HuggingFace: {REPO_ID}")

        for filename in files:
            hf_path = f"checkpoints/{filename}"
            print(f"[SAM3DObjects] Downloading {filename}...")

            try:
                hf_hub_download(
                    repo_id=REPO_ID,
                    filename=hf_path,
                    local_dir=str(target_dir),
                    local_dir_use_symlinks=False,
                )
            except Exception as e:
                raise RuntimeError(f"Failed to download {filename}: {e}") from e

        print(f"[SAM3DObjects] Download complete")
