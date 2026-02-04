"""LoadSAM3DModel node - downloads checkpoints and passes config paths."""

import os
from pathlib import Path

try:
    from .comfy_utils import get_sam3d_models_path
except ImportError:
    from comfy_utils import get_sam3d_models_path


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
    "ss_generator.ckpt": 6_690_000_000,
    "slat_generator.ckpt": 4_910_000_000,
    "ss_decoder.ckpt": 147_600_000,
    "slat_decoder_gs.ckpt": 171_000_000,
    "slat_decoder_gs_4.ckpt": 170_000_000,
    "slat_decoder_mesh.ckpt": 364_000_000,
}

# Config files to always download
ALWAYS_DOWNLOAD = ["pipeline.yaml"]


class LoadSAM3DModel:
    """
    Load SAM 3D Objects model configuration.

    Downloads checkpoints if needed and passes config paths to downstream nodes.
    Actual model loading happens in isolated subprocesses.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "compile": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable PyTorch model compilation for faster inference"
                }),
                "use_gpu_cache": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep models on GPU between stages for faster inference"
                }),
            },
            "optional": {
                "depth_backend": (["moge2", "moge"], {
                    "default": "moge2",
                    "tooltip": "Depth model: moge2 (newer, metric scale) or moge (original)"
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
        "Depth estimation model (MoGe) - use with SAM3DDepthEstimate",
        "SLAT generator (Stage 1 + 2) - use with SAM3DGenerateSLAT",
        "Gaussian decoder - use with SAM3DGaussianDecode",
        "Mesh decoder - use with SAM3DMeshDecode",
    )
    FUNCTION = "load_model"
    CATEGORY = "SAM3DObjects"
    DESCRIPTION = "Load SAM 3D Objects model configuration. Downloads checkpoints if needed."

    def load_model(
        self,
        compile: bool,
        use_gpu_cache: bool,
        depth_backend: str = "moge2",
        unique_id: str = None,
        prompt: dict = None,
    ):
        print(f"[SAM3DObjects] Loading SAM3D model...")

        # Detect which outputs are connected
        used_outputs = self._detect_used_outputs(prompt, unique_id)
        if used_outputs:
            print(f"[SAM3DObjects] Connected outputs: {', '.join(used_outputs)}")

        # Download checkpoints if needed
        checkpoint_path = self._get_or_download_checkpoint(used_outputs)
        config_path = str(checkpoint_path / "checkpoints" / "pipeline.yaml")

        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        print(f"[SAM3DObjects] Model loaded successfully")

        # Return simple config dict
        model = {
            "config_path": config_path,
            "depth_backend": depth_backend,
            "compile": compile,
            "use_gpu_cache": use_gpu_cache,
        }
        return (model, model, model, model)

    def _detect_used_outputs(self, prompt: dict, unique_id: str) -> set:
        """Detect which outputs are connected downstream."""
        if not prompt or not unique_id:
            return set()

        used = set()
        for node_id, node_info in prompt.items():
            if not isinstance(node_info, dict):
                continue
            for input_name, input_value in node_info.get("inputs", {}).items():
                if isinstance(input_value, list) and len(input_value) == 2:
                    linked_node_id, output_index = input_value
                    if str(linked_node_id) == str(unique_id) and output_index < len(OUTPUT_NAMES):
                        used.add(OUTPUT_NAMES[output_index])
        return used

    @classmethod
    def _get_or_download_checkpoint(cls, used_outputs: set = None) -> Path:
        """Get checkpoint path, downloading required files if necessary."""
        models_dir = get_sam3d_models_path()
        checkpoint_dir = models_dir / "sam-3d-objects"
        checkpoints_path = checkpoint_dir / "checkpoints"

        # Determine required files
        required_files = set(ALWAYS_DOWNLOAD)
        if used_outputs:
            for output in used_outputs:
                required_files.update(REQUIRED_FILES.get(output, []))
        else:
            for files in REQUIRED_FILES.values():
                required_files.update(files)

        # Check which files are missing
        missing_files = []
        for filename in required_files:
            filepath = checkpoints_path / filename
            if not cls._verify_checkpoint(filepath, filename):
                missing_files.append(filename)

        if missing_files:
            print(f"[SAM3DObjects] Need to download {len(missing_files)} file(s)...")
            cls._download_files(checkpoint_dir, missing_files)
        else:
            print(f"[SAM3DObjects] All required checkpoints present")

        return checkpoint_dir

    @classmethod
    def _verify_checkpoint(cls, filepath: Path, filename: str) -> bool:
        """Verify a checkpoint file exists and has expected size."""
        if not filepath.exists():
            return False

        if filename.endswith('.yaml'):
            return filepath.stat().st_size > 0

        expected = EXPECTED_SIZES.get(filename)
        if expected:
            actual = filepath.stat().st_size
            if abs(actual - expected) > expected * 0.1:
                return False

        return True

    @classmethod
    def _download_files(cls, target_dir: Path, files: list):
        """Download specific files from HuggingFace."""
        target_dir.mkdir(parents=True, exist_ok=True)

        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError("huggingface_hub required: pip install huggingface-hub")

        print(f"[SAM3DObjects] Downloading from HuggingFace: {REPO_ID}")

        for filename in files:
            hf_path = f"checkpoints/{filename}"
            print(f"[SAM3DObjects] Downloading {filename}...")
            hf_hub_download(
                repo_id=REPO_ID,
                filename=hf_path,
                local_dir=str(target_dir),
                local_dir_use_symlinks=False,
            )

        print(f"[SAM3DObjects] Download complete")
