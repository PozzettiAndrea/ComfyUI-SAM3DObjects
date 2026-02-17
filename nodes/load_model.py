"""LoadSAM3DModel node - downloads checkpoints and passes config paths."""

import logging
import os
from pathlib import Path

import comfy.model_management as mm

log = logging.getLogger("sam3dobjects")

try:
    from .comfy_utils import get_sam3d_models_path
except ImportError:
    from comfy_utils import get_sam3d_models_path


# HuggingFace repo for SAM3D checkpoints (safetensors format, mmap-friendly)
REPO_ID = "apozz/sam-3d-objects-safetensors"

# Output names for detecting which are connected
OUTPUT_NAMES = ("depth_model", "generator", "slat_decoder_gs", "slat_decoder_mesh")

# Map outputs to required checkpoint files (safetensors preferred)
REQUIRED_FILES = {
    "depth_model": [],  # Depth uses MoGe v1 (Ruicheng/moge-vitl) from HuggingFace
    "generator": [
        "ss_generator.safetensors",
        "ss_generator.yaml",
        "ss_decoder.safetensors",
        "ss_decoder.yaml",
        "slat_generator.safetensors",
        "slat_generator.yaml",
    ],
    "slat_decoder_gs": [
        "slat_decoder_gs.safetensors",
        "slat_decoder_gs.yaml",
        "slat_decoder_gs_4.safetensors",
        "slat_decoder_gs_4.yaml",
    ],
    "slat_decoder_mesh": [
        "slat_decoder_mesh.safetensors",
        "slat_decoder_mesh.yaml",
    ],
}

# Expected file sizes for verification (within 10% tolerance)
EXPECTED_SIZES = {
    "ss_generator.safetensors": 6_690_000_000,
    "slat_generator.safetensors": 4_910_000_000,
    "ss_decoder.safetensors": 147_600_000,
    "slat_decoder_gs.safetensors": 171_000_000,
    "slat_decoder_gs_4.safetensors": 170_000_000,
    "slat_decoder_mesh.safetensors": 364_000_000,
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
                "attn_backend": (["auto", "flash_attn", "sdpa", "xformers", "torch_flash_attn"], {
                    "default": "auto",
                    "tooltip": "Deprecated - attention is now auto-detected by ComfyUI. This setting has no effect."
                }),
                "compile": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable PyTorch model compilation for faster inference"
                }),
                "precision": (["auto", "bf16", "fp16", "fp32"], {
                    "default": "auto",
                    "tooltip": "Model precision. auto: bf16 on Ampere+, fp16 on Volta/Turing, fp32 on older."
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
        attn_backend: str,
        compile: bool,
        precision: str = "auto",
        unique_id: str = None,
        prompt: dict = None,
    ):
        log.info("Loading SAM3D model...")

        # Resolve precision "auto" using GPU capabilities
        # spconv's compiled CUDA kernels don't include bf16 GEMM, so auto picks fp16.
        if precision == "auto":
            device = mm.get_torch_device()
            if mm.should_use_fp16(device) or mm.should_use_bf16(device):
                precision = "fp16"
            else:
                precision = "fp32"
        log.info("Precision: %s", precision)

        # Detect which outputs are connected
        used_outputs = self._detect_used_outputs(prompt, unique_id)
        if used_outputs:
            log.info("Connected outputs: %s", ', '.join(used_outputs))

        # Download checkpoints if needed
        checkpoint_path = self._get_or_download_checkpoint(used_outputs)
        config_path = str(checkpoint_path / "pipeline.yaml")

        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Download MoGe / DINOv2 safetensors to ComfyUI/models/sam3d/sam-3d-objects/
        if "generator" in used_outputs or not used_outputs:
            self._ensure_dinov2_safetensors()

        if "depth_model" in used_outputs or not used_outputs:
            self._ensure_moge_safetensors()

        log.info("Model loaded successfully")

        # Return simple config dict
        model = {
            "config_path": config_path,
            "compile": compile,
            "precision": precision,
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
            filepath = models_dir / filename
            if not cls._verify_checkpoint(filepath, filename):
                missing_files.append(filename)

        if missing_files:
            log.info("Need to download %d file(s)...", len(missing_files))
            cls._download_files(models_dir, missing_files)
        else:
            log.info("All required checkpoints present")

        return models_dir

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

        log.info("Downloading from HuggingFace: %s", REPO_ID)

        for filename in files:
            log.info("Downloading %s...", filename)
            hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                local_dir=str(target_dir),
                local_dir_use_symlinks=False,
            )

        log.info("Download complete")

    @classmethod
    def _ensure_dinov2_safetensors(cls):
        """Download DINOv2 ViT-L/14 register weights (safetensors) to models folder."""
        from huggingface_hub import hf_hub_download

        models_dir = get_sam3d_models_path()
        target = models_dir / "dinov2_vitl14_reg.safetensors"

        if target.exists():
            log.info("DINOv2 safetensors already present")
            return

        log.info("Downloading DINOv2 safetensors...")
        hf_hub_download(
            repo_id=REPO_ID,
            filename="dinov2_vitl14_reg.safetensors",
            local_dir=str(models_dir),
            local_dir_use_symlinks=False,
        )
        log.info("DINOv2 safetensors downloaded")

    @classmethod
    def _ensure_moge_safetensors(cls):
        """Download MoGe ViT-L weights + config (safetensors) to models folder."""
        from huggingface_hub import hf_hub_download

        models_dir = get_sam3d_models_path()
        target = models_dir / "moge_vitl.safetensors"

        if target.exists():
            log.info("MoGe safetensors already present")
            return

        log.info("Downloading MoGe safetensors...")
        for fname in ("moge_vitl.safetensors", "moge_vitl_config.json"):
            hf_hub_download(
                repo_id=REPO_ID,
                filename=fname,
                local_dir=str(models_dir),
                local_dir_use_symlinks=False,
            )
        log.info("MoGe safetensors downloaded")
