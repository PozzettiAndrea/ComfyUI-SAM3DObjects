"""LoadSAM3DModel node - downloads checkpoints and passes config paths."""

import logging
import os
from pathlib import Path

import torch
import comfy.model_management as mm
from comfy_api.latest import io

log = logging.getLogger("sam3dobjects")

try:
    from .comfy_utils import get_sam3d_models_path
except ImportError:
    from comfy_utils import get_sam3d_models_path


# HuggingFace repo for SAM3D checkpoints (safetensors format, mmap-friendly)
REPO_ID = "apozz/sam-3d-objects-safetensors"

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


class LoadSAM3DModel(io.ComfyNode):
    """
    Load SAM 3D Objects model configuration.

    Downloads checkpoints if needed and passes config paths to downstream nodes.
    Actual model loading happens in isolated subprocesses.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoadSAM3DModel",
            display_name="(Down)Load SAM3D Model",
            category="SAM3DObjects",
            description="Load SAM 3D Objects model configuration. Downloads checkpoints if needed.",
            inputs=[
                io.Combo.Input("attn_backend", options=["auto", "flash_attn", "sdpa", "xformers", "torch_flash_attn"],
                    default="auto",
                    tooltip="Deprecated - attention is now auto-detected by ComfyUI. This setting has no effect."
                ),
                io.Boolean.Input("compile",
                    default=False,
                    tooltip="Enable PyTorch model compilation for faster inference"
                ),
                io.Combo.Input("precision", options=["auto", "bf16", "fp16", "fp32"],
                    default="auto",
                    tooltip="Model precision. auto: bf16 on Ampere+, fp16 on Volta/Turing, fp32 on older."
                ),
            ],
            outputs=[
                io.Custom("SAM3D_MODEL").Output(display_name="depth_model", tooltip="Depth estimation model (MoGe) - use with SAM3DDepthEstimate"),
                io.Custom("SAM3D_MODEL").Output(display_name="generator", tooltip="SLAT generator (Stage 1 + 2) - use with SAM3DGenerateSLAT"),
                io.Custom("SAM3D_MODEL").Output(display_name="slat_decoder_gs", tooltip="Gaussian decoder - use with SAM3DGaussianDecode"),
                io.Custom("SAM3D_MODEL").Output(display_name="slat_decoder_mesh", tooltip="Mesh decoder - use with SAM3DMeshDecode"),
            ],
        )

    @classmethod
    @torch.no_grad()
    def execute(
        cls,
        attn_backend: str,
        compile: bool,
        precision: str = "auto",
        **kwargs,
    ):
        log.info("Loading SAM3D model...")

        # Resolve precision "auto" using GPU capabilities
        # Prefer bf16 (better dynamic range, native on Ampere+), fall back to fp16/fp32.
        if precision == "auto":
            device = mm.get_torch_device()
            if mm.should_use_bf16(device):
                precision = "bf16"
            elif mm.should_use_fp16(device):
                precision = "fp16"
            else:
                precision = "fp32"
        log.info("Precision: %s", precision)

        # Download all checkpoints if needed (always download everything to avoid
        # missing files when outputs are connected/disconnected between runs)
        checkpoint_path = cls._get_or_download_checkpoint()
        config_path = str(checkpoint_path / "pipeline.yaml")

        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Download MoGe / DINOv2 safetensors to ComfyUI/models/sam3d/sam-3d-objects/
        cls._ensure_dinov2_safetensors()
        cls._ensure_moge_safetensors()

        log.info("Model loaded successfully")

        # Return simple config dict
        model = {
            "config_path": config_path,
            "compile": compile,
            "precision": precision,
        }
        return io.NodeOutput(model, model, model, model)

    @staticmethod
    def _get_or_download_checkpoint() -> Path:
        """Get checkpoint path, downloading all required files if necessary."""
        models_dir = get_sam3d_models_path()

        # Always download all files
        required_files = set(ALWAYS_DOWNLOAD)
        for files in REQUIRED_FILES.values():
            required_files.update(files)

        # Check which files are missing
        missing_files = []
        for filename in required_files:
            filepath = models_dir / filename
            if not LoadSAM3DModel._verify_checkpoint(filepath, filename):
                missing_files.append(filename)

        if missing_files:
            log.info("Need to download %d file(s)...", len(missing_files))
            LoadSAM3DModel._download_files(models_dir, missing_files)
        else:
            log.info("All required checkpoints present")

        return models_dir

    @staticmethod
    def _verify_checkpoint(filepath: Path, filename: str) -> bool:
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

    @staticmethod
    def _download_files(target_dir: Path, files: list):
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

    @staticmethod
    def _ensure_dinov2_safetensors():
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

    @staticmethod
    def _ensure_moge_safetensors():
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
