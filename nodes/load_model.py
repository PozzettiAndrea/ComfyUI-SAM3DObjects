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
    "depth_model": [],  # Depth uses MoGe v1 (Ruicheng/moge-vitl) from HuggingFace
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
                "attn_backend": (["flash_attn", "sdpa", "xformers", "torch_flash_attn"], {
                    "default": "flash_attn",
                    "tooltip": "Attention backend (flash_attn recommended for A100/H100/H200)"
                }),
                "compile": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable PyTorch model compilation for faster inference"
                }),
                "memory": (["cache_gpu", "cpu_offload", "delete"], {
                    "default": "cpu_offload",
                    "tooltip": "Model memory strategy: cache_gpu = keep on GPU between runs, cpu_offload = move to CPU RAM after use, delete = free after use"
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
        memory: str,
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

        # Download models to ComfyUI/models/sam3d/ (subprocesses may have SSL issues)
        if "generator" in used_outputs or not used_outputs:
            self._ensure_dinov2_downloaded()

        if "depth_model" in used_outputs or not used_outputs:
            self._ensure_moge_downloaded()

        print(f"[SAM3DObjects] Model loaded successfully")

        # Return simple config dict
        model = {
            "config_path": config_path,
            "compile": compile,
            "memory": memory,
            "attn_backend": attn_backend,
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

    @classmethod
    def _ensure_dinov2_downloaded(cls):
        """Download DINOv2 repo and weights to ComfyUI models folder.

        Downloads both the code repo and the pretrained weights.
        This avoids SSL issues in isolated subprocesses.
        """
        import subprocess

        models_dir = get_sam3d_models_path()
        dinov2_dir = models_dir / "sam-3d-objects" / "dinov2"
        weights_file = dinov2_dir / "dinov2_vitl14_reg4_pretrain.pth"

        # Download repo if not present
        if not (dinov2_dir / "hubconf.py").exists():
            print(f"[SAM3DObjects] Downloading DINOv2 repo to {dinov2_dir}...")
            try:
                dinov2_dir.mkdir(parents=True, exist_ok=True)
                result = subprocess.run(
                    ["git", "clone", "--depth", "1", "https://github.com/facebookresearch/dinov2.git", str(dinov2_dir)],
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    raise RuntimeError(f"git clone failed: {result.stderr}")
                print(f"[SAM3DObjects] DINOv2 repo downloaded")
            except Exception as e:
                print(f"[SAM3DObjects] Warning: Failed to download DINOv2 repo: {e}")
                return
        else:
            print(f"[SAM3DObjects] DINOv2 repo already present")

        # Download weights if not present (use wget to avoid SSL issues in isolated subprocess)
        if not weights_file.exists():
            print(f"[SAM3DObjects] Downloading DINOv2 weights...")
            try:
                weights_url = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth"
                # Use wget/curl to bypass Python SSL certificate issues in isolated subprocess
                result = subprocess.run(
                    ["wget", "-q", "-O", str(weights_file), weights_url],
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    # Try curl as fallback
                    result = subprocess.run(
                        ["curl", "-sL", "-o", str(weights_file), weights_url],
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode != 0:
                        raise RuntimeError(f"wget/curl failed: {result.stderr}")
                print(f"[SAM3DObjects] DINOv2 weights downloaded")
            except Exception as e:
                print(f"[SAM3DObjects] Warning: Failed to download DINOv2 weights: {e}")
        else:
            print(f"[SAM3DObjects] DINOv2 weights already present")

    @classmethod
    def _ensure_moge_downloaded(cls):
        """Download MoGe to ComfyUI models folder.

        SAM-3D was trained on MoGe v1 (Ruicheng/moge-vitl), so we use that.
        """
        from huggingface_hub import snapshot_download

        models_dir = get_sam3d_models_path()
        moge_dir = models_dir / "sam-3d-objects" / "moge-vitl"

        # Check if already downloaded
        if (moge_dir / "model.pt").exists():
            print(f"[SAM3DObjects] MoGe already downloaded")
            return

        print(f"[SAM3DObjects] Downloading MoGe to {moge_dir}...")
        try:
            snapshot_download(
                "Ruicheng/moge-vitl",
                local_dir=str(moge_dir),
                local_dir_use_symlinks=False,
            )
            print(f"[SAM3DObjects] MoGe downloaded successfully")
        except Exception as e:
            print(f"[SAM3DObjects] Warning: Failed to download MoGe: {e}")
