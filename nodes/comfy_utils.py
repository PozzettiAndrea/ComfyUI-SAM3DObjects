"""Utility functions for SAM3DObjects nodes."""

from pathlib import Path
import folder_paths


def get_sam3d_models_path() -> Path:
    """Get the path to SAM3D models directory within ComfyUI models folder."""
    models_dir = Path(folder_paths.models_dir) / "sam3d"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir
