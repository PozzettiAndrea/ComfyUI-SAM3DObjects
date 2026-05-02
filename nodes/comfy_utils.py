"""Utility functions for SAM3DObjects nodes."""

import os
from pathlib import Path
import folder_paths

# Register model folder with ComfyUI's folder_paths system
_sam3d_models_dir = os.path.join(folder_paths.models_dir, "sam3dobjects")
os.makedirs(_sam3d_models_dir, exist_ok=True)
folder_paths.add_model_folder_path("sam3d", _sam3d_models_dir)


def get_sam3d_models_path() -> Path:
    """Get the path to SAM3D models directory within ComfyUI models folder."""
    models_dir = Path(folder_paths.models_dir) / "sam3dobjects"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir
