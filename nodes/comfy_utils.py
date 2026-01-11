"""Utility functions for SAM3DObjects nodes."""

import torch
from pathlib import Path
from typing import Dict, Any
import folder_paths


# Global model cache to avoid reloading models
_MODEL_CACHE: Dict[str, Any] = {}


def get_sam3d_models_path() -> Path:
    """
    Get the path to SAM3D models directory within ComfyUI models folder.

    Returns:
        Path to ComfyUI/models/sam3d/
    """
    models_dir = Path(folder_paths.models_dir) / "sam3d"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def get_device() -> torch.device:
    """
    Get the appropriate torch device.

    Returns:
        torch.device (cuda if available, else cpu)
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
