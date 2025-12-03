"""Utility functions and data type conversions for SAM3DObjects nodes."""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
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


def comfy_image_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert ComfyUI IMAGE tensor to PIL Image.

    ComfyUI IMAGE format: [B, H, W, C], float32, range [0, 1]

    Args:
        tensor: ComfyUI image tensor

    Returns:
        PIL Image in RGB mode
    """
    # Take first image if batch
    if len(tensor.shape) == 4:
        tensor = tensor[0]

    # Convert from [H, W, C] float [0,1] to [H, W, C] uint8 [0,255]
    img_np = (tensor.cpu().numpy() * 255).astype(np.uint8)

    # Convert to PIL
    return Image.fromarray(img_np, mode='RGB')


def comfy_image_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert ComfyUI IMAGE tensor to numpy array.

    Args:
        tensor: ComfyUI image tensor [B, H, W, C]

    Returns:
        Numpy array [H, W, C], float32, range [0, 1]
    """
    # Take first image if batch
    if len(tensor.shape) == 4:
        tensor = tensor[0]

    return tensor.cpu().numpy()


def comfy_mask_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert ComfyUI MASK tensor to numpy array.

    ComfyUI MASK format: [N, H, W], float32, range [0, 1]

    Args:
        tensor: ComfyUI mask tensor

    Returns:
        Numpy array [H, W], float32, range [0, 1]
    """
    # Take first mask if batch
    if len(tensor.shape) == 3:
        tensor = tensor[0]

    return tensor.cpu().numpy()


def pil_to_comfy_image(pil_image: Image.Image) -> torch.Tensor:
    """
    Convert PIL Image to ComfyUI IMAGE tensor.

    Args:
        pil_image: PIL Image

    Returns:
        ComfyUI image tensor [1, H, W, C], float32, range [0, 1]
    """
    # Convert to RGB if needed
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')

    # Convert to numpy array
    img_np = np.array(pil_image).astype(np.float32) / 255.0

    # Convert to torch tensor and add batch dimension
    tensor = torch.from_numpy(img_np).unsqueeze(0)

    return tensor


def numpy_to_comfy_image(array: np.ndarray) -> torch.Tensor:
    """
    Convert numpy array to ComfyUI IMAGE tensor.

    Args:
        array: Numpy array, can be [H, W, C] or [H, W]

    Returns:
        ComfyUI image tensor [1, H, W, C], float32, range [0, 1]
    """
    # Handle grayscale images
    if len(array.shape) == 2:
        array = np.stack([array] * 3, axis=-1)

    # Ensure float32 and range [0, 1]
    if array.dtype == np.uint8:
        array = array.astype(np.float32) / 255.0
    else:
        array = array.astype(np.float32)

    # Ensure range [0, 1]
    array = np.clip(array, 0.0, 1.0)

    # Convert to torch and add batch dimension
    tensor = torch.from_numpy(array).unsqueeze(0)

    return tensor


def numpy_to_comfy_mask(array: np.ndarray) -> torch.Tensor:
    """
    Convert numpy array to ComfyUI MASK tensor.

    Args:
        array: Numpy array [H, W]

    Returns:
        ComfyUI mask tensor [1, H, W], float32, range [0, 1]
    """
    # Ensure 2D
    if len(array.shape) == 3:
        # Take first channel if multi-channel
        array = array[:, :, 0]

    # Ensure float32 and range [0, 1]
    if array.dtype == np.uint8:
        array = array.astype(np.float32) / 255.0
    else:
        array = array.astype(np.float32)

    # Ensure range [0, 1]
    array = np.clip(array, 0.0, 1.0)

    # Convert to torch and add batch dimension
    tensor = torch.from_numpy(array).unsqueeze(0)

    return tensor


def get_device() -> torch.device:
    """
    Get the appropriate torch device.

    Returns:
        torch.device (cuda if available, else cpu)
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in bytes to human-readable string.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"
