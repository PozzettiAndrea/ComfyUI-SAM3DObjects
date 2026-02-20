"""
Utility functions for SAM3D inference worker.

This module contains:
- Pointmap loading and image preprocessing
- File I/O helpers
"""

import logging
import sys
import json
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import torch
import comfy.utils

log = logging.getLogger("sam3dobjects")


# =============================================================================
# Preprocessing functions (merged from preprocessing.py)
# =============================================================================

def load_pointmap_from_file(pointmap_path: str) -> torch.Tensor:
    """
    Load pointmap from a .pt tensor file.

    Args:
        pointmap_path: Path to .pt file (dict format with "pointmap" key)

    Returns:
        Pointmap tensor in HWC format (H, W, 3)
    """
    import comfy.model_management as mm

    data = comfy.utils.load_torch_file(pointmap_path)
    if isinstance(data, dict):
        pointmap = data.get("pointmap") or data.get("data")
    else:
        pointmap = data
    log.info("Loaded pointmap tensor: shape=%s", pointmap.shape)

    pointmap = pointmap.to(mm.get_torch_device())

    return pointmap


def preprocess_image_lazy(
    image_np: np.ndarray,
    mask_np: Optional[np.ndarray],
    preprocessor,
    pointmap: Optional[torch.Tensor] = None,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Preprocess image using a preprocessor (extracted from InferencePipeline).

    Args:
        image_np: Image as numpy array (H, W, 4) RGBA format
        mask_np: Mask as numpy array (unused - mask comes from alpha channel)
        preprocessor: Preprocessor instance
        pointmap: Optional pointmap tensor (H, W, 3) or numpy array
        device: Target device for tensors

    Returns:
        dict with preprocessed image, mask, pointmap etc.
    """
    from ..sam3d.transforms import get_mask

    # Ensure RGBA format
    assert image_np.ndim == 3, f"Expected 3D image, got {image_np.ndim}D"
    assert image_np.shape[-1] == 4, f"Expected RGBA (4 channels), got {image_np.shape[-1]}"
    assert image_np.dtype == np.uint8, f"Expected uint8, got {image_np.dtype}"

    # Convert to float [0, 1] and tensor (CHW format)
    image_float = (image_np / 255.0).astype(np.float32)
    rgba_image = torch.from_numpy(image_float).permute(2, 0, 1).contiguous()
    rgb_image = rgba_image[:3]
    rgb_image_mask = get_mask(rgba_image, None, "ALPHA_CHANNEL")

    # Convert pointmap to tensor if needed (should be CHW format for preprocessor)
    if pointmap is not None:
        if isinstance(pointmap, np.ndarray):
            pointmap = torch.from_numpy(pointmap).float()
        # If HWC, convert to CHW
        if pointmap.dim() == 3 and pointmap.shape[-1] == 3:
            pointmap = pointmap.permute(2, 0, 1).contiguous()

    # Use the preprocessor's internal method
    preprocessor_return_dict = preprocessor._process_image_mask_pointmap_mess(
        rgb_image, rgb_image_mask, pointmap
    )

    # Build result dict with batch dimension and move to device
    _item = preprocessor_return_dict
    item = {
        "mask": _item["mask"][None].to(device),
        "image": _item["image"][None].to(device),
        "rgb_image": _item["rgb_image"][None].to(device),
        "rgb_image_mask": _item["rgb_image_mask"][None].to(device),
    }

    # Add pointmap-related fields if available
    if pointmap is not None and hasattr(preprocessor, 'pointmap_transform') and preprocessor.pointmap_transform != (None,):
        if "pointmap" in _item:
            item["pointmap"] = _item["pointmap"][None].to(device)
        if "rgb_pointmap" in _item:
            item["rgb_pointmap"] = _item["rgb_pointmap"][None].to(device)
        if "pointmap_scale" in _item:
            item["pointmap_scale"] = _item["pointmap_scale"][None].to(device)
        if "pointmap_shift" in _item:
            item["pointmap_shift"] = _item["pointmap_shift"][None].to(device)
        if "rgb_pointmap_scale" in _item:
            item["rgb_pointmap_scale"] = _item["rgb_pointmap_scale"][None].to(device)
        if "rgb_pointmap_shift" in _item:
            item["rgb_pointmap_shift"] = _item["rgb_pointmap_shift"][None].to(device)

    return item


# =============================================================================
# File I/O helpers
# =============================================================================

def save_output_to_disk(output: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    """
    Save output to disk and return file paths.

    This is much more robust than trying to serialize complex objects through IPC.
    Following ComfyUI's standard pattern of saving outputs to disk.

    The output_dir should be a sam3d_inference_N directory created by depth_estimate.
    Files are saved directly to this directory (no subdirectory creation).
    """
    # Use provided directory directly (created by depth_estimate node)
    save_dir = Path(output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    log.info("Saving outputs to: %s", save_dir)

    result = {
        "output_dir": str(save_dir),
        "files": {},
        "metadata": {}
    }

    # Save sparse structure (Stage 1 output)
    # Identify sparse structure by keys present in stage 1 but not others
    if "coords" in output and "pointmap" in output and "slat" not in output:
        sparse_path = save_dir / "sparse_structure.pt"
        torch.save(output, sparse_path)
        result["files"]["sparse_structure"] = str(sparse_path)
        log.info("Saved sparse structure: %s", sparse_path)

    # Save SLAT (Stage 2 intermediate output)
    if "slat" in output:
        slat_path = save_dir / "slat.pt"
        torch.save(output, slat_path)
        result["files"]["slat"] = str(slat_path)
        log.info("Saved SLAT: %s", slat_path)

    # Save GLB file (textured mesh)
    if "glb" in output and output["glb"] is not None:
        glb_path = save_dir / "mesh.glb"

        # Check if it's a Trimesh object that needs to be exported
        import trimesh
        if isinstance(output["glb"], trimesh.Trimesh):
            # Export Trimesh to GLB format
            glb_bytes = output["glb"].export(file_type="glb")
            with open(glb_path, 'wb') as f:
                f.write(glb_bytes)
            log.info("Saved GLB: %s (%d bytes)", glb_path, len(glb_bytes))
        else:
            # Already bytes
            with open(glb_path, 'wb') as f:
                f.write(output["glb"])
            log.info("Saved GLB: %s (%d bytes)", glb_path, len(output['glb']))

        result["files"]["glb"] = str(glb_path)

    # Save Gaussian Splat PLY file (colored point cloud)
    if "gs" in output and output["gs"] is not None:
        ply_path = save_dir / "gaussian.ply"
        try:
            output["gs"].save_ply(str(ply_path))
            result["files"]["ply"] = str(ply_path)
            log.info("Saved Gaussian PLY: %s", ply_path)
        except Exception as e:
            log.warning("Failed to save Gaussian PLY: %s", e)

    # Save metadata (simple types only)
    metadata = {}
    for key, value in output.items():
        if isinstance(value, (int, float, str, bool)):
            metadata[key] = value
        elif isinstance(value, torch.Tensor):
            # Convert torch tensors to lists for JSON serialization
            metadata[key] = value.cpu().tolist()
        elif isinstance(value, np.ndarray):
            # Convert numpy arrays to lists for JSON serialization
            metadata[key] = value.tolist()
        elif isinstance(value, dict) and key not in ["glb", "gaussian_splat", "mesh"]:
            # Save simple dict metadata
            try:
                json.dumps(value)  # Test if it's JSON-serializable
                metadata[key] = value
            except Exception as e:
                log.debug("Skipping non-serializable metadata key %r: %s", key, e)

    if metadata:
        metadata_path = save_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        result["files"]["metadata"] = str(metadata_path)
        result["metadata"] = metadata

    return result


# =============================================================================
# Model download helpers
# =============================================================================

# HuggingFace repo for SAM3D checkpoints
REPO_ID = "jetjodh/sam-3d-objects"

# Decoder files
DECODER_FILES = {
    "gaussian": [
        "slat_decoder_gs.yaml",
        "slat_decoder_gs.ckpt",
        "slat_decoder_gs_4.yaml",
        "slat_decoder_gs_4.ckpt",
    ],
    "mesh": [
        "slat_decoder_mesh.yaml",
        "slat_decoder_mesh.ckpt",
    ],
}


def ensure_decoder_files(config_path: str, decoder_type: str):
    """
    Ensure decoder files exist, downloading if missing.

    Args:
        config_path: Path to pipeline.yaml
        decoder_type: "gaussian" or "mesh"
    """
    checkpoint_dir = Path(config_path).parent
    files = DECODER_FILES.get(decoder_type, [])

    missing = [f for f in files if not (checkpoint_dir / f).exists()]
    if missing:
        log.info("Downloading missing %s decoder files: %s", decoder_type, missing)
        _download_decoder_files(checkpoint_dir, missing)


def _download_decoder_files(checkpoint_dir: Path, files: list):
    """
    Download specific decoder files from HuggingFace.

    Args:
        checkpoint_dir: Directory containing checkpoints (e.g., .../checkpoints/)
        files: List of filenames to download
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for downloading checkpoints. "
            "Please install it: pip install huggingface-hub"
        )

    # target_dir is parent of checkpoints/ for hf_hub_download
    target_dir = checkpoint_dir.parent

    log.info("Downloading from HuggingFace: %s", REPO_ID)

    for filename in files:
        hf_path = f"checkpoints/{filename}"
        log.info("Downloading %s...", filename)

        try:
            hf_hub_download(
                repo_id=REPO_ID,
                filename=hf_path,
                local_dir=str(target_dir),
                local_dir_use_symlinks=False,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to download {filename}: {e}") from e

    log.info("Download complete")

