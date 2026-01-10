"""
Depth estimation for SAM3D inference.

Loads only the MoGe depth model on-demand (~2GB VRAM) rather than the full pipeline.
"""

import sys
import base64
import pickle
from typing import Any, Dict
import numpy as np
import torch


def run_depth_only(model_manager, image, unload_after: bool = True, depth_backend: str = "moge2") -> Dict[str, Any]:
    """
    Run depth estimation (loads only MoGe model, ~2GB VRAM).

    Args:
        model_manager: ModelManager instance
        image: PIL Image
        unload_after: Whether to unload depth model after use (frees VRAM)
        depth_backend: "moge2" (newer, metric scale) or "moge" (original)

    Returns:
        Dict with pointmap, intrinsics, depth
    """
    from pytorch3d.renderer import look_at_view_transform
    from pytorch3d.transforms import Transform3d

    print(f"[Worker] Running depth estimation (backend={depth_backend})", file=sys.stderr)

    # Load only the depth model
    depth_model = model_manager.load_depth_model(backend=depth_backend)

    # Convert image to tensor format expected by depth model
    image_np = np.array(image)
    if image_np.ndim == 2:
        image_np = np.stack([image_np] * 3 + [np.full_like(image_np, 255)], axis=-1)
    elif image_np.shape[-1] == 3:
        alpha = np.full((image_np.shape[0], image_np.shape[1], 1), 255, dtype=np.uint8)
        image_np = np.concatenate([image_np, alpha], axis=-1)

    print(f"[Worker] Image shape: {image_np.shape}", file=sys.stderr)

    # Convert to float and tensor
    loaded_image = image_np.astype(np.float32) / 255.0
    loaded_image = torch.from_numpy(loaded_image)
    loaded_image = loaded_image.permute(2, 0, 1).contiguous()[:3]  # CHW, RGB only

    # Run depth model
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=model_manager._get_dtype()):
            output = depth_model(loaded_image)

    pointmaps = output["pointmaps"]

    # Apply camera convention transform (R3 -> PyTorch3D camera space)
    device = pointmaps.device
    r3_to_p3d_R, r3_to_p3d_T = look_at_view_transform(
        eye=np.array([[0, 0, -1]]),
        at=np.array([[0, 0, 0]]),
        up=np.array([[0, -1, 0]]),
        device=device,
    )
    camera_transform = Transform3d().rotate(r3_to_p3d_R).to(device)
    points_tensor = camera_transform.transform_points(pointmaps)

    intrinsics = output.get("intrinsics", None)

    # Convert to CHW format
    points_tensor = points_tensor.permute(2, 0, 1)

    # Infer intrinsics if not provided
    if intrinsics is None:
        from sam3d_objects.pipeline.utils.pointmap import infer_intrinsics_from_pointmap
        intrinsics_result = infer_intrinsics_from_pointmap(
            points_tensor.permute(1, 2, 0), device=device
        )
        intrinsics = intrinsics_result["intrinsics"]

    print(f"[Worker] Pointmap computed: shape={points_tensor.shape}", file=sys.stderr)
    print(f"[Worker] Intrinsics available: {intrinsics is not None}", file=sys.stderr)

    # Unload depth model if requested (frees ~2GB VRAM)
    if unload_after:
        model_manager.unload_depth_model()
        print(f"[Worker] Depth model unloaded", file=sys.stderr)

    # Serialize for transfer
    # Transpose from CHW (3, H, W) to HWC (H, W, 3)
    pointmap_hwc = points_tensor.permute(1, 2, 0).contiguous()
    pointmap_np = pointmap_hwc.cpu().numpy()
    print(f"[Worker] Pointmap transposed to HWC: {pointmap_np.shape}", file=sys.stderr)

    if intrinsics is not None and hasattr(intrinsics, 'cpu'):
        intrinsics_np = intrinsics.cpu().numpy()
    else:
        intrinsics_np = intrinsics

    pointmap_b64 = base64.b64encode(pickle.dumps(pointmap_np)).decode('utf-8')
    intrinsics_b64 = base64.b64encode(pickle.dumps(intrinsics_np)).decode('utf-8') if intrinsics_np is not None else None

    return {
        "status": "success",
        "depth_only": True,
        "pointmap": pointmap_b64,
        "intrinsics": intrinsics_b64,
    }
