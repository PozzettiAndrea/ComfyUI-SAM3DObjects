"""SAM3D_DepthEstimate node for running MoGe depth estimation separately."""

import torch
import numpy as np
import os
from typing import Any

import folder_paths

from .utils import comfy_image_to_pil


class SAM3D_DepthEstimate:
    """
    Depth Estimation using MoGe model.

    Runs depth estimation separately from sparse structure generation.
    This allows unloading the depth model before running other stages,
    saving VRAM on memory-constrained GPUs.

    Outputs:
    - pointmap: 3D point cloud (HxWx3) - pass to SAM3D_SparseGen
    - intrinsics: Camera intrinsics matrix (3x3)
    - pointcloud_glb: Path to GLB file containing the point cloud
    - depth_mask: Depth visualization as mask (normalized 0-1)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth_model": ("SAM3D_MODEL", {"tooltip": "Depth model from LoadSAM3DModel"}),
                "image": ("IMAGE", {"tooltip": "Input RGB image"}),
            },
        }

    RETURN_TYPES = ("SAM3D_INTRINSICS", "STRING", "STRING", "MASK")
    RETURN_NAMES = ("intrinsics", "pointmap_path", "pointcloud_ply", "depth_mask")
    OUTPUT_TOOLTIPS = (
        "Camera intrinsics matrix (3x3)",
        "Path to pointmap tensor file (.pt) - pass to SAM3D_SparseGen",
        "Path to PLY file for visualization",
        "Depth map as mask (normalized 0-1)"
    )
    FUNCTION = "estimate_depth"
    CATEGORY = "SAM3DObjects"
    DESCRIPTION = "Run MoGe depth estimation to get camera intrinsics and point cloud PLY."

    def _get_next_inference_dir(self, base_output_dir: str) -> str:
        """
        Find the next available sam3d_inference_N directory.

        Args:
            base_output_dir: ComfyUI output directory

        Returns:
            Path to the new sam3d_inference_N directory (created)
        """
        # Find existing sam3d_inference_N directories
        existing = []
        if os.path.exists(base_output_dir):
            for name in os.listdir(base_output_dir):
                if name.startswith("sam3d_inference_") and os.path.isdir(os.path.join(base_output_dir, name)):
                    try:
                        num = int(name.split("_")[-1])
                        existing.append(num)
                    except ValueError:
                        pass

        # Get next number
        next_num = max(existing) + 1 if existing else 1
        inference_dir = os.path.join(base_output_dir, f"sam3d_inference_{next_num}")
        os.makedirs(inference_dir, exist_ok=True)

        print(f"[SAM3DObjects] Created inference directory: {inference_dir}")
        return inference_dir

    def _save_pointcloud_ply(self, pointmap: np.ndarray, image_pil, output_dir: str) -> str:
        """
        Save pointmap as a PLY file with vertex colors from the image.

        Args:
            pointmap: Point cloud data (H, W, 3)
            image_pil: PIL Image for vertex colors
            output_dir: Directory to save the PLY file

        Returns:
            Path to the saved PLY file
        """
        # Get image as numpy for colors
        image_np = np.array(image_pil)
        if image_np.shape[-1] == 4:
            # RGBA - use alpha as mask
            alpha = image_np[..., 3]
            rgb = image_np[..., :3]
        else:
            # RGB - no mask
            alpha = np.ones(image_np.shape[:2], dtype=np.uint8) * 255
            rgb = image_np

        # Flatten pointmap and colors
        H, W = pointmap.shape[:2]
        points = pointmap.reshape(-1, 3)
        colors = rgb.reshape(-1, 3)
        alpha_flat = alpha.reshape(-1)

        # Filter out invalid points (NaN, inf, or masked out)
        valid_mask = (
            ~np.isnan(points).any(axis=1) &
            ~np.isinf(points).any(axis=1) &
            (alpha_flat > 128)  # Only keep non-transparent points
        )

        valid_points = points[valid_mask]
        valid_colors = colors[valid_mask]

        print(f"[SAM3DObjects] Point cloud: {len(valid_points)} valid points out of {len(points)}")

        if len(valid_points) == 0:
            raise RuntimeError("No valid points in pointmap")

        # Save as pointcloud.ply in the inference directory
        filepath = os.path.join(output_dir, "pointcloud.ply")

        # Write PLY file manually (simple ASCII format with colors)
        with open(filepath, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(valid_points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")

            for i in range(len(valid_points)):
                x, y, z = valid_points[i]
                r, g, b = valid_colors[i]
                f.write(f"{x} {y} {z} {r} {g} {b}\n")

        print(f"[SAM3DObjects] Saved point cloud PLY: {filepath}")

        return filepath

    def estimate_depth(
        self,
        depth_model: Any,
        image: torch.Tensor,
    ):
        """
        Run depth estimation.

        Args:
            depth_model: SAM3D model wrapper (IsolatedSAM3DModel)
            image: Input image tensor [B, H, W, C]

        Returns:
            Tuple of (intrinsics, pointcloud_ply, depth_mask)
        """
        print("[SAM3DObjects] DepthEstimate: Running depth estimation...")

        # Convert ComfyUI tensor to PIL
        image_pil = comfy_image_to_pil(image)
        print(f"[SAM3DObjects] Image size: {image_pil.size}")

        # Run depth-only inference
        try:
            result = depth_model(
                image_pil,
                None,  # No mask needed for depth
                depth_only=True,  # depth-only mode
            )
        except Exception as e:
            raise RuntimeError(f"Depth estimation failed: {e}") from e

        # Extract results
        pointmap = result.get("pointmap")
        intrinsics = result.get("intrinsics")

        if pointmap is None:
            raise RuntimeError("Depth estimation did not return pointmap")

        print(f"[SAM3DObjects] Pointmap shape: {pointmap.shape if hasattr(pointmap, 'shape') else 'unknown'}")

        # Convert pointmap to numpy for PLY export
        if isinstance(pointmap, torch.Tensor):
            pointmap_np = pointmap.cpu().numpy() if pointmap.is_cuda else pointmap.numpy()
        else:
            pointmap_np = pointmap

        # Create sam3d_inference_N directory
        base_output_dir = folder_paths.get_output_directory()
        inference_dir = self._get_next_inference_dir(base_output_dir)

        # Save pointmap tensor for SparseGen (preserves H×W structure)
        pointmap_path = os.path.join(inference_dir, "pointmap.pt")
        torch.save(torch.from_numpy(pointmap_np), pointmap_path)
        print(f"[SAM3DObjects] Saved pointmap tensor: {pointmap_path}")

        # Save PLY file for visualization
        pointcloud_ply = self._save_pointcloud_ply(pointmap_np, image_pil, inference_dir)

        # Create depth visualization
        # Pointmap is in HWC format (H, W, 3) where channel 2 is Z (depth)
        # Normalize Z channel to 0-1 for visualization
        depth_np = pointmap_np[..., 2]

        # Normalize depth for visualization
        depth_min = np.nanmin(depth_np)
        depth_max = np.nanmax(depth_np)
        if depth_max - depth_min > 0:
            depth_normalized = (depth_np - depth_min) / (depth_max - depth_min)
        else:
            depth_normalized = np.zeros_like(depth_np)

        # Handle NaN values
        depth_normalized = np.nan_to_num(depth_normalized, nan=0.0)

        # Convert to ComfyUI MASK format [B, H, W]
        depth_mask = torch.from_numpy(depth_normalized).unsqueeze(0).float()

        print("[SAM3DObjects] Depth estimation completed!")

        return (intrinsics, pointmap_path, pointcloud_ply, depth_mask)
