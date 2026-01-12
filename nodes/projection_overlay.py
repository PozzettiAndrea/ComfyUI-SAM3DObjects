"""SAM3D_ProjectionOverlay node - visualize projected meshes over original image."""

import os
import re
from typing import Tuple

from comfyui_isolation import isolated


@isolated(env="sam3dobjects", import_paths=[".", "../vendor"])
class SAM3D_ProjectionOverlay:
    """
    Projection Overlay - Visualize pose-optimized meshes projected onto the original image.

    Takes the output folder from SAM3D_ScenePoseOptimize and creates an overlay
    visualization showing how the 3D meshes project onto the 2D image.

    Useful for debugging and verifying pose optimization quality.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_opt_folder": ("STRING", {
                    "default": "",
                    "tooltip": "Path to pose-optimized folder (output from SAM3D_ScenePoseOptimize)"
                }),
                "point_size": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 10,
                    "tooltip": "Size of projected points in pixels"
                }),
                "alpha": ("FLOAT", {
                    "default": 0.6,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Opacity of projected mesh points"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("overlay_image",)
    OUTPUT_TOOLTIPS = ("Overlay image showing projected meshes on original image",)
    FUNCTION = "create_overlay"
    CATEGORY = "SAM3DObjects"
    DESCRIPTION = "Visualize pose-optimized meshes projected onto the original image."

    def create_overlay(
        self,
        pose_opt_folder: str,
        point_size: int = 2,
        alpha: float = 0.6,
    ) -> Tuple:
        """
        Create overlay visualization of projected meshes.

        Args:
            pose_opt_folder: Path to pose-optimized folder
            point_size: Size of projected points
            alpha: Opacity of mesh points

        Returns:
            Tuple containing ComfyUI IMAGE tensor
        """
        import os
        import re
        import torch
        import numpy as np
        import trimesh
        from PIL import Image

        print(f"[SAM3DObjects] ProjectionOverlay: Creating overlay from {pose_opt_folder}")

        # Validate folder exists
        if not os.path.exists(pose_opt_folder):
            raise ValueError(f"Pose optimization folder does not exist: {pose_opt_folder}")

        # Derive source folder (remove _pose_optimized suffix)
        if pose_opt_folder.endswith("_pose_optimized"):
            source_folder = pose_opt_folder[:-len("_pose_optimized")]
        else:
            # Try to find source folder
            source_folder = pose_opt_folder
            print(f"[SAM3DObjects] ProjectionOverlay: Warning - folder doesn't end with _pose_optimized")

        # Load original image
        image_path = os.path.join(source_folder, "image.png")
        if not os.path.exists(image_path):
            raise ValueError(f"Original image not found: {image_path}")

        orig_img = Image.open(image_path).convert("RGB")
        img_array = np.array(orig_img).astype(np.float32) / 255.0
        H, W = img_array.shape[:2]
        print(f"[SAM3DObjects] ProjectionOverlay: Loaded image {W}x{H}")

        # Load intrinsics
        intrinsics_path = os.path.join(source_folder, "intrinsics.pt")
        if not os.path.exists(intrinsics_path):
            raise ValueError(f"Intrinsics not found: {intrinsics_path}")

        intrinsics = torch.load(intrinsics_path, weights_only=False)
        intrinsics = np.array(intrinsics)

        # Denormalize intrinsics
        K = intrinsics.copy()
        K[0, 0] *= W  # fx
        K[1, 1] *= H  # fy
        K[0, 2] *= W  # cx
        K[1, 2] *= H  # cy
        print(f"[SAM3DObjects] ProjectionOverlay: Intrinsics loaded")

        # Find all mesh files
        mesh_files = []
        pattern = re.compile(r'^object_(\d+)\.glb$')
        for name in os.listdir(pose_opt_folder):
            match = pattern.match(name)
            if match:
                idx = int(match.group(1))
                mesh_files.append((idx, os.path.join(pose_opt_folder, name)))

        mesh_files.sort(key=lambda x: x[0])
        print(f"[SAM3DObjects] ProjectionOverlay: Found {len(mesh_files)} meshes")

        if not mesh_files:
            print("[SAM3DObjects] ProjectionOverlay: No meshes found, returning original image")
            img_tensor = torch.from_numpy(img_array).unsqueeze(0)
            return (img_tensor,)

        # Generate distinct colors for each object
        colors = [
            [1.0, 0.5, 0.0],  # Orange
            [0.0, 0.5, 1.0],  # Blue
            [0.0, 0.8, 0.2],  # Green
            [0.8, 0.0, 0.2],  # Red
            [0.6, 0.0, 0.8],  # Purple
            [0.6, 0.4, 0.2],  # Brown
            [1.0, 0.0, 0.5],  # Pink
            [0.0, 0.8, 0.8],  # Cyan
            [0.8, 0.8, 0.0],  # Yellow
            [0.5, 0.5, 0.5],  # Gray
        ]

        # Create overlay
        overlay = img_array.copy()

        for idx, mesh_path in mesh_files:
            try:
                mesh = trimesh.load(mesh_path)
                if isinstance(mesh, trimesh.Scene):
                    meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
                    if not meshes:
                        continue
                    mesh = meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)

                verts = mesh.vertices

                # Filter vertices in front of camera (Z > 0)
                mask = verts[:, 2] > 0.1
                v = verts[mask]

                if len(v) == 0:
                    continue

                # Project vertices
                X, Y, Z = v[:, 0], v[:, 1], v[:, 2]
                x = (K[0, 0] * X / Z + K[0, 2]).astype(int)
                y = (K[1, 1] * Y / Z + K[1, 2]).astype(int)

                # Filter points within image bounds
                valid = (x >= 0) & (x < W) & (y >= 0) & (y < H)
                x, y = x[valid], y[valid]

                # Get color for this object
                color = np.array(colors[idx % len(colors)])

                # Draw points with size
                for px, py in zip(x, y):
                    for dx in range(-point_size, point_size + 1):
                        for dy in range(-point_size, point_size + 1):
                            nx, ny = px + dx, py + dy
                            if 0 <= nx < W and 0 <= ny < H:
                                overlay[ny, nx] = overlay[ny, nx] * (1 - alpha) + color * alpha

                print(f"[SAM3DObjects] ProjectionOverlay: Object {idx}: {len(x)} points projected")

            except Exception as e:
                print(f"[SAM3DObjects] ProjectionOverlay: Error loading object {idx}: {e}")
                continue

        # Convert to ComfyUI IMAGE format (B, H, W, C)
        overlay = np.clip(overlay, 0, 1)
        img_tensor = torch.from_numpy(overlay).float().unsqueeze(0)

        print(f"[SAM3DObjects] ProjectionOverlay: Done, output shape {img_tensor.shape}")
        return (img_tensor,)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "SAM3D_ProjectionOverlay": SAM3D_ProjectionOverlay,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3D_ProjectionOverlay": "SAM3D Projection Overlay",
}
