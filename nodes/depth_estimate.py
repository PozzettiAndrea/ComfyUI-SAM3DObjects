"""SAM3D_DepthEstimate node for running MoGe depth estimation separately."""

import logging
import os
from typing import Any

log = logging.getLogger("sam3dobjects")

# Module-level cache for MoGe ModelPatcher (persists across node executions)
_MOGE_PATCHER = None

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

    def estimate_depth(
        self,
        depth_model: Any,
        image,  # torch.Tensor [B, H, W, C] - arrives deserialized
    ):
        """
        Run depth estimation.

        This method runs in an isolated subprocess with its own Python environment.
        All imports happen inside the method body.

        Args:
            depth_model: SAM3DModelConfig with config_path, etc.
            image: Input image tensor [B, H, W, C]

        Returns:
            Tuple of (intrinsics, pointmap_path, pointcloud_ply, depth_mask)
        """
        global _MOGE_PATCHER

        # These imports happen in the isolated subprocess
        import sys
        import time
        import torch
        import numpy as np
        from pathlib import Path
        from PIL import Image
        import folder_paths
        import comfy.utils
        import comfy.model_management as mm
        import comfy.model_patcher
        from pytorch3d.renderer import look_at_view_transform
        from pytorch3d.transforms import Transform3d

        # Add vendor/ to sys.path for consistent imports
        _VENDOR_PATH = str(Path(__file__).parent / "vendor")
        if _VENDOR_PATH not in sys.path:
            sys.path.insert(0, _VENDOR_PATH)

        pbar = comfy.utils.ProgressBar(4)
        start_time = time.time()
        log.info("DepthEstimate: Running depth estimation with MoGe...")

        # Convert ComfyUI tensor to PIL Image
        # ComfyUI IMAGE format: [B, H, W, C] float32 0-1
        if image.dim() == 4:
            image_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        else:
            image_np = (image.cpu().numpy() * 255).astype(np.uint8)
        image_pil = Image.fromarray(image_np)

        # Load MoGe depth model (cached in ModelPatcher for VRAM management)
        if _MOGE_PATCHER is None:
            from moge.model.v1 import MoGeModel

            moge_path = Path(folder_paths.models_dir) / "sam3dobjects" / "moge-vitl" / "model.pt"
            log.info("Loading MoGe model from %s...", moge_path)
            raw_model = MoGeModel.from_pretrained(str(moge_path))

            from .utils.stages import _enable_lowvram_cast
            _enable_lowvram_cast(raw_model)

            _MOGE_PATCHER = comfy.model_patcher.ModelPatcher(
                raw_model,
                load_device=mm.get_torch_device(),
                offload_device=mm.unet_offload_device(),
            )
            log.info("MoGe wrapped in ModelPatcher")

        device = mm.get_torch_device()
        mm.load_models_gpu([_MOGE_PATCHER])

        from sam3d_objects.pipeline.depth_models.moge import MoGe
        model = MoGe(_MOGE_PATCHER.model, device=str(device))
        pbar.update(1)  # Model loaded

        # Prepare image tensor
        loaded_image = image_np.astype(np.float32) / 255.0
        loaded_image = torch.from_numpy(loaded_image).permute(2, 0, 1).contiguous()[:3]  # CHW, RGB

        # Run depth model
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                output = model(loaded_image)

        pointmaps = output["pointmaps"]
        pbar.update(1)  # Inference done

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

        # Convert to CHW then HWC for output
        points_tensor = points_tensor.permute(2, 0, 1)

        # Infer intrinsics if not provided
        if intrinsics is None:
            from sam3d_objects.pipeline.utils.pointmap import infer_intrinsics_from_pointmap
            intrinsics_result = infer_intrinsics_from_pointmap(
                points_tensor.permute(1, 2, 0), device=device
            )
            intrinsics = intrinsics_result["intrinsics"]

        # Convert to numpy
        pointmap_np = points_tensor.permute(1, 2, 0).cpu().numpy()  # HWC
        intrinsics_np = intrinsics.cpu().numpy() if intrinsics is not None else None

        pbar.update(1)  # Transforms done

        # Create output directory
        base_output_dir = folder_paths.get_output_directory()
        inference_dir = self._get_next_inference_dir(base_output_dir)

        # Save pointmap tensor for SparseGen (preserves H×W structure)
        pointmap_path = os.path.join(inference_dir, "pointmap.pt")
        torch.save(torch.from_numpy(pointmap_np), pointmap_path)

        # Save PLY file for visualization
        pointcloud_ply = self._save_pointcloud_ply(pointmap_np, image_pil, inference_dir)

        # Create depth visualization
        # Pointmap is in HWC format (H, W, 3) where channel 2 is Z (depth)
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

        pbar.update(1)  # Outputs saved
        elapsed = time.time() - start_time
        log.info("Depth estimation done: %.0fs", elapsed)
        return (intrinsics_np, pointmap_path, pointcloud_ply, depth_mask)

    def _get_next_inference_dir(self, base_output_dir: str) -> str:
        """
        Find the next available sam3d_inference_N directory.

        Args:
            base_output_dir: ComfyUI output directory

        Returns:
            Path to the new sam3d_inference_N directory (created)
        """
        import os

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

        return inference_dir

    def _save_pointcloud_ply(self, pointmap, image_pil, output_dir: str) -> str:
        """
        Save pointmap as a PLY file with vertex colors from the image.

        Args:
            pointmap: Point cloud data (H, W, 3)
            image_pil: PIL Image for vertex colors
            output_dir: Directory to save the PLY file

        Returns:
            Path to the saved PLY file
        """
        import numpy as np

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

        return filepath
