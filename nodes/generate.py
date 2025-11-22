"""SAM3DGenerate node for generating 3D objects from images."""

import torch
import numpy as np
from typing import Any, Dict

from .utils import (
    comfy_image_to_pil,
    comfy_image_to_numpy,
    comfy_mask_to_numpy,
)


class SAM3DGenerate:
    """
    Generate 3D object from image and mask using SAM3D.

    Takes an image and mask as input and generates a 3D Gaussian Splat,
    mesh, and pose information.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SAM3D_MODEL",),
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**31 - 1}),
            }
        }

    RETURN_TYPES = ("SAM3D_GAUSSIAN", "SAM3D_MESH", "SAM3D_POSE")
    RETURN_NAMES = ("gaussian_splat", "mesh", "pose_data")
    FUNCTION = "generate"
    CATEGORY = "SAM3DObjects"
    DESCRIPTION = "Generate 3D object from image and mask using SAM3D."

    def generate(
        self,
        model: Any,
        image: torch.Tensor,
        mask: torch.Tensor,
        seed: int,
    ):
        """
        Generate 3D object from image and mask.

        Args:
            model: SAM3D inference pipeline
            image: Input image tensor [B, H, W, C]
            mask: Input mask tensor [N, H, W]
            seed: Random seed

        Returns:
            Tuple of (gaussian_splat, mesh, pose_data)
        """
        print(f"[SAM3DObjects] Generating 3D object (seed: {seed})")

        # Convert ComfyUI tensors to formats expected by SAM3D
        # SAM3D expects PIL Image and numpy mask
        image_pil = comfy_image_to_pil(image)
        mask_np = comfy_mask_to_numpy(mask)

        print(f"[SAM3DObjects] Image size: {image_pil.size}")
        print(f"[SAM3DObjects] Mask shape: {mask_np.shape}")

        # Run inference
        try:
            print("[SAM3DObjects] Running inference...")
            # Note: with_mesh_postprocess=False to avoid nvdiffrast CUDA compilation
            # This skips hole filling and mesh simplification but still generates valid 3D models
            output = model(image_pil, mask_np, seed=seed, with_mesh_postprocess=False)

        except Exception as e:
            raise RuntimeError(f"SAM3D inference failed: {e}") from e

        # Extract outputs
        # Based on demo.py, output is a dict containing:
        # - "gs": Gaussian splat object
        # - "gaussian": Gaussian model
        # - "rotation": rotation quaternion
        # - "translation": translation vector
        # - "scale": scale value

        gaussian_splat = output.get("gs")
        gaussian_model = output.get("gaussian")

        # Create pose data dict
        pose_data = {
            "rotation": output.get("rotation"),
            "translation": output.get("translation"),
            "scale": output.get("scale"),
        }

        print("[SAM3DObjects] 3D generation completed!")
        print(f"[SAM3DObjects] - Gaussian Splat: {type(gaussian_splat)}")
        print(f"[SAM3DObjects] - Rotation: {pose_data['rotation']}")
        print(f"[SAM3DObjects] - Translation: {pose_data['translation']}")
        print(f"[SAM3DObjects] - Scale: {pose_data['scale']}")

        # Return outputs
        # Note: We pass both gaussian_splat (gs) and the full output dict as "mesh"
        # since we'll need the full output for mesh export
        return (
            gaussian_splat,  # For PLY export and visualization
            output,          # Full output dict (contains mesh data)
            pose_data,       # Pose information
        )


class SAM3DGenerateRGBA:
    """
    Generate 3D object from RGBA image (alpha channel as mask).

    Convenience node that extracts the alpha channel as mask automatically.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SAM3D_MODEL",),
                "rgba_image": ("IMAGE",),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**31 - 1}),
                "alpha_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("SAM3D_GAUSSIAN", "SAM3D_MESH", "SAM3D_POSE")
    RETURN_NAMES = ("gaussian_splat", "mesh", "pose_data")
    FUNCTION = "generate_rgba"
    CATEGORY = "SAM3DObjects"
    DESCRIPTION = "Generate 3D object from RGBA image (alpha channel as mask)."

    def generate_rgba(
        self,
        model: Any,
        rgba_image: torch.Tensor,
        seed: int,
        alpha_threshold: float,
    ):
        """
        Generate 3D object from RGBA image.

        Args:
            model: SAM3D inference pipeline
            rgba_image: RGBA image tensor [B, H, W, 4]
            seed: Random seed
            alpha_threshold: Threshold for alpha to mask conversion

        Returns:
            Tuple of (gaussian_splat, mesh, pose_data)
        """
        print(f"[SAM3DObjects] Generating 3D object from RGBA (seed: {seed})")

        # Convert to numpy
        rgba_np = comfy_image_to_numpy(rgba_image)

        # Check if image has alpha channel
        if rgba_np.shape[-1] < 4:
            raise ValueError(
                f"Expected RGBA image with 4 channels, got {rgba_np.shape[-1]} channels. "
                "Use SAM3DGenerate node for RGB images with separate mask."
            )

        # Split into RGB and alpha
        rgb_np = rgba_np[:, :, :3]
        alpha_np = rgba_np[:, :, 3]

        # Convert alpha to binary mask
        mask_np = (alpha_np > alpha_threshold).astype(np.float32)

        print(f"[SAM3DObjects] Image size: {rgb_np.shape[:2]}")
        print(f"[SAM3DObjects] Alpha threshold: {alpha_threshold}")
        print(f"[SAM3DObjects] Mask coverage: {mask_np.mean()*100:.1f}%")

        # Convert RGB numpy to PIL
        from PIL import Image
        rgb_uint8 = (rgb_np * 255).astype(np.uint8)
        image_pil = Image.fromarray(rgb_uint8, mode='RGB')

        # Run inference
        try:
            print("[SAM3DObjects] Running inference...")
            # Note: with_mesh_postprocess=False to avoid nvdiffrast CUDA compilation
            # This skips hole filling and mesh simplification but still generates valid 3D models
            output = model(image_pil, mask_np, seed=seed, with_mesh_postprocess=False)

        except Exception as e:
            raise RuntimeError(f"SAM3D inference failed: {e}") from e

        # Extract outputs
        gaussian_splat = output.get("gs")

        pose_data = {
            "rotation": output.get("rotation"),
            "translation": output.get("translation"),
            "scale": output.get("scale"),
        }

        print("[SAM3DObjects] 3D generation completed!")

        return (
            gaussian_splat,
            output,
            pose_data,
        )
