"""SAM3DGenerate node for generating 3D objects from images."""

import torch
import numpy as np
from typing import Any, Dict
from comfy_api.latest import io

from .utils import (
    comfy_image_to_pil,
    comfy_image_to_numpy,
    comfy_mask_to_numpy,
)


class SAM3DGenerate(io.ComfyNode):
    """
    Generate 3D object from image and mask using SAM3D.

    Takes an image and mask as input and generates a 3D Gaussian Splat,
    mesh, and pose information.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SAM3DGenerate",
            display_name="SAM3D Generate",
            category="SAM3DObjects",
            inputs=[
                io.Any.Input(
                    "model",
                    tooltip="SAM3D inference pipeline from LoadSAM3DModel node."
                ),
                io.Image.Input(
                    "image",
                    tooltip="Input image containing the object to reconstruct."
                ),
                io.Mask.Input(
                    "mask",
                    tooltip="Binary mask indicating the object region in the image."
                ),
                io.Int.Input(
                    "seed",
                    default=42,
                    min=0,
                    max=2**31 - 1,
                    tooltip="Random seed for reproducible generation."
                ),
            ],
            outputs=[
                io.Any.Output(
                    "gaussian_splat",
                    tooltip="3D Gaussian Splat representation. Use with SAM3DExportPLY or SAM3DVisualizer."
                ),
                io.Any.Output(
                    "mesh",
                    tooltip="3D mesh representation. Use with SAM3DExportMesh."
                ),
                io.Any.Output(
                    "pose_data",
                    tooltip="Pose data containing rotation, translation, and scale."
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        model: Any,
        image: torch.Tensor,
        mask: torch.Tensor,
        seed: int,
    ) -> io.NodeOutput:
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
            output = model(image_pil, mask_np, seed=seed)

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
        return io.NodeOutput(
            gaussian_splat,  # For PLY export and visualization
            output,          # Full output dict (contains mesh data)
            pose_data,       # Pose information
        )


class SAM3DGenerateRGBA(io.ComfyNode):
    """
    Generate 3D object from RGBA image (alpha channel as mask).

    Convenience node that extracts the alpha channel as mask automatically.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SAM3DGenerateRGBA",
            display_name="SAM3D Generate (RGBA)",
            category="SAM3DObjects",
            inputs=[
                io.Any.Input(
                    "model",
                    tooltip="SAM3D inference pipeline from LoadSAM3DModel node."
                ),
                io.Image.Input(
                    "rgba_image",
                    tooltip="RGBA input image (alpha channel will be used as mask)."
                ),
                io.Int.Input(
                    "seed",
                    default=42,
                    min=0,
                    max=2**31 - 1,
                    tooltip="Random seed for reproducible generation."
                ),
                io.Float.Input(
                    "alpha_threshold",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Threshold for converting alpha channel to binary mask."
                ),
            ],
            outputs=[
                io.Any.Output(
                    "gaussian_splat",
                    tooltip="3D Gaussian Splat representation."
                ),
                io.Any.Output(
                    "mesh",
                    tooltip="3D mesh representation."
                ),
                io.Any.Output(
                    "pose_data",
                    tooltip="Pose data containing rotation, translation, and scale."
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        model: Any,
        rgba_image: torch.Tensor,
        seed: int,
        alpha_threshold: float,
    ) -> io.NodeOutput:
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
            output = model(image_pil, mask_np, seed=seed)

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

        return io.NodeOutput(
            gaussian_splat,
            output,
            pose_data,
        )
