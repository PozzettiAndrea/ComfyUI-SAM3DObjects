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
                "model": ("SAM3D_MODEL", {"tooltip": "SAM3D model loaded from checkpoint"}),
                "image": ("IMAGE", {"tooltip": "Input RGB image"}),
                "mask": ("MASK", {"tooltip": "Binary mask indicating the object region"}),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 2**31 - 1,
                    "tooltip": "Random seed for reproducible generation"
                }),
            },
            "optional": {
                "stage1_inference_steps": ("INT", {
                    "default": 25,
                    "min": 1,
                    "max": 100,
                    "tooltip": "Number of denoising steps for Stage 1 (sparse structure). Higher = better quality but slower"
                }),
                "stage2_inference_steps": ("INT", {
                    "default": 25,
                    "min": 1,
                    "max": 100,
                    "tooltip": "Number of denoising steps for Stage 2 (SLAT generation). Higher = better quality but slower"
                }),
                "stage1_cfg_strength": ("FLOAT", {
                    "default": 7.0,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.1,
                    "tooltip": "Classifier-free guidance strength for Stage 1. Higher = stronger adherence to input image"
                }),
                "stage2_cfg_strength": ("FLOAT", {
                    "default": 5.0,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.1,
                    "tooltip": "Classifier-free guidance strength for Stage 2. Higher = stronger adherence to conditions"
                }),
                "texture_size": ([512, 1024, 2048, 4096], {
                    "default": 1024,
                    "tooltip": "Texture resolution for baked textures. Higher = better quality but larger file size"
                }),
                "simplify": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Mesh simplification ratio (1.0 = no simplification, 0.95 = keep 95% of faces)"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "SAM3D_POSE")
    RETURN_NAMES = ("glb_filepath", "ply_filepath", "pose_data")
    OUTPUT_TOOLTIPS = (
        "Path to the generated GLB mesh file (textured 3D mesh)",
        "Path to the generated PLY file (colored Gaussian point cloud)",
        "Pose information containing rotation, translation, and scale"
    )
    FUNCTION = "generate"
    CATEGORY = "SAM3DObjects"
    DESCRIPTION = "Generate 3D object from image and mask using SAM3D."

    def generate(
        self,
        model: Any,
        image: torch.Tensor,
        mask: torch.Tensor,
        seed: int,
        stage1_inference_steps: int = 25,
        stage2_inference_steps: int = 25,
        stage1_cfg_strength: float = 7.0,
        stage2_cfg_strength: float = 5.0,
        texture_size: int = 1024,
        simplify: float = 0.95,
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
            output = model(
                image_pil, mask_np,
                seed=seed,
                stage1_inference_steps=stage1_inference_steps,
                stage2_inference_steps=stage2_inference_steps,
                stage1_cfg_strength=stage1_cfg_strength,
                stage2_cfg_strength=stage2_cfg_strength,
                texture_size=texture_size,
                simplify=simplify,
            )

        except Exception as e:
            raise RuntimeError(f"SAM3D inference failed: {e}") from e

        # Extract outputs - now these are file paths!
        # Output dict now contains:
        # - "glb_path": Path to saved GLB file
        # - "ply_path": Path to saved Gaussian PLY file
        # - "output_dir": Directory containing all outputs
        # - "metadata": Simple metadata from inference

        glb_path = output.get("glb_path")
        ply_path = output.get("ply_path")
        output_dir = output.get("output_dir")
        metadata = output.get("metadata", {})

        # Create pose data dict from metadata
        pose_data = {
            "rotation": metadata.get("rotation"),
            "translation": metadata.get("translation"),
            "scale": metadata.get("scale"),
        }

        print("[SAM3DObjects] 3D generation completed!")
        print(f"[SAM3DObjects] - GLB file: {glb_path}")
        print(f"[SAM3DObjects] - Gaussian PLY file: {ply_path}")
        print(f"[SAM3DObjects] - Output directory: {output_dir}")
        print(f"[SAM3DObjects] - Rotation: {pose_data['rotation']}")
        print(f"[SAM3DObjects] - Translation: {pose_data['translation']}")
        print(f"[SAM3DObjects] - Scale: {pose_data['scale']}")

        # Return file paths as outputs
        return (
            glb_path,  # Path to GLB mesh file
            ply_path,  # Path to Gaussian PLY file (colored point cloud)
            pose_data, # Pose information
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
                "model": ("SAM3D_MODEL", {"tooltip": "SAM3D model loaded from checkpoint"}),
                "rgba_image": ("IMAGE", {"tooltip": "Input RGBA image with alpha channel"}),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 2**31 - 1,
                    "tooltip": "Random seed for reproducible generation"
                }),
                "alpha_threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Threshold for converting alpha channel to binary mask (pixels above this value are considered foreground)"
                }),
            },
            "optional": {
                "stage1_inference_steps": ("INT", {
                    "default": 25,
                    "min": 1,
                    "max": 100,
                    "tooltip": "Number of denoising steps for Stage 1 (sparse structure). Higher = better quality but slower"
                }),
                "stage2_inference_steps": ("INT", {
                    "default": 25,
                    "min": 1,
                    "max": 100,
                    "tooltip": "Number of denoising steps for Stage 2 (SLAT generation). Higher = better quality but slower"
                }),
                "stage1_cfg_strength": ("FLOAT", {
                    "default": 7.0,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.1,
                    "tooltip": "Classifier-free guidance strength for Stage 1. Higher = stronger adherence to input image"
                }),
                "stage2_cfg_strength": ("FLOAT", {
                    "default": 5.0,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.1,
                    "tooltip": "Classifier-free guidance strength for Stage 2. Higher = stronger adherence to conditions"
                }),
                "texture_size": ([512, 1024, 2048, 4096], {
                    "default": 1024,
                    "tooltip": "Texture resolution for baked textures. Higher = better quality but larger file size"
                }),
                "simplify": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Mesh simplification ratio (1.0 = no simplification, 0.95 = keep 95% of faces)"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "SAM3D_POSE")
    RETURN_NAMES = ("glb_filepath", "ply_filepath", "pose_data")
    OUTPUT_TOOLTIPS = (
        "Path to the generated GLB mesh file (textured 3D mesh)",
        "Path to the generated PLY file (colored Gaussian point cloud)",
        "Pose information containing rotation, translation, and scale"
    )
    FUNCTION = "generate_rgba"
    CATEGORY = "SAM3DObjects"
    DESCRIPTION = "Generate 3D object from RGBA image (alpha channel as mask)."

    def generate_rgba(
        self,
        model: Any,
        rgba_image: torch.Tensor,
        seed: int,
        alpha_threshold: float,
        stage1_inference_steps: int = 25,
        stage2_inference_steps: int = 25,
        stage1_cfg_strength: float = 7.0,
        stage2_cfg_strength: float = 5.0,
        texture_size: int = 1024,
        simplify: float = 0.95,
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
            output = model(
                image_pil, mask_np,
                seed=seed,
                stage1_inference_steps=stage1_inference_steps,
                stage2_inference_steps=stage2_inference_steps,
                stage1_cfg_strength=stage1_cfg_strength,
                stage2_cfg_strength=stage2_cfg_strength,
                texture_size=texture_size,
                simplify=simplify,
            )

        except Exception as e:
            raise RuntimeError(f"SAM3D inference failed: {e}") from e

        # Extract outputs - now these are file paths!
        glb_path = output.get("glb_path")
        ply_path = output.get("ply_path")
        output_dir = output.get("output_dir")
        metadata = output.get("metadata", {})

        # Create pose data dict from metadata
        pose_data = {
            "rotation": metadata.get("rotation"),
            "translation": metadata.get("translation"),
            "scale": metadata.get("scale"),
        }

        print("[SAM3DObjects] 3D generation completed!")
        print(f"[SAM3DObjects] - GLB file: {glb_path}")
        print(f"[SAM3DObjects] - Gaussian PLY file: {ply_path}")
        print(f"[SAM3DObjects] - Output directory: {output_dir}")

        # Return file paths as outputs
        return (
            glb_path,  # Path to GLB mesh file
            ply_path,  # Path to Gaussian PLY file (colored point cloud)
            pose_data, # Pose information
        )
