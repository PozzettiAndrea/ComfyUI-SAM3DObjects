"""SAM3DGenerateStage2 node for completing generation from Stage 1 output."""

import torch
from typing import Any

from .utils import (
    comfy_image_to_pil,
    comfy_mask_to_numpy,
)


class SAM3DGenerateStage2:
    """
    Generate Stage 2: Complete generation from Stage 1 output.

    This node takes Stage 1 output and runs the remaining pipeline:
    - Stage 2 diffusion (SLAT generation)
    - Decode (Gaussian Splat + Mesh generation)
    - Postprocessing (Texture baking + Mesh simplification)

    Cache-efficient: Allows iterating on Stage 2 parameters without re-running Stage 1.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SAM3D_MODEL", {"tooltip": "SAM3D model loaded from checkpoint"}),
                "stage1_data": ("SAM3D_STAGE1_DATA", {"tooltip": "Stage 1 output from SAM3DGenerateStage1"}),
                "image": ("IMAGE", {"tooltip": "Input RGB image (must match Stage 1)"}),
                "mask": ("MASK", {"tooltip": "Binary mask (must match Stage 1)"}),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 2**31 - 1,
                    "tooltip": "Random seed (must match Stage 1 for consistency)"
                }),
            },
            "optional": {
                "stage2_inference_steps": ("INT", {
                    "default": 25,
                    "min": 1,
                    "max": 100,
                    "tooltip": "Number of denoising steps for Stage 2 (SLAT). Higher = better quality but slower"
                }),
                "stage2_cfg_strength": ("FLOAT", {
                    "default": 5.0,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.1,
                    "tooltip": "Classifier-free guidance strength for Stage 2. Higher = stronger adherence to input"
                }),
                "texture_size": ("INT", {
                    "default": 1024,
                    "min": 512,
                    "max": 2048,
                    "step": 512,
                    "tooltip": "Texture resolution for mesh. Higher = better quality but more memory"
                }),
                "simplify": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.9,
                    "max": 0.98,
                    "step": 0.01,
                    "tooltip": "Mesh simplification ratio (0.9 = aggressive, 0.98 = gentle)"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "SAM3D_POSE_DATA")
    RETURN_NAMES = ("glb_filepath", "ply_filepath", "pose_data")
    OUTPUT_TOOLTIPS = (
        "Path to saved GLB mesh file",
        "Path to saved Gaussian PLY file",
        "Camera pose and scale metadata"
    )
    FUNCTION = "generate_stage2"
    CATEGORY = "SAM3DObjects"
    DESCRIPTION = "Complete 3D generation from Stage 1 output (Stage 2 + postprocess, ~90 seconds)."

    def generate_stage2(
        self,
        model: Any,
        stage1_data: dict,
        image: torch.Tensor,
        mask: torch.Tensor,
        seed: int,
        stage2_inference_steps: int = 25,
        stage2_cfg_strength: float = 5.0,
        texture_size: int = 1024,
        simplify: float = 0.95,
    ):
        """
        Complete generation from Stage 1 output.

        Args:
            model: SAM3D inference pipeline
            stage1_data: Stage 1 output from SAM3DGenerateStage1
            image: Input image tensor [B, H, W, C] (must match Stage 1)
            mask: Input mask tensor [N, H, W] (must match Stage 1)
            seed: Random seed (must match Stage 1)
            stage2_inference_steps: Denoising steps for Stage 2
            stage2_cfg_strength: CFG strength for Stage 2
            texture_size: Texture resolution
            simplify: Mesh simplification ratio

        Returns:
            Tuple of (glb_filepath, ply_filepath, pose_data)
        """
        print(f"[SAM3DObjects] Stage 2: Resuming from Stage 1 output")
        print(f"[SAM3DObjects] Stage 2 parameters: steps={stage2_inference_steps}, cfg={stage2_cfg_strength}")
        print(f"[SAM3DObjects] Postprocess parameters: texture_size={texture_size}, simplify={simplify}")

        # Convert ComfyUI tensors to formats expected by SAM3D
        image_pil = comfy_image_to_pil(image)
        mask_np = comfy_mask_to_numpy(mask)

        # Run from Stage 2 onwards using stage1_output
        try:
            print("[SAM3DObjects] Running Stage 2 + postprocessing...")
            output = model(
                image_pil, mask_np,
                seed=seed,
                stage1_output=stage1_data,  # CRITICAL: Resume from Stage 1
                stage2_inference_steps=stage2_inference_steps,
                stage2_cfg_strength=stage2_cfg_strength,
                texture_size=texture_size,
                simplify=simplify,
            )

        except Exception as e:
            raise RuntimeError(f"SAM3D Stage 2 inference failed: {e}") from e

        print("[SAM3DObjects] Stage 2 + postprocessing completed!")

        # Extract outputs
        glb_path = output.get("glb_path")
        ply_path = output.get("ply_path")
        pose_data = output.get("metadata", {})

        if glb_path is None:
            raise RuntimeError("GLB file was not generated")
        if ply_path is None:
            raise RuntimeError("PLY file was not generated")

        return (glb_path, ply_path, pose_data)
