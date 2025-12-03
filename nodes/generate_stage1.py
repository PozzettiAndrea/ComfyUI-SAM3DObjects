"""SAM3DGenerateStage1 node for generating sparse 3D structure (Stage 1 only)."""

import torch
from typing import Any

from .utils import (
    comfy_image_to_pil,
    comfy_mask_to_numpy,
)


class SAM3DGenerateStage1:
    """
    Generate Stage 1: Sparse 3D Structure.

    This node runs only the first diffusion stage to generate the sparse voxel structure.
    Output can be passed to SAM3DGenerateStage2 for completion.

    Cache-efficient: Allows iterating on Stage 2 parameters without re-running Stage 1.
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
                "stage1_cfg_strength": ("FLOAT", {
                    "default": 7.0,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.1,
                    "tooltip": "Classifier-free guidance strength for Stage 1. Higher = stronger adherence to input image"
                }),
            }
        }

    RETURN_TYPES = ("SAM3D_STAGE1_DATA",)
    RETURN_NAMES = ("stage1_data",)
    OUTPUT_TOOLTIPS = ("Stage 1 output (sparse structure) - pass to SAM3DGenerateStage2",)
    FUNCTION = "generate_stage1"
    CATEGORY = "SAM3DObjects"
    DESCRIPTION = "Generate Stage 1: Sparse 3D structure from image and mask (fast, ~3 seconds)."

    def generate_stage1(
        self,
        model: Any,
        image: torch.Tensor,
        mask: torch.Tensor,
        seed: int,
        stage1_inference_steps: int = 25,
        stage1_cfg_strength: float = 7.0,
    ):
        """
        Generate Stage 1 sparse structure.

        Args:
            model: SAM3D inference pipeline
            image: Input image tensor [B, H, W, C]
            mask: Input mask tensor [N, H, W]
            seed: Random seed
            stage1_inference_steps: Denoising steps for Stage 1
            stage1_cfg_strength: CFG strength for Stage 1

        Returns:
            Tuple of (stage1_data,) - intermediate data for Stage 2
        """
        print(f"[SAM3DObjects] Stage 1: Generating sparse structure (seed: {seed})")

        # Convert ComfyUI tensors to formats expected by SAM3D
        image_pil = comfy_image_to_pil(image)
        mask_np = comfy_mask_to_numpy(mask)

        print(f"[SAM3DObjects] Image size: {image_pil.size}")
        print(f"[SAM3DObjects] Mask shape: {mask_np.shape}")
        print(f"[SAM3DObjects] Stage 1 parameters: steps={stage1_inference_steps}, cfg={stage1_cfg_strength}")

        # Run Stage 1 only
        try:
            print("[SAM3DObjects] Running Stage 1 inference...")
            stage1_output = model(
                image_pil, mask_np,
                seed=seed,
                stage1_only=True,  # CRITICAL: Only run Stage 1
                stage1_inference_steps=stage1_inference_steps,
                stage1_cfg_strength=stage1_cfg_strength,
            )

        except Exception as e:
            raise RuntimeError(f"SAM3D Stage 1 inference failed: {e}") from e

        print("[SAM3DObjects] Stage 1 completed!")
        print(f"[SAM3DObjects] - Sparse structure generated with {stage1_output.get('coords', torch.tensor([])).shape[0]} voxels")

        # Return the raw Stage 1 output as an opaque object
        # ComfyUI will pass this Python dict through to the next node
        return (stage1_output,)
