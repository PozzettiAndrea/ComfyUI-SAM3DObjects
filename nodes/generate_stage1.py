"""SAM3DSparseGen node for generating sparse 3D structure."""

import torch
from typing import Any

from .utils import (
    comfy_image_to_pil,
    comfy_mask_to_numpy,
)


class SAM3DSparseGen:
    """
    Sparse Structure Generation.

    Generates sparse voxel coordinates using the sparse structure diffusion model.
    Fast stage (~3 seconds) that produces the structural skeleton for 3D generation.

    Output can be passed to SAM3DSLATGen to continue the pipeline.
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
                    "control_after_generate": "fixed",
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

    RETURN_TYPES = ("SAM3D_SPARSE",)
    RETURN_NAMES = ("sparse_structure",)
    OUTPUT_TOOLTIPS = ("Sparse voxel structure - pass to SAM3DSLATGen",)
    FUNCTION = "generate_sparse"
    CATEGORY = "SAM3DObjects"
    DESCRIPTION = "Generate sparse voxel structure from image and mask (~3 seconds)."

    def generate_sparse(
        self,
        model: Any,
        image: torch.Tensor,
        mask: torch.Tensor,
        seed: int,
        stage1_inference_steps: int = 25,
        stage1_cfg_strength: float = 7.0,
    ):
        """
        Generate sparse voxel structure.

        Args:
            model: SAM3D inference pipeline
            image: Input image tensor [B, H, W, C]
            mask: Input mask tensor [N, H, W]
            seed: Random seed
            stage1_inference_steps: Denoising steps for Stage 1
            stage1_cfg_strength: CFG strength for Stage 1

        Returns:
            Tuple of (sparse_structure,) - sparse voxel data for SAM3DSLATGen
        """
        print(f"[SAM3DObjects] SparseGen: Generating sparse structure (seed: {seed})")

        # Convert ComfyUI tensors to formats expected by SAM3D
        image_pil = comfy_image_to_pil(image)
        mask_np = comfy_mask_to_numpy(mask)

        print(f"[SAM3DObjects] Image size: {image_pil.size}")
        print(f"[SAM3DObjects] Mask shape: {mask_np.shape}")
        print(f"[SAM3DObjects] Sparse parameters: steps={stage1_inference_steps}, cfg={stage1_cfg_strength}")

        # Run sparse structure generation only
        try:
            print("[SAM3DObjects] Running sparse structure generation...")
            sparse_output = model(
                image_pil, mask_np,
                seed=seed,
                stage1_only=True,  # CRITICAL: Only run sparse generation
                stage1_inference_steps=stage1_inference_steps,
                stage1_cfg_strength=stage1_cfg_strength,
            )

        except Exception as e:
            raise RuntimeError(f"SAM3D sparse generation failed: {e}") from e

        print("[SAM3DObjects] Sparse structure generation completed!")
        print(f"[SAM3DObjects] - Generated {sparse_output.get('coords', torch.tensor([])).shape[0]} voxels")

        # Return the sparse structure as an opaque object
        # ComfyUI will pass this Python dict through to the next node
        return (sparse_output,)
