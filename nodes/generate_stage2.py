"""SAM3DSLATGen node for SLAT generation via diffusion."""

import torch
from typing import Any

from .utils import (
    comfy_image_to_pil,
    comfy_mask_to_numpy,
)


class SAM3DSLATGen:
    """
    SLAT Generation.

    Generates SLAT (Sparse LATent) using diffusion model conditioned on sparse structure.
    This is the expensive diffusion stage (~60 seconds) that produces the latent representation.

    Output can be decoded to Gaussian/Mesh using SAM3DGaussianDecode / SAM3DMeshDecode nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SAM3D_MODEL", {"tooltip": "SAM3D model loaded from checkpoint"}),
                "sparse_structure": ("SAM3D_SPARSE", {"tooltip": "Sparse structure from SAM3DSparseGen"}),
                "image": ("IMAGE", {"tooltip": "Input RGB image (must match SparseGen)"}),
                "mask": ("MASK", {"tooltip": "Binary mask (must match SparseGen)"}),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 2**31 - 1,
                    "control_after_generate": "fixed",
                    "tooltip": "Random seed (must match SparseGen for consistency)"
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
            }
        }

    RETURN_TYPES = ("SAM3D_SLAT",)
    RETURN_NAMES = ("slat",)
    OUTPUT_TOOLTIPS = ("SLAT latent - pass to SAM3DGaussianDecode or SAM3DMeshDecode",)
    FUNCTION = "generate_slat"
    CATEGORY = "SAM3DObjects"
    DESCRIPTION = "Generate SLAT latents via diffusion (~60 seconds)."

    def generate_slat(
        self,
        model: Any,
        sparse_structure: dict,
        image: torch.Tensor,
        mask: torch.Tensor,
        seed: int,
        stage2_inference_steps: int = 25,
        stage2_cfg_strength: float = 5.0,
    ):
        """
        Generate SLAT latents via diffusion.

        Args:
            model: SAM3D inference pipeline
            sparse_structure: Sparse structure from SAM3DSparseGen
            image: Input image tensor [B, H, W, C] (must match SparseGen)
            mask: Input mask tensor [N, H, W] (must match SparseGen)
            seed: Random seed (must match SparseGen)
            stage2_inference_steps: Denoising steps for SLAT generation
            stage2_cfg_strength: CFG strength for SLAT generation

        Returns:
            Tuple of (slat,) - SLAT latent for decoder nodes
        """
        print(f"[SAM3DObjects] SLATGen: Generating SLAT from sparse structure")
        print(f"[SAM3DObjects] SLAT parameters: steps={stage2_inference_steps}, cfg={stage2_cfg_strength}")

        # Convert ComfyUI tensors to formats expected by SAM3D
        image_pil = comfy_image_to_pil(image)
        mask_np = comfy_mask_to_numpy(mask)

        # Run SLAT generation only (no decoding)
        try:
            print("[SAM3DObjects] Running SLAT generation...")
            slat_output = model(
                image_pil, mask_np,
                seed=seed,
                stage1_output=sparse_structure,  # Resume from sparse generation
                slat_only=True,  # CRITICAL: Only generate SLAT, skip decoding
                stage2_inference_steps=stage2_inference_steps,
                stage2_cfg_strength=stage2_cfg_strength,
            )

        except Exception as e:
            raise RuntimeError(f"SAM3D SLAT generation failed: {e}") from e

        print("[SAM3DObjects] SLAT generation completed!")
        print(f"[SAM3DObjects] - SLAT latent ready for decoding")

        # Return the SLAT as an opaque object
        # ComfyUI will pass this Python dict through to decoder nodes
        return (slat_output,)
