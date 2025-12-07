"""SAM3DSLATGen node for SLAT generation via diffusion."""

import os
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
                "slat_generator": ("SAM3D_MODEL", {"tooltip": "SLAT generator from LoadSAM3DModel"}),
                "sparse_structure": ("STRING", {"tooltip": "Path to sparse structure from SAM3DSparseGen"}),
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
                "use_stage2_distillation": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable distillation mode for faster inference. Disables CFG guidance but uses learned shortcuts."
                }),
                "merge_mask_with_image": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If False, skip merging mask with image. Useful when mask size doesn't match image size (e.g., 64x64 mask with 1500x1500 image)."
                }),
                "auto_resize_mask": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If True, automatically resize mask to match image size if they differ. If False, will raise error on size mismatch."
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("slat_path",)
    OUTPUT_TOOLTIPS = ("Path to saved SLAT latent - pass to SAM3DGaussianDecode or SAM3DMeshDecode",)
    FUNCTION = "generate_slat"
    CATEGORY = "SAM3DObjects"
    DESCRIPTION = "Generate SLAT latents via diffusion (~60 seconds)."

    def generate_slat(
        self,
        slat_generator: Any,
        sparse_structure: str,
        image: torch.Tensor,
        mask: torch.Tensor,
        seed: int,
        stage2_inference_steps: int = 25,
        stage2_cfg_strength: float = 5.0,
        use_stage2_distillation: bool = False,
        merge_mask_with_image: bool = True,
        auto_resize_mask: bool = True,
    ):
        """
        Generate SLAT latents via diffusion.

        Args:
            slat_generator: SAM3D SLAT generator
            sparse_structure: Path to sparse structure from SAM3DSparseGen
            image: Input image tensor [B, H, W, C] (must match SparseGen)
            mask: Input mask tensor [N, H, W] (must match SparseGen)
            seed: Random seed (must match SparseGen)
            stage2_inference_steps: Denoising steps for SLAT generation
            stage2_cfg_strength: CFG strength for SLAT generation
            use_stage2_distillation: Enable distillation mode for faster inference

        Returns:
            Tuple of (slat_path,) - path to SLAT latent for decoder nodes
        """
        print(f"[SAM3DObjects] SLATGen: Generating SLAT...")

        # Convert ComfyUI tensors to formats expected by SAM3D
        image_pil = comfy_image_to_pil(image)
        mask_np = comfy_mask_to_numpy(mask)

        # Derive output_dir from sparse_structure path (same directory)
        output_dir = os.path.dirname(sparse_structure)

        # Run SLAT generation only (no decoding)
        try:
            slat_output = slat_generator(
                image_pil, mask_np,
                seed=seed,
                stage1_output=sparse_structure,  # Resume from sparse generation (now a path)
                slat_only=True,  # CRITICAL: Only generate SLAT, skip decoding
                stage2_inference_steps=stage2_inference_steps,
                stage2_cfg_strength=stage2_cfg_strength,
                use_stage2_distillation=use_stage2_distillation,
                output_dir=output_dir,  # Use same directory as sparse_structure
                merge_mask=merge_mask_with_image,
                auto_resize_mask=auto_resize_mask,
            )

        except Exception as e:
            raise RuntimeError(f"SAM3D SLAT generation failed: {e}") from e

        # Extract file path from output
        if isinstance(slat_output, dict) and "files" in slat_output and "slat" in slat_output["files"]:
            slat_path = slat_output["files"]["slat"]
            print(f"[SAM3DObjects] SLATGen completed: {slat_path}")
            return (slat_path,)
            
        # Fallback/Error
        print(f"[SAM3DObjects] Warning: Could not find file path in output: {slat_output.keys() if isinstance(slat_output, dict) else slat_output}")
        raise RuntimeError("Failed to get SLAT file path from worker")
