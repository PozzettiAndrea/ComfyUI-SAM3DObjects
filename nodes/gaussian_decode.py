"""SAM3DGaussianDecode node for decoding SLAT to Gaussian splats."""

import os
import torch
from typing import Any

from .utils import (
    comfy_image_to_pil,
    comfy_mask_to_numpy,
)


class SAM3DGaussianDecode:
    """
    Gaussian Decoding.

    Decodes SLAT latent to Gaussian splats using the Gaussian decoder.
    Fast decoding stage (~15 seconds) that produces colored point cloud representation.

    Output can be saved as PLY file or passed to SAM3DTextureBake for texture baking.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "slat_decoder_gs": ("SAM3D_MODEL", {"tooltip": "Gaussian decoder from LoadSAM3DModel"}),
                "slat": ("STRING", {"tooltip": "Path to SLAT from SAM3DSLATGen"}),
                "image": ("IMAGE", {"tooltip": "Input RGB image (must match SLATGen)"}),
                "mask": ("MASK", {"tooltip": "Binary mask (must match SLATGen)"}),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 2**31 - 1,
                    "control_after_generate": "fixed",
                    "tooltip": "Random seed (must match previous stages)"
                }),
            },
            "optional": {
                "save_ply": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Save Gaussian as PLY file"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "SAM3D_GAUSSIAN")
    RETURN_NAMES = ("ply_filepath", "gaussian_data")
    OUTPUT_TOOLTIPS = (
        "Path to saved Gaussian PLY file (if save_ply=True)",
        "Gaussian data - pass to SAM3DTextureBake"
    )
    FUNCTION = "decode_gaussian"
    CATEGORY = "SAM3DObjects"
    DESCRIPTION = "Decode SLAT to Gaussian splats (~15 seconds)."

    def decode_gaussian(
        self,
        slat_decoder_gs: Any,
        slat: str,
        image: torch.Tensor,
        mask: torch.Tensor,
        seed: int,
        save_ply: bool = True,
    ):
        """
        Decode SLAT to Gaussian splats.

        Args:
            slat_decoder_gs: SAM3D Gaussian decoder
            slat: Path to SLAT from SAM3DSLATGen
            image: Input image tensor [B, H, W, C]
            mask: Input mask tensor [N, H, W]
            seed: Random seed
            save_ply: Whether to save PLY file

        Returns:
            Tuple of (ply_filepath, gaussian_data)
        """
        print(f"[SAM3DObjects] GaussianDecode: Decoding SLAT to Gaussian...")

        # Convert ComfyUI tensors to formats expected by SAM3D
        image_pil = comfy_image_to_pil(image)
        mask_np = comfy_mask_to_numpy(mask)

        # Derive output_dir from slat path (same directory)
        output_dir = os.path.dirname(slat)

        # Run Gaussian decoding only
        try:
            gaussian_output = slat_decoder_gs(
                image_pil, mask_np,
                seed=seed,
                slat_output=slat,  # Resume from SLAT path
                gaussian_only=True,  # CRITICAL: Only decode to Gaussian
                save_files=save_ply,  # Save PLY if requested
                output_dir=output_dir,  # Use same directory as SLAT
            )

        except Exception as e:
            raise RuntimeError(f"SAM3D Gaussian decode failed: {e}") from e

        # Extract PLY path if saved
        ply_path = gaussian_output.get("ply_path", None)

        # If not found directly, check files dict (bridge returns this structure)
        if not ply_path and "files" in gaussian_output and "ply" in gaussian_output["files"]:
             ply_path = gaussian_output["files"]["ply"]

        if save_ply and ply_path:
            print(f"[SAM3DObjects] GaussianDecode completed: {ply_path}")
        elif save_ply:
            print(f"[SAM3DObjects] Warning: PLY file not saved")
        else:
            print(f"[SAM3DObjects] GaussianDecode completed")

        # Return PLY path and Gaussian data
        return (ply_path, gaussian_output)
