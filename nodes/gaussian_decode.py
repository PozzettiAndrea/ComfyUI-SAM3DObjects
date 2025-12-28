"""SAM3DGaussianDecode node for decoding SLAT to Gaussian splats."""

import os
from typing import Any


class SAM3DGaussianDecode:
    """
    Gaussian Decoding.

    Decodes SLAT latent to Gaussian splats using the Gaussian decoder.
    Fast decoding stage (~15 seconds) that produces colored point cloud representation.

    Output PLY file can be passed to SAM3DTextureBake for texture baking.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "slat_decoder_gs": ("SAM3D_MODEL", {"tooltip": "Gaussian decoder from LoadSAM3DModel"}),
                "slat": ("STRING", {"tooltip": "Path to SLAT from SAM3DSLATGen"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("ply_filepath",)
    OUTPUT_TOOLTIPS = (
        "Path to saved Gaussian PLY file",
    )
    FUNCTION = "decode_gaussian"
    CATEGORY = "SAM3DObjects"
    DESCRIPTION = "Decode SLAT to Gaussian splats (~15 seconds)."

    def decode_gaussian(
        self,
        slat_decoder_gs: Any,
        slat: str,
    ):
        """
        Decode SLAT to Gaussian splats.

        Args:
            slat_decoder_gs: SAM3D Gaussian decoder
            slat: Path to SLAT from SAM3DSLATGen

        Returns:
            ply_filepath
        """
        print(f"[SAM3DObjects] GaussianDecode: Decoding SLAT to Gaussian...")

        # Derive output_dir from slat path (same directory)
        output_dir = os.path.dirname(slat)

        # Run Gaussian decoding
        try:
            gaussian_output = slat_decoder_gs(
                slat_output=slat,  # SLAT path
                gaussian_only=True,  # Only decode to Gaussian
                save_files=True,  # Always save PLY
                output_dir=output_dir,  # Use same directory as SLAT
            )

        except Exception as e:
            raise RuntimeError(f"SAM3D Gaussian decode failed: {e}") from e

        # Extract PLY path
        ply_path = gaussian_output.get("ply_path", None)

        # Check files dict (bridge returns this structure)
        if not ply_path and "files" in gaussian_output and "ply" in gaussian_output["files"]:
            ply_path = gaussian_output["files"]["ply"]

        if not ply_path:
            raise RuntimeError("PLY file was not generated")

        print(f"[SAM3DObjects] GaussianDecode completed: {ply_path}")
        return (ply_path,)
