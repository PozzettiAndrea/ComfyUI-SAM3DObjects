"""SAM3DGaussianDecode node for decoding SLAT to Gaussian splats."""

import os
from pathlib import Path
from typing import Any

from .subprocess_bridge import get_bridge, run_decode
from .load_model import LoadSAM3DModel, REQUIRED_FILES


class SAM3DGaussianDecode:
    """
    Gaussian Decoding.

    Decodes SLAT latent to Gaussian splats using the Gaussian decoder.
    Fast decoding stage (~15 seconds) that produces colored point cloud representation.

    Output PLY file can be passed to SAM3DTextureBake for texture baking.
    """

    @classmethod
    def get_bridge(cls):
        return get_bridge()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "slat_decoder_gs": ("SAM3D_MODEL", {"tooltip": "Gaussian decoder from LoadSAM3DModel"}),
                "slat": ("STRING", {"forceInput": True, "tooltip": "Path to SLAT from SAM3DGenerateSLAT"}),
            },
            "optional": {
                "up_axis": (["Y-up (standard)", "Z-up"], {
                    "default": "Y-up (standard)",
                    "tooltip": "Coordinate system for PLY output. Y-up is common for viewers."
                }),
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
        up_axis: str = "Y-up (standard)",
    ):
        """
        Decode SLAT to Gaussian splats.

        Args:
            slat_decoder_gs: SAM3D model (provides config_path)
            slat: Path to SLAT from SAM3DGenerateSLAT

        Returns:
            ply_filepath
        """
        print(f"[SAM3DObjects] GaussianDecode: Decoding SLAT to Gaussian...")

        # Ensure Gaussian decoder files are downloaded
        # (LoadSAM3DModel might have been cached without downloading these)
        LoadSAM3DModel._get_or_download_checkpoint({"slat_decoder_gs"})

        # Derive output_dir from slat path (same directory)
        output_dir = os.path.dirname(slat)

        # Get config path from model and bridge
        config_path = slat_decoder_gs.config_path
        bridge = self.get_bridge()
        bridge.start_worker()

        # Run Gaussian decoding via dedicated decode command
        try:
            result = run_decode(
                bridge=bridge,
                config_path=config_path,
                slat_path=slat,
                output_dir=output_dir,
                decode_format="gaussian",
                up_axis=up_axis,
            )
        except Exception as e:
            raise RuntimeError(f"SAM3D Gaussian decode failed: {e}") from e

        # Extract PLY path from nested result structure
        ply_path = result.get("ply_path", None)

        # Check nested output.files structure (from run_decode_lazy)
        if not ply_path and "output" in result:
            output = result["output"]
            if isinstance(output, dict) and "files" in output:
                ply_path = output["files"].get("ply")

        # Check file_output.files structure (alternative key)
        if not ply_path and "file_output" in result:
            file_output = result["file_output"]
            if isinstance(file_output, dict) and "files" in file_output:
                ply_path = file_output["files"].get("ply")

        # Fallback: check direct files dict
        if not ply_path and "files" in result and "ply" in result["files"]:
            ply_path = result["files"]["ply"]

        if not ply_path:
            raise RuntimeError("PLY file was not generated")

        print(f"[SAM3DObjects] GaussianDecode completed: {ply_path}")
        return (ply_path,)
