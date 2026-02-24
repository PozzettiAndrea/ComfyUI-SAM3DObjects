"""SAM3DGaussianDecode node for decoding SLAT to Gaussian splats."""

import logging
import os
from typing import Any

log = logging.getLogger("sam3dobjects")

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
                "slat": ("STRING", {"forceInput": True, "tooltip": "Path to SLAT from SAM3DGenerateSLAT"}),
            },
            "optional": {
                "up_axis": (["Y-up (standard)", "Z-up"], {
                    "default": "Y-up (standard)",
                    "tooltip": "Coordinate system for PLY output. Y-up is common for viewers."
                }),
                "world_coordinates": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Output in world coordinates (from depth estimation). Disabled = centered at origin."
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
        world_coordinates: bool = False,
    ):
        """
        Decode SLAT to Gaussian splats.

        This method runs in an isolated subprocess with its own Python environment.

        Args:
            slat_decoder_gs: SAM3DModelConfig (provides config_path)
            slat: Path to SLAT from SAM3DGenerateSLAT

        Returns:
            ply_filepath
        """
        # These imports happen in the isolated subprocess
        import os
        import torch
        import comfy.utils
        from pathlib import Path

        import folder_paths
        from .utils.stages import run_decode
        from .utils.helpers import ensure_decoder_files

        log.info("GaussianDecode: Decoding SLAT to Gaussian...")

        # Resolve path to absolute (handles both absolute and relative inputs)
        if not os.path.isabs(slat):
            comfyui_base = os.path.dirname(folder_paths.get_output_directory())
            slat = os.path.join(comfyui_base, slat)
        output_dir = os.path.dirname(slat)

        # Get config path from model
        config_path = slat_decoder_gs["config_path"]

        # Ensure decoder files exist (download if missing)
        ensure_decoder_files(config_path, "gaussian")

        # Load SLAT (our own intermediate file, not an untrusted checkpoint)
        slat_data = torch.load(slat, weights_only=False)

        # Run Gaussian decoding
        result = run_decode(
            config_path,
            slat_data=slat_data,
            decode_format="gaussian",
            output_dir=output_dir,
            up_axis=up_axis,
            world_coordinates=world_coordinates,
            precision=slat_decoder_gs.get("precision", "bf16"),
        )

        # Extract PLY path from result
        ply_path = None

        # Check file_output structure (from run_decode_lazy)
        if "file_output" in result:
            file_output = result["file_output"]
            if isinstance(file_output, dict) and "files" in file_output:
                ply_path = file_output["files"].get("ply")

        # Fallback: check direct files dict
        if not ply_path and "files" in result:
            ply_path = result["files"].get("ply")

        if not ply_path:
            raise RuntimeError("PLY file was not generated")

        log.info("GaussianDecode completed: %s", ply_path)
        return (ply_path,)
