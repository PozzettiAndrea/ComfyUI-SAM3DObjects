"""SAM3DGaussianDecode node for decoding SLAT to Gaussian splats."""

import os
from typing import Any

from comfyui_isolation import isolated

from .load_model import LoadSAM3DModel


@isolated(
    env="sam3dobjects",
    config="comfyui_isolation_reqs.toml",
    import_paths=[".", "../vendor"],
    timeout=300.0,  # 5 minutes for Gaussian decode
)
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
        from pathlib import Path

        from worker.lazy_manager import get_model_manager
        from worker.stages import run_decode_lazy

        print(f"[SAM3DObjects] GaussianDecode: Decoding SLAT to Gaussian...")

        # Derive output_dir from slat path (same directory)
        output_dir = os.path.dirname(slat)

        # Get config path from model
        config_path = slat_decoder_gs.config_path
        config_dir = str(Path(config_path).parent)

        # Load SLAT
        slat_data = torch.load(slat, weights_only=False)

        # Get lazy manager
        lazy_manager = get_model_manager(config_dir, compile=slat_decoder_gs.compile)

        # Run Gaussian decoding
        result = run_decode_lazy(
            lazy_manager,
            slat_data={"slat": slat_data},
            decode_format="gaussian",
            unload_after=True,
            output_dir=output_dir,
            up_axis=up_axis,
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

        print(f"[SAM3DObjects] GaussianDecode completed: {ply_path}")
        return (ply_path,)
