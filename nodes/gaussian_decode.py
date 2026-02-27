"""SAM3DGaussianDecode node for decoding SLAT to Gaussian splats."""

import logging
import os
from typing import Any

import torch
from comfy_api.latest import io

log = logging.getLogger("sam3dobjects")

class SAM3DGaussianDecode(io.ComfyNode):
    """
    Gaussian Decoding.

    Decodes SLAT latent to Gaussian splats using the Gaussian decoder.
    Fast decoding stage (~15 seconds) that produces colored point cloud representation.

    Output PLY file can be passed to SAM3DTextureBake for texture baking.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SAM3DGaussianDecode",
            category="SAM3DObjects",
            description="Decode SLAT to Gaussian splats (~15 seconds).",
            inputs=[
                io.Custom("SAM3D_MODEL").Input("slat_decoder_gs", tooltip="Gaussian decoder from LoadSAM3DModel"),
                io.String.Input("slat", forceInput=True, tooltip="Path to SLAT from SAM3DGenerateSLAT"),
                io.Combo.Input("up_axis", options=["Y-up (standard)", "Z-up"],
                    default="Y-up (standard)",
                    tooltip="Coordinate system for PLY output. Y-up is common for viewers.",
                    optional=True,
                ),
                io.Boolean.Input("world_coordinates",
                    default=False,
                    tooltip="Output in world coordinates (from depth estimation). Disabled = centered at origin.",
                    optional=True,
                ),
            ],
            outputs=[
                io.String.Output(display_name="ply_filepath", tooltip="Path to saved Gaussian PLY file"),
            ],
        )

    @classmethod
    @torch.no_grad()
    def execute(
        cls,
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
            NodeOutput of ply_filepath
        """
        # These imports happen in the isolated subprocess
        import os
        import torch
        import comfy.utils
        from pathlib import Path

        import folder_paths
        from .utils.stages import run_decode
        from .utils.helpers import ensure_decoder_files
        from .utils.vram_log import vram

        log.info("GaussianDecode: Decoding SLAT to Gaussian...")
        vram("GaussianDecode: start")

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
        slat_data = comfy.utils.load_torch_file(slat, safe_load=True)

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

        vram("GaussianDecode: done")
        log.info("GaussianDecode completed: %s", ply_path)
        return io.NodeOutput(ply_path,)
