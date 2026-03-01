"""SAM3DMeshDecode node for decoding SLAT to mesh."""

import logging
import os
from typing import Any

import torch
from comfy_api.latest import io

log = logging.getLogger("sam3dobjects")

class SAM3DMeshDecode(io.ComfyNode):
    """
    Mesh Decoding.

    Decodes SLAT latent to mesh using the mesh decoder.
    Fast decoding stage (~15 seconds) that produces vertex-colored mesh.

    Optionally applies mesh postprocessing (simplification + hole filling).
    Output GLB file can be passed to SAM3DTextureBake for texture baking.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SAM3DMeshDecode",
            category="SAM3DObjects",
            description="Decode SLAT to mesh (~15 seconds). Optionally simplify + fill holes.",
            inputs=[
                io.Custom("SAM3D_MODEL").Input("slat_decoder_mesh", tooltip="Mesh decoder from LoadSAM3DModel"),
                io.String.Input("slat", force_input=True, tooltip="Path to SLAT from SAM3DGenerateSLAT"),
                io.Boolean.Input("with_postprocess",
                    default=False,
                    tooltip="Apply mesh simplification + hole filling. Reduces poly count for faster downstream processing.",
                    optional=True,
                ),
                io.Float.Input("simplify",
                    default=0.95,
                    min=0.5,
                    max=0.98,
                    step=0.01,
                    tooltip="Fraction of faces to remove (only used when with_postprocess=True). 0.5 = keep 50% (gentle), 0.95 = keep 5% (aggressive)",
                    optional=True,
                ),
                io.Combo.Input("up_axis", options=["Y-up (standard)", "Z-up"],
                    default="Y-up (standard)",
                    tooltip="Coordinate system for GLB output. Y-up is glTF standard.",
                    optional=True,
                ),
                io.Boolean.Input("world_coordinates",
                    default=False,
                    tooltip="Output in world coordinates (from depth estimation). Disabled = centered at origin.",
                    optional=True,
                ),
                io.Boolean.Input("use_sparse_flexicubes",
                    default=True,
                    tooltip="Use SparseFlex mesh extraction (saves ~2GB VRAM). Disable for dense grid fallback if you see holes.",
                    optional=True,
                ),
            ],
            outputs=[
                io.String.Output(display_name="glb_filepath", tooltip="Path to saved vertex-colored GLB file"),
            ],
        )

    @classmethod
    @torch.no_grad()
    def execute(
        cls,
        slat_decoder_mesh: Any,
        slat: str,
        with_postprocess: bool = False,
        simplify: float = 0.95,
        up_axis: str = "Y-up (standard)",
        world_coordinates: bool = False,
        use_sparse_flexicubes: bool = True,
    ):
        """
        Decode SLAT to mesh.

        This method runs in an isolated subprocess with its own Python environment.

        Args:
            slat_decoder_mesh: SAM3DModelConfig (provides config_path)
            slat: Path to SLAT from SAM3DGenerateSLAT
            with_postprocess: Apply mesh simplification + hole filling
            simplify: Fraction of faces to remove (0.5-0.98)

        Returns:
            NodeOutput of glb_filepath
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

        log.info("MeshDecode: Decoding SLAT to Mesh...")
        vram("MeshDecode: start")
        if with_postprocess:
            log.info("MeshDecode: Will apply postprocessing (simplify=%s)", simplify)

        # Resolve path to absolute (handles both absolute and relative inputs)
        if not os.path.isabs(slat):
            comfyui_base = os.path.dirname(folder_paths.get_output_directory())
            slat = os.path.join(comfyui_base, slat)
        output_dir = os.path.dirname(slat)

        # Get config path from model
        config_path = slat_decoder_mesh["config_path"]

        # Ensure decoder files exist (download if missing)
        ensure_decoder_files(config_path, "mesh")

        # Load SLAT (our own intermediate file, not an untrusted checkpoint)
        slat_data = comfy.utils.load_torch_file(slat, safe_load=True)

        # Run Mesh decoding
        result = run_decode(
            config_path,
            slat_data=slat_data,
            decode_format="mesh",
            output_dir=output_dir,
            with_postprocess=with_postprocess,
            simplify=simplify,
            up_axis=up_axis,
            world_coordinates=world_coordinates,
            precision=slat_decoder_mesh.get("precision", "bf16"),
            use_sparse_flexicubes=use_sparse_flexicubes,
        )

        # Extract GLB path from result
        glb_path = None

        # Check file_output structure (from run_decode_lazy)
        if "file_output" in result:
            file_output = result["file_output"]
            if isinstance(file_output, dict) and "files" in file_output:
                glb_path = file_output["files"].get("glb")

        # Fallback: check direct files dict
        if not glb_path and "files" in result:
            glb_path = result["files"].get("glb")

        if not glb_path:
            raise RuntimeError("GLB file was not generated")

        vram("MeshDecode: done")
        log.info("MeshDecode completed: %s", glb_path)
        return io.NodeOutput(glb_path,)
