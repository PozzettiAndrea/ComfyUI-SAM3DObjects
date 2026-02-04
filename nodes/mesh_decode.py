"""SAM3DMeshDecode node for decoding SLAT to mesh."""

import os
from typing import Any

from comfy_env import isolated


@isolated(env="sam3dobjects", import_paths=[".", "vendor"])
class SAM3DMeshDecode:
    """
    Mesh Decoding.

    Decodes SLAT latent to mesh using the mesh decoder.
    Fast decoding stage (~15 seconds) that produces vertex-colored mesh.

    Optionally applies mesh postprocessing (simplification + hole filling).
    Output GLB file can be passed to SAM3DTextureBake for texture baking.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "slat_decoder_mesh": ("SAM3D_MODEL", {"tooltip": "Mesh decoder from LoadSAM3DModel"}),
                "slat": ("STRING", {"forceInput": True, "tooltip": "Path to SLAT from SAM3DGenerateSLAT"}),
            },
            "optional": {
                "with_postprocess": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Apply mesh simplification + hole filling. Reduces poly count for faster downstream processing."
                }),
                "simplify": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.5,
                    "max": 0.98,
                    "step": 0.01,
                    "tooltip": "Fraction of faces to remove (only used when with_postprocess=True). 0.5 = keep 50% (gentle), 0.95 = keep 5% (aggressive)"
                }),
                "up_axis": (["Y-up (standard)", "Z-up"], {
                    "default": "Y-up (standard)",
                    "tooltip": "Coordinate system for GLB output. Y-up is glTF standard."
                }),
                "world_coordinates": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Output in world coordinates (from depth estimation). Disabled = centered at origin."
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("glb_filepath",)
    OUTPUT_TOOLTIPS = (
        "Path to saved vertex-colored GLB file",
    )
    FUNCTION = "decode_mesh"
    CATEGORY = "SAM3DObjects"
    DESCRIPTION = "Decode SLAT to mesh (~15 seconds). Optionally simplify + fill holes."

    def decode_mesh(
        self,
        slat_decoder_mesh: Any,
        slat: str,
        with_postprocess: bool = False,
        simplify: float = 0.95,
        up_axis: str = "Y-up (standard)",
        world_coordinates: bool = False,
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
            glb_filepath
        """
        # These imports happen in the isolated subprocess
        import os
        import torch
        from pathlib import Path

        from utils.lazy_manager import get_model_manager
        from utils.stages import run_decode_lazy
        from utils.helpers import ensure_decoder_files

        print(f"[SAM3DObjects] MeshDecode: Decoding SLAT to Mesh...")
        if with_postprocess:
            print(f"[SAM3DObjects] MeshDecode: Will apply postprocessing (simplify={simplify})")

        # Derive output_dir from slat path (same directory)
        output_dir = os.path.dirname(slat)

        # Get config path from model
        config_path = slat_decoder_mesh.config_path

        # Ensure decoder files exist (download if missing)
        ensure_decoder_files(config_path, "mesh")

        # Load SLAT
        slat_data = torch.load(slat, weights_only=False)

        # Get lazy manager
        lazy_manager = get_model_manager(config_path, compile=slat_decoder_mesh.compile)

        # Run Mesh decoding
        result = run_decode_lazy(
            lazy_manager,
            slat_data=slat_data,
            decode_format="mesh",
            unload_after=True,
            output_dir=output_dir,
            with_postprocess=with_postprocess,
            simplify=simplify,
            up_axis=up_axis,
            world_coordinates=world_coordinates,
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

        print(f"[SAM3DObjects] MeshDecode completed: {glb_path}")
        return (glb_path,)
