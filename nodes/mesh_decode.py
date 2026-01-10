"""SAM3DMeshDecode node for decoding SLAT to mesh."""

import os
from pathlib import Path
from typing import Any

from .subprocess_bridge import get_bridge, run_decode
from .load_model import LoadSAM3DModel


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
    ):
        """
        Decode SLAT to mesh.

        Args:
            slat_decoder_mesh: SAM3D model (provides config_path)
            slat: Path to SLAT from SAM3DGenerateSLAT
            with_postprocess: Apply mesh simplification + hole filling
            simplify: Fraction of faces to remove (0.5-0.98)

        Returns:
            glb_filepath
        """
        print(f"[SAM3DObjects] MeshDecode: Decoding SLAT to Mesh...")
        if with_postprocess:
            print(f"[SAM3DObjects] MeshDecode: Will apply postprocessing (simplify={simplify})")

        # Ensure Mesh decoder files are downloaded
        # (LoadSAM3DModel might have been cached without downloading these)
        LoadSAM3DModel._get_or_download_checkpoint({"slat_decoder_mesh"})

        # Derive output_dir from slat path (same directory)
        output_dir = os.path.dirname(slat)

        # Get config path from model and bridge
        config_path = slat_decoder_mesh.config_path
        bridge = get_bridge()

        # Run Mesh decoding via dedicated decode command
        try:
            result = run_decode(
                bridge=bridge,
                config_path=config_path,
                slat_path=slat,
                output_dir=output_dir,
                decode_format="mesh",
                with_postprocess=with_postprocess,
                simplify=simplify,
                up_axis=up_axis,
            )
        except Exception as e:
            raise RuntimeError(f"SAM3D Mesh decode failed: {e}") from e

        # Extract GLB path from nested result structure
        glb_path = result.get("glb_path", None)

        # Check nested output.files structure (from run_decode_lazy)
        if not glb_path and "output" in result:
            output = result["output"]
            if isinstance(output, dict) and "files" in output:
                glb_path = output["files"].get("glb")

        # Check file_output.files structure (alternative key)
        if not glb_path and "file_output" in result:
            file_output = result["file_output"]
            if isinstance(file_output, dict) and "files" in file_output:
                glb_path = file_output["files"].get("glb")

        # Fallback: check direct files dict
        if not glb_path and "files" in result and "glb" in result["files"]:
            glb_path = result["files"]["glb"]

        if not glb_path:
            raise RuntimeError("GLB file was not generated")

        print(f"[SAM3DObjects] MeshDecode completed: {glb_path}")
        return (glb_path,)
