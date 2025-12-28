"""SAM3DMeshDecode node for decoding SLAT to mesh."""

import os
from typing import Any


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
                "slat": ("STRING", {"tooltip": "Path to SLAT from SAM3DSLATGen"}),
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
    ):
        """
        Decode SLAT to mesh.

        Args:
            slat_decoder_mesh: SAM3D mesh decoder
            slat: Path to SLAT from SAM3DSLATGen
            with_postprocess: Apply mesh simplification + hole filling
            simplify: Fraction of faces to remove (0.5-0.98)

        Returns:
            glb_filepath
        """
        print(f"[SAM3DObjects] MeshDecode: Decoding SLAT to Mesh...")
        if with_postprocess:
            print(f"[SAM3DObjects] MeshDecode: Will apply postprocessing (simplify={simplify})")

        # Derive output_dir from slat path (same directory)
        output_dir = os.path.dirname(slat)

        # Run Mesh decoding
        try:
            mesh_output = slat_decoder_mesh(
                slat_output=slat,  # SLAT path
                mesh_only=True,  # Only decode to Mesh
                save_files=True,  # Always save GLB
                use_vertex_color=True,  # Use vertex colors (no texture baking)
                output_dir=output_dir,  # Use same directory as SLAT
                with_mesh_postprocess=with_postprocess,  # Simplify + fill holes
                simplify=simplify,  # Simplification ratio
            )

        except Exception as e:
            raise RuntimeError(f"SAM3D Mesh decode failed: {e}") from e

        # Extract GLB path
        glb_path = mesh_output.get("glb_path", None)

        # Check files dict (bridge returns this structure)
        if not glb_path and "files" in mesh_output and "glb" in mesh_output["files"]:
            glb_path = mesh_output["files"]["glb"]

        if not glb_path:
            raise RuntimeError("GLB file was not generated")

        print(f"[SAM3DObjects] MeshDecode completed: {glb_path}")
        return (glb_path,)
