"""SAM3DTextureBake node for texture baking."""

import logging
import os
from typing import Any

log = logging.getLogger("sam3dobjects")

class SAM3DTextureBake:
    """
    Texture Baking.

    Bakes Gaussian appearance into mesh UV textures.
    Uses gradient descent optimization ('opt') or fast nearest neighbor ('fast').

    Requires GLB and PLY file paths as inputs.
    Final stage that produces textured GLB output.

    NOTE: This node does NOT require any models - it directly loads the Gaussian
    and Mesh from files and performs texture baking.

    TIP: Apply mesh simplification in SAM3DMeshDecode BEFORE texture baking
    for faster processing and lower memory usage.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "glb_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Path to GLB mesh file from SAM3DMeshDecode"
                }),
                "ply_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Path to PLY Gaussian file from SAM3DGaussianDecode"
                }),
            },
            "optional": {
                "texture_mode": (["opt", "fast"], {
                    "default": "opt",
                    "tooltip": "Texture baking mode: 'opt' = gradient descent (30-60s, better quality), 'fast' = nearest neighbor (5s)"
                }),
                "texture_size": ("INT", {
                    "default": 1024,
                    "min": 512,
                    "max": 4096,
                    "step": 512,
                    "tooltip": "Texture resolution. Higher = better quality but more memory"
                }),
                "rendering_engine": (["nvdiffrast", "pytorch3d"], {
                    "default": "nvdiffrast",
                    "tooltip": "Rendering backend for texture baking. nvdiffrast = faster/better quality, pytorch3d = fallback"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("glb_filepath",)
    OUTPUT_TOOLTIPS = (
        "Path to saved textured GLB mesh file",
    )
    FUNCTION = "bake_texture"
    CATEGORY = "SAM3DObjects"
    DESCRIPTION = "Bake Gaussian appearance into mesh UV textures. 'opt' mode: 30-60s, 'fast' mode: ~5s."

    def bake_texture(
        self,
        glb_path: str,
        ply_path: str,
        texture_mode: str = "opt",
        texture_size: int = 1024,
        rendering_engine: str = "nvdiffrast",
    ):
        """
        Bake Gaussian appearance into mesh UV textures.

        This method runs in an isolated subprocess with its own Python environment.

        Args:
            glb_path: Path to input GLB mesh file
            ply_path: Path to input PLY Gaussian file
            texture_mode: Texture baking mode ("opt" or "fast")
            texture_size: Texture resolution
            rendering_engine: Rendering backend ("pytorch3d" or "nvdiffrast")

        Returns:
            Tuple of (glb_filepath,)
        """
        # These imports happen in the isolated subprocess
        import os

        from .utils.stages import run_texture_bake_direct

        log.info("TextureBake: Baking textures (mode=%s, size=%d)", texture_mode, texture_size)

        # Validate file paths
        if not glb_path or not ply_path:
            raise RuntimeError("Both glb_path and ply_path are required")

        if not os.path.exists(glb_path):
            raise RuntimeError(f"GLB file not found: {glb_path}")
        if not os.path.exists(ply_path):
            raise RuntimeError(f"PLY file not found: {ply_path}")

        # Derive output_dir from glb_path (same directory)
        output_dir = os.path.dirname(glb_path)

        # Run texture baking (no models needed!)
        output = run_texture_bake_direct({
            "ply_path": ply_path,
            "glb_path": glb_path,
            "output_dir": output_dir,
            "texture_mode": texture_mode,
            "texture_size": texture_size,
            "rendering_engine": rendering_engine,
        })

        # Extract outputs (nested under "output" key from run_texture_bake_direct)
        output_glb_path = output.get("output", {}).get("glb_path")
        if output_glb_path is None:
            raise RuntimeError("GLB file was not generated")

        log.info("TextureBake completed: %s", output_glb_path)
        return (output_glb_path,)
