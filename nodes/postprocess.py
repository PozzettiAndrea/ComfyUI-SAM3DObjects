"""SAM3DTextureBake node for texture baking and mesh postprocessing."""

import os
import torch
from typing import Any

from .subprocess_bridge import run_texture_bake_direct


class SAM3DTextureBake:
    """
    Texture Baking.

    Bakes Gaussian appearance into mesh UV textures using gradient descent optimization.
    Also performs mesh simplification and optional hole filling.

    Requires GLB and PLY file paths as inputs.
    Final stage that produces textured GLB output (~30-60 seconds).

    NOTE: This node does NOT require any models - it directly loads the Gaussian
    and Mesh from files and performs texture baking.
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
                "with_mesh_postprocess": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Simplify mesh + fill holes. Enable for faster texture baking; disable to preserve full mesh detail."
                }),
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
                "simplify": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.9,
                    "max": 0.98,
                    "step": 0.01,
                    "tooltip": "Mesh simplification ratio (0.9 = aggressive, 0.98 = gentle)"
                }),
                "rendering_engine": (["nvdiffrast", "pytorch3d"], {
                    "default": "nvdiffrast",
                    "tooltip": "Rendering backend for texture baking. nvdiffrast = faster/better quality, pytorch3d = fallback"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("glb_filepath", "ply_filepath")
    OUTPUT_TOOLTIPS = (
        "Path to saved textured GLB mesh file",
        "Path to PLY Gaussian file (unchanged)",
    )
    FUNCTION = "bake_texture"
    CATEGORY = "SAM3DObjects"
    DESCRIPTION = "Bake Gaussian appearance into mesh UV textures (~30-60 seconds). No models required."

    def bake_texture(
        self,
        glb_path: str,
        ply_path: str,
        with_mesh_postprocess: bool = False,
        texture_mode: str = "opt",
        texture_size: int = 1024,
        simplify: float = 0.95,
        rendering_engine: str = "nvdiffrast",
    ):
        """
        Bake Gaussian appearance into mesh UV textures.

        Args:
            glb_path: Path to input GLB mesh file
            ply_path: Path to input PLY Gaussian file
            with_mesh_postprocess: Enable mesh hole filling + cleanup
            texture_mode: Texture baking mode ("opt" or "fast")
            texture_size: Texture resolution
            simplify: Mesh simplification ratio
            rendering_engine: Rendering backend ("pytorch3d" or "nvdiffrast")

        Returns:
            Tuple of (glb_filepath, ply_filepath)
        """
        print(f"[SAM3DObjects] TextureBake: Baking textures (mode={texture_mode}, size={texture_size})")

        # Validate file paths
        if not glb_path or not ply_path:
            raise RuntimeError("Both glb_path and ply_path are required")

        if not os.path.exists(glb_path):
            raise RuntimeError(f"GLB file not found: {glb_path}")
        if not os.path.exists(ply_path):
            raise RuntimeError(f"PLY file not found: {ply_path}")

        # Derive output_dir from glb_path (same directory)
        output_dir = os.path.dirname(glb_path)

        # Run texture baking directly (no models needed!)
        try:
            output = run_texture_bake_direct(
                ply_path=ply_path,
                glb_path=glb_path,
                output_dir=output_dir,
                texture_mode=texture_mode,
                texture_size=texture_size,
                simplify=simplify,
                with_mesh_postprocess=with_mesh_postprocess,
                rendering_engine=rendering_engine,
            )

        except Exception as e:
            raise RuntimeError(f"SAM3D texture baking failed: {e}") from e

        # Extract outputs
        output_glb_path = output.get("glb_path")
        if output_glb_path is None:
            raise RuntimeError("GLB file was not generated")

        print(f"[SAM3DObjects] TextureBake completed: {output_glb_path}")
        return (output_glb_path, ply_path)
