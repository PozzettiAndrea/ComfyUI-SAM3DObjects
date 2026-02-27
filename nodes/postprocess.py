"""SAM3DTextureBake node for texture baking."""

import logging
import os
from typing import Any

import torch
from comfy_api.latest import io

log = logging.getLogger("sam3dobjects")

class SAM3DTextureBake(io.ComfyNode):
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
    def define_schema(cls):
        return io.Schema(
            node_id="SAM3DTextureBake",
            category="SAM3DObjects",
            description="Bake Gaussian appearance into mesh UV textures. 'opt' mode: 30-60s, 'fast' mode: ~5s.",
            inputs=[
                io.String.Input("glb_path",
                    default="",
                    multiline=False,
                    tooltip="Path to GLB mesh file from SAM3DMeshDecode"
                ),
                io.String.Input("ply_path",
                    default="",
                    multiline=False,
                    tooltip="Path to PLY Gaussian file from SAM3DGaussianDecode"
                ),
                io.Combo.Input("texture_mode", options=["opt", "fast"],
                    default="opt",
                    tooltip="Texture baking mode: 'opt' = gradient descent (30-60s, better quality), 'fast' = nearest neighbor (5s)",
                    optional=True,
                ),
                io.Int.Input("texture_size",
                    default=1024,
                    min=512,
                    max=4096,
                    step=512,
                    tooltip="Texture resolution. Higher = better quality but more memory",
                    optional=True,
                ),
                io.Combo.Input("rendering_engine", options=["nvdiffrast", "pytorch3d"],
                    default="nvdiffrast",
                    tooltip="Rendering backend for texture baking. nvdiffrast = faster/better quality, pytorch3d = fallback",
                    optional=True,
                ),
            ],
            outputs=[
                io.String.Output(display_name="glb_filepath", tooltip="Path to saved textured GLB mesh file"),
            ],
        )

    @classmethod
    @torch.no_grad()
    def execute(
        cls,
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
            NodeOutput of (glb_filepath,)
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
        return io.NodeOutput(output_glb_path,)
