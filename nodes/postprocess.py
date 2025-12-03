"""SAM3DTextureBake node for texture baking and mesh postprocessing."""

import torch
from typing import Any

from .utils import (
    comfy_image_to_pil,
    comfy_mask_to_numpy,
)


class SAM3DTextureBake:
    """
    Texture Baking.

    Bakes Gaussian appearance into mesh UV textures using gradient descent optimization.
    Also performs mesh simplification and optional hole filling.

    Requires BOTH Gaussian and Mesh inputs from decoder nodes.
    Final stage that produces textured GLB output (~30-60 seconds).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "embedders": ("SAM3D_MODEL", {"tooltip": "Embedders from LoadSAM3DModel (for preprocessing)"}),
                "gaussian_data": ("SAM3D_GAUSSIAN", {"tooltip": "Gaussian from SAM3DGaussianDecode"}),
                "mesh_data": ("SAM3D_MESH", {"tooltip": "Mesh from SAM3DMeshDecode"}),
                "image": ("IMAGE", {"tooltip": "Input RGB image (must match previous stages)"}),
                "mask": ("MASK", {"tooltip": "Binary mask (must match previous stages)"}),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 2**31 - 1,
                    "control_after_generate": "fixed",
                    "tooltip": "Random seed (must match previous stages)"
                }),
            },
            "optional": {
                "with_mesh_postprocess": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable expensive mesh hole filling + cleanup (~30s extra)"
                }),
                "with_texture_baking": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Bake Gaussian appearance into UV texture. If False, uses vertex colors (faster, lower quality)"
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
                "rendering_engine": (["pytorch3d", "nvdiffrast"], {
                    "default": "pytorch3d",
                    "tooltip": "Rendering backend for texture baking. pytorch3d = portable, nvdiffrast = faster but needs CUDA compilation"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "SAM3D_POSE_DATA")
    RETURN_NAMES = ("glb_filepath", "ply_filepath", "pose_data")
    OUTPUT_TOOLTIPS = (
        "Path to saved GLB mesh file",
        "Path to saved Gaussian PLY file",
        "Camera pose and scale metadata"
    )
    FUNCTION = "bake_texture"
    CATEGORY = "SAM3DObjects"
    DESCRIPTION = "Bake Gaussian appearance into mesh UV textures (~30-60 seconds)."

    def bake_texture(
        self,
        embedders: Any,
        gaussian_data: dict,
        mesh_data: dict,
        image: torch.Tensor,
        mask: torch.Tensor,
        seed: int,
        with_mesh_postprocess: bool = False,
        with_texture_baking: bool = True,
        texture_mode: str = "opt",
        texture_size: int = 1024,
        simplify: float = 0.95,
        rendering_engine: str = "pytorch3d",
    ):
        """
        Bake Gaussian appearance into mesh UV textures.

        Args:
            embedders: SAM3D model with embedders (for preprocessing)
            gaussian_data: Gaussian from SAM3DGaussianDecode
            mesh_data: Mesh from SAM3DMeshDecode
            image: Input image tensor [B, H, W, C] (must match previous stages)
            mask: Input mask tensor [N, H, W] (must match previous stages)
            seed: Random seed (must match previous stages)
            with_mesh_postprocess: Enable mesh hole filling + cleanup
            with_texture_baking: Enable texture baking
            texture_mode: Texture baking mode ("opt" or "fast")
            texture_size: Texture resolution
            simplify: Mesh simplification ratio
            rendering_engine: Rendering backend ("pytorch3d" or "nvdiffrast")

        Returns:
            Tuple of (glb_filepath, ply_filepath, pose_data)
        """
        print(f"[SAM3DObjects] TextureBake: Baking Gaussian into mesh texture")
        print(f"[SAM3DObjects] Baking parameters:")
        print(f"[SAM3DObjects]   - with_mesh_postprocess: {with_mesh_postprocess}")
        print(f"[SAM3DObjects]   - with_texture_baking: {with_texture_baking}")
        print(f"[SAM3DObjects]   - texture_mode: {texture_mode}")
        print(f"[SAM3DObjects]   - texture_size: {texture_size}")
        print(f"[SAM3DObjects]   - simplify: {simplify}")
        print(f"[SAM3DObjects]   - rendering_engine: {rendering_engine}")

        # Convert ComfyUI tensors to formats expected by SAM3D
        image_pil = comfy_image_to_pil(image)
        mask_np = comfy_mask_to_numpy(mask)

        # TODO: Implement handling of texture_mode and rendering_engine
        # These require modifications to the underlying pipeline
        use_vertex_color = not with_texture_baking

        # Extract serialized data from Gaussian and Mesh decode outputs
        # These outputs contain "_serialized_stage2_output" with base64-encoded pickle data
        # We can't deserialize here because it requires sam3d_objects module (worker-only)
        # Instead, pass the serialized data to the model wrapper which will send to worker
        gaussian_serialized = gaussian_data.get("_serialized_stage2_output")
        mesh_serialized = mesh_data.get("_serialized_stage2_output")

        if not gaussian_serialized or not mesh_serialized:
            raise RuntimeError(
                "Texture baking requires both Gaussian and Mesh data in serialized format. "
                "Ensure SAM3DGaussianDecode and SAM3DMeshDecode outputs are connected."
            )

        # Create a special marker dict that tells the model wrapper to combine these
        stage2_output = {
            "_gaussian_serialized": gaussian_serialized,
            "_mesh_serialized": mesh_serialized,
            "_needs_combination": True
        }

        # Run texture baking using combined output
        try:
            print("[SAM3DObjects] Running texture baking...")
            output = embedders(
                image_pil, mask_np,
                seed=seed,
                stage2_output=stage2_output,  # Combined Gaussian + Mesh
                with_mesh_postprocess=with_mesh_postprocess,
                with_texture_baking=with_texture_baking,
                use_vertex_color=use_vertex_color,
                texture_size=texture_size,
                simplify=simplify,
            )

        except Exception as e:
            raise RuntimeError(f"SAM3D texture baking failed: {e}") from e

        print("[SAM3DObjects] Texture baking completed!")

        # Extract outputs
        glb_path = output.get("glb_path")
        ply_path = output.get("ply_path")
        pose_data = output.get("metadata", {})

        if glb_path is None:
            raise RuntimeError("GLB file was not generated")
        if ply_path is None:
            raise RuntimeError("PLY file was not generated")

        return (glb_path, ply_path, pose_data)
