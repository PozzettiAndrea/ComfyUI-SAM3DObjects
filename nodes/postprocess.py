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

    Requires GLB and PLY file paths as inputs.
    Final stage that produces textured GLB output (~30-60 seconds).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "embedders": ("SAM3D_MODEL", {"tooltip": "Embedders from LoadSAM3DModel (for preprocessing)"}),
                "glb_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Path to GLB mesh file"
                }),
                "ply_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Path to PLY Gaussian file"
                }),
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
                    "tooltip": "Simplify mesh + fill holes. Enable for faster texture baking; disable to preserve full mesh detail."
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
        glb_path: str,
        ply_path: str,
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
            glb_path: Path to input GLB mesh file
            ply_path: Path to input PLY Gaussian file
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
        print(f"[SAM3DObjects]   - seed: {seed}")

        print(f"[SAM3DObjects] Input tensors:")
        print(f"[SAM3DObjects]   - image shape: {image.shape}, dtype: {image.dtype}")
        print(f"[SAM3DObjects]   - mask shape: {mask.shape}, dtype: {mask.dtype}")

        # Convert ComfyUI tensors to formats expected by SAM3D
        print(f"[SAM3DObjects] Converting ComfyUI tensors...")
        image_pil = comfy_image_to_pil(image)
        mask_np = comfy_mask_to_numpy(mask)
        print(f"[SAM3DObjects] Converted image: mode={image_pil.mode}, size={image_pil.size}")
        print(f"[SAM3DObjects] Converted mask: shape={mask_np.shape}, dtype={mask_np.dtype}, range=[{mask_np.min()}, {mask_np.max()}]")

        # TODO: Implement handling of texture_mode and rendering_engine
        # These require modifications to the underlying pipeline
        use_vertex_color = not with_texture_baking

        # Validate file paths
        import os
        print(f"[SAM3DObjects] Validating file paths...")
        print(f"[SAM3DObjects]   - glb_path: '{glb_path}'")
        print(f"[SAM3DObjects]   - ply_path: '{ply_path}'")

        if not glb_path or not ply_path:
            raise RuntimeError("Both glb_path and ply_path are required")

        if not os.path.exists(glb_path):
            raise RuntimeError(f"GLB file not found: {glb_path}")
        if not os.path.exists(ply_path):
            raise RuntimeError(f"PLY file not found: {ply_path}")

        glb_size = os.path.getsize(glb_path)
        ply_size = os.path.getsize(ply_path)
        print(f"[SAM3DObjects] File validation passed:")
        print(f"[SAM3DObjects]   - GLB: {glb_path} ({glb_size:,} bytes)")
        print(f"[SAM3DObjects]   - PLY: {ply_path} ({ply_size:,} bytes)")

        # Create a marker dict with file paths
        stage2_output = {
            "_glb_path": glb_path,
            "_ply_path": ply_path,
            "_needs_file_loading": True
        }

        # Run texture baking using combined output
        try:
            print("[SAM3DObjects] Running texture baking via embedders...")
            print(f"[SAM3DObjects] Embedders call parameters:")
            print(f"[SAM3DObjects]   - stage2_output keys: {list(stage2_output.keys())}")
            print(f"[SAM3DObjects]   - with_mesh_postprocess: {with_mesh_postprocess}")
            print(f"[SAM3DObjects]   - with_texture_baking: {with_texture_baking}")
            print(f"[SAM3DObjects]   - use_vertex_color: {use_vertex_color}")
            print(f"[SAM3DObjects]   - texture_size: {texture_size}")
            print(f"[SAM3DObjects]   - simplify: {simplify}")
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
            print(f"[SAM3DObjects] Embedders returned output with keys: {list(output.keys()) if isinstance(output, dict) else type(output)}")

        except Exception as e:
            print(f"[SAM3DObjects] ERROR during texture baking: {type(e).__name__}: {e}")
            raise RuntimeError(f"SAM3D texture baking failed: {e}") from e

        print("[SAM3DObjects] Texture baking completed successfully!")

        # Extract outputs
        print(f"[SAM3DObjects] Extracting outputs from result...")
        output_glb_path = output.get("glb_path")
        output_ply_path = output.get("ply_path")
        pose_data = output.get("metadata", {})

        print(f"[SAM3DObjects] Output paths:")
        print(f"[SAM3DObjects]   - output_glb_path: {output_glb_path}")
        print(f"[SAM3DObjects]   - output_ply_path: {output_ply_path}")
        print(f"[SAM3DObjects]   - pose_data keys: {list(pose_data.keys()) if pose_data else 'None'}")

        if output_glb_path is None:
            print(f"[SAM3DObjects] ERROR: GLB file was not generated")
            raise RuntimeError("GLB file was not generated")
        if output_ply_path is None:
            print(f"[SAM3DObjects] ERROR: PLY file was not generated")
            raise RuntimeError("PLY file was not generated")

        print(f"[SAM3DObjects] TextureBake node completed successfully!")
        return (output_glb_path, output_ply_path, pose_data)
