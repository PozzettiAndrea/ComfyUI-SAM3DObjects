"""SAM3DMeshDecode node for decoding SLAT to mesh."""

import os
import torch
from typing import Any

from .utils import (
    comfy_image_to_pil,
    comfy_mask_to_numpy,
)


class SAM3DMeshDecode:
    """
    Mesh Decoding.

    Decodes SLAT latent to mesh using the mesh decoder.
    Fast decoding stage (~15 seconds) that produces vertex-colored mesh.

    Output can be saved as GLB file or passed to SAM3DTextureBake for texture baking.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "slat_decoder_mesh": ("SAM3D_MODEL", {"tooltip": "Mesh decoder from LoadSAM3DModel"}),
                "slat": ("STRING", {"tooltip": "Path to SLAT from SAM3DSLATGen"}),
                "image": ("IMAGE", {"tooltip": "Input RGB image (must match SLATGen)"}),
                "mask": ("MASK", {"tooltip": "Binary mask (must match SLATGen)"}),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 2**31 - 1,
                    "control_after_generate": "fixed",
                    "tooltip": "Random seed (must match previous stages)"
                }),
            },
            "optional": {
                "save_glb": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Save mesh as vertex-colored GLB file"
                }),
                "simplify": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.9,
                    "max": 0.98,
                    "step": 0.01,
                    "tooltip": "Mesh simplification ratio (0.9 = aggressive, 0.98 = gentle)"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "SAM3D_MESH")
    RETURN_NAMES = ("glb_filepath", "mesh_data")
    OUTPUT_TOOLTIPS = (
        "Path to saved vertex-colored GLB file (if save_glb=True)",
        "Mesh data - pass to SAM3DTextureBake"
    )
    FUNCTION = "decode_mesh"
    CATEGORY = "SAM3DObjects"
    DESCRIPTION = "Decode SLAT to mesh (~15 seconds)."

    def decode_mesh(
        self,
        slat_decoder_mesh: Any,
        slat: str,
        image: torch.Tensor,
        mask: torch.Tensor,
        seed: int,
        save_glb: bool = True,
        simplify: float = 0.95,
    ):
        """
        Decode SLAT to mesh.

        Args:
            slat_decoder_mesh: SAM3D mesh decoder
            slat: Path to SLAT from SAM3DSLATGen
            image: Input image tensor [B, H, W, C]
            mask: Input mask tensor [N, H, W]
            seed: Random seed
            save_glb: Whether to save GLB file
            simplify: Mesh simplification ratio

        Returns:
            Tuple of (glb_filepath, mesh_data)
        """
        print(f"[SAM3DObjects] MeshDecode: Decoding SLAT to Mesh...")

        # Convert ComfyUI tensors to formats expected by SAM3D
        image_pil = comfy_image_to_pil(image)
        mask_np = comfy_mask_to_numpy(mask)

        # Derive output_dir from slat path (same directory)
        output_dir = os.path.dirname(slat)

        # Run Mesh decoding only
        try:
            mesh_output = slat_decoder_mesh(
                image_pil, mask_np,
                seed=seed,
                slat_output=slat,  # Resume from SLAT path
                mesh_only=True,  # CRITICAL: Only decode to Mesh
                save_files=save_glb,  # Save GLB if requested
                simplify=simplify,
                use_vertex_color=True,  # Use vertex colors (no texture baking)
                output_dir=output_dir,  # Use same directory as SLAT
            )

        except Exception as e:
            raise RuntimeError(f"SAM3D Mesh decode failed: {e}") from e

        # Extract GLB path if saved
        glb_path = mesh_output.get("glb_path", None)

        # If not found directly, check files dict (bridge returns this structure)
        if not glb_path and "files" in mesh_output and "glb" in mesh_output["files"]:
             glb_path = mesh_output["files"]["glb"]

        if save_glb and glb_path:
            print(f"[SAM3DObjects] MeshDecode completed: {glb_path}")
        elif save_glb:
            print(f"[SAM3DObjects] Warning: GLB file not saved")
        else:
            print(f"[SAM3DObjects] MeshDecode completed")

        # Return GLB path and Mesh data
        return (glb_path, mesh_output)
