"""SAM3DMeshDecode node for decoding SLAT to mesh."""

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
                "model": ("SAM3D_MODEL", {"tooltip": "SAM3D model loaded from checkpoint"}),
                "slat": ("SAM3D_SLAT", {"tooltip": "SLAT from SAM3DSLATGen"}),
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
        model: Any,
        slat: dict,
        image: torch.Tensor,
        mask: torch.Tensor,
        seed: int,
        save_glb: bool = True,
        simplify: float = 0.95,
    ):
        """
        Decode SLAT to mesh.

        Args:
            model: SAM3D inference pipeline
            slat: SLAT from SAM3DSLATGen
            image: Input image tensor [B, H, W, C]
            mask: Input mask tensor [N, H, W]
            seed: Random seed
            save_glb: Whether to save GLB file
            simplify: Mesh simplification ratio

        Returns:
            Tuple of (glb_filepath, mesh_data)
        """
        print(f"[SAM3DObjects] MeshDecode: Decoding SLAT to Mesh")
        print(f"[SAM3DObjects] Save GLB: {save_glb}, Simplify: {simplify}")

        # Convert ComfyUI tensors to formats expected by SAM3D
        image_pil = comfy_image_to_pil(image)
        mask_np = comfy_mask_to_numpy(mask)

        # Run Mesh decoding only
        try:
            print("[SAM3DObjects] Running Mesh decode...")
            mesh_output = model(
                image_pil, mask_np,
                seed=seed,
                slat_output=slat,  # Resume from SLAT
                mesh_only=True,  # CRITICAL: Only decode to Mesh
                save_files=save_glb,  # Save GLB if requested
                simplify=simplify,
                use_vertex_color=True,  # Use vertex colors (no texture baking)
            )

        except Exception as e:
            raise RuntimeError(f"SAM3D Mesh decode failed: {e}") from e

        print("[SAM3DObjects] Mesh decoding completed!")

        # Extract GLB path if saved
        glb_path = mesh_output.get("glb_path", None)
        if save_glb and glb_path:
            print(f"[SAM3DObjects] - Vertex-colored GLB saved to: {glb_path}")
        elif save_glb:
            print(f"[SAM3DObjects] - Warning: GLB file not saved")

        # Return GLB path and Mesh data
        return (glb_path, mesh_output)
