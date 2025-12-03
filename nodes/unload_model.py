"""SAM3D_UnloadModel node for selectively unloading models to free VRAM."""

from typing import Any


class SAM3D_UnloadModel:
    """
    Unload specific SAM3D model components to free VRAM.

    Use this node after a stage completes to free GPU memory before
    running the next stage. This is useful for memory-constrained GPUs.

    Model types:
    - depth: MoGe depth estimation model
    - sparse: Sparse structure generator (Stage 1)
    - slat: SLAT generator (Stage 2)
    - decoders: Gaussian and Mesh decoders (Stage 3)
    - all: Unload all models
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SAM3D_MODEL", {"tooltip": "Model to unload (any output from LoadSAM3DModel)"}),
                "model_type": (["depth", "sparse", "slat", "decoders", "all"], {
                    "default": "depth",
                    "tooltip": "Which model component to unload"
                }),
            },
            "optional": {
                # Pass-through inputs to allow chaining in workflows
                "pointmap": ("SAM3D_POINTMAP", {"tooltip": "Pass-through pointmap"}),
                "sparse_structure_path": ("STRING", {"tooltip": "Pass-through sparse structure path"}),
                "slat_path": ("STRING", {"tooltip": "Pass-through SLAT path"}),
            }
        }

    RETURN_TYPES = ("SAM3D_MODEL", "SAM3D_POINTMAP", "STRING", "STRING")
    RETURN_NAMES = ("model", "pointmap", "sparse_structure_path", "slat_path")
    OUTPUT_TOOLTIPS = (
        "Model (unchanged, for chaining)",
        "Pass-through pointmap",
        "Pass-through sparse structure path",
        "Pass-through SLAT path"
    )
    FUNCTION = "unload"
    CATEGORY = "SAM3DObjects"
    DESCRIPTION = "Unload specific model components to free VRAM."

    def unload(
        self,
        model: Any,
        model_type: str,
        pointmap: Any = None,
        sparse_structure_path: str = None,
        slat_path: str = None,
    ):
        """
        Unload specified model component.

        Args:
            model: SAM3D model wrapper (IsolatedSAM3DModel)
            model_type: Which component to unload (depth/sparse/slat/decoders/all)
            pointmap: Pass-through
            sparse_structure_path: Pass-through
            slat_path: Pass-through

        Returns:
            Tuple of (model, pointmap, sparse_structure_path, slat_path) - all pass-through
        """
        print(f"[SAM3DObjects] UnloadModel: Unloading {model_type}...")

        try:
            # Call the model wrapper to unload
            result = model(
                None,  # No image
                None,  # No mask
                unload_model=model_type,  # NEW: unload command
            )

            status = result.get("status", "unknown")
            unloaded = result.get("model", model_type)
            print(f"[SAM3DObjects] Unload status: {status}, model: {unloaded}")

        except Exception as e:
            print(f"[SAM3DObjects] Warning: Unload failed: {e}")
            # Don't raise - allow workflow to continue

        # Pass through all inputs
        return (model, pointmap, sparse_structure_path, slat_path)
