"""SAM3D_UnloadModel node - now a no-op pass-through.

With the @isolated decorator pattern, each node method runs in its own
subprocess that exits when done, automatically freeing VRAM.
This node is kept for workflow compatibility but doesn't do anything.
"""

from typing import Any


class SAM3D_UnloadModel:
    """
    Unload Model (No-op Pass-through).

    With the @isolated decorator pattern, models are automatically unloaded
    when each node's subprocess exits. This node is kept for workflow
    compatibility but no longer performs any action.

    To free VRAM between stages, workflows naturally do this automatically
    since each @isolated node runs in a fresh subprocess.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SAM3D_MODEL", {"tooltip": "Model config (pass-through)"}),
                "model_type": (["depth", "sparse", "slat", "decoders", "all"], {
                    "default": "depth",
                    "tooltip": "(No longer used) Which model component to unload"
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
        "Model config (unchanged, for chaining)",
        "Pass-through pointmap",
        "Pass-through sparse structure path",
        "Pass-through SLAT path"
    )
    FUNCTION = "unload"
    CATEGORY = "SAM3DObjects"
    DESCRIPTION = "No-op pass-through. Models auto-unload when subprocess exits."

    def unload(
        self,
        model: Any,
        model_type: str,
        pointmap: Any = None,
        sparse_structure_path: str = None,
        slat_path: str = None,
    ):
        """
        Pass-through node (no longer performs unloading).

        With @isolated pattern, each node runs in a subprocess that exits
        when done, automatically freeing VRAM.

        Args:
            model: SAM3DModelConfig (pass-through)
            model_type: (Ignored) Which component to unload
            pointmap: Pass-through
            sparse_structure_path: Pass-through
            slat_path: Pass-through

        Returns:
            Tuple of (model, pointmap, sparse_structure_path, slat_path) - all pass-through
        """
        print(f"[SAM3DObjects] UnloadModel: No-op (models auto-unload with @isolated pattern)")

        # Pass through all inputs unchanged
        return (model, pointmap, sparse_structure_path, slat_path)
