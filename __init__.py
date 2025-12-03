"""
ComfyUI-SAM3DObjects: ComfyUI custom nodes for SAM 3D Objects

Generate 3D objects from single images using SAM 3D Objects.

Nodes:
- LoadSAM3DModel: Load SAM3D inference pipeline
- SAM3DGenerate: Generate 3D object from image + mask
- SAM3DGenerateRGBA: Generate 3D object from RGBA image
- SAM3DExportPLY: Export Gaussian Splat to PLY file
- SAM3DExportPLYBatch: Batch export PLY files
- SAM3DExportMesh: Export mesh to OBJ/GLB/PLY format
- SAM3DExtractMesh: Extract mesh data for processing
- SAM3DVisualizer: Render views of 3D object
- SAM3DRenderSingle: Render single view with camera control
"""

from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

# Import all node classes
from .nodes.load_model import LoadSAM3DModel
from .nodes.generate import SAM3DGenerate, SAM3DGenerateRGBA
from .nodes.export_ply import SAM3DExportPLY, SAM3DExportPLYBatch
from .nodes.export_mesh import SAM3DExportMesh, SAM3DExtractMesh
from .nodes.visualizer import SAM3DVisualizer, SAM3DRenderSingle


__version__ = "1.0.0"
__author__ = "ComfyUI-SAM3DObjects Contributors"


class SAM3DObjectsExtension(ComfyExtension):
    """
    ComfyUI Extension for SAM 3D Objects nodes.

    Provides nodes for generating and working with 3D objects
    from single images using the SAM 3D Objects model.
    """

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        """
        Return list of all node classes in this extension.

        Returns:
            List of ComfyNode classes
        """
        return [
            # Core nodes
            LoadSAM3DModel,
            SAM3DGenerate,
            SAM3DGenerateRGBA,

            # Export nodes
            SAM3DExportPLY,
            SAM3DExportPLYBatch,
            SAM3DExportMesh,
            SAM3DExtractMesh,

            # Visualization nodes
            SAM3DVisualizer,
            SAM3DRenderSingle,
        ]


async def comfy_entrypoint() -> SAM3DObjectsExtension:
    """
    ComfyUI extension entry point.

    This function is called by ComfyUI to load the extension and its nodes.

    Returns:
        SAM3DObjectsExtension instance
    """
    print("[SAM3DObjects] Loading ComfyUI-SAM3DObjects extension")
    print(f"[SAM3DObjects] Version: {__version__}")
    print("[SAM3DObjects] ")
    print("[SAM3DObjects] Nodes loaded:")
    print("[SAM3DObjects]   - LoadSAM3DModel")
    print("[SAM3DObjects]   - SAM3DGenerate")
    print("[SAM3DObjects]   - SAM3DGenerateRGBA")
    print("[SAM3DObjects]   - SAM3DExportPLY")
    print("[SAM3DObjects]   - SAM3DExportPLYBatch")
    print("[SAM3DObjects]   - SAM3DExportMesh")
    print("[SAM3DObjects]   - SAM3DExtractMesh")
    print("[SAM3DObjects]   - SAM3DVisualizer")
    print("[SAM3DObjects]   - SAM3DRenderSingle")
    print("[SAM3DObjects] ")
    print("[SAM3DObjects] Requirements:")
    print("[SAM3DObjects]   - NVIDIA GPU with 32GB+ VRAM recommended")
    print("[SAM3DObjects]   - sam-3d-objects package must be available")
    print("[SAM3DObjects]   - Model checkpoints will auto-download on first use")
    print("[SAM3DObjects] ")

    return SAM3DObjectsExtension()
