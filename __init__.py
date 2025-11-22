"""
ComfyUI-SAM3DObjects: ComfyUI custom nodes for SAM 3D Objects

Generate 3D objects from single images using SAM 3D Objects.

Nodes:
- LoadSAM3DModel: Load SAM3D inference pipeline
- SAM3DGenerate: Generate 3D object from image + mask (all-in-one)
- SAM3DGenerateRGBA: Generate 3D object from RGBA image
- SAM3DGenerateStage1: Generate Stage 1 only (cache-efficient)
- SAM3DGenerateStage2: Complete from Stage 1 output (cache-efficient)
- SAM3DExportPLY: Export Gaussian Splat to PLY file
- SAM3DExportPLYBatch: Batch export PLY files
- SAM3DExportMesh: Export mesh to OBJ/GLB/PLY format
- SAM3DExtractMesh: Extract mesh data for processing
- SAM3DVisualizer: Render views of 3D object
- SAM3DRenderSingle: Render single view with camera control
"""

# Import all node classes
from .nodes.load_model import LoadSAM3DModel
from .nodes.generate import SAM3DGenerate, SAM3DGenerateRGBA
from .nodes.generate_stage1 import SAM3DGenerateStage1
from .nodes.generate_stage2 import SAM3DGenerateStage2
from .nodes.export_ply import SAM3DExportPLY, SAM3DExportPLYBatch
from .nodes.export_mesh import SAM3DExportMesh, SAM3DExtractMesh
from .nodes.visualizer import SAM3DVisualizer, SAM3DRenderSingle


__version__ = "1.0.0"
__author__ = "ComfyUI-SAM3DObjects Contributors"


# Standard ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "LoadSAM3DModel": LoadSAM3DModel,
    "SAM3DGenerate": SAM3DGenerate,
    "SAM3DGenerateRGBA": SAM3DGenerateRGBA,
    "SAM3DGenerateStage1": SAM3DGenerateStage1,
    "SAM3DGenerateStage2": SAM3DGenerateStage2,
    "SAM3DExportPLY": SAM3DExportPLY,
    "SAM3DExportPLYBatch": SAM3DExportPLYBatch,
    "SAM3DExportMesh": SAM3DExportMesh,
    "SAM3DExtractMesh": SAM3DExtractMesh,
    "SAM3DVisualizer": SAM3DVisualizer,
    "SAM3DRenderSingle": SAM3DRenderSingle,
}

# Optional: Human-readable names for nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadSAM3DModel": "Load SAM3D Model",
    "SAM3DGenerate": "SAM3D Generate",
    "SAM3DGenerateRGBA": "SAM3D Generate (RGBA)",
    "SAM3DGenerateStage1": "SAM3D Generate Stage 1 (Cache-Efficient)",
    "SAM3DGenerateStage2": "SAM3D Generate Stage 2 (Cache-Efficient)",
    "SAM3DExportPLY": "SAM3D Export PLY",
    "SAM3DExportPLYBatch": "SAM3D Export PLY (Batch)",
    "SAM3DExportMesh": "SAM3D Export Mesh",
    "SAM3DExtractMesh": "SAM3D Extract Mesh",
    "SAM3DVisualizer": "SAM3D Visualizer",
    "SAM3DRenderSingle": "SAM3D Render Single",
}

# Print info when loaded
print("[SAM3DObjects] Loading ComfyUI-SAM3DObjects extension")
print(f"[SAM3DObjects] Version: {__version__}")
print("[SAM3DObjects] ")
print("[SAM3DObjects] Nodes loaded:")
print("[SAM3DObjects]   - LoadSAM3DModel")
print("[SAM3DObjects]   - SAM3DGenerate (all-in-one)")
print("[SAM3DObjects]   - SAM3DGenerateRGBA")
print("[SAM3DObjects]   - SAM3DGenerateStage1 (cache-efficient)")
print("[SAM3DObjects]   - SAM3DGenerateStage2 (cache-efficient)")
print("[SAM3DObjects]   - SAM3DExportPLY")
print("[SAM3DObjects]   - SAM3DExportPLYBatch")
print("[SAM3DObjects]   - SAM3DExportMesh")
print("[SAM3DObjects]   - SAM3DExtractMesh")
print("[SAM3DObjects]   - SAM3DVisualizer")
print("[SAM3DObjects]   - SAM3DRenderSingle")
print("[SAM3DObjects] ")
print("[SAM3DObjects] Cache-efficient workflow:")
print("[SAM3DObjects]   Use Stage1 + Stage2 nodes to avoid re-running expensive Stage 1")
print("[SAM3DObjects]   when iterating on Stage 2 parameters (CFG, texture, simplify)")
print("[SAM3DObjects] ")
print("[SAM3DObjects] Requirements:")
print("[SAM3DObjects]   - NVIDIA GPU with 32GB+ VRAM recommended")
print("[SAM3DObjects]   - Model checkpoints will auto-download on first use")
print("[SAM3DObjects] ")
