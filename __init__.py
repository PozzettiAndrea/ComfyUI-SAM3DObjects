"""
ComfyUI-SAM3DObjects: ComfyUI custom nodes for SAM 3D Objects

Generate 3D objects from single images using SAM 3D Objects.

Nodes:
- LoadSAM3DModel: Load SAM3D inference pipeline
- SAM3DGenerate: Generate 3D object from image + mask (all-in-one)
- SAM3DGenerateRGBA: Generate 3D object from RGBA image
- SAM3D_DepthEstimate: Run MoGe depth estimation (outputs pointmap + intrinsics)
- SAM3DSparseGen: Generate sparse structure (outputs sparse coords + pose)
- SAM3D_UnloadModel: Unload models to free VRAM
- SAM3DSLATGen: Generate SLAT latents (cache-efficient)
- SAM3DGaussianDecode: Decode SLAT to Gaussian (cache-efficient)
- SAM3DMeshDecode: Decode SLAT to Mesh (cache-efficient)
- SAM3DTextureBake: Bake texture from Gaussian + Mesh (cache-efficient)
- SAM3DExportPLY: Export Gaussian Splat to PLY file
- SAM3DExportPLYBatch: Batch export PLY files
- SAM3DExportMesh: Export mesh to OBJ/GLB/PLY format
- SAM3DExtractMesh: Extract mesh data for processing
- SAM3DVisualizer: Render views of 3D object
- SAM3DRenderSingle: Render single view with camera control
"""

import os

# Define web directory for ComfyUI extension loading
WEB_DIRECTORY = os.path.join(os.path.dirname(__file__), "web")

# Import all node classes
from .nodes.load_model import LoadSAM3DModel
from .nodes.generate import SAM3DGenerate, SAM3DGenerateRGBA
from .nodes.depth_estimate import SAM3D_DepthEstimate
from .nodes.generate_stage1 import SAM3DSparseGen
from .nodes.unload_model import SAM3D_UnloadModel
from .nodes.generate_stage2 import SAM3DSLATGen
from .nodes.gaussian_decode import SAM3DGaussianDecode
from .nodes.mesh_decode import SAM3DMeshDecode
from .nodes.postprocess import SAM3DTextureBake
from .nodes.export_ply import SAM3DExportPLY, SAM3DExportPLYBatch
from .nodes.export_mesh import SAM3DExportMesh, SAM3DExtractMesh
from .nodes.visualizer import SAM3DVisualizer, SAM3DRenderSingle
from .nodes.preview_nodes import SAM3D_PreviewPointCloud


__version__ = "1.0.0"
__author__ = "ComfyUI-SAM3DObjects Contributors"


# Standard ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "LoadSAM3DModel": LoadSAM3DModel,
    "SAM3DGenerate": SAM3DGenerate,
    "SAM3DGenerateRGBA": SAM3DGenerateRGBA,
    "SAM3D_DepthEstimate": SAM3D_DepthEstimate,
    "SAM3DSparseGen": SAM3DSparseGen,
    "SAM3D_UnloadModel": SAM3D_UnloadModel,
    "SAM3DSLATGen": SAM3DSLATGen,
    "SAM3DGaussianDecode": SAM3DGaussianDecode,
    "SAM3DMeshDecode": SAM3DMeshDecode,
    "SAM3DTextureBake": SAM3DTextureBake,
    "SAM3DExportPLY": SAM3DExportPLY,
    "SAM3DExportPLYBatch": SAM3DExportPLYBatch,
    "SAM3DExportMesh": SAM3DExportMesh,
    "SAM3DExtractMesh": SAM3DExtractMesh,
    "SAM3DVisualizer": SAM3DVisualizer,
    "SAM3DRenderSingle": SAM3DRenderSingle,
    "SAM3D_PreviewPointCloud": SAM3D_PreviewPointCloud,
}

# Optional: Human-readable names for nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadSAM3DModel": "Load SAM3D Model",
    "SAM3DGenerate": "SAM3D Generate",
    "SAM3DGenerateRGBA": "SAM3D Generate (RGBA)",
    "SAM3D_DepthEstimate": "SAM3D Depth Estimate",
    "SAM3DSparseGen": "SAM3D Sparse Gen",
    "SAM3D_UnloadModel": "SAM3D Unload Model",
    "SAM3DSLATGen": "SAM3D SLAT Gen",
    "SAM3DGaussianDecode": "SAM3D Gaussian Decode",
    "SAM3DMeshDecode": "SAM3D Mesh Decode",
    "SAM3DTextureBake": "SAM3D Texture Bake",
    "SAM3DExportPLY": "SAM3D Export PLY",
    "SAM3DExportPLYBatch": "SAM3D Export PLY (Batch)",
    "SAM3DExportMesh": "SAM3D Export Mesh",
    "SAM3DExtractMesh": "SAM3D Extract Mesh",
    "SAM3DVisualizer": "SAM3D Visualizer",
    "SAM3DRenderSingle": "SAM3D Render Single",
    "SAM3D_PreviewPointCloud": "SAM3D Preview Point Cloud",
}

# Print info when loaded
print("[SAM3DObjects] Loading ComfyUI-SAM3DObjects extension")
print(f"[SAM3DObjects] Version: {__version__}")
print("[SAM3DObjects] ")
print("[SAM3DObjects] Nodes loaded:")
print("[SAM3DObjects]   - LoadSAM3DModel")
print("[SAM3DObjects]   - SAM3DGenerate (all-in-one)")
print("[SAM3DObjects]   - SAM3DGenerateRGBA")
print("[SAM3DObjects]   - SAM3D_DepthEstimate (MoGe depth → pointmap + intrinsics)")
print("[SAM3DObjects]   - SAM3DSparseGen (sparse structure → pose)")
print("[SAM3DObjects]   - SAM3D_UnloadModel (VRAM management)")
print("[SAM3DObjects]   - SAM3DSLATGen (cache-efficient)")
print("[SAM3DObjects]   - SAM3DGaussianDecode (cache-efficient)")
print("[SAM3DObjects]   - SAM3DMeshDecode (cache-efficient)")
print("[SAM3DObjects]   - SAM3DTextureBake (cache-efficient)")
print("[SAM3DObjects]   - SAM3DExportPLY")
print("[SAM3DObjects]   - SAM3DExportPLYBatch")
print("[SAM3DObjects]   - SAM3DExportMesh")
print("[SAM3DObjects]   - SAM3DExtractMesh")
print("[SAM3DObjects]   - SAM3DVisualizer")
print("[SAM3DObjects]   - SAM3DRenderSingle")
print("[SAM3DObjects]   - SAM3D Preview Point Cloud")
print("[SAM3DObjects] ")
print("[SAM3DObjects] Memory-efficient workflow (6 stages):")
print("[SAM3DObjects]   1. DepthEstimate: MoGe depth → pointmap + intrinsics")
print("[SAM3DObjects]   2. UnloadModel(depth) → free VRAM")
print("[SAM3DObjects]   3. SparseGen: Sparse structure + pose (~3s)")
print("[SAM3DObjects]   4. SLATGen: SLAT latent generation via diffusion (~60s)")
print("[SAM3DObjects]   5. GaussianDecode/MeshDecode: Decode SLAT (~15s each)")
print("[SAM3DObjects]   6. TextureBake: Bake Gaussian into mesh texture (~30-60s)")
print("[SAM3DObjects] ")
print("[SAM3DObjects] Requirements:")
print("[SAM3DObjects]   - NVIDIA GPU with 32GB+ VRAM recommended")
print("[SAM3DObjects]   - Model checkpoints will auto-download on first use")
print("[SAM3DObjects] ")

# Export required attributes for ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
