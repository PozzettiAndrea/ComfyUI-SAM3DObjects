"""
ComfyUI-SAM3DObjects: ComfyUI custom nodes for SAM 3D Objects

Generate 3D objects from single images using SAM 3D Objects.

Nodes:
- LoadSAM3DModel: Load SAM3D inference pipeline
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
from .nodes.pose_optimization import SAM3D_PoseOptimization


__version__ = "1.0.0"
__author__ = "ComfyUI-SAM3DObjects Contributors"


# Standard ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "LoadSAM3DModel": LoadSAM3DModel,
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
    "SAM3D_PoseOptimization": SAM3D_PoseOptimization,
}

# Optional: Human-readable names for nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadSAM3DModel": "Load SAM3D Model",
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
    "SAM3D_PoseOptimization": "SAM3D Pose Optimization",
}

# Print info when loaded
print("[SAM3DObjects] Loading ComfyUI-SAM3DObjects extension")
print(f"[SAM3DObjects] Version: {__version__}")
print("[SAM3DObjects] ")
print("[SAM3DObjects] Modular pipeline nodes:")
print("[SAM3DObjects]   1. LoadSAM3DModel - Load model")
print("[SAM3DObjects]   2. SAM3D_DepthEstimate - MoGe depth → pointmap + intrinsics")
print("[SAM3DObjects]   3. SAM3DSparseGen - Sparse structure + pose (~3s)")
print("[SAM3DObjects]   4. SAM3DSLATGen - SLAT latent generation via diffusion (~60s)")
print("[SAM3DObjects]   5. SAM3DGaussianDecode - Decode SLAT to Gaussian (~15s)")
print("[SAM3DObjects]   6. SAM3DMeshDecode - Decode SLAT to Mesh (~15s)")
print("[SAM3DObjects]   7. SAM3DTextureBake - Bake Gaussian into mesh texture (~30-60s)")
print("[SAM3DObjects] ")
print("[SAM3DObjects] Utility nodes:")
print("[SAM3DObjects]   - SAM3D_UnloadModel (VRAM management)")
print("[SAM3DObjects]   - SAM3D_PoseOptimization (ICP + render optimization)")
print("[SAM3DObjects]   - SAM3DExportPLY / SAM3DExportPLYBatch")
print("[SAM3DObjects]   - SAM3DExportMesh / SAM3DExtractMesh")
print("[SAM3DObjects]   - SAM3DVisualizer / SAM3DRenderSingle")
print("[SAM3DObjects]   - SAM3D_PreviewPointCloud")
print("[SAM3DObjects] ")

# Export required attributes for ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
