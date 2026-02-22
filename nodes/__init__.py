import pathlib
import comfy_sparse_attn
from comfy_sparse_attn import setup_link
_PKG = pathlib.Path(comfy_sparse_attn.__file__).parent
setup_link(_PKG / "sparse.py",           "sparse.py")
setup_link(_PKG / "ops_sparse.py",       "ops_sparse.py")
setup_link(_PKG / "attention_sparse.py", "attention_sparse.py")
del pathlib, comfy_sparse_attn, setup_link, _PKG

from .load_model import LoadSAM3DModel
from .depth_estimate import SAM3D_DepthEstimate
from .generate_slat import SAM3DGenerateSLAT
from .gaussian_decode import SAM3DGaussianDecode
from .mesh_decode import SAM3DMeshDecode
from .postprocess import SAM3DTextureBake
from .export_ply import SAM3DExportPLY, SAM3DExportPLYBatch
from .preview_nodes import SAM3D_PreviewPointCloud
from .pose_optimization import SAM3D_PoseOptimization
from .scene_generate import SAM3DSceneGenerate
from .scene_pose_optimize import SAM3D_ScenePoseOptimize
from .projection_overlay import SAM3D_ProjectionOverlay

NODE_CLASS_MAPPINGS = {
    "LoadSAM3DModel": LoadSAM3DModel,
    "SAM3D_DepthEstimate": SAM3D_DepthEstimate,
    "SAM3DGenerateSLAT": SAM3DGenerateSLAT,
    "SAM3DGaussianDecode": SAM3DGaussianDecode,
    "SAM3DMeshDecode": SAM3DMeshDecode,
    "SAM3DTextureBake": SAM3DTextureBake,
    "SAM3DExportPLY": SAM3DExportPLY,
    "SAM3DExportPLYBatch": SAM3DExportPLYBatch,
    "SAM3D_PreviewPointCloud": SAM3D_PreviewPointCloud,
    "SAM3D_PoseOptimization": SAM3D_PoseOptimization,
    "SAM3DSceneGenerate": SAM3DSceneGenerate,
    "SAM3D_ScenePoseOptimize": SAM3D_ScenePoseOptimize,
    "SAM3D_ProjectionOverlay": SAM3D_ProjectionOverlay,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadSAM3DModel": "(down)Load SAM3D Model",
    "SAM3D_DepthEstimate": "SAM3D Depth Estimate",
    "SAM3DGenerateSLAT": "SAM3D Generate SLAT",
    "SAM3DGaussianDecode": "SAM3D Gaussian Decode",
    "SAM3DMeshDecode": "SAM3D Mesh Decode",
    "SAM3DTextureBake": "SAM3D Texture Bake",
    "SAM3DExportPLY": "SAM3D Export PLY",
    "SAM3DExportPLYBatch": "SAM3D Export PLY (Batch)",
    "SAM3D_PreviewPointCloud": "SAM3D Preview Point Cloud",
    "SAM3D_PoseOptimization": "SAM3D Pose Optimization",
    "SAM3DSceneGenerate": "SAM3D Scene Generate",
    "SAM3D_ScenePoseOptimize": "SAM3D Scene Pose Optimize",
    "SAM3D_ProjectionOverlay": "SAM3D Projection Overlay",
}
