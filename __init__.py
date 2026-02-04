"""
ComfyUI-SAM3DObjects: ComfyUI custom nodes for SAM 3D Objects

Generate 3D objects from single images using SAM 3D Objects.

Pipeline Nodes:
- LoadSAM3DModel: Load SAM3D inference pipeline
- SAM3D_DepthEstimate: Run MoGe depth estimation (outputs pointmap)
- SAM3DGenerateSLAT: Generate SLAT latent (Stage 1 + 2 combined with caching)
- SAM3DGaussianDecode: Decode SLAT to Gaussian splat
- SAM3DMeshDecode: Decode SLAT to Mesh
- SAM3DTextureBake: Bake texture from Gaussian onto Mesh
- SAM3DSceneGenerate: Batch process multiple masks to 3D objects

Utility Nodes:
- SAM3DExportPLY: Export Gaussian Splat to PLY file
- SAM3DExportPLYBatch: Batch export PLY files
"""

import os
import sys
from pathlib import Path

__version__ = "1.0.0"
__author__ = "ComfyUI-SAM3DObjects Contributors"
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

# Define web directory for ComfyUI extension loading
WEB_DIRECTORY = "./web"

# Only run initialization when loaded by ComfyUI, not during pytest
if 'PYTEST_CURRENT_TEST' not in os.environ:
    _node_dir = Path(__file__).resolve().parent

    # Add vendor directory to path FIRST, before any other imports
    # This ensures our cv2 shim is found before the pip cv2 (which may have DLL issues on Windows)
    _VENDOR_PATH = str(_node_dir / "nodes" / "vendor")
    if _VENDOR_PATH not in sys.path:
        sys.path.insert(0, _VENDOR_PATH)

    # Check for isolated environment
    _env_path = _node_dir / "_env_sam3dobjects"
    if _env_path.exists():
        print("[SAM3DObjects] Isolated environment found")
    else:
        print("[SAM3DObjects] WARNING: No isolated environment at _env_sam3dobjects")

    # Setup import stubs BEFORE importing nodes
    try:
        from comfy_env import setup_isolated_imports
        setup_isolated_imports(__file__)
    except ImportError:
        print("[SAM3DObjects] comfy-env not installed")
    except Exception as e:
        print(f"[SAM3DObjects] Failed to setup import stubs: {e}")

    # Import all node classes
    from .nodes.load_model import LoadSAM3DModel
    from .nodes.depth_estimate import SAM3D_DepthEstimate
    from .nodes.generate_slat import SAM3DGenerateSLAT
    from .nodes.gaussian_decode import SAM3DGaussianDecode
    from .nodes.mesh_decode import SAM3DMeshDecode
    from .nodes.postprocess import SAM3DTextureBake
    from .nodes.export_ply import SAM3DExportPLY, SAM3DExportPLYBatch
    from .nodes.preview_nodes import SAM3D_PreviewPointCloud
    from .nodes.pose_optimization import SAM3D_PoseOptimization
    from .nodes.scene_generate import SAM3DSceneGenerate
    from .nodes.scene_pose_optimize import SAM3D_ScenePoseOptimize
    from .nodes.projection_overlay import SAM3D_ProjectionOverlay

    # Standard ComfyUI node registration
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

    # Optional: Human-readable names for nodes
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

    # Enable process isolation
    try:
        from comfy_env import enable_isolation
        enable_isolation(NODE_CLASS_MAPPINGS)
    except ImportError:
        pass
    except Exception as e:
        print(f"[SAM3DObjects] Failed to enable isolation: {e}")

    # Print info when loaded
    print("[SAM3DObjects] Loading ComfyUI-SAM3DObjects extension")
    print(f"[SAM3DObjects] Version: {__version__}")
    print("[SAM3DObjects] ")
    print("[SAM3DObjects] Simple pipeline (single object):")
    print("[SAM3DObjects]   1. LoadSAM3DModel - Load model")
    print("[SAM3DObjects]   2. SAM3D_DepthEstimate - MoGe depth -> pointmap")
    print("[SAM3DObjects]   3. SAM3DGenerateSLAT - Generate SLAT latent (~60s)")
    print("[SAM3DObjects]   4. SAM3DGaussianDecode - Decode SLAT to Gaussian (~15s)")
    print("[SAM3DObjects]   5. SAM3DMeshDecode - Decode SLAT to Mesh (~15s)")
    print("[SAM3DObjects]   6. SAM3DTextureBake - Bake Gaussian into mesh texture (~30-60s)")
    print("[SAM3DObjects] ")
    print("[SAM3DObjects] Batch pipeline (multiple objects):")
    print("[SAM3DObjects]   - SAM3DSceneGenerate: Process batch of masks -> multiple GLB meshes")
    print("[SAM3DObjects] ")
    print("[SAM3DObjects] Utility nodes:")
    print("[SAM3DObjects]   - SAM3D_PoseOptimization (ICP + render optimization)")
    print("[SAM3DObjects]   - SAM3D_ProjectionOverlay (visualize projected meshes)")
    print("[SAM3DObjects]   - SAM3DExportPLY / SAM3DExportPLYBatch")
    print("[SAM3DObjects]   - SAM3D_PreviewPointCloud")
    print("[SAM3DObjects] ")

else:
    # Pytest mode - empty mappings
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
