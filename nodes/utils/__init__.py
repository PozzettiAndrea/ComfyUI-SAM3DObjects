"""
Utility modules for SAM3D inference.

This package contains:
- stages: All pipeline stages (depth, stage1, stage2, decode, texture baking)
- lazy_manager: Model loading/caching
- helpers: Preprocessing and file I/O utilities
- pose_optimization: Pose refinement
- scene_batch: Batch scene processing
"""

# Set up vendor path before any imports that need sam3d_objects
import sys
from pathlib import Path

_VENDOR_PATH = str(Path(__file__).parent.parent / "vendor")
if _VENDOR_PATH not in sys.path:
    sys.path.insert(0, _VENDOR_PATH)

from .lazy_manager import LazyModelManager, get_model_manager, get_lazy_manager
from .helpers import (
    load_pointmap_from_file,
    preprocess_image_lazy,
    save_output_to_disk,
)
from .stages import (
    run_stage1_lazy,
    run_stage2_lazy,
    run_decode_lazy,
    run_depth_only,
    run_texture_bake_direct,
)
from .scene_batch import run_scene_generate_batch
from .pose_optimization import run_pose_optimization, run_pose_optimization_batch

__all__ = [
    # Model manager
    "LazyModelManager",
    "get_model_manager",
    "get_lazy_manager",
    # Helpers
    "load_pointmap_from_file",
    "preprocess_image_lazy",
    "save_output_to_disk",
    # Pipeline stages
    "run_depth_only",
    "run_stage1_lazy",
    "run_stage2_lazy",
    "run_decode_lazy",
    "run_texture_bake_direct",
    # Scene batch
    "run_scene_generate_batch",
    # Pose optimization
    "run_pose_optimization",
    "run_pose_optimization_batch",
]
