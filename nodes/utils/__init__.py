"""
Utility modules for SAM3D inference.

This package contains:
- stages: All pipeline stages (stage1, stage2, decode, texture baking)
- helpers: Preprocessing and file I/O utilities
- pose_optimization: Pose refinement
- scene_batch: Batch scene processing
"""

from .helpers import (
    load_pointmap_from_file,
    preprocess_image_lazy,
    save_output_to_disk,
)
from .stages import (
    run_stage1,
    run_stage2,
    run_decode,
    run_texture_bake_direct,
)
from .scene_batch import run_scene_generate_batch
from .pose_optimization import run_pose_optimization, run_pose_optimization_batch

__all__ = [
    # Helpers
    "load_pointmap_from_file",
    "preprocess_image_lazy",
    "save_output_to_disk",
    # Pipeline stages
    "run_stage1",
    "run_stage2",
    "run_decode",
    "run_texture_bake_direct",
    # Scene batch
    "run_scene_generate_batch",
    # Pose optimization
    "run_pose_optimization",
    "run_pose_optimization_batch",
]
