"""
Inference scripts for SAM3D worker.

This package contains modular components for the inference worker:
- lazy_manager: Lazy loading of models for low-VRAM GPUs
- utils: Utility functions for serialization, coordinate transforms, etc.
- preprocessing: Image/mask preprocessing
- stages: Pipeline stages (sparse gen, SLAT gen, decode)
- depth: Depth estimation
- texture_baking: Texture baking from Gaussian to mesh
- pose_optimization: Pose optimization for scene generation
- inference: Main inference orchestration
"""

# Set up vendor path before any imports that need sam3d_objects
import sys
from pathlib import Path

_VENDOR_PATH = str(Path(__file__).parent.parent / "vendor")
if _VENDOR_PATH not in sys.path:
    sys.path.insert(0, _VENDOR_PATH)

from lazy_manager import LazyModelManager, get_model_manager, get_lazy_manager
from utils import (
    deserialize_image,
    deserialize_mask,
    transform_to_global_coordinates,
    save_output_to_disk,
)
from preprocessing import load_pointmap_from_file, preprocess_image_lazy
from stages import run_stage1_lazy, run_stage2_lazy, run_decode_lazy, run_generate_slat, run_decode
from scene_batch import run_scene_generate_batch
from depth import run_depth_only
from texture_baking import run_texture_bake_direct
from pose_optimization import run_pose_optimization, run_pose_optimization_batch
from inference import run_inference

__all__ = [
    # Model manager
    "LazyModelManager",
    "get_model_manager",
    "get_lazy_manager",  # Backward compatibility alias
    # Utils
    "deserialize_image",
    "deserialize_mask",
    "transform_to_global_coordinates",
    "save_output_to_disk",
    # Preprocessing
    "load_pointmap_from_file",
    "preprocess_image_lazy",
    # Stages
    "run_stage1_lazy",
    "run_stage2_lazy",
    "run_decode_lazy",
    "run_generate_slat",
    "run_decode",
    # Scene batch
    "run_scene_generate_batch",
    # Depth
    "run_depth_only",
    # Texture baking
    "run_texture_bake_direct",
    # Pose optimization
    "run_pose_optimization",
    "run_pose_optimization_batch",
    # Main inference
    "run_inference",
]
