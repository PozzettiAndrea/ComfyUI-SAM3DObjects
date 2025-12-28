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

from .lazy_manager import LazyModelManager, get_lazy_manager, load_model
from .utils import (
    deserialize_image,
    deserialize_mask,
    transform_to_global_coordinates,
    save_output_to_disk,
    unload_model,
)
from .preprocessing import load_pointmap_from_file, preprocess_image_lazy
from .stages import run_stage1_lazy, run_stage2_lazy, run_decode_lazy, run_generate_slat, run_decode
from .depth import run_depth_only_lazy, run_depth_only
from .texture_baking import run_texture_bake_direct
from .pose_optimization import run_pose_optimization
from .inference import run_inference

__all__ = [
    # Lazy manager
    "LazyModelManager",
    "get_lazy_manager",
    "load_model",
    # Utils
    "deserialize_image",
    "deserialize_mask",
    "transform_to_global_coordinates",
    "save_output_to_disk",
    "unload_model",
    # Preprocessing
    "load_pointmap_from_file",
    "preprocess_image_lazy",
    # Stages
    "run_stage1_lazy",
    "run_stage2_lazy",
    "run_decode_lazy",
    "run_generate_slat",
    "run_decode",
    # Depth
    "run_depth_only_lazy",
    "run_depth_only",
    # Texture baking
    "run_texture_bake_direct",
    # Pose optimization
    "run_pose_optimization",
    # Main inference
    "run_inference",
]
