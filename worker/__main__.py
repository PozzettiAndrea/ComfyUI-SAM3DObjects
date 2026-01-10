"""
Inference worker for SAM3D that runs in the isolated environment.

This worker loads the SAM3D model and handles inference requests
via IPC (stdin/stdout communication).

Uses comfyui-isolation's BaseWorker for the IPC protocol.

Modules in this package:
- lazy_manager: On-demand model loading for low-VRAM GPUs
- utils: Serialization, coordinate transforms, file I/O
- preprocessing: Image/mask preprocessing
- stages: Pipeline stages (sparse gen, SLAT gen, decode)
- depth: Depth estimation
- texture_baking: Texture baking from Gaussian to mesh
- pose_optimization: Pose optimization for scene generation
- inference: Main inference orchestration
"""

import sys
import os
from pathlib import Path

# CRITICAL: Suppress all library output BEFORE any imports
# Libraries like OmegaConf, Hydra, PyTorch, CUDA can print to stdout,
# which interferes with our JSON-based IPC protocol
import warnings
import logging

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['HYDRA_FULL_ERROR'] = '0'
logging.disable(logging.CRITICAL)

# Configure loguru early
try:
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, level="ERROR", format="{message}")
except ImportError:
    pass

from comfyui_isolation import BaseWorker, register

# Import from this package (relative imports)
from . import (
    run_inference,
    run_texture_bake_direct,
    run_pose_optimization,
    run_pose_optimization_batch,
    run_generate_slat,
    run_decode,
    run_scene_generate_batch,
)


class SAM3DWorker(BaseWorker):
    """
    SAM3D inference worker.

    Handles all inference requests for the SAM3DObjects ComfyUI node.
    Each method receives kwargs and passes them as a dict to the
    underlying run_* functions.
    """

    def setup(self):
        """Called once when worker starts - verify critical dependencies."""
        self.log("SAM3D inference worker starting...")
        self.log(f"Python: {sys.executable}")
        self.log(f"Working directory: {Path.cwd()}")

        # Verify critical imports
        try:
            import torch
            import pytorch3d
            self.log(f"PyTorch version: {torch.__version__}")
            self.log(f"PyTorch3D version: {pytorch3d.__version__}")
            self.log(f"CUDA available: {torch.cuda.is_available()}")
        except Exception as e:
            self.log(f"Warning: Could not verify dependencies: {e}")

        self.log("Ready for requests")

    @register("inference")
    def inference(self, **kwargs):
        """Run main inference pipeline."""
        return run_inference(kwargs)

    @register("decode")
    def decode(self, **kwargs):
        """Decode SLAT to Gaussian or Mesh."""
        return run_decode(kwargs)

    @register("generate_slat")
    def generate_slat(self, **kwargs):
        """Generate SLAT from image."""
        return run_generate_slat(kwargs)

    @register("texture_bake_direct")
    def texture_bake_direct(self, **kwargs):
        """Bake texture directly from Gaussian PLY to Mesh GLB."""
        return run_texture_bake_direct(kwargs)

    @register("pose_optimization")
    def pose_optimization(self, **kwargs):
        """Run pose optimization."""
        return run_pose_optimization(kwargs)

    @register("pose_optimization_batch")
    def pose_optimization_batch(self, **kwargs):
        """Run batch pose optimization."""
        return run_pose_optimization_batch(kwargs)

    @register("scene_generate_batch")
    def scene_generate_batch(self, **kwargs):
        """Run batch scene generation."""
        self.log("Starting scene_generate_batch")
        result = run_scene_generate_batch(kwargs)
        self.log("scene_generate_batch complete")
        return result


if __name__ == "__main__":
    SAM3DWorker().run()
