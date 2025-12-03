"""
Isolated model wrapper that runs inference via subprocess.
"""

from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image


class IsolatedSAM3DModel:
    """
    Wrapper for SAM3D model that runs in an isolated subprocess.

    This allows SAM3D to have its own dependency environment without
    conflicting with ComfyUI or other custom nodes.
    """

    def __init__(self, config_path: str, compile: bool = False, use_gpu_cache: bool = True):
        """
        Initialize the isolated model.

        Args:
            config_path: Path to pipeline config
            compile: Whether to compile the model
            use_gpu_cache: Keep models on GPU between stages (higher VRAM, faster)
        """
        self.config_path = str(config_path)
        self.compile = compile
        self.use_gpu_cache = use_gpu_cache
        self._bridge = None

    def get_bridge(self):
        """Get or create the subprocess bridge."""
        if self._bridge is None:
            from pathlib import Path
            from .subprocess_bridge import InferenceWorkerBridge

            # Get node root (parent of nodes/ directory)
            node_root = Path(__file__).parent.parent

            self._bridge = InferenceWorkerBridge.get_instance(node_root)

        return self._bridge

    def __call__(
        self,
        image: Image.Image,
        mask: np.ndarray,
        seed: int = 42,
        stage1_inference_steps: int = 25,
        stage2_inference_steps: int = 25,
        stage1_cfg_strength: float = 7.0,
        stage2_cfg_strength: float = 5.0,
        texture_size: int = 1024,
        simplify: float = 0.95,
        stage1_only: bool = False,
        stage1_output: dict = None,
        stage2_only: bool = False,
        stage2_output: dict = None,
        slat_only: bool = False,
        slat_output: dict = None,
        gaussian_only: bool = False,
        mesh_only: bool = False,
        save_files: bool = False,
        with_mesh_postprocess: bool = False,
        with_texture_baking: bool = True,
        use_vertex_color: bool = False,
        use_stage1_distillation: bool = False,
        use_stage2_distillation: bool = False,
        # NEW: Depth estimation and memory management
        depth_only: bool = False,
        unload_model: str = None,
        pointmap_path: str = None,
        intrinsics: Any = None,
        # Texture baking mode
        texture_mode: str = "opt",
        # Rendering engine
        rendering_engine: str = "pytorch3d",
        # Output directory (from depth_estimate)
        output_dir: str = None,
    ) -> dict[str, Any]:
        """
        Run inference on the given image and mask.

        Args:
            image: Input PIL image
            mask: Input numpy mask
            seed: Random seed
            stage1_inference_steps: Denoising steps for Stage 1
            stage2_inference_steps: Denoising steps for Stage 2
            stage1_cfg_strength: CFG strength for Stage 1
            stage2_cfg_strength: CFG strength for Stage 2
            texture_size: Texture resolution
            simplify: Mesh simplification ratio
            stage1_only: If True, only run Stage 1 (sparse structure generation)
            stage1_output: Stage 1 output to resume from (for Stage 2 only mode)

        Returns:
            Output dictionary with gaussian splats, mesh, and pose data
        """
        bridge = self.get_bridge()
        return bridge.run_inference(
            config_path=self.config_path,
            image=image,
            mask=mask,
            seed=seed,
            compile=self.compile,
            use_gpu_cache=self.use_gpu_cache,
            stage1_inference_steps=stage1_inference_steps,
            stage2_inference_steps=stage2_inference_steps,
            stage1_cfg_strength=stage1_cfg_strength,
            stage2_cfg_strength=stage2_cfg_strength,
            texture_size=texture_size,
            simplify=simplify,
            stage1_only=stage1_only,
            stage1_output=stage1_output,
            stage2_only=stage2_only,
            stage2_output=stage2_output,
            slat_only=slat_only,
            slat_output=slat_output,
            gaussian_only=gaussian_only,
            mesh_only=mesh_only,
            save_files=save_files,
            with_mesh_postprocess=with_mesh_postprocess,
            with_texture_baking=with_texture_baking,
            use_vertex_color=use_vertex_color,
            use_stage1_distillation=use_stage1_distillation,
            use_stage2_distillation=use_stage2_distillation,
            # NEW: Depth estimation and memory management
            depth_only=depth_only,
            unload_model=unload_model,
            pointmap_path=pointmap_path,
            intrinsics=intrinsics,
            # Texture baking mode
            texture_mode=texture_mode,
            # Rendering engine
            rendering_engine=rendering_engine,
            # Output directory (from depth_estimate)
            output_dir=output_dir,
        )

    def __repr__(self) -> str:
        return f"IsolatedSAM3DModel(config={self.config_path}, compile={self.compile}, use_gpu_cache={self.use_gpu_cache})"
