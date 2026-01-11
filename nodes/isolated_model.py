"""
SAM3D model configuration holder.

This is a simple data class that holds model configuration.
Actual inference happens in @isolated decorated node methods.
"""

from typing import Any


class SAM3DModelConfig:
    """
    Configuration holder for SAM3D models.

    This doesn't load any models - it just stores the configuration
    that @isolated node methods need to run inference.
    """

    def __init__(
        self,
        config_path: str,
        compile: bool = False,
        use_gpu_cache: bool = True,
        depth_backend: str = "moge2",
    ):
        """
        Initialize model configuration.

        Args:
            config_path: Path to pipeline config (pipeline.yaml)
            compile: Whether to compile the model
            use_gpu_cache: Keep models on GPU between stages (higher VRAM, faster)
            depth_backend: Depth model backend (moge2 or moge)
        """
        self.config_path = str(config_path)
        self.compile = compile
        self.use_gpu_cache = use_gpu_cache
        self.depth_backend = depth_backend

    def __repr__(self) -> str:
        return (
            f"SAM3DModelConfig(config={self.config_path}, "
            f"compile={self.compile}, use_gpu_cache={self.use_gpu_cache}, "
            f"depth_backend={self.depth_backend})"
        )


# Backward compatibility alias
IsolatedSAM3DModel = SAM3DModelConfig
