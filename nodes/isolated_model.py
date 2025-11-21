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

    def __init__(self, config_path: str, compile: bool = False):
        """
        Initialize the isolated model.

        Args:
            config_path: Path to pipeline config
            compile: Whether to compile the model
        """
        self.config_path = str(config_path)
        self.compile = compile
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

    def __call__(self, image: Image.Image, mask: np.ndarray, seed: int = 42) -> dict[str, Any]:
        """
        Run inference on the given image and mask.

        Args:
            image: Input PIL image
            mask: Input numpy mask
            seed: Random seed

        Returns:
            Output dictionary with gaussian splats, mesh, and pose data
        """
        bridge = self.get_bridge()
        return bridge.run_inference(
            config_path=self.config_path,
            image=image,
            mask=mask,
            seed=seed,
            compile=self.compile
        )

    def __repr__(self) -> str:
        return f"IsolatedSAM3DModel(config={self.config_path}, compile={self.compile})"
