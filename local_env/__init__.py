"""
SAM3D Isolated Environment Manager.

This package handles creation and management of the isolated Python
environment for SAM3D inference, completely separate from ComfyUI.
"""

from .manager import SAM3DEnvironmentManager
from .config import InstallConfig

def install(node_root, config: InstallConfig = None) -> bool:
    """
    Install SAM3D dependencies in an isolated environment.

    Args:
        node_root: Root directory of the ComfyUI-SAM3DObjects node
        config: Optional configuration overrides

    Returns:
        True if installation succeeded
    """
    from pathlib import Path

    config = config or InstallConfig()
    manager = SAM3DEnvironmentManager(Path(node_root), config)

    try:
        manager.setup_environment()
        return True
    except Exception as e:
        print(f"[SAM3DObjects] Installation failed: {e}")
        return False

__all__ = ['install', 'SAM3DEnvironmentManager', 'InstallConfig']
