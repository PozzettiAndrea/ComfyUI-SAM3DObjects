#!/usr/bin/env python3
"""
Installation script for ComfyUI-SAM3DObjects with isolated environment.

This script sets up an isolated Python virtual environment with all dependencies
required for SAM 3D Objects. The environment is completely isolated from
ComfyUI's main environment, preventing any dependency conflicts.
"""

import sys
from pathlib import Path


def main():
    """Main installation function."""
    node_root = Path(__file__).parent.absolute()
    sys.path.insert(0, str(node_root))

    from local_env import SAM3DEnvironmentManager, InstallConfig

    env_mgr = SAM3DEnvironmentManager(node_root, InstallConfig())

    # Check if already ready
    if env_mgr.is_environment_ready():
        print("[SAM3DObjects] Isolated environment already exists and is ready!")
        print(f"[SAM3DObjects] Location: {env_mgr.env_dir}")
        print("[SAM3DObjects] To reinstall, delete the _env directory.")
        return 0

    # Setup environment
    try:
        env_mgr.setup_environment()
        return 0
    except Exception as e:
        print(f"\n[SAM3DObjects] Installation FAILED: {e}")
        print("[SAM3DObjects] Report issues at: https://github.com/PozzettiAndrea/ComfyUI-SAM3DObjects/issues")
        return 1


if __name__ == "__main__":
    sys.exit(main())
