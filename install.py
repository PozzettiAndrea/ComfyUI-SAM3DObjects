#!/usr/bin/env python3
"""
Installation script for ComfyUI-SAM3DObjects with isolated environment.

This script sets up an isolated micromamba environment with all dependencies
required for SAM 3D Objects. The environment is completely isolated from
ComfyUI's main environment, preventing any dependency conflicts.

SAM3D will run inference in its own subprocess using this isolated environment.
"""

import sys
from pathlib import Path


def main():
    """Main installation function."""
    print("[SAM3DObjects] ========================================")
    print("[SAM3DObjects] ComfyUI-SAM3DObjects Installation")
    print("[SAM3DObjects] ========================================")
    print()
    print("[SAM3DObjects] This will create an isolated micromamba environment")
    print("[SAM3DObjects] for SAM3D inference, completely separate from ComfyUI.")
    print()

    # Get node root directory
    node_root = Path(__file__).parent.absolute()

    # Import environment manager from nodes package
    sys.path.insert(0, str(node_root))
    from nodes.env_manager import SAM3DEnvironmentManager

    # Create environment manager
    env_mgr = SAM3DEnvironmentManager(node_root)

    # Check if environment already exists and is ready
    if env_mgr.is_environment_ready():
        print("[SAM3DObjects] Isolated environment already exists and is ready!")
        print(f"[SAM3DObjects] Location: {env_mgr.env_dir}")
        print()
        print("[SAM3DObjects] If you want to reinstall, delete the following directories:")
        print(f"[SAM3DObjects]   - {env_mgr.env_dir}")
        print(f"[SAM3DObjects]   - {env_mgr.micromamba_dir}")
        print()
        print("[SAM3DObjects] Installation complete!")
        return 0

    # Setup environment
    try:
        env_mgr.setup_environment()
    except Exception as e:
        print()
        print("[SAM3DObjects] ========================================")
        print("[SAM3DObjects] Installation FAILED")
        print("[SAM3DObjects] ========================================")
        print(f"[SAM3DObjects] Error: {e}")
        print()
        print("[SAM3DObjects] Please check the error above and try again.")
        print("[SAM3DObjects] If the problem persists, please report it at:")
        print("[SAM3DObjects]   https://github.com/your-repo/ComfyUI-SAM3DObjects/issues")
        return 1

    print()
    print("[SAM3DObjects] ========================================")
    print("[SAM3DObjects] Installation Complete!")
    print("[SAM3DObjects] ========================================")
    print()
    print("[SAM3DObjects] The isolated environment has been created at:")
    print(f"[SAM3DObjects]   {env_mgr.env_dir}")
    print()
    print("[SAM3DObjects] SAM3D nodes will automatically use this isolated")
    print("[SAM3DObjects] environment for inference, preventing any conflicts")
    print("[SAM3DObjects] with your main ComfyUI installation.")
    print()
    print("[SAM3DObjects] You can now use SAM3D nodes in ComfyUI!")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
