#!/usr/bin/env python3
"""
Installation script for ComfyUI-SAM3DObjects with isolated environment.

This script sets up an isolated Python virtual environment with all dependencies
required for SAM 3D Objects. The environment is completely isolated from
ComfyUI's main environment, preventing any dependency conflicts.
"""

import sys
import os
import platform
from pathlib import Path


# =============================================================================
# VC++ Redistributable Check (Windows only)
# =============================================================================

VCREDIST_URL = "https://aka.ms/vs/17/release/vc_redist.x64.exe"


def check_vcredist_installed():
    """Check if VC++ Redistributable 2015-2022 x64 is installed."""
    if platform.system() != "Windows":
        return True  # Not needed on non-Windows

    try:
        import winreg
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\X64"
        )
        winreg.CloseKey(key)
        return True
    except (FileNotFoundError, ImportError):
        return False


def install_vcredist():
    """Download and install VC++ Redistributable with UAC elevation."""
    import urllib.request
    import subprocess
    import tempfile

    print("[SAM3DObjects] Downloading VC++ Redistributable...")

    # Download to temp file
    temp_path = os.path.join(tempfile.gettempdir(), "vc_redist.x64.exe")
    try:
        urllib.request.urlretrieve(VCREDIST_URL, temp_path)
    except Exception as e:
        print(f"[SAM3DObjects] Failed to download VC++ Redistributable: {e}")
        print(f"[SAM3DObjects] Please download manually from: {VCREDIST_URL}")
        return False

    print("[SAM3DObjects] Installing VC++ Redistributable (UAC prompt may appear)...")

    # Run with elevation - /passive shows progress, /quiet is fully silent
    try:
        result = subprocess.run(
            [temp_path, '/install', '/passive', '/norestart'],
            capture_output=True
        )
    except Exception as e:
        print(f"[SAM3DObjects] Failed to run installer: {e}")
        print(f"[SAM3DObjects] Please run manually: {temp_path}")
        return False

    # Clean up
    try:
        os.remove(temp_path)
    except:
        pass

    if result.returncode == 0:
        print("[SAM3DObjects] VC++ Redistributable installed successfully!")
        return True
    elif result.returncode == 1638:
        # 1638 = newer version already installed
        print("[SAM3DObjects] VC++ Redistributable already installed (newer version)")
        return True
    else:
        print(f"[SAM3DObjects] Installation returned code {result.returncode}")
        print(f"[SAM3DObjects] Please install manually from: {VCREDIST_URL}")
        return False


def ensure_vcredist():
    """Check and install VC++ Redistributable if needed (Windows only)."""
    if platform.system() != "Windows":
        return True

    if check_vcredist_installed():
        print("[SAM3DObjects] VC++ Redistributable: OK")
        return True

    print("[SAM3DObjects] VC++ Redistributable not found - installing...")
    return install_vcredist()


# =============================================================================
# Main Installation
# =============================================================================

def main():
    """Main installation function."""
    # Check VC++ Redistributable first (required for OpenCV, Open3D, etc.)
    if not ensure_vcredist():
        print("[SAM3DObjects] WARNING: VC++ Redistributable installation failed.")
        print("[SAM3DObjects] Some features may not work. Continuing anyway...")

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
