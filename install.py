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
    """Check if VC++ Redistributable DLLs are actually present on the system."""
    if platform.system() != "Windows":
        return True  # Not needed on non-Windows

    required_dlls = ['vcruntime140.dll', 'msvcp140.dll']

    # Search locations in order of preference
    search_paths = []

    # 1. System directory (most reliable)
    system_root = os.environ.get('SystemRoot', r'C:\Windows')
    search_paths.append(os.path.join(system_root, 'System32'))

    # 2. Python environment directories
    if hasattr(sys, 'base_prefix'):
        search_paths.append(os.path.join(sys.base_prefix, 'Library', 'bin'))
        search_paths.append(os.path.join(sys.base_prefix, 'DLLs'))
    if hasattr(sys, 'prefix') and sys.prefix != getattr(sys, 'base_prefix', sys.prefix):
        search_paths.append(os.path.join(sys.prefix, 'Library', 'bin'))
        search_paths.append(os.path.join(sys.prefix, 'DLLs'))

    # Check each required DLL
    for dll in required_dlls:
        found = False
        for search_path in search_paths:
            dll_path = os.path.join(search_path, dll)
            if os.path.exists(dll_path):
                found = True
                break
        if not found:
            return False

    return True


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
        print("[SAM3DObjects] VC++ Redistributable installer completed.")
    elif result.returncode == 1638:
        # 1638 = newer version already installed
        print("[SAM3DObjects] VC++ Redistributable already installed (newer version)")
    else:
        print(f"[SAM3DObjects] Installation returned code {result.returncode}")
        print(f"[SAM3DObjects] Please install manually from: {VCREDIST_URL}")
        return False

    # Verify DLLs are actually present after installation
    if check_vcredist_installed():
        print("[SAM3DObjects] VC++ Redistributable DLLs verified!")
        return True
    else:
        print("[SAM3DObjects] Installation completed but DLLs not found in expected locations.")
        print("[SAM3DObjects] You may need to restart your system or terminal.")
        return False


def ensure_vcredist():
    """Check and install VC++ Redistributable if needed (Windows only)."""
    if platform.system() != "Windows":
        return True

    if check_vcredist_installed():
        print("[SAM3DObjects] VC++ Redistributable: OK (DLLs found)")
        return True

    print("[SAM3DObjects] VC++ Redistributable DLLs not found - attempting automatic install...")

    if install_vcredist():
        return True

    # Fallback: provide clear manual instructions
    print("")
    print("=" * 70)
    print("[SAM3DObjects] MANUAL INSTALLATION REQUIRED")
    print("=" * 70)
    print("")
    print("  The automatic installation of VC++ Redistributable failed.")
    print("  This is required for PyTorch CUDA and other native extensions.")
    print("")
    print("  Please download and install manually:")
    print(f"    {VCREDIST_URL}")
    print("")
    print("  After installation, restart your terminal and try again.")
    print("=" * 70)
    print("")
    return False


# =============================================================================
# Main Installation
# =============================================================================

def main():
    """Main installation function."""
    # Check VC++ Redistributable first (required for PyTorch CUDA and native extensions)
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
