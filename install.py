"""Installation script for ComfyUI-SAM3DObjects custom node."""

import subprocess
import sys
import os
import shutil
from pathlib import Path


def find_conda_command():
    """
    Find available conda/mamba/micromamba command.

    Returns:
        str or None: Command name if found, None otherwise
    """
    for cmd in ['mamba', 'micromamba', 'conda']:
        if shutil.which(cmd):
            return cmd
    return None


def check_pytorch3d():
    """Check if pytorch3d is already installed."""
    try:
        import pytorch3d
        print(f"[SAM3DObjects] pytorch3d {pytorch3d.__version__} is already installed")
        return True
    except ImportError:
        return False


def install_pytorch3d_conda():
    """
    Try to install pytorch3d via conda/mamba/micromamba.

    Returns:
        bool: True if successful, False otherwise
    """
    conda_cmd = find_conda_command()

    if not conda_cmd:
        print("[SAM3DObjects] No conda/mamba/micromamba found")
        return False

    print(f"[SAM3DObjects] Found '{conda_cmd}' package manager")
    print(f"[SAM3DObjects] Installing pytorch3d via {conda_cmd}...")
    print("[SAM3DObjects] This is much faster than building from source!")
    print("[SAM3DObjects] ")

    try:
        # Try facebookresearch channel first
        subprocess.check_call([
            conda_cmd,
            "install",
            "-y",
            "-c", "facebookresearch",
            "pytorch3d"
        ])
        print("[SAM3DObjects] pytorch3d installed successfully via conda!")
        return True
    except subprocess.CalledProcessError:
        print(f"[SAM3DObjects] Failed to install from facebookresearch channel")

        # Try conda-forge as fallback
        try:
            print("[SAM3DObjects] Trying conda-forge channel...")
            subprocess.check_call([
                conda_cmd,
                "install",
                "-y",
                "-c", "conda-forge",
                "pytorch3d"
            ])
            print("[SAM3DObjects] pytorch3d installed successfully via conda-forge!")
            return True
        except subprocess.CalledProcessError:
            print(f"[SAM3DObjects] Failed to install via {conda_cmd}")
            return False


def install_pytorch3d_pip():
    """Install pytorch3d from source via pip."""
    print("[SAM3DObjects] ")
    print("[SAM3DObjects] Installing pytorch3d from source (pip)...")
    print("[SAM3DObjects] This may take 10-20 minutes and requires:")
    print("[SAM3DObjects]   - CUDA toolkit installed")
    print("[SAM3DObjects]   - C++ compiler (gcc/g++)")
    print("[SAM3DObjects]   - PyTorch with CUDA support")
    print("[SAM3DObjects] ")

    try:
        # Install build dependencies
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "ninja",  # Build accelerator
        ])

        # Install pytorch3d from source
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47",
        ])
        print("[SAM3DObjects] pytorch3d installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[SAM3DObjects] Failed to install pytorch3d via pip: {e}")
        return False


def install_pytorch3d():
    """
    Install pytorch3d using the best available method.

    Tries in order:
    1. conda/mamba/micromamba (fastest, most reliable)
    2. pip from source (slower, requires build tools)

    Returns:
        bool: True if successful, False otherwise
    """
    # Try conda first (fastest and most reliable)
    if install_pytorch3d_conda():
        return True

    # Fall back to pip installation
    print("[SAM3DObjects] ")
    print("[SAM3DObjects] Conda installation failed, trying pip installation...")

    if install_pytorch3d_pip():
        return True

    # All methods failed
    print("[SAM3DObjects] ")
    print("[SAM3DObjects] TROUBLESHOOTING:")
    print("[SAM3DObjects] All automatic installation methods failed. Try manually:")
    print("[SAM3DObjects] ")

    conda_cmd = find_conda_command()
    if conda_cmd:
        print(f"[SAM3DObjects] 1. Install via {conda_cmd}:")
        print(f"[SAM3DObjects]    {conda_cmd} install -c facebookresearch pytorch3d")

    print("[SAM3DObjects] 2. Install build tools and try pip:")
    print("[SAM3DObjects]    conda install cuda-toolkit -c nvidia")
    print("[SAM3DObjects]    conda install gcc_linux-64 gxx_linux-64")
    print("[SAM3DObjects]    pip install 'git+https://github.com/facebookresearch/pytorch3d.git'")

    return False


def install():
    """Install dependencies for SAM3DObjects node."""
    print("[SAM3DObjects] Starting installation...")

    # Get the directory of this script
    script_dir = Path(__file__).parent
    requirements_file = script_dir / "requirements.txt"

    if not requirements_file.exists():
        print(f"[SAM3DObjects] Error: requirements.txt not found at {requirements_file}")
        return False

    # Check/install pytorch3d separately first
    if not check_pytorch3d():
        if not install_pytorch3d():
            print("[SAM3DObjects] ")
            print("[SAM3DObjects] WARNING: pytorch3d installation failed!")
            print("[SAM3DObjects] You may need to install it manually before using this node.")
            print("[SAM3DObjects] Continuing with other dependencies...")
            print("[SAM3DObjects] ")

    # sam3d_objects is VENDORED (no installation needed!)
    print("[SAM3DObjects] sam3d_objects code is vendored (included in vendor/ directory)")
    print("[SAM3DObjects] No external installation required!")

    try:
        # Install other requirements (excluding pytorch3d line)
        print("[SAM3DObjects] Installing remaining Python dependencies...")

        # Read requirements and filter out pytorch3d and sam3d_objects (already installed)
        with open(requirements_file) as f:
            requirements = [
                line.strip()
                for line in f
                if line.strip()
                and not line.strip().startswith('#')
                and 'pytorch3d' not in line.lower()
                and 'sam-3d-objects' not in line.lower()
                and 'sam3d_objects' not in line.lower()
            ]

        # Install each requirement
        for req in requirements:
            try:
                print(f"[SAM3DObjects] Installing: {req}")
                subprocess.check_call([
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    req
                ])
            except subprocess.CalledProcessError as e:
                print(f"[SAM3DObjects] Warning: Failed to install {req}: {e}")
                print("[SAM3DObjects] Continuing with other packages...")

        print("[SAM3DObjects] ")
        print("[SAM3DObjects] Installation completed successfully!")
        print("[SAM3DObjects] ")
        print("[SAM3DObjects] ✅ COMPLETELY STANDALONE!")
        print("[SAM3DObjects] - sam3d_objects code is vendored (no GitHub cloning!)")
        print("[SAM3DObjects] - No external dependencies on repositories")
        print("[SAM3DObjects] - All code included in this package")
        print("[SAM3DObjects] ")
        print("[SAM3DObjects] REQUIREMENTS:")
        print("[SAM3DObjects] - NVIDIA GPU with at least 32GB VRAM")
        print("[SAM3DObjects] - Model checkpoints will be auto-downloaded on first use")
        print("[SAM3DObjects] - Checkpoints saved to: ComfyUI/models/sam3d/")
        print("[SAM3DObjects] ")

        return True

    except Exception as e:
        print(f"[SAM3DObjects] Error during installation: {e}")
        return False


if __name__ == "__main__":
    success = install()
    sys.exit(0 if success else 1)
