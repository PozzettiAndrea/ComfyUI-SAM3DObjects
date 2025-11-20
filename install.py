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


def check_pytorch_cuda():
    """
    Check if PyTorch with CUDA is installed.

    Returns:
        bool: True if PyTorch with CUDA is available, False otherwise
    """
    try:
        import torch
        # Check if torch has cuda attribute (may be missing in broken installations)
        if not hasattr(torch, 'cuda'):
            print("[SAM3DObjects] WARNING: PyTorch installation is broken (no cuda module)")
            return False

        if not torch.cuda.is_available():
            print("[SAM3DObjects] WARNING: PyTorch is installed but CUDA is not available!")
            print(f"[SAM3DObjects] PyTorch version: {torch.__version__}")
            cuda_version = torch.version.cuda if hasattr(torch.version, 'cuda') else 'No'
            print(f"[SAM3DObjects] CUDA compiled: {cuda_version}")
            return False
        print(f"[SAM3DObjects] ✓ PyTorch {torch.__version__} with CUDA {torch.version.cuda} detected")
        return True
    except (ImportError, AttributeError, OSError) as e:
        print(f"[SAM3DObjects] PyTorch not installed or broken: {type(e).__name__}")
        return False


def install_pytorch_cuda():
    """
    Install PyTorch with CUDA 12.1 support.

    Returns:
        bool: True if successful, False otherwise
    """
    print("[SAM3DObjects] ")
    print("[SAM3DObjects] Installing PyTorch with CUDA 12.1 support...")
    print("[SAM3DObjects] This is required for SAM3DObjects to function properly.")
    print("[SAM3DObjects] ")

    try:
        # Clean up any broken installations first
        print("[SAM3DObjects] Cleaning up any existing/broken PyTorch installation...")

        # Force remove torch directories if they exist
        import site
        site_packages = site.getsitepackages()[0] if site.getsitepackages() else None
        if site_packages:
            import shutil
            torch_dirs = [
                os.path.join(site_packages, 'torch'),
                os.path.join(site_packages, 'torchvision'),
                os.path.join(site_packages, 'torchaudio'),
            ]
            for torch_dir in torch_dirs:
                if os.path.exists(torch_dir):
                    try:
                        shutil.rmtree(torch_dir)
                        print(f"[SAM3DObjects]   Removed: {torch_dir}")
                    except Exception as e:
                        print(f"[SAM3DObjects]   Could not remove {torch_dir}: {e}")

        # Uninstall via pip
        subprocess.run([
            sys.executable,
            "-m",
            "pip",
            "uninstall",
            "-y",
            "torch",
            "torchvision",
            "torchaudio"
        ], check=False, capture_output=True)

        # Install CUDA-enabled PyTorch
        print("[SAM3DObjects] Installing PyTorch with CUDA 12.1...")
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "torch",
            "torchvision",
            "torchaudio",
            "--index-url",
            "https://download.pytorch.org/whl/cu121"
        ])

        # Verify installation
        print("[SAM3DObjects] Verifying CUDA installation...")
        result = subprocess.run([
            sys.executable,
            "-c",
            "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
        ], capture_output=True, text=True)

        if "CUDA available: True" in result.stdout:
            print("[SAM3DObjects] ✓ PyTorch with CUDA installed successfully!")
            return True
        else:
            print("[SAM3DObjects] ✗ CUDA verification failed!")
            print(f"[SAM3DObjects] Output: {result.stdout}")
            if result.stderr:
                print(f"[SAM3DObjects] Error: {result.stderr}")
            return False

    except subprocess.CalledProcessError as e:
        print(f"[SAM3DObjects] ✗ Failed to install PyTorch with CUDA: {e}")
        return False


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

    # Check PyTorch CUDA FIRST - this is critical for all other dependencies
    print("[SAM3DObjects] ")
    print("[SAM3DObjects] Step 1/3: Checking PyTorch CUDA installation...")
    if not check_pytorch_cuda():
        print("[SAM3DObjects] ")
        print("[SAM3DObjects] ⚠️  PyTorch with CUDA is REQUIRED for SAM3DObjects")
        print("[SAM3DObjects] Attempting automatic installation...")
        print("[SAM3DObjects] ")

        if not install_pytorch_cuda():
            print("[SAM3DObjects] ")
            print("[SAM3DObjects] ✗ ERROR: Failed to install PyTorch with CUDA!")
            print("[SAM3DObjects] ")
            print("[SAM3DObjects] MANUAL INSTALLATION REQUIRED:")
            print("[SAM3DObjects]   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
            print("[SAM3DObjects] ")
            print("[SAM3DObjects] After installing PyTorch with CUDA, run this script again.")
            return False

        # Verify the installation worked
        if not check_pytorch_cuda():
            print("[SAM3DObjects] ")
            print("[SAM3DObjects] ✗ ERROR: PyTorch CUDA verification failed after installation!")
            print("[SAM3DObjects] This may indicate a system CUDA compatibility issue.")
            print("[SAM3DObjects] Please check that CUDA drivers are properly installed.")
            return False

    print("[SAM3DObjects] ")
    print("[SAM3DObjects] Step 2/3: Checking pytorch3d installation...")
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
        print("[SAM3DObjects] ")
        print("[SAM3DObjects] Step 3/3: Installing remaining Python dependencies...")
        print("[SAM3DObjects] ")

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

        # Install each requirement with proper CUDA support
        for req in requirements:
            try:
                print(f"[SAM3DObjects] Installing: {req}")

                # Build install command
                install_cmd = [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    req
                ]

                # For xformers, use CUDA 12.1 index to ensure CUDA-enabled version
                if "xformers" in req.lower():
                    install_cmd.extend(["--index-url", "https://download.pytorch.org/whl/cu121"])
                    print(f"[SAM3DObjects]   → Using CUDA 12.1 index for xformers")

                subprocess.check_call(install_cmd)

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
