"""Installation script for ComfyUI-SAM3DObjects custom node."""

import subprocess
import sys
import os
import shutil
from pathlib import Path
import platform
import urllib.request
import tarfile
import stat


def get_micromamba_platform():
    """
    Determine the correct micromamba platform string for current OS/architecture.

    Returns:
        str: Platform string (e.g., 'linux-64', 'osx-arm64', 'win-64')

    Raises:
        ValueError: If platform is not supported
    """
    system = platform.system()
    machine = platform.machine().lower()

    if system == "Linux":
        if machine in ["x86_64", "amd64"]:
            return "linux-64"
        elif machine in ["aarch64", "arm64"]:
            return "linux-aarch64"
        elif machine in ["ppc64le"]:
            return "linux-ppc64le"
    elif system == "Darwin":  # macOS
        if machine in ["x86_64", "amd64"]:
            return "osx-64"
        elif machine in ["arm64", "aarch64"]:
            return "osx-arm64"
    elif system == "Windows":
        if machine in ["x86_64", "amd64", "amd64"]:
            return "win-64"

    raise ValueError(f"Unsupported platform: {system} {machine}")


def download_micromamba(install_dir):
    """
    Download and install micromamba to the specified directory.

    Args:
        install_dir (Path): Directory to install micromamba

    Returns:
        str: Path to the micromamba executable

    Raises:
        Exception: If download or extraction fails
    """
    platform_str = get_micromamba_platform()
    url = f"https://micro.mamba.pm/api/micromamba/{platform_str}/latest"

    install_path = Path(install_dir)
    install_path.mkdir(parents=True, exist_ok=True)

    # Download micromamba
    print(f"[SAM3DObjects] Downloading micromamba for {platform_str}...")
    tar_path = install_path / "micromamba.tar.bz2"

    urllib.request.urlretrieve(url, tar_path)

    # Extract the binary
    print("[SAM3DObjects] Extracting micromamba...")
    with tarfile.open(tar_path, "r:bz2") as tar:
        # Find and extract the micromamba executable
        for member in tar.getmembers():
            if member.name.endswith("bin/micromamba") or member.name.endswith("micromamba.exe"):
                # Extract with basename only (remove directory structure)
                member.name = os.path.basename(member.name)
                tar.extract(member, install_path)
                break

    # Determine executable name based on platform
    exe_name = "micromamba.exe" if platform.system() == "Windows" else "micromamba"
    micromamba_path = install_path / exe_name

    # Make executable on Unix systems
    if platform.system() != "Windows":
        current_permissions = os.stat(micromamba_path).st_mode
        os.chmod(
            micromamba_path,
            current_permissions | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
        )

    # Clean up tar file
    tar_path.unlink()

    print(f"[SAM3DObjects] ✓ Micromamba installed to: {micromamba_path}")
    return str(micromamba_path)


def find_conda_command():
    """
    Find available conda/mamba/micromamba command.
    Downloads micromamba automatically if none are found.

    Returns:
        str or None: Command name/path if found, None if all methods fail
    """
    # First check for existing installations in PATH
    for cmd in ['mamba', 'micromamba', 'conda']:
        if shutil.which(cmd):
            print(f"[SAM3DObjects] Found '{cmd}' package manager in PATH")
            return cmd

    # Check if we already downloaded micromamba
    script_dir = Path(__file__).parent
    bin_dir = script_dir / ".bin"
    exe_name = "micromamba.exe" if platform.system() == "Windows" else "micromamba"
    local_micromamba = bin_dir / exe_name

    if local_micromamba.exists():
        print(f"[SAM3DObjects] Using previously downloaded micromamba at: {local_micromamba}")
        return str(local_micromamba)

    # None found - download micromamba automatically
    print("[SAM3DObjects] No conda/mamba/micromamba found in PATH")
    print("[SAM3DObjects] Downloading micromamba automatically...")

    try:
        micromamba_path = download_micromamba(bin_dir)
        return micromamba_path
    except Exception as e:
        print(f"[SAM3DObjects] Failed to download micromamba: {e}")
        print("[SAM3DObjects] Installation will fall back to pip (slower build from source)")
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


def check_kaolin():
    """Check if kaolin is already installed."""
    try:
        import kaolin
        print(f"[SAM3DObjects] ✓ kaolin {kaolin.__version__} is already installed")
        return True
    except ImportError:
        print("[SAM3DObjects] kaolin not installed")
        return False


def check_gsplat():
    """Check if gsplat is already installed."""
    try:
        import gsplat
        print("[SAM3DObjects] ✓ gsplat is already installed")
        return True
    except ImportError:
        print("[SAM3DObjects] gsplat not installed")
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


def install_kaolin():
    """
    Install kaolin using conda (preferred) or pip (fallback).

    Returns:
        bool: True if successful, False otherwise
    """
    conda_cmd = find_conda_command()

    # Try conda installation first
    if conda_cmd:
        print(f"[SAM3DObjects] Installing kaolin via {conda_cmd}...")
        try:
            subprocess.check_call([
                conda_cmd, "install", "-y",
                "-c", "nvidia",
                "-c", "conda-forge",
                "kaolin"
            ])
            if check_kaolin():
                return True
        except subprocess.CalledProcessError as e:
            print(f"[SAM3DObjects] Conda installation failed: {e}")

    # Fallback to pip
    print("[SAM3DObjects] Trying pip installation for kaolin...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "kaolin>=0.17.0"
        ])
        return check_kaolin()
    except subprocess.CalledProcessError as e:
        print(f"[SAM3DObjects] Pip installation failed: {e}")
        return False


def install_gsplat():
    """
    Install gsplat using conda (preferred) or pip (fallback).

    Returns:
        bool: True if successful, False otherwise
    """
    conda_cmd = find_conda_command()

    # Try conda installation first
    if conda_cmd:
        print(f"[SAM3DObjects] Installing gsplat via {conda_cmd}...")
        try:
            subprocess.check_call([
                conda_cmd, "install", "-y",
                "-c", "conda-forge",
                "gsplat"
            ])
            if check_gsplat():
                return True
        except subprocess.CalledProcessError as e:
            print(f"[SAM3DObjects] Conda installation failed: {e}")

    # Fallback to pip
    print("[SAM3DObjects] Trying pip installation for gsplat...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "gsplat>=1.4.0"
        ])
        return check_gsplat()
    except subprocess.CalledProcessError as e:
        print(f"[SAM3DObjects] Pip installation failed: {e}")
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
    print("[SAM3DObjects] Step 2/5: Checking pytorch3d installation...")
    # Check/install pytorch3d separately first
    if not check_pytorch3d():
        if not install_pytorch3d():
            print("[SAM3DObjects] ")
            print("[SAM3DObjects] WARNING: pytorch3d installation failed!")
            print("[SAM3DObjects] You may need to install it manually before using this node.")
            print("[SAM3DObjects] Continuing with other dependencies...")
            print("[SAM3DObjects] ")

    print("[SAM3DObjects] ")
    print("[SAM3DObjects] Step 3/5: Checking kaolin installation...")
    if not check_kaolin():
        if not install_kaolin():
            print("[SAM3DObjects] ")
            print("[SAM3DObjects] WARNING: kaolin installation failed!")
            print("[SAM3DObjects] Visualization features may be limited.")
            print("[SAM3DObjects] ")

    print("[SAM3DObjects] ")
    print("[SAM3DObjects] Step 4/5: Checking gsplat installation...")
    if not check_gsplat():
        if not install_gsplat():
            print("[SAM3DObjects] ")
            print("[SAM3DObjects] WARNING: gsplat installation failed!")
            print("[SAM3DObjects] Gaussian splatting rendering may not work.")
            print("[SAM3DObjects] ")

    # sam3d_objects is VENDORED (no installation needed!)
    print("[SAM3DObjects] sam3d_objects code is vendored (included in vendor/ directory)")
    print("[SAM3DObjects] No external installation required!")

    try:
        print("[SAM3DObjects] ")
        print("[SAM3DObjects] Step 5/5: Installing remaining Python dependencies...")
        print("[SAM3DObjects] ")

        # Read requirements and filter out packages we install separately
        with open(requirements_file) as f:
            requirements = [
                line.strip()
                for line in f
                if line.strip()
                and not line.strip().startswith('#')
                and 'pytorch3d' not in line.lower()
                and 'kaolin' not in line.lower()
                and 'gsplat' not in line.lower()
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
