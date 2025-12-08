"""
Configuration constants and dataclasses for SAM3D installation.

This module centralizes all version numbers, URLs, and configuration
options for the installation process.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

from .gpu_detection import get_recommended_cuda_version, get_gpu_summary


def _get_pytorch_version(cuda_version: str) -> str:
    """Get PyTorch version based on CUDA version."""
    if cuda_version == "12.8":
        return "2.8.0"
    return "2.4.1"


def _get_torchvision_version(cuda_version: str) -> str:
    """Get torchvision version based on CUDA version."""
    if cuda_version == "12.8":
        return "0.23.0"
    return "0.19.1"


@dataclass
class InstallConfig:
    """Installation configuration options."""

    # Python version for the isolated environment
    python_version: str = "3.10"

    # CUDA version - auto-detected based on GPU
    # Blackwell GPUs (RTX 50xx) need 12.8, others use 12.4
    cuda_version: str = field(default_factory=get_recommended_cuda_version)

    # PyTorch version - depends on CUDA version
    # cu124 -> PyTorch 2.4.1, cu128 -> PyTorch 2.8.0
    pytorch_version: str = field(default="")
    torchvision_version: str = field(default="")

    # PyTorch3D version
    pytorch3d_version: str = "0.7.8"

    # Directory names within the node root
    env_name: str = "_env"
    tools_dir: str = "_tools"
    log_file: str = "install.log"

    # Package versions
    gsplat_version: str = "1.5.0"
    nvdiffrast_version: str = "0.4.0"

    def __post_init__(self):
        """Set PyTorch versions based on detected CUDA version."""
        if not self.pytorch_version:
            self.pytorch_version = _get_pytorch_version(self.cuda_version)
        if not self.torchvision_version:
            self.torchvision_version = _get_torchvision_version(self.cuda_version)

    def get_gpu_info(self) -> str:
        """Get GPU detection summary for logging."""
        return get_gpu_summary()


# sam3dobjects-wheels GitHub releases base URL
SAM3DO_WHEELS_BASE = "https://github.com/PozzettiAndrea/sam3dobjects-wheels/releases/download"


def get_nvdiffrast_wheel_url(cuda_version: str, platform: str) -> Optional[str]:
    """
    Get nvdiffrast wheel URL for specific CUDA version and platform.

    Args:
        cuda_version: "12.4" or "12.8"
        platform: "Linux" or "Windows"

    Returns:
        Wheel URL or None if not available
    """
    cu = cuda_version.replace(".", "")  # "12.4" -> "124"
    tag = f"nvdiffrast-cu{cu}"

    if platform == "Linux":
        wheel = f"nvdiffrast-0.4.0%2Bcu{cu}-cp310-cp310-linux_x86_64.whl"
    elif platform == "Windows":
        wheel = f"nvdiffrast-0.4.0%2Bcu{cu}-cp310-cp310-win_amd64.whl"
    else:
        return None  # macOS - fall back to source

    return f"{SAM3DO_WHEELS_BASE}/{tag}/{wheel}"


def get_gsplat_wheel_url(cuda_version: str, platform: str) -> Optional[str]:
    """
    Get gsplat wheel URL for specific CUDA version and platform.

    Args:
        cuda_version: "12.4" or "12.8"
        platform: "Linux" or "Windows"

    Returns:
        Wheel URL or None if not available
    """
    cu = cuda_version.replace(".", "")  # "12.4" -> "124"
    tag = f"gsplat-cu{cu}"

    if platform == "Linux":
        wheel = f"gsplat-1.5.3%2Bcu{cu}-cp310-cp310-linux_x86_64.whl"
    elif platform == "Windows":
        wheel = f"gsplat-1.5.3%2Bcu{cu}-cp310-cp310-win_amd64.whl"
    else:
        return None  # macOS - fall back to source

    return f"{SAM3DO_WHEELS_BASE}/{tag}/{wheel}"


# Legacy: gsplat wheel index URL template (fallback to official)
def get_gsplat_index_url(pytorch_version: str, cuda_version: str) -> str:
    """Get gsplat wheel index URL for specific PyTorch/CUDA versions."""
    pt = pytorch_version.replace(".", "")[:2]  # "2.4.1" -> "24"
    cu = cuda_version.replace(".", "")  # "12.1" -> "121"
    return f"https://docs.gsplat.studio/whl/pt{pt}cu{cu}"


def get_pytorch_index_url(cuda_version: str) -> str:
    """Get PyTorch wheel index URL for specific CUDA version."""
    cu = cuda_version.replace(".", "")  # "12.4" -> "124"
    return f"https://download.pytorch.org/whl/cu{cu}"


# Legacy constant for backwards compatibility
PYTORCH_PIP_INDEX_URL = "https://download.pytorch.org/whl/cu124"

# PyTorch3D third-party wheel index (MiroPsota's repository)
# See: https://github.com/MiroPsota/torch_packages_builder
PYTORCH3D_PIP_INDEX_URL = "https://miropsota.github.io/torch_packages_builder"


def get_pytorch3d_pip_version(pytorch_version: str, cuda_version: str) -> str:
    """
    Get PyTorch3D version string for pip install.

    The third-party wheels use a specific version format that includes
    a commit hash and the PyTorch version.

    Args:
        pytorch_version: PyTorch version (e.g., "2.4.1")
        cuda_version: CUDA version (e.g., "12.1")

    Returns:
        Version string like "0.7.8+5043d15pt2.4.1cu121"
    """
    cu = cuda_version.replace(".", "")  # "12.1" -> "121"
    return f"0.7.8+5043d15pt{pytorch_version}cu{cu}"
