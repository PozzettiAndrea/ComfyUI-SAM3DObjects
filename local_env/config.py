"""
Configuration constants and dataclasses for SAM3D installation.

This module centralizes all version numbers, URLs, and configuration
options for the installation process.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class InstallConfig:
    """Installation configuration options."""

    # Python version for the isolated environment
    python_version: str = "3.10"

    # PyTorch version - pinned for PyTorch3D compatibility
    # NOTE: PyTorch3D 0.7.8 requires PyTorch 2.4.x
    pytorch_version: str = "2.4.1"
    torchvision_version: str = "0.19.1"

    # CUDA version for GPU support
    cuda_version: str = "12.4"

    # PyTorch3D version
    pytorch3d_version: str = "0.7.8"

    # Directory names within the node root
    env_name: str = "_env"
    tools_dir: str = "_tools"
    log_file: str = "install.log"

    # Package versions
    gsplat_version: str = "1.4.0"
    nvdiffrast_version: str = "0.3.5"


# Micromamba download URLs by platform
MICROMAMBA_URLS: Dict[str, Dict[str, str]] = {
    "Linux": {
        "x86_64": "https://micro.mamba.pm/api/micromamba/linux-64/latest",
        "aarch64": "https://micro.mamba.pm/api/micromamba/linux-aarch64/latest",
    },
    "Darwin": {
        "arm64": "https://micro.mamba.pm/api/micromamba/osx-arm64/latest",
        "x86_64": "https://micro.mamba.pm/api/micromamba/osx-64/latest",
    },
    "Windows": {
        "x86_64": "https://micro.mamba.pm/api/micromamba/win-64/latest",
        "AMD64": "https://micro.mamba.pm/api/micromamba/win-64/latest",
    },
}

# nvdiffrast prebuilt wheel URLs by platform
NVDIFFRAST_WHEEL_URLS: Dict[str, str] = {
    "Linux": "https://huggingface.co/spaces/microsoft/TRELLIS/resolve/main/wheels/nvdiffrast-0.3.3-cp310-cp310-linux_x86_64.whl",
    "Windows": "https://pozzettiandrea.github.io/nvdiffrast-wheels/cu124/nvdiffrast-0.3.5%2Bpt2.4.1cu124-py3-none-any.whl",
    # Darwin: No prebuilt wheel, falls back to source compilation
}

# CUDA toolkit conda packages by platform
CUDA_TOOLKIT_URLS: Dict[str, str] = {
    "Linux": "https://conda.anaconda.org/conda-forge/linux-64/cudatoolkit-dev-12.1.0-h4b99516_3.conda",
    "Windows": "https://conda.anaconda.org/conda-forge/win-64/cudatoolkit-dev-12.1.0-hd020da6_3.conda",
    "Darwin": "https://conda.anaconda.org/conda-forge/osx-64/cudatoolkit-dev-12.1.0-h2e7b6a8_3.conda",
}

# gsplat wheel index URL template
def get_gsplat_index_url(pytorch_version: str, cuda_version: str) -> str:
    """Get gsplat wheel index URL for specific PyTorch/CUDA versions."""
    pt = pytorch_version.replace(".", "")[:2]  # "2.4.1" -> "24"
    cu = cuda_version.replace(".", "")  # "12.1" -> "121"
    return f"https://docs.gsplat.studio/whl/pt{pt}cu{cu}"


# PyTorch pip wheel index URL (with CUDA support)
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
