"""
PyTorch and PyTorch3D installation via pip wheels.

Used on Windows instead of micromamba-based installation.
"""

import subprocess

from .base import Installer
from ..config import (
    PYTORCH_PIP_INDEX_URL,
    PYTORCH3D_PIP_INDEX_URL,
    get_pytorch3d_pip_version,
)


class PyTorchPipInstaller(Installer):
    """
    Install PyTorch, torchvision, and PyTorch3D via pip wheels.

    This is used on Windows instead of micromamba, installing from:
    - PyTorch: Official PyTorch wheel index (with CUDA support)
    - PyTorch3D: Third-party prebuilt wheels from MiroPsota's repository
    """

    @property
    def name(self) -> str:
        return "PyTorch + PyTorch3D (pip)"

    def is_installed(self) -> bool:
        """Check if PyTorch and PyTorch3D are installed with CUDA."""
        result = self.run_python(
            "import torch, pytorch3d; "
            "print(f'PyTorch {torch.__version__}, PyTorch3D {pytorch3d.__version__}, CUDA {torch.cuda.is_available()}')"
        )
        if result.returncode != 0:
            return False
        # Verify CUDA is available
        return 'True' in result.stdout

    def install(self) -> bool:
        """Install PyTorch + PyTorch3D via pip wheels."""
        self.logger.info(f"Installing PyTorch {self.config.pytorch_version} + PyTorch3D {self.config.pytorch3d_version} via pip...")

        try:
            # Step 1: Upgrade pip and install uv for faster installs
            self.logger.info("Upgrading pip...")
            self.run_pip(["install", "--upgrade", "pip"], step_name="Upgrade pip", check=True)
            self.logger.info("Installing uv package manager...")
            self.run_pip(["install", "uv"], step_name="Install uv", check=True)

            # Step 2: Install PyTorch with CUDA from official index (using uv)
            self.logger.info(f"Installing PyTorch {self.config.pytorch_version} with CUDA {self.config.cuda_version}...")
            self.run_uv_pip(
                [
                    "install",
                    f"torch=={self.config.pytorch_version}",
                    f"torchvision=={self.config.torchvision_version}",
                    "--index-url", PYTORCH_PIP_INDEX_URL,
                ],
                step_name=f"Install PyTorch {self.config.pytorch_version} via uv",
                check=True
            )

            # Verify PyTorch installation and CUDA
            result = self.run_python(
                "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
            )
            if result.returncode != 0:
                raise RuntimeError(f"PyTorch verification failed: {result.stderr}")
            self.logger.info(f"Verified: {result.stdout.strip()}")

            # Step 3: Install PyTorch3D from third-party wheel index (using uv)
            pytorch3d_version = get_pytorch3d_pip_version(
                self.config.pytorch_version,
                self.config.cuda_version
            )
            self.logger.info(f"Installing PyTorch3D {pytorch3d_version}...")
            self.logger.info("(Using prebuilt wheel from MiroPsota's repository)")

            self.run_uv_pip(
                [
                    "install",
                    f"pytorch3d=={pytorch3d_version}",
                    "--extra-index-url", PYTORCH3D_PIP_INDEX_URL,
                ],
                step_name=f"Install PyTorch3D via uv",
                check=True
            )

            # Final verification
            result = self.run_python(
                "import torch, pytorch3d; "
                f"print(f'PyTorch {{torch.__version__}}, PyTorch3D {{pytorch3d.__version__}}')"
            )

            if result.returncode == 0:
                self.logger.success(f"Verified: {result.stdout.strip()}")
                return True
            else:
                raise RuntimeError(f"Verification failed: {result.stderr}")

        except subprocess.CalledProcessError as e:
            self.logger.error(f"PyTorch installation failed: {e}")
            return False
