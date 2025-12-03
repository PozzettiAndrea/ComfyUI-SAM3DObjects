"""
PyTorch, torchvision, and PyTorch3D installers.
"""

import subprocess
from pathlib import Path

from .base import Installer


class PyTorchInstaller(Installer):
    """
    Install PyTorch, torchvision, and PyTorch3D via micromamba.

    Uses micromamba to install from conda channels, ensuring
    CUDA compatibility between torch, torchvision, and pytorch3d.
    """

    @property
    def name(self) -> str:
        return "PyTorch + PyTorch3D"

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
        """Install PyTorch + PyTorch3D via micromamba."""
        self.logger.info(f"Installing PyTorch {self.config.pytorch_version} + PyTorch3D {self.config.pytorch3d_version}...")
        self.logger.info(f"(Using PyTorch {self.config.pytorch_version} - latest with PyTorch3D prebuilt support)")

        try:
            # Install PyTorch + PyTorch3D together via micromamba
            # This ensures CUDA version compatibility
            self.run_micromamba(
                [
                    "install",
                    "-p", str(self.env_dir),
                    "-c", "pytorch",       # PRIMARY: PyTorch official channel
                    "-c", "pytorch3d",     # PyTorch3D channel
                    "-c", "nvidia",        # NVIDIA CUDA packages
                    "-c", "fvcore",        # PyTorch3D dependency
                    "-c", "conda-forge",   # Fallback
                    f"pytorch=={self.config.pytorch_version}",
                    f"pytorch-cuda={self.config.cuda_version}",
                    f"torchvision=={self.config.torchvision_version}",
                    f"pytorch3d=={self.config.pytorch3d_version}",
                    "-y"
                ],
                step_name=f"Install PyTorch {self.config.pytorch_version} + PyTorch3D via micromamba",
                check=True
            )

            # Verify installation
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


class PipDependenciesInstaller(Installer):
    """
    Install pip dependencies from requirements file.

    Installs packages while protecting PyTorch version from being upgraded.
    """

    def __init__(self, *args, requirements_file: Path, **kwargs):
        super().__init__(*args, **kwargs)
        self.requirements_file = requirements_file

    @property
    def name(self) -> str:
        return "Pip Dependencies"

    def is_installed(self) -> bool:
        # Always run to ensure all dependencies are present
        return False

    def install(self) -> bool:
        """Install pip dependencies with PyTorch constraints."""
        self.logger.info("Installing pip dependencies...")

        if not self.requirements_file.exists():
            self.logger.error(f"Requirements file not found: {self.requirements_file}")
            return False

        try:
            # Step 1: Upgrade pip
            self.logger.info("Upgrading pip...")
            self.run_pip(["install", "--upgrade", "pip"], step_name="Upgrade pip", check=True)

            # Step 2: Install uv for faster package installation
            self.logger.info("Installing uv package manager...")
            self.run_pip(["install", "uv"], step_name="Install uv", check=True)

            # Step 3: Install packages without deps first to avoid PyTorch upgrade
            self.logger.info("Installing packages (protecting PyTorch version)...")
            self.run_uv_pip(
                [
                    "install",
                    "--no-deps",
                    "-r", str(self.requirements_file)
                ],
                step_name="Install packages (no deps)",
                check=True
            )

            # Step 4: Install with constraints to keep PyTorch at specific version
            constraints_file = self.env_dir.parent / "_pytorch_constraints.txt"
            with open(constraints_file, 'w') as f:
                f.write(f"torch=={self.config.pytorch_version}\n")
                f.write(f"torchvision=={self.config.torchvision_version}\n")

            self.run_pip(
                [
                    "install",
                    "-r", str(self.requirements_file),
                    "-c", str(constraints_file),
                    "--upgrade",
                ],
                step_name="Install dependencies with constraints",
                check=True
            )

            # Clean up
            if constraints_file.exists():
                constraints_file.unlink()

            self.logger.success("Pip dependencies installed")
            return True

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Pip dependencies installation failed: {e}")
            return False
