"""
SAM3D Environment Manager - Orchestrates the installation process.
"""

import subprocess
from pathlib import Path
from typing import List

from .config import InstallConfig
from .platform import get_platform
from .platform.base import PlatformProvider
from .utils import Logger
from .installers.base import Installer
from .installers.micromamba import MicromambaInstaller
from .installers.pytorch import PyTorchInstaller, PipDependenciesInstaller
from .installers.cuda import CudaToolkitInstaller, CompilerInstaller
from .installers.specialized import GsplatInstaller, NvdiffrastInstaller, KaolinInstaller


class SAM3DEnvironmentManager:
    """
    Manages the isolated environment for SAM3D.

    Orchestrates the installation of all dependencies in the correct order.
    """

    def __init__(self, node_root: Path, config: InstallConfig = None):
        """
        Initialize environment manager.

        Args:
            node_root: Root directory of the ComfyUI-SAM3DObjects node
            config: Installation configuration (optional, uses defaults)
        """
        self.node_root = Path(node_root)
        self.config = config or InstallConfig()
        self.platform = get_platform()

        # Check platform compatibility
        is_compatible, error = self.platform.check_prerequisites()
        if not is_compatible:
            self._print_error_banner("PLATFORM COMPATIBILITY ERROR", error)
            raise RuntimeError(f"Incompatible platform: {error}")

        # Setup paths
        self.env_dir = self.node_root / self.config.env_name
        self.tools_dir = self.node_root / self.config.tools_dir
        self.log_file = self.node_root / self.config.log_file

        # Initialize logger
        self.logger = Logger(self.log_file)

        # Micromamba installer (handles both download and env creation)
        self._micromamba_installer = None

    def _print_error_banner(self, title: str, message: str):
        """Print a formatted error banner."""
        print(f"\n{'='*60}")
        print(f"[SAM3DObjects] {title}")
        print('='*60)
        print(message)
        print('='*60)

    def get_python_executable(self) -> Path:
        """Get path to Python executable in isolated environment."""
        return self.platform.get_env_paths(self.env_dir).python

    def get_pip_executable(self) -> Path:
        """Get path to pip executable in isolated environment."""
        return self.platform.get_env_paths(self.env_dir).pip

    def is_environment_ready(self) -> bool:
        """Check if environment is fully set up."""
        python_exe = self.get_python_executable()
        if not python_exe.exists():
            return False

        # Verify critical packages
        try:
            result = subprocess.run(
                [str(python_exe), "-c", "import torch, pytorch3d"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, OSError):
            return False

    def setup_environment(self) -> None:
        """
        Run the complete installation process.

        Raises:
            RuntimeError: If installation fails
        """
        self.logger.info("=" * 40)
        self.logger.info("ComfyUI-SAM3DObjects Installation")
        self.logger.info("=" * 40)
        self.logger.info("")
        self.logger.info("This will create an isolated Python venv")
        self.logger.info("for SAM3D inference, completely separate from ComfyUI.")
        self.logger.info("")
        self.logger.info("Starting installation...")
        self.logger.info(f"Full logs will be saved to: {self.log_file}")

        # Step 1: Setup micromamba
        self._micromamba_installer = MicromambaInstaller(
            self.tools_dir, self.env_dir, self.platform, self.config, self.logger
        )

        if not self._micromamba_installer.is_installed():
            if not self._micromamba_installer.install():
                raise RuntimeError("Failed to download micromamba")

        # Step 2: Create environment
        if not self._micromamba_installer.create_environment():
            raise RuntimeError("Failed to create Python environment")

        # Step 3: Run installers in order
        installers = self._get_installers()

        for installer in installers:
            self.logger.info(f"")
            self.logger.info(f"--- {installer.name} ---")

            if installer.is_installed():
                self.logger.info(f"Already installed, skipping")
                continue

            try:
                if not installer.install():
                    raise RuntimeError(f"{installer.name} installation failed")
            except Exception as e:
                self.logger.error(f"Error installing {installer.name}: {e}")
                raise RuntimeError(f"Installation failed at {installer.name}: {e}") from e

        # Final verification
        self.logger.info("")
        self.logger.info("=" * 40)

        if self.is_environment_ready():
            self.logger.info("")
            self.logger.success("Installation complete!")
            self.logger.info(f"Full logs: {self.log_file}")
        else:
            raise RuntimeError("Installation complete but verification failed")

    def _get_installers(self) -> List[Installer]:
        """
        Get ordered list of installers.

        Returns:
            List of Installer instances in execution order
        """
        micromamba_path = self._micromamba_installer.micromamba_path

        # Common arguments for all installers
        common_kwargs = {
            'env_dir': self.env_dir,
            'platform': self.platform,
            'config': self.config,
            'logger': self.logger,
            'micromamba_exe': micromamba_path,
        }

        # Requirements file path
        requirements_file = self.node_root / "local_env_settings" / "requirements_env.txt"

        return [
            # PyTorch + PyTorch3D (via micromamba for CUDA compatibility)
            PyTorchInstaller(**common_kwargs),

            # Pip dependencies (with PyTorch constraints)
            PipDependenciesInstaller(
                requirements_file=requirements_file,
                **common_kwargs
            ),

            # gsplat (prebuilt wheel)
            GsplatInstaller(**common_kwargs),

            # nvdiffrast (prebuilt wheel or source)
            NvdiffrastInstaller(**common_kwargs),

            # kaolin (from NVIDIA S3)
            KaolinInstaller(**common_kwargs),

            # CUDA toolkit (for JIT compilation)
            CudaToolkitInstaller(**common_kwargs),

            # C++ compiler (for CUDA JIT)
            CompilerInstaller(**common_kwargs),
        ]
