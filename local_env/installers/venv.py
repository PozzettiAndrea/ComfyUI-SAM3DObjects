"""
Virtual environment creation using Python's venv module.

Used on Windows instead of micromamba.
"""

import subprocess
import sys
from pathlib import Path

from .base import Installer


class VenvInstaller(Installer):
    """
    Create Python virtual environment using venv module.

    This is used on Windows instead of micromamba, which has
    compatibility issues on some Windows systems.
    """

    def __init__(self, env_dir: Path, platform_provider, config, logger):
        """
        Initialize venv installer.

        Args:
            env_dir: Environment directory to create
            platform_provider: Platform provider
            config: Installation config
            logger: Logger instance
        """
        super().__init__(
            env_dir=env_dir,
            platform=platform_provider,
            config=config,
            logger=logger,
        )

    @property
    def name(self) -> str:
        return "Python Virtual Environment"

    def is_installed(self) -> bool:
        """Check if venv already exists and is working."""
        if not self.env_dir.exists():
            return False

        python_exe = self._paths.python
        if not python_exe.exists():
            return False

        # Verify it works
        try:
            result = subprocess.run(
                [str(python_exe), "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, OSError):
            return False

    def install(self) -> bool:
        """Create Python virtual environment using venv."""
        self.logger.info("Creating Python virtual environment...")

        # Use the current Python interpreter to create venv
        python_exe = sys.executable

        try:
            # Create venv
            self.logger.run_logged(
                [python_exe, "-m", "venv", str(self.env_dir)],
                step_name="Create virtual environment with venv",
                check=True
            )

            # Verify Python in venv works
            venv_python = self._paths.python
            result = subprocess.run(
                [str(venv_python), "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                self.logger.success(f"Virtual environment created: {result.stdout.strip()}")
                return True
            else:
                raise RuntimeError("Python executable not working in new venv")

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to create virtual environment: {e}")
            return False

    def create_environment(self) -> bool:
        """
        Alias for install() to match MicromambaInstaller interface.

        Returns:
            True if environment was created successfully
        """
        if self.is_installed():
            self.logger.info("Virtual environment already exists, skipping creation")
            return True
        return self.install()
