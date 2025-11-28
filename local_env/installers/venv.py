"""
Virtual environment creation using UV.

UV can manage Python versions without system-wide installation,
which avoids Windows permission issues.
"""

import subprocess
import sys
import shutil
from pathlib import Path

from .base import Installer


class VenvInstaller(Installer):
    """
    Create Python virtual environment using UV.

    UV automatically downloads and manages Python versions without
    requiring system-wide installation, avoiding Windows permission issues.
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
        return "Python Virtual Environment (UV)"

    def is_installed(self) -> bool:
        """Check if venv already exists and is working."""
        if not self.env_dir.exists():
            return False

        python_exe = self._paths.python
        if not python_exe.exists():
            return False

        # Verify it works and is Python 3.10
        try:
            result = subprocess.run(
                [str(python_exe), "-c", "import sys; print(sys.version_info[:2])"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0 and "(3, 10)" in result.stdout:
                return True
            return False
        except (subprocess.SubprocessError, OSError):
            return False

    def _ensure_uv(self) -> str:
        """Ensure UV is available, return path to uv executable."""
        # Check if uv is already in PATH
        uv_path = shutil.which("uv")
        if uv_path:
            return uv_path

        # Check tools directory (env_dir is _env, parent is node root)
        tools_dir = self.env_dir.parent / "_tools"
        if self.platform.name == "windows":
            uv_exe = tools_dir / "uv.exe"
        else:
            uv_exe = tools_dir / "uv"

        if uv_exe.exists():
            return str(uv_exe)

        # Install UV using pip
        self.logger.info("Installing UV package manager...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "uv", "--quiet"],
                check=True,
                capture_output=True,
                timeout=120
            )
            # After pip install, uv should be in PATH or Scripts
            uv_path = shutil.which("uv")
            if uv_path:
                return uv_path

            # Check Scripts directory
            scripts_dir = Path(sys.executable).parent / "Scripts"
            uv_exe = scripts_dir / "uv.exe"
            if uv_exe.exists():
                return str(uv_exe)

            raise RuntimeError("UV installed but executable not found")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to install UV: {e}")

    def install(self) -> bool:
        """Create Python 3.10 virtual environment using UV."""
        self.logger.info("Creating Python 3.10 virtual environment with UV...")

        try:
            uv_path = self._ensure_uv()
            self.logger.info(f"Using UV: {uv_path}")
        except RuntimeError as e:
            self.logger.error(str(e))
            return False

        try:
            # UV can create venv with specific Python version
            # It will automatically download Python 3.10 if needed
            # --seed installs pip/setuptools/wheel
            self.logger.run_logged(
                [uv_path, "venv", str(self.env_dir), "--python", "3.10", "--seed"],
                step_name="Create Python 3.10 venv with UV",
                check=True
            )

            # Verify Python in venv works and is 3.10
            venv_python = self._paths.python
            result = subprocess.run(
                [str(venv_python), "-c", "import sys; print(sys.version_info[:2])"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0 and "(3, 10)" in result.stdout:
                self.logger.success(f"Virtual environment created with Python 3.10")
                return True
            else:
                self.logger.error(f"Wrong Python version: {result.stdout.strip()}")
                return False

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
