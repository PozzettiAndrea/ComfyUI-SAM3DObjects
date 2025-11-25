"""
Micromamba download and environment creation.
"""

import platform
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from .base import Installer


class MicromambaInstaller(Installer):
    """
    Download micromamba and create Python environment.

    Micromamba is a tiny standalone executable that can create conda
    environments without requiring conda/mamba to be installed.
    """

    def __init__(self, tools_dir: Path, env_dir: Path, platform_provider, config, logger):
        """
        Initialize micromamba installer.

        Args:
            tools_dir: Directory to store micromamba executable
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
        self._tools_dir = tools_dir
        self._micromamba_path = tools_dir / platform_provider.micromamba_exe_name

    @property
    def name(self) -> str:
        return "Micromamba"

    @property
    def micromamba_path(self) -> Path:
        """Get path to micromamba executable."""
        return self._micromamba_path

    def is_installed(self) -> bool:
        """Check if micromamba is already downloaded."""
        if not self._micromamba_path.exists():
            return False

        # Verify it works
        try:
            result = subprocess.run(
                [str(self._micromamba_path), "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, OSError):
            return False

    def install(self) -> bool:
        """Download micromamba for the current platform."""
        self._tools_dir.mkdir(parents=True, exist_ok=True)

        machine = platform.machine()
        url = self.platform.get_micromamba_url(machine)

        self.logger.info(f"Downloading micromamba for {self.platform.name} {machine}...")
        self.logger.info("This is a one-time download (~70MB)")

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                archive_path = tmpdir_path / "micromamba.tar.bz2"

                # Download
                urllib.request.urlretrieve(url, archive_path)

                # Extract (micromamba is distributed as tar.bz2)
                with tarfile.open(archive_path, "r:bz2") as tar:
                    tar.extractall(tmpdir_path)

                    # Find the micromamba binary
                    exe_name = self.platform.micromamba_exe_name
                    extracted_binary = tmpdir_path / "bin" / exe_name

                    if not extracted_binary.exists():
                        # Sometimes it's at root
                        extracted_binary = tmpdir_path / exe_name

                    if not extracted_binary.exists():
                        # Search for it
                        for item in tmpdir_path.rglob(exe_name):
                            extracted_binary = item
                            break

                    if not extracted_binary.exists():
                        raise RuntimeError("Could not find micromamba binary in archive")

                    # Copy to tools directory
                    import shutil
                    shutil.copy2(extracted_binary, self._micromamba_path)

            # Make executable
            self.platform.make_executable(self._micromamba_path)

            # Verify it works
            result = subprocess.run(
                [str(self._micromamba_path), "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode != 0:
                raise RuntimeError(f"Micromamba binary is not working: {result.stderr}")

            self.logger.success(f"Micromamba {result.stdout.strip()} downloaded")
            return True

        except Exception as e:
            self.logger.error(f"Failed to download micromamba: {e}")
            return False

    def create_environment(self) -> bool:
        """
        Create Python 3.10 environment using micromamba.

        Returns:
            True if environment was created successfully
        """
        if self.env_dir.exists():
            # Check if it's working
            if self._paths.python.exists():
                try:
                    result = subprocess.run(
                        [str(self._paths.python), "--version"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if result.returncode == 0:
                        self.logger.info("Environment already exists, skipping creation")
                        return True
                except (subprocess.SubprocessError, OSError):
                    pass

            self.logger.info("Recreating incomplete environment")

        self.logger.info(f"Creating Python {self.config.python_version} environment...")

        try:
            self.logger.run_logged(
                [
                    str(self._micromamba_path), "create",
                    "-p", str(self.env_dir),
                    f"python={self.config.python_version}",
                    "-c", "conda-forge",
                    "-y"
                ],
                step_name="Create Python environment with micromamba",
                check=True
            )

            # Verify Python version
            result = subprocess.run(
                [str(self._paths.python), "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                self.logger.success(f"Environment created: {result.stdout.strip()}")
                return True
            else:
                raise RuntimeError("Python executable not working in new environment")

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to create environment: {e}")
            return False
