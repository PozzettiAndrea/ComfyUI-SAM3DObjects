"""
Abstract base class for package installers.
"""

from abc import ABC, abstractmethod
from pathlib import Path
import subprocess

from ..platform.base import PlatformProvider
from ..config import InstallConfig
from ..utils import Logger


class Installer(ABC):
    """
    Abstract base class for package installers.

    Each installer handles one package or related group of packages.
    Installers are designed to be idempotent - they check if already
    installed before doing work.
    """

    def __init__(
        self,
        env_dir: Path,
        platform: PlatformProvider,
        config: InstallConfig,
        logger: Logger,
        micromamba_exe: Path = None,
    ):
        """
        Initialize installer.

        Args:
            env_dir: Path to the isolated environment
            platform: Platform provider for OS-specific operations
            config: Installation configuration
            logger: Logger for output
            micromamba_exe: Path to micromamba executable (optional)
        """
        self.env_dir = env_dir
        self.platform = platform
        self.config = config
        self.logger = logger
        self.micromamba_exe = micromamba_exe
        self._paths = platform.get_env_paths(env_dir)

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this installer."""
        pass

    @abstractmethod
    def is_installed(self) -> bool:
        """
        Check if the package is already installed.

        Returns:
            True if installed and working
        """
        pass

    @abstractmethod
    def install(self) -> bool:
        """
        Install the package.

        Returns:
            True if installation succeeded
        """
        pass

    def run_pip(self, args: list, **kwargs) -> subprocess.CompletedProcess:
        """
        Run pip in the isolated environment.

        Args:
            args: Arguments to pass to pip
            **kwargs: Additional subprocess arguments

        Returns:
            CompletedProcess result
        """
        cmd = [str(self._paths.python), "-m", "pip"] + args
        return self.logger.run_logged(cmd, **kwargs)

    def run_uv_pip(self, args: list, **kwargs) -> subprocess.CompletedProcess:
        """
        Run uv pip in the isolated environment.

        Args:
            args: Arguments to pass to uv pip
            **kwargs: Additional subprocess arguments

        Returns:
            CompletedProcess result
        """
        cmd = [str(self._paths.python), "-m", "uv", "pip"] + args
        return self.logger.run_logged(cmd, **kwargs)

    def run_python(self, code: str, **kwargs) -> subprocess.CompletedProcess:
        """
        Run Python code in the isolated environment.

        Args:
            code: Python code to execute
            **kwargs: Additional subprocess arguments

        Returns:
            CompletedProcess result
        """
        cmd = [str(self._paths.python), "-c", code]
        kwargs.setdefault('capture_output', True)
        kwargs.setdefault('text', True)
        kwargs.setdefault('timeout', 30)
        return subprocess.run(cmd, **kwargs)

    def verify_import(self, module: str) -> bool:
        """
        Verify a module can be imported.

        Args:
            module: Module name to import

        Returns:
            True if import succeeds
        """
        result = self.run_python(f"import {module}")
        return result.returncode == 0

    def run_micromamba(self, args: list, **kwargs) -> subprocess.CompletedProcess:
        """
        Run micromamba command.

        Args:
            args: Arguments to pass to micromamba
            **kwargs: Additional subprocess arguments

        Returns:
            CompletedProcess result
        """
        if self.micromamba_exe is None:
            raise RuntimeError("Micromamba not available")

        cmd = [str(self.micromamba_exe)] + args
        return self.logger.run_logged(cmd, **kwargs)
