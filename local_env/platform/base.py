"""
Abstract base class for platform-specific operations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple


@dataclass
class PlatformPaths:
    """Platform-specific paths within an environment."""
    python: Path
    pip: Path
    site_packages: Path
    bin_dir: Path


@dataclass
class CompilerInfo:
    """Information about a detected compiler."""
    path: Path
    name: str  # 'gcc', 'clang', 'msvc'
    version: str
    is_compatible: bool


class PlatformProvider(ABC):
    """
    Abstract base class for platform-specific operations.

    Each platform (Linux, Windows, macOS) implements this interface
    to provide consistent behavior across operating systems.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Platform name: 'linux', 'windows', 'darwin'."""
        pass

    @property
    @abstractmethod
    def executable_suffix(self) -> str:
        """Executable suffix: '' for Unix, '.exe' for Windows."""
        pass

    @property
    @abstractmethod
    def shared_lib_extension(self) -> str:
        """Shared library extension: '.so', '.dll', '.dylib'."""
        pass

    @property
    @abstractmethod
    def micromamba_exe_name(self) -> str:
        """Micromamba executable name for this platform."""
        pass

    @abstractmethod
    def get_micromamba_url(self, machine: str) -> str:
        """
        Get micromamba download URL for this platform and architecture.

        Args:
            machine: Machine architecture (e.g., 'x86_64', 'arm64')

        Returns:
            URL to download micromamba
        """
        pass

    @abstractmethod
    def get_env_paths(self, env_dir: Path) -> PlatformPaths:
        """
        Get platform-specific paths for an environment.

        Args:
            env_dir: Root directory of the environment

        Returns:
            PlatformPaths with python, pip, site_packages, bin_dir
        """
        pass

    @abstractmethod
    def check_prerequisites(self) -> Tuple[bool, Optional[str]]:
        """
        Check platform-specific prerequisites.

        Returns:
            Tuple of (is_compatible, error_message)
            error_message is None if compatible
        """
        pass

    @abstractmethod
    def detect_cxx_compiler(self) -> Optional[CompilerInfo]:
        """
        Detect available C++ compiler on the system.

        Returns:
            CompilerInfo if found, None otherwise
        """
        pass

    @abstractmethod
    def get_conda_compiler_packages(self) -> List[str]:
        """
        Get conda package names for C++ compiler on this platform.

        Returns:
            List of package names to install via micromamba
        """
        pass

    @abstractmethod
    def make_executable(self, path: Path) -> None:
        """
        Make a file executable.

        Args:
            path: Path to the file
        """
        pass

    @abstractmethod
    def rmtree_robust(self, path: Path) -> bool:
        """
        Remove directory tree with platform-specific error handling.

        Args:
            path: Directory to remove

        Returns:
            True if successful
        """
        pass

    def get_nvcc_exe_name(self) -> str:
        """Get nvcc executable name for this platform."""
        return f"nvcc{self.executable_suffix}"
