"""
Linux platform provider implementation.
"""

import os
import stat
import shutil
import subprocess
from pathlib import Path
from typing import Optional, List, Tuple

from .base import PlatformProvider, PlatformPaths, CompilerInfo


class LinuxPlatformProvider(PlatformProvider):
    """Platform provider for Linux systems."""

    @property
    def name(self) -> str:
        return 'linux'

    @property
    def executable_suffix(self) -> str:
        return ''

    @property
    def shared_lib_extension(self) -> str:
        return '.so'

    @property
    def micromamba_exe_name(self) -> str:
        return 'micromamba'

    def get_micromamba_url(self, machine: str) -> str:
        machine_lower = machine.lower()
        if 'aarch64' in machine_lower or 'arm64' in machine_lower:
            return "https://micro.mamba.pm/api/micromamba/linux-aarch64/latest"
        return "https://micro.mamba.pm/api/micromamba/linux-64/latest"

    def get_env_paths(self, env_dir: Path) -> PlatformPaths:
        return PlatformPaths(
            python=env_dir / "bin" / "python",
            pip=env_dir / "bin" / "pip",
            site_packages=env_dir / "lib" / "python3.10" / "site-packages",
            bin_dir=env_dir / "bin"
        )

    def check_prerequisites(self) -> Tuple[bool, Optional[str]]:
        # Check for WSL
        if self._detect_wsl():
            return (False,
                    "Running in Windows Subsystem for Linux (WSL).\n"
                    "This package requires native Windows Python, not WSL.\n"
                    "Please install and run from Windows PowerShell or Command Prompt.")
        return (True, None)

    def _detect_wsl(self) -> bool:
        """Detect if running under Windows Subsystem for Linux."""
        # Method 1: Check /proc/sys/kernel/osrelease
        try:
            with open('/proc/sys/kernel/osrelease', 'r') as f:
                kernel_release = f.read().lower()
                if 'microsoft' in kernel_release or 'wsl' in kernel_release:
                    return True
        except (FileNotFoundError, PermissionError):
            pass

        # Method 2: Check for WSLInterop
        if os.path.exists('/proc/sys/fs/binfmt_misc/WSLInterop'):
            return True

        # Method 3: Check environment variable
        if 'WSL_DISTRO_NAME' in os.environ:
            return True

        return False

    def detect_cxx_compiler(self) -> Optional[CompilerInfo]:
        for compiler in ['g++', 'clang++']:
            path = shutil.which(compiler)
            if path:
                try:
                    result = subprocess.run(
                        [path, '--version'],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if result.returncode == 0:
                        version_line = result.stdout.splitlines()[0] if result.stdout else "unknown"
                        return CompilerInfo(
                            path=Path(path),
                            name=compiler.replace('++', ''),
                            version=version_line,
                            is_compatible=True  # Could add version checking
                        )
                except (subprocess.SubprocessError, OSError):
                    continue
        return None

    def get_conda_compiler_packages(self) -> List[str]:
        return ['gxx_linux-64', 'sysroot_linux-64']

    def make_executable(self, path: Path) -> None:
        current = os.stat(path).st_mode
        os.chmod(path, current | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    def rmtree_robust(self, path: Path) -> bool:
        shutil.rmtree(path)
        return True
