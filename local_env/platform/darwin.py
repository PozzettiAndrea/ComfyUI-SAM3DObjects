"""
macOS (Darwin) platform provider implementation.
"""

import os
import stat
import shutil
import subprocess
from pathlib import Path
from typing import Optional, List, Tuple

from .base import PlatformProvider, PlatformPaths, CompilerInfo


class DarwinPlatformProvider(PlatformProvider):
    """Platform provider for macOS systems."""

    @property
    def name(self) -> str:
        return 'darwin'

    @property
    def executable_suffix(self) -> str:
        return ''

    @property
    def shared_lib_extension(self) -> str:
        return '.dylib'

    @property
    def micromamba_exe_name(self) -> str:
        return 'micromamba'

    def get_micromamba_url(self, machine: str) -> str:
        machine_lower = machine.lower()
        if 'arm64' in machine_lower or 'aarch64' in machine_lower:
            return "https://micro.mamba.pm/api/micromamba/osx-arm64/latest"
        return "https://micro.mamba.pm/api/micromamba/osx-64/latest"

    def get_env_paths(self, env_dir: Path) -> PlatformPaths:
        return PlatformPaths(
            python=env_dir / "bin" / "python",
            pip=env_dir / "bin" / "pip",
            site_packages=env_dir / "lib" / "python3.10" / "site-packages",
            bin_dir=env_dir / "bin"
        )

    def check_prerequisites(self) -> Tuple[bool, Optional[str]]:
        # macOS typically has fewer compatibility issues
        # Could add check for Xcode Command Line Tools here
        return (True, None)

    def detect_cxx_compiler(self) -> Optional[CompilerInfo]:
        # macOS uses clang by default (even when called as g++)
        for compiler in ['clang++', 'g++']:
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
                            name='clang',  # On macOS, g++ is usually clang
                            version=version_line,
                            is_compatible=True
                        )
                except (subprocess.SubprocessError, OSError):
                    continue
        return None

    def get_conda_compiler_packages(self) -> List[str]:
        return ['clang_osx-64', 'clangxx_osx-64']

    def make_executable(self, path: Path) -> None:
        current = os.stat(path).st_mode
        os.chmod(path, current | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    def rmtree_robust(self, path: Path) -> bool:
        shutil.rmtree(path)
        return True
