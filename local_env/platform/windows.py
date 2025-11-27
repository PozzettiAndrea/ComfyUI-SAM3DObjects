"""
Windows platform provider implementation.
"""

import os
import stat
import shutil
import subprocess
import sys
import time
import ctypes.util
from pathlib import Path
from typing import Dict, Optional, List, Tuple

from .base import PlatformProvider, PlatformPaths, CompilerInfo


class WindowsPlatformProvider(PlatformProvider):
    """Platform provider for Windows systems."""

    @property
    def name(self) -> str:
        return 'windows'

    @property
    def executable_suffix(self) -> str:
        return '.exe'

    @property
    def shared_lib_extension(self) -> str:
        return '.dll'

    @property
    def micromamba_exe_name(self) -> str:
        return 'micromamba.exe'

    def get_micromamba_url(self, machine: str) -> str:
        return "https://micro.mamba.pm/api/micromamba/win-64/latest"

    def get_env_paths(self, env_dir: Path) -> PlatformPaths:
        return PlatformPaths(
            python=env_dir / "Scripts" / "python.exe",
            pip=env_dir / "Scripts" / "pip.exe",
            site_packages=env_dir / "Lib" / "site-packages",
            bin_dir=env_dir / "Scripts"
        )

    def check_prerequisites(self) -> Tuple[bool, Optional[str]]:
        # Check for MSYS2/Cygwin/Git Bash
        shell_env = self._detect_shell_environment()
        if shell_env in ('msys2', 'cygwin', 'git-bash'):
            return (False,
                    f"Running in {shell_env.upper()} environment.\n"
                    f"This package requires native Windows Python.\n"
                    f"Please use PowerShell, Command Prompt, or native Windows terminal.")

        # Check Visual C++ Redistributable
        vc_ok, vc_error = self._check_vc_redistributable()
        if not vc_ok:
            return (False, vc_error)

        return (True, None)

    def _detect_shell_environment(self) -> str:
        """Detect if running in MSYS2, Cygwin, Git Bash, or native Windows."""
        msystem = os.environ.get('MSYSTEM', '')
        if msystem:
            if 'MINGW' in msystem:
                return 'git-bash'
            return 'msys2'

        term = os.environ.get('TERM', '')
        if term and 'cygwin' in term:
            return 'cygwin'

        return 'native-windows'

    def _find_vc_dlls(self) -> Dict[str, Optional[Path]]:
        """Find VC++ runtime DLLs in common locations."""
        required_dlls = ['vcruntime140.dll', 'msvcp140.dll']
        found = {}

        # Search locations in order of preference
        search_paths = []

        # 1. Current Python environment (conda/venv)
        if hasattr(sys, 'base_prefix'):
            search_paths.append(Path(sys.base_prefix) / 'Library' / 'bin')
            search_paths.append(Path(sys.base_prefix) / 'DLLs')
        if hasattr(sys, 'prefix'):
            search_paths.append(Path(sys.prefix) / 'Library' / 'bin')
            search_paths.append(Path(sys.prefix) / 'DLLs')

        # 2. System directories
        system_root = os.environ.get('SystemRoot', r'C:\Windows')
        search_paths.append(Path(system_root) / 'System32')

        # 3. Visual Studio redistributable directories
        program_files = os.environ.get('ProgramFiles', r'C:\Program Files')
        vc_redist = Path(program_files) / 'Microsoft Visual Studio' / '2022' / 'Community' / 'VC' / 'Redist' / 'MSVC'
        if vc_redist.exists():
            for version_dir in vc_redist.iterdir():
                search_paths.append(version_dir / 'x64' / 'Microsoft.VC143.CRT')

        for dll_name in required_dlls:
            found[dll_name] = None
            for search_path in search_paths:
                dll_path = search_path / dll_name
                if dll_path.exists():
                    found[dll_name] = dll_path
                    break

        return found

    def bundle_vc_dlls_to_env(self, env_dir: Path) -> Tuple[bool, Optional[str]]:
        """Bundle VC++ runtime DLLs into the isolated environment."""
        import sys

        required_dlls = ['vcruntime140.dll', 'msvcp140.dll']
        found_dlls = self._find_vc_dlls()

        # Check which DLLs are missing
        missing = [dll for dll, path in found_dlls.items() if path is None]

        if missing:
            return (False,
                f"Could not find VC++ DLLs to bundle: {', '.join(missing)}\n\n"
                f"Please install Visual C++ Redistributable:\n"
                f"  Download: https://aka.ms/vs/17/release/vc_redist.x64.exe\n"
                f"\nAfter installation, delete _env and try again.")

        # Copy DLLs to the environment's site-packages/open3d/ directory
        # This is where Open3D adds its DLL search path
        target_dir = env_dir / 'Lib' / 'site-packages' / 'open3d'

        # Also copy to Scripts as a fallback
        scripts_dir = env_dir / 'Scripts'

        copied = []
        for dll_name, source_path in found_dlls.items():
            if source_path:
                try:
                    # Copy to open3d directory (primary)
                    if target_dir.exists():
                        target_path = target_dir / dll_name
                        if not target_path.exists():
                            shutil.copy2(source_path, target_path)
                            copied.append(f"{dll_name} -> open3d/")

                    # Copy to Scripts directory (fallback)
                    if scripts_dir.exists():
                        scripts_target = scripts_dir / dll_name
                        if not scripts_target.exists():
                            shutil.copy2(source_path, scripts_target)
                            copied.append(f"{dll_name} -> Scripts/")

                except (OSError, IOError) as e:
                    return (False, f"Failed to copy {dll_name}: {e}")

        if copied:
            print(f"[SAM3DObjects] Bundled VC++ DLLs: {', '.join(copied)}")

        return (True, None)

    def _check_vc_redistributable(self) -> Tuple[bool, Optional[str]]:
        """Check if Visual C++ Redistributable DLLs are available."""
        # We just check if the DLLs exist somewhere - bundling happens later
        required_dlls = ['vcruntime140.dll', 'msvcp140.dll']
        found_dlls = self._find_vc_dlls()

        missing = [dll for dll, path in found_dlls.items() if path is None]

        if missing:
            error_msg = (
                f"Visual C++ Redistributable DLLs not found!\n"
                f"\nMissing: {', '.join(missing)}\n"
                f"\nPlease install Visual C++ Redistributable for Visual Studio 2015-2022:\n"
                f"\n  Download (64-bit): https://aka.ms/vs/17/release/vc_redist.x64.exe\n"
                f"\nAfter installation, restart your terminal and try again."
            )
            return (False, error_msg)

        return (True, None)

    def detect_cxx_compiler(self) -> Optional[CompilerInfo]:
        """Detect MSVC compiler on Windows."""
        cl_path = self._find_msvc()
        if cl_path:
            version = self._get_msvc_version(cl_path)
            return CompilerInfo(
                path=cl_path,
                name='msvc',
                version=version,
                is_compatible=self._test_msvc(cl_path)
            )
        return None

    def _find_msvc(self) -> Optional[Path]:
        """Find MSVC compiler (cl.exe) in common locations."""
        # Check PATH first
        cl_path = shutil.which('cl.exe')
        if cl_path:
            return Path(cl_path)

        # Search Visual Studio installations
        program_files = [
            os.environ.get('ProgramFiles(x86)', 'C:\\Program Files (x86)'),
            os.environ.get('ProgramFiles', 'C:\\Program Files'),
        ]

        for pf in program_files:
            vs_base = Path(pf) / 'Microsoft Visual Studio'
            if not vs_base.exists():
                continue

            try:
                for year_dir in vs_base.iterdir():
                    if not year_dir.is_dir():
                        continue
                    for edition_dir in year_dir.iterdir():
                        if not edition_dir.is_dir():
                            continue
                        vc_tools = edition_dir / 'VC' / 'Tools' / 'MSVC'
                        if not vc_tools.exists():
                            continue
                        for version_dir in vc_tools.iterdir():
                            cl_exe = version_dir / 'bin' / 'Hostx64' / 'x64' / 'cl.exe'
                            if cl_exe.exists():
                                return cl_exe
            except (PermissionError, OSError):
                continue

        return None

    def _get_msvc_version(self, cl_path: Path) -> str:
        """Get MSVC version string."""
        try:
            result = subprocess.run(
                [str(cl_path)],
                capture_output=True,
                text=True,
                timeout=5
            )
            # MSVC prints version to stderr
            return result.stderr.splitlines()[0] if result.stderr else "unknown"
        except (subprocess.SubprocessError, OSError):
            return "unknown"

    def _test_msvc(self, cl_path: Path) -> bool:
        """Test compile with MSVC to verify it works."""
        import tempfile

        test_code = """
#include <iostream>
int main() {
    std::cout << "Hello from MSVC!" << std::endl;
    return 0;
}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            source_file = tmpdir / 'test.cpp'
            exe_file = tmpdir / 'test.exe'

            source_file.write_text(test_code)

            try:
                result = subprocess.run(
                    [str(cl_path), '/EHsc', '/nologo', str(source_file), f'/Fe{exe_file}'],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=tmpdir
                )
                return result.returncode == 0
            except (subprocess.SubprocessError, OSError):
                return False

    def get_conda_compiler_packages(self) -> List[str]:
        return ['m2w64-gcc', 'm2w64-gcc-fortran']

    def make_executable(self, path: Path) -> None:
        # No-op on Windows - executables are determined by extension
        pass

    def rmtree_robust(self, path: Path, max_retries: int = 5, delay: float = 0.5) -> bool:
        """
        Windows-specific rmtree with retry logic for file locking issues.

        Handles Windows file locking, read-only files, and antivirus interference.
        """
        def handle_remove_readonly(func, fpath, exc):
            """Error handler for removing read-only files."""
            if isinstance(exc[1], PermissionError):
                try:
                    os.chmod(fpath, stat.S_IWRITE)
                    func(fpath)
                except Exception:
                    raise exc[1]
            else:
                raise exc[1]

        for attempt in range(max_retries):
            try:
                shutil.rmtree(path, onerror=handle_remove_readonly)
                return True
            except PermissionError:
                if attempt < max_retries - 1:
                    wait_time = delay * (2 ** attempt)
                    time.sleep(wait_time)
                else:
                    raise
            except OSError:
                if attempt < max_retries - 1:
                    wait_time = delay * (2 ** attempt)
                    time.sleep(wait_time)
                else:
                    raise

        return False
