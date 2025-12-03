"""
Environment manager for SAM3D isolated venv environment.

This module handles creation, management, and validation of the isolated
Python virtual environment for SAM3D inference.
"""

import os
import sys
import subprocess
import platform
import venv
import importlib
from pathlib import Path
from typing import Optional


def _log_subprocess_output(log_file: Path, message: str, stdout: str = "", stderr: str = ""):
    """Write subprocess output to log file."""
    with open(log_file, 'a') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"{message}\n")
        f.write(f"{'='*80}\n")
        if stdout:
            f.write(f"STDOUT:\n{stdout}\n")
        if stderr:
            f.write(f"STDERR:\n{stderr}\n")


def _run_subprocess_logged(cmd: list, log_file: Path, step_name: str, **kwargs):
    """Run subprocess with output logged to file, minimal console output."""
    # Force capture_output if not specified
    kwargs['capture_output'] = True
    kwargs['text'] = True

    try:
        result = subprocess.run(cmd, **kwargs)
        _log_subprocess_output(log_file, f"{step_name} - SUCCESS", result.stdout, result.stderr)
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
        return result
    except subprocess.CalledProcessError as e:
        _log_subprocess_output(log_file, f"{step_name} - FAILED", e.stdout if hasattr(e, 'stdout') else "", e.stderr if hasattr(e, 'stderr') else "")
        print(f"\n[SAM3DObjects] ERROR: {step_name} failed!")
        print(f"[SAM3DObjects] Check logs at: {log_file}")
        if hasattr(e, 'stdout') and e.stdout:
            print(f"\nLast output:\n{e.stdout[-500:]}")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"\nError output:\n{e.stderr[-500:]}")
        raise


def detect_wsl():
    """
    Detect if running under Windows Subsystem for Linux.

    Returns:
        bool: True if running under WSL
    """
    # Method 1: Check /proc/sys/kernel/osrelease (most reliable)
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


def detect_windows_shell_environment():
    """
    Detect if running in MSYS2, Cygwin, Git Bash, or native Windows.

    Returns:
        str: 'msys2', 'cygwin', 'git-bash', 'native-windows', or 'other'
    """
    system = platform.system()

    # Check if running on actual Windows
    if system == 'Windows':
        # Check environment variables
        msystem = os.environ.get('MSYSTEM', '')
        if msystem:
            # MSYS2 sets MSYSTEM to MINGW64, MINGW32, etc.
            if 'MINGW' in msystem:
                return 'git-bash'  # Git Bash also uses MINGW
            return 'msys2'

        # Check TERM variable
        term = os.environ.get('TERM', '')
        if term and 'cygwin' in term:
            return 'cygwin'

        return 'native-windows'

    # Check OSTYPE shell variable
    ostype = os.environ.get('OSTYPE', '')
    if ostype:
        if ostype.startswith('msys'):
            return 'msys2'
        elif ostype.startswith('cygwin'):
            return 'cygwin'

    # Check if platform.system() reports Cygwin/MSYS variants
    if 'CYGWIN' in system:
        return 'cygwin'
    elif 'MSYS' in system or 'MINGW' in system:
        return 'msys2'

    return 'other'


def check_platform_compatibility():
    """
    Check if the platform is compatible with the installation.
    Warns about WSL, MSYS2, Cygwin which may have compatibility issues.

    Returns:
        tuple: (is_compatible, warning_message)
    """
    # Check for WSL
    if detect_wsl():
        return (False,
                "Running in Windows Subsystem for Linux (WSL).\n"
                "This package requires native Windows Python, not WSL.\n"
                "Please install and run from Windows PowerShell or Command Prompt.")

    # Check for MSYS2/Cygwin
    shell_env = detect_windows_shell_environment()
    if shell_env in ('msys2', 'cygwin', 'git-bash'):
        return (False,
                f"Running in {shell_env.upper()} environment.\n"
                f"This package requires native Windows Python.\n"
                f"Please use PowerShell, Command Prompt, or native Windows terminal.")

    return (True, None)


def validate_wheel_url(url, timeout=10):
    """
    Validate that a wheel URL exists and is accessible.

    Args:
        url: URL to validate
        timeout: Request timeout in seconds

    Returns:
        dict: Validation result with status, size, etc.
    """
    import urllib.request
    import urllib.error

    result = {
        'valid': False,
        'url': url,
        'status_code': None,
        'content_length': None,
        'error': None
    }

    # Create HEAD request
    request = urllib.request.Request(url, method='HEAD')

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result['status_code'] = response.status
            result['valid'] = 200 <= response.status < 300
            result['content_length'] = response.headers.get('Content-Length')

        return result

    except urllib.error.HTTPError as e:
        result['status_code'] = e.code
        result['error'] = f'HTTP {e.code}: {e.reason}'

        # Try GET if HEAD failed with 405
        if e.code == 405:
            try:
                request = urllib.request.Request(url)
                request.add_header('Range', 'bytes=0-0')

                with urllib.request.urlopen(request, timeout=timeout) as response:
                    result['status_code'] = response.status
                    result['valid'] = 200 <= response.status < 300
                    result['error'] = None

            except Exception:
                pass  # Keep original error

        return result

    except urllib.error.URLError as e:
        result['error'] = f'URL error: {e.reason}'
        return result

    except Exception as e:
        result['error'] = f'Unexpected error: {e}'
        return result


def check_vcruntime_dlls():
    """
    Check if required Visual C++ runtime DLLs are available.

    Returns:
        dict: Status of each required DLL
    """
    import ctypes.util

    required_dlls = [
        'VCRUNTIME140.dll',
        'MSVCP140.dll',
    ]

    dll_status = {}
    for dll_name in required_dlls:
        # Try to find DLL in system path
        dll_path = ctypes.util.find_library(dll_name.replace('.dll', ''))
        dll_status[dll_name] = dll_path is not None

    return dll_status


def check_vc_redistributable():
    """
    Check if Visual C++ Redistributable is installed.

    Returns:
        tuple: (is_installed, error_message)
    """
    if platform.system() != 'Windows':
        return (True, None)  # Not needed on non-Windows

    print("[SAM3DObjects] Checking Visual C++ Redistributable...")

    dll_status = check_vcruntime_dlls()
    missing_dlls = [dll for dll, found in dll_status.items() if not found]

    if missing_dlls:
        error_msg = (
            f"Visual C++ Redistributable is not installed or incomplete!\n"
            f"\nMissing DLLs: {', '.join(missing_dlls)}\n"
            f"\nThe prebuilt Python wheels require Visual C++ runtime libraries.\n"
            f"Please install Visual C++ Redistributable for Visual Studio 2015-2022:\n"
            f"\n  Download (64-bit): https://aka.ms/vs/17/release/vc_redist.x64.exe\n"
            f"\nAfter installation, restart your terminal and try again."
        )
        return (False, error_msg)

    print("[SAM3DObjects] ✓ Visual C++ Redistributable found")
    return (True, None)


def find_msvc_compiler():
    """
    Find MSVC compiler (cl.exe) on the system.

    Returns:
        Path or None: Path to cl.exe if found
    """
    import shutil

    # Method 1: Check if already in PATH
    cl_path = shutil.which('cl.exe')
    if cl_path:
        return Path(cl_path)

    # Method 2: Search common MSVC installation paths
    program_files = [
        os.environ.get('ProgramFiles(x86)', 'C:\\Program Files (x86)'),
        os.environ.get('ProgramFiles', 'C:\\Program Files'),
    ]

    for pf in program_files:
        vs_base = Path(pf) / 'Microsoft Visual Studio'
        if not vs_base.exists():
            continue

        # Search for cl.exe in VS installations
        # Pattern: VS/{year}/{edition}/VC/Tools/MSVC/{version}/bin/Host{arch}/{target}/cl.exe
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
                        # Check for 64-bit compiler
                        cl_exe = version_dir / 'bin' / 'Hostx64' / 'x64' / 'cl.exe'
                        if cl_exe.exists():
                            return cl_exe

                        # Check for 32-bit compiler
                        cl_exe = version_dir / 'bin' / 'Hostx86' / 'x86' / 'cl.exe'
                        if cl_exe.exists():
                            return cl_exe
        except (PermissionError, OSError):
            continue

    return None


def test_compile_cpp(cl_path=None):
    """
    Test compile a simple C++ program to verify MSVC works.

    Args:
        cl_path: Path to cl.exe (optional, will search if not provided)

    Returns:
        bool: True if compilation succeeded
    """
    import tempfile

    if cl_path is None:
        cl_path = find_msvc_compiler()

    if cl_path is None:
        return False

    # Simple test program
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

        # Write test source
        source_file.write_text(test_code)

        try:
            # Compile
            result = subprocess.run(
                [str(cl_path), '/EHsc', '/nologo', str(source_file), f'/Fe{exe_file}'],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=tmpdir
            )

            if result.returncode != 0:
                print(f"[SAM3DObjects] MSVC test compilation failed: {result.stderr}")
                return False

            # Run the compiled program
            result = subprocess.run(
                [str(exe_file)],
                capture_output=True,
                text=True,
                timeout=5
            )

            return result.returncode == 0 and 'Hello from MSVC!' in result.stdout

        except (subprocess.SubprocessError, OSError) as e:
            print(f"[SAM3DObjects] MSVC test compilation error: {e}")
            return False


def rmtree_windows_robust(path, max_retries=5, delay=0.5):
    """
    Robust directory removal for Windows with error handling.

    Handles Windows file locking, read-only files, and antivirus interference.

    Args:
        path: Path to directory
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)

    Returns:
        bool: True if successful
    """
    import time
    import stat
    import shutil

    path = Path(path)

    def handle_remove_readonly(func, path, exc):
        """Error handler for removing read-only files."""
        if isinstance(exc[1], PermissionError):
            # Try to change permissions and retry
            try:
                os.chmod(path, stat.S_IWRITE)
                func(path)
            except Exception:
                raise exc[1]
        else:
            raise exc[1]

    for attempt in range(max_retries):
        try:
            shutil.rmtree(path, onerror=handle_remove_readonly)
            return True

        except PermissionError as e:
            if attempt < max_retries - 1:
                # Wait with exponential backoff
                wait_time = delay * (2 ** attempt)
                print(f"[SAM3DObjects] Retrying file operation ({attempt + 1}/{max_retries}) after {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                # Last attempt failed
                print(f"[SAM3DObjects] Failed to remove {path} after {max_retries} attempts")
                print(f"[SAM3DObjects] This may be caused by:")
                print(f"[SAM3DObjects]   - Antivirus software scanning files")
                print(f"[SAM3DObjects]   - Windows Search indexer")
                print(f"[SAM3DObjects]   - File Explorer having the directory open")
                print(f"[SAM3DObjects]   - Another process using the files")
                raise

        except OSError as e:
            if attempt < max_retries - 1:
                wait_time = delay * (2 ** attempt)
                print(f"[SAM3DObjects] OS error, retrying ({attempt + 1}/{max_retries}) after {wait_time:.1f}s: {e}")
                time.sleep(wait_time)
            else:
                raise

    return False


class SAM3DEnvironmentManager:
    """Manages the isolated venv environment for SAM3D."""

    def __init__(self, node_root: Path):
        """
        Initialize environment manager.

        Args:
            node_root: Root directory of the ComfyUI-SAM3DObjects node
        """
        # Check platform compatibility first
        is_compatible, warning_msg = check_platform_compatibility()
        if not is_compatible:
            print(f"\n{'='*60}")
            print("[SAM3DObjects] PLATFORM COMPATIBILITY ERROR")
            print('='*60)
            print(warning_msg)
            print('='*60)
            raise RuntimeError(f"Incompatible platform detected:\n{warning_msg}")

        self.node_root = Path(node_root)
        self.env_dir = self.node_root / "_env"
        self.log_file = self.node_root / "install.log"
        self.tools_dir = self.node_root / "_tools"
        self.micromamba_exe = None  # Will be set by _download_micromamba()

    def get_python_executable(self) -> Path:
        """Get path to Python executable in isolated environment."""
        if platform.system() == "Windows":
            return self.env_dir / "Scripts" / "python.exe"
        else:
            return self.env_dir / "bin" / "python"

    def get_pip_executable(self) -> Path:
        """Get path to pip executable in isolated environment."""
        if platform.system() == "Windows":
            return self.env_dir / "Scripts" / "pip.exe"
        else:
            return self.env_dir / "bin" / "pip"

    def is_environment_ready(self) -> bool:
        """Check if the isolated environment is ready to use."""
        python_exe = self.get_python_executable()
        if not python_exe.exists():
            return False

        # Verify critical packages are installed
        # Note: sam3d_objects is vendored, so we only check torch and pytorch3d
        try:
            result = subprocess.run(
                [str(python_exe), "-c", "import torch, pytorch3d"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except Exception:
            return False

    def _download_micromamba(self) -> Path:
        """
        Download micromamba binary for the current platform.

        Micromamba is a tiny standalone executable that can create conda environments
        without requiring conda/mamba to be installed. This allows us to:
        - Install Python 3.10 consistently across all platforms
        - Use conda packages (compilers, etc.) without user having conda

        Returns:
            Path to micromamba executable
        """
        import urllib.request
        import stat

        # Create tools directory if it doesn't exist
        self.tools_dir.mkdir(exist_ok=True)

        # Determine platform and micromamba URL
        system = platform.system()
        machine = platform.machine().lower()

        if system == "Linux":
            if "x86_64" in machine or "amd64" in machine:
                url = "https://micro.mamba.pm/api/micromamba/linux-64/latest"
                exe_name = "micromamba"
            elif "aarch64" in machine or "arm64" in machine:
                url = "https://micro.mamba.pm/api/micromamba/linux-aarch64/latest"
                exe_name = "micromamba"
            else:
                raise RuntimeError(f"Unsupported Linux architecture: {machine}")
        elif system == "Darwin":  # macOS
            if "arm64" in machine or "aarch64" in machine:
                url = "https://micro.mamba.pm/api/micromamba/osx-arm64/latest"
            else:
                url = "https://micro.mamba.pm/api/micromamba/osx-64/latest"
            exe_name = "micromamba"
        elif system == "Windows":
            url = "https://micro.mamba.pm/api/micromamba/win-64/latest"
            exe_name = "micromamba.exe"
        else:
            raise RuntimeError(f"Unsupported operating system: {system}")

        micromamba_path = self.tools_dir / exe_name

        # Check if already downloaded
        if micromamba_path.exists():
            print(f"[SAM3DObjects] Micromamba already downloaded")
            self.micromamba_exe = micromamba_path
            return micromamba_path

        print(f"[SAM3DObjects] Downloading micromamba for {system} {machine}...")
        print(f"[SAM3DObjects] This is a one-time download (~70MB)")

        try:
            # Download with progress
            import tempfile
            import tarfile

            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                archive_path = tmpdir_path / "micromamba.tar.bz2"

                # Download
                urllib.request.urlretrieve(url, archive_path)

                # Extract (micromamba is distributed as tar.bz2 with binary inside)
                with tarfile.open(archive_path, "r:bz2") as tar:
                    # Extract all to tmpdir
                    tar.extractall(tmpdir_path)

                    # Find the micromamba binary
                    extracted_binary = tmpdir_path / "bin" / exe_name
                    if not extracted_binary.exists():
                        # Sometimes it's at root
                        extracted_binary = tmpdir_path / exe_name

                    if not extracted_binary.exists():
                        raise RuntimeError(f"Could not find micromamba binary in archive")

                    # Copy to tools directory
                    import shutil
                    shutil.copy2(extracted_binary, micromamba_path)

            # Make executable (Unix/Mac)
            if system in ["Linux", "Darwin"]:
                micromamba_path.chmod(micromamba_path.stat().st_mode | stat.S_IEXEC)

            # Verify it works
            result = subprocess.run(
                [str(micromamba_path), "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode != 0:
                raise RuntimeError(f"Micromamba binary is not executable: {result.stderr}")

            print(f"[SAM3DObjects] Micromamba downloaded successfully!")
            print(f"[SAM3DObjects] Version: {result.stdout.strip()}")

            self.micromamba_exe = micromamba_path
            return micromamba_path

        except Exception as e:
            raise RuntimeError(f"Failed to download micromamba: {e}") from e

    def create_environment(self) -> None:
        """
        Create the isolated Python environment using micromamba.

        This uses micromamba to create a conda environment with Python 3.10,
        ensuring consistent Python version across all platforms.
        """
        if self.env_dir.exists():
            if self.is_environment_ready():
                print("[SAM3DObjects] Environment already exists, skipping creation")
                # Still need to download micromamba for later use
                self._download_micromamba()
                return
            else:
                print("[SAM3DObjects] Recreating incomplete environment")

        print("[SAM3DObjects] Creating Python 3.10 environment using micromamba...")

        # Download micromamba first
        micromamba_exe = self._download_micromamba()

        try:
            # Create conda environment with Python 3.10
            # Using -p (prefix) instead of -n (name) to create in specific directory
            # -y = yes to all prompts
            # -c conda-forge = use conda-forge channel
            _run_subprocess_logged(
                [
                    str(micromamba_exe), "create",
                    "-p", str(self.env_dir),
                    "python=3.10",
                    "-c", "conda-forge",
                    "-y"
                ],
                self.log_file,
                "Create Python 3.10 environment with micromamba",
                check=True
            )

            print("[SAM3DObjects] Python 3.10 environment created successfully!")

            # Verify Python version
            python_exe = self.get_python_executable()
            result = subprocess.run(
                [str(python_exe), "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                print(f"[SAM3DObjects] Python version: {result.stdout.strip()}")
            else:
                raise RuntimeError("Python executable not working in new environment")

        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to create conda environment with micromamba: {e}\n"
                f"Check logs at: {self.log_file}"
            ) from e

    def install_dependencies(self) -> None:
        """Install all dependencies using pip/uv."""
        print("[SAM3DObjects] Installing dependencies...")
        print("[SAM3DObjects] This will take 5-10 minutes...")

        python_exe = self.get_python_executable()
        pip_exe = self.get_pip_executable()

        # Step 1: Upgrade pip
        print("[SAM3DObjects] Upgrading pip...")
        _run_subprocess_logged(
            [str(python_exe), "-m", "pip", "install", "--upgrade", "pip"],
            self.log_file,
            "Upgrade pip",
            check=True
        )

        # Step 2: Install uv for faster package installation
        print("[SAM3DObjects] Installing uv package manager...")
        _run_subprocess_logged(
            [str(pip_exe), "install", "uv"],
            self.log_file,
            "Install uv",
            check=True
        )

        # Step 3: Install PyTorch + PyTorch3D together via micromamba
        # This ensures CUDA version compatibility!
        # NOTE: Using PyTorch 2.4.1 because PyTorch3D hasn't built packages for 2.5.x yet
        print("[SAM3DObjects] Installing PyTorch 2.4.1 + PyTorch3D via micromamba...")
        print("[SAM3DObjects] (Using PyTorch 2.4.1 - latest with PyTorch3D prebuilt support)")

        _run_subprocess_logged(
            [
                str(self.micromamba_exe), "install",
                "-p", str(self.env_dir),
                "-c", "pytorch",       # PRIMARY: PyTorch official channel (MUST BE FIRST!)
                "-c", "pytorch3d",     # PyTorch3D channel
                "-c", "nvidia",        # NVIDIA CUDA packages
                "-c", "fvcore",        # PyTorch3D dependency
                "-c", "conda-forge",   # Fallback (LAST!)
                "pytorch==2.4.1",      # PyTorch 2.4.1 - latest with PyTorch3D support
                "pytorch-cuda=12.1",   # Explicitly specify CUDA build variant
                "torchvision==0.19.1", # Matching version for PyTorch 2.4.1
                "pytorch3d==0.7.8",    # Latest PyTorch3D with 2.4.1 support
                "-y"
            ],
            self.log_file,
            "Install PyTorch 2.4.1 + torchvision 0.19.1 + PyTorch3D 0.7.8 via micromamba",
            check=True
        )

        # Verify installations
        verify_result = subprocess.run(
            [str(python_exe), "-c", "import torch, pytorch3d; print(f'PyTorch {torch.__version__}, PyTorch3D {pytorch3d.__version__}')"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if verify_result.returncode == 0:
            print(f"[SAM3DObjects] Verified: {verify_result.stdout.strip()}")
        else:
            raise RuntimeError(f"PyTorch/PyTorch3D verification failed: {verify_result.stderr}")

        # Step 4: Install all dependencies via pip (BEFORE gsplat to ensure dependencies are available)
        print("[SAM3DObjects] Installing dependencies via pip...")
        print("[SAM3DObjects] (PyTorch is pinned, will not be upgraded)")

        requirements_file = self.node_root / "local_env_settings" / "requirements_env.txt"
        if not requirements_file.exists():
            raise RuntimeError(f"requirements_env.txt not found: {requirements_file}")

        # Install with --no-deps for packages that might upgrade PyTorch, then install their deps separately
        _run_subprocess_logged(
            [
                str(python_exe), "-m", "uv", "pip", "install",
                "--no-deps",  # Don't install dependencies to avoid PyTorch upgrade
                "-r", str(requirements_file)
            ],
            self.log_file,
            "Install packages (no deps to protect PyTorch)",
            check=True
        )

        # Note: We skip `pip check` because some packages (like xformers) may complain
        # about PyTorch version, but we intentionally pin it to 2.4.1 for PyTorch3D compatibility

        # Install dependencies but with constraints to keep PyTorch at 2.4.1
        constraints_file = self.node_root / "_pytorch_constraints.txt"
        with open(constraints_file, 'w') as f:
            f.write("torch==2.4.1\n")
            f.write("torchvision==0.19.1\n")

        _run_subprocess_logged(
            [
                str(python_exe), "-m", "pip", "install",
                "-r", str(requirements_file),
                "-c", str(constraints_file),  # Use constraints to pin PyTorch
                "--upgrade",  # Upgrade other packages but respect constraints
            ],
            self.log_file,
            "Install remaining dependencies with PyTorch constraints",
            check=True
        )

        # Clean up constraints file
        if constraints_file.exists():
            constraints_file.unlink()

        # Step 5: Install gsplat from prebuilt wheel to avoid JIT compilation
        print("[SAM3DObjects] Installing gsplat (prebuilt wheel for PyTorch 2.4.1 + CUDA 12.1)...")
        _run_subprocess_logged(
            [
                str(python_exe), "-m", "pip", "install",
                "gsplat>=1.4.0",
                "--index-url", "https://docs.gsplat.studio/whl/pt24cu121",
            ],
            self.log_file,
            "Install gsplat prebuilt wheel",
            check=True
        )

        # Step 4.5: Install nvdiffrast from prebuilt wheel (platform-specific)
        # Using prebuilt wheel from Unique3D HuggingFace for Python 3.10
        # Note: The wheel has invalid version format, so we extract and install directly

        # Check Visual C++ Redistributable on Windows before installing wheels
        vc_installed, vc_error = check_vc_redistributable()
        if not vc_installed:
            print(f"\n{'='*60}")
            print("[SAM3DObjects] DEPENDENCY ERROR")
            print('='*60)
            print(vc_error)
            print('='*60)
            raise RuntimeError(f"Missing Visual C++ Redistributable:\n{vc_error}")

        print("[SAM3DObjects] Installing nvdiffrast (prebuilt wheel for Python 3.10)...")
        import urllib.request
        import tempfile
        import zipfile
        import shutil
        import platform

        system = platform.system()

        # Platform-specific wheel URLs
        nvdiffrast_wheel_urls = {
            "Linux": "https://huggingface.co/spaces/microsoft/TRELLIS/resolve/main/wheels/nvdiffrast-0.3.3-cp310-cp310-linux_x86_64.whl",
            "Windows": "https://huggingface.co/MonsterMMORPG/SECourses_Premium_Flash_Attention/resolve/main/nvdiffrast-0.3.3-cp310-cp310-win_amd64.whl",
            # Darwin: No prebuilt wheel available, will fall back to pip install from source
        }

        nvdiffrast_wheel_url = nvdiffrast_wheel_urls.get(system)

        if nvdiffrast_wheel_url:
            # Validate wheel URL before downloading
            print(f"[SAM3DObjects] Validating nvdiffrast wheel URL...")
            validation = validate_wheel_url(nvdiffrast_wheel_url, timeout=10)

            if not validation['valid']:
                error_msg = (
                    f"nvdiffrast wheel URL validation failed!\n"
                    f"  URL: {nvdiffrast_wheel_url}\n"
                    f"  Error: {validation['error']}\n"
                )
                if validation['status_code']:
                    error_msg += f"  HTTP Status: {validation['status_code']}\n"
                error_msg += (
                    f"\nThe prebuilt wheel for {system} is not available.\n"
                    f"This may be because:\n"
                    f"  - The URL has changed or the file was removed\n"
                    f"  - Network connectivity issues\n"
                    f"  - The hosting service is temporarily unavailable\n"
                    f"\nPlease report this issue at:\n"
                    f"  https://github.com/anthropics/claude-code/issues"
                )
                raise RuntimeError(error_msg)

            # Log file size if available
            if validation['content_length']:
                size_mb = int(validation['content_length']) / (1024 * 1024)
                print(f"[SAM3DObjects] Downloading nvdiffrast wheel ({size_mb:.1f} MB)...")
            else:
                print(f"[SAM3DObjects] Downloading nvdiffrast wheel...")

            # Install from prebuilt wheel
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                wheel_path = tmpdir_path / "nvdiffrast.whl"
                urllib.request.urlretrieve(nvdiffrast_wheel_url, wheel_path)

            # Extract wheel completely (it's a zip file)
            extract_dir = tmpdir_path / "extracted"
            extract_dir.mkdir()
            with zipfile.ZipFile(wheel_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)

            # Copy ALL package files to site-packages
            # Windows: Lib/site-packages, Unix: lib/python3.10/site-packages
            if platform.system() == "Windows":
                site_packages = self.env_dir / "Lib" / "site-packages"
            else:
                site_packages = self.env_dir / "lib" / "python3.10" / "site-packages"

            # Copy the nvdiffrast package directory
            nvdiffrast_src = extract_dir / "nvdiffrast"
            if nvdiffrast_src.exists():
                nvdiffrast_dest = site_packages / "nvdiffrast"
                if nvdiffrast_dest.exists():
                    rmtree_windows_robust(nvdiffrast_dest)
                shutil.copytree(nvdiffrast_src, nvdiffrast_dest)
                print(f"[SAM3DObjects] Installed nvdiffrast package to {nvdiffrast_dest}")

            # Copy ALL compiled plugin files (platform-specific extensions)
            # Linux/macOS: .so files, Windows: .dll/.pyd files
            lib_ext = "*.dll" if system == "Windows" else "*.so"
            plugin_count = 0
            for plugin_file in extract_dir.glob(lib_ext):
                shutil.copy2(plugin_file, site_packages / plugin_file.name)
                print(f"[SAM3DObjects] Installed compiled plugin: {plugin_file.name}")
                plugin_count += 1

            # Also check for .pyd files on Windows (Python extension modules)
            if system == "Windows":
                for pyd_file in extract_dir.glob("*.pyd"):
                    shutil.copy2(pyd_file, site_packages / pyd_file.name)
                    print(f"[SAM3DObjects] Installed compiled plugin: {pyd_file.name}")
                    plugin_count += 1

            if plugin_count == 0:
                print(f"[SAM3DObjects] Warning: No compiled plugins ({lib_ext}) found in wheel!")

            print(f"[SAM3DObjects] nvdiffrast installation complete ({plugin_count} plugin(s) installed)")
        else:
            # Fallback: Install from source (for Windows/macOS where no prebuilt wheel available)
            print(f"[SAM3DObjects] No prebuilt wheel for {system}, installing nvdiffrast from source...")
            print("[SAM3DObjects] Warning: This requires a working C++ compiler and may take several minutes")
            _run_subprocess_logged(
                [str(python_exe), "-m", "pip", "install", "nvdiffrast"],
                self.log_file,
                "Install nvdiffrast from source",
                check=True
            )
            print("[SAM3DObjects] nvdiffrast installed from source")

        # Step 6: Install kaolin (NVIDIA library with special wheel location)
        print("[SAM3DObjects] Installing Kaolin...")
        _run_subprocess_logged(
            [
                str(python_exe), "-m", "pip", "install",
                "kaolin==0.17.0",
                "-f", "https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.1_cu121.html"
            ],
            self.log_file,
            "Install kaolin from NVIDIA S3 for PyTorch 2.4.1",
            check=True
        )

        # Step 6: Install CUDA toolkit (nvcc compiler + headers) for JIT compilation
        # MUST be before PyTorch3D in case it needs to build from source!
        # gsplat and other packages also need nvcc for CUDA extension compilation
        print("[SAM3DObjects] Installing CUDA toolkit for JIT compilation...")
        print("[SAM3DObjects] (Installing compilers BEFORE PyTorch3D in case build needed)")

        # Try Tier 1 first (PyPI), fallback to Tier 2 (conda-forge extraction)
        if not self._install_cuda_toolkit_pypi(python_exe):
            print("[SAM3DObjects] PyPI cuda-toolkit incomplete, using conda-forge extraction...")
            self._install_cuda_toolkit_from_conda(python_exe)

        # Step 7: Install g++ compiler for CUDA JIT compilation
        # nvcc needs g++ as the host compiler to compile C++ code
        # NOTE: PyTorch3D is already installed (step 3 with PyTorch)
        print("[SAM3DObjects] Installing g++ compiler for CUDA JIT compilation...")
        self._install_gcc_from_conda(python_exe)

        print(f"[SAM3DObjects] All dependencies installed! (Full logs: {self.log_file})")

    def _install_pytorch3d_from_conda(self, python_exe: Path) -> None:
        """
        Install PyTorch3D using micromamba.

        This uses micromamba to find and install the correct prebuilt PyTorch3D
        package from conda channels, avoiding manual URL management and extraction.
        """
        print("[SAM3DObjects] Installing PyTorch3D via micromamba from conda channels...")

        try:
            # Use micromamba to install PyTorch3D from pytorch3d channel
            # Micromamba will automatically find the right build for Python 3.10 + CUDA 12.1
            _run_subprocess_logged(
                [
                    str(self.micromamba_exe), "install",
                    "-p", str(self.env_dir),
                    "-c", "pytorch3d",     # Official pytorch3d channel
                    "-c", "fvcore",        # Required dependency channel
                    "-c", "conda-forge",   # Fallback channel
                    "pytorch3d",
                    "-y"
                ],
                self.log_file,
                "Install PyTorch3D via micromamba",
                check=True
            )

            # Verify PyTorch3D was installed
            verify_result = subprocess.run(
                [str(python_exe), "-c", "import pytorch3d; print(pytorch3d.__version__)"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if verify_result.returncode == 0:
                print(f"[SAM3DObjects] PyTorch3D installed successfully via micromamba!")
                print(f"[SAM3DObjects] PyTorch3D version: {verify_result.stdout.strip()}")
            else:
                raise RuntimeError(f"PyTorch3D import failed: {verify_result.stderr}")

        except Exception as e:
            print(f"[SAM3DObjects] Failed to install PyTorch3D via micromamba: {e}")

            # Fallback: Use third-party prebuilt wheels (NO source building!)
            # MiroPsota's repository provides Windows/Linux wheels for PyTorch 2.4.1 + CUDA 12.1
            print("[SAM3DObjects] Using prebuilt PyTorch3D wheels from third-party repository...")
            print("[SAM3DObjects] Repository: https://github.com/MiroPsota/torch_packages_builder")

            _run_subprocess_logged(
                [
                    str(python_exe), "-m", "pip", "install",
                    "--extra-index-url", "https://miropsota.github.io/torch_packages_builder",
                    "pytorch3d==0.7.7+pt2.4.1cu121"
                ],
                self.log_file,
                "Install PyTorch3D from prebuilt wheel",
                check=True
            )
            print("[SAM3DObjects] PyTorch3D installed from prebuilt wheel!")

    def _install_cuda_toolkit_pypi(self, python_exe: Path) -> bool:
        """
        Install CUDA toolkit from PyPI (Tier 1).

        Returns:
            True if successful and nvcc is available, False otherwise
        """
        print("[SAM3DObjects] Attempting to install CUDA toolkit from PyPI...")

        try:
            # Try installing cuda-toolkit package from PyPI
            _run_subprocess_logged(
                [str(python_exe), "-m", "pip", "install", "cuda-toolkit[nvcc,cudart,crt]"],
                self.log_file,
                "Install cuda-toolkit from PyPI",
                check=True
            )

            # Verify nvcc exists in the venv (cross-platform using Path.glob)
            import platform
            nvcc_pattern = "nvcc.exe" if platform.system() == "Windows" else "nvcc"
            nvcc_paths = list(self.env_dir.glob(f"**/{nvcc_pattern}"))

            if nvcc_paths:
                nvcc_path = str(nvcc_paths[0])  # Take first match
                print(f"[SAM3DObjects] nvcc found at: {nvcc_path}")

                # Verify it's executable
                test_result = subprocess.run(
                    [nvcc_path, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )

                if test_result.returncode == 0:
                    print(f"[SAM3DObjects] CUDA toolkit from PyPI installed successfully!")
                    print(f"[SAM3DObjects] nvcc version: {test_result.stdout.splitlines()[0]}")
                    return True
                else:
                    print("[SAM3DObjects] nvcc found but not executable")
                    return False
            else:
                print("[SAM3DObjects] PyPI cuda-toolkit package incomplete (no nvcc)")
                return False

        except Exception as e:
            print(f"[SAM3DObjects] PyPI cuda-toolkit installation failed: {e}")
            _log_subprocess_output(self.log_file, f"PyPI cuda-toolkit failed: {e}")
            return False

    def _install_cuda_toolkit_from_conda(self, python_exe: Path) -> None:
        """
        Download and extract CUDA toolkit from conda-forge (Tier 2).

        This uses the same proven approach as pytorch3d installation.
        """
        import urllib.request
        import tarfile
        import tempfile
        import shutil

        # CUDA toolkit dev package from conda-forge for CUDA 12.1 (platform-specific)
        # From: https://anaconda.org/conda-forge/cudatoolkit-dev
        import platform
        system = platform.system()

        cuda_toolkit_urls = {
            "Linux": "https://conda.anaconda.org/conda-forge/linux-64/cudatoolkit-dev-12.1.0-h4b99516_3.conda",
            "Windows": "https://conda.anaconda.org/conda-forge/win-64/cudatoolkit-dev-12.1.0-hd020da6_3.conda",
            "Darwin": "https://conda.anaconda.org/conda-forge/osx-64/cudatoolkit-dev-12.1.0-h2e7b6a8_3.conda",
        }

        cuda_toolkit_url = cuda_toolkit_urls.get(system)
        if not cuda_toolkit_url:
            raise RuntimeError(f"Unsupported platform for CUDA toolkit: {system}")

        # Platform-specific nvcc executable name
        nvcc_exe = "nvcc.exe" if system == "Windows" else "nvcc"

        print("[SAM3DObjects] Downloading CUDA toolkit from conda-forge...")

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                archive_path = tmpdir_path / "cudatoolkit-dev.conda"

                # Download conda package (.conda format is a zip file containing a tar.zst)
                urllib.request.urlretrieve(cuda_toolkit_url, archive_path)

                # Extract .conda file (it's a zip containing pkg-*.tar.zst)
                print("[SAM3DObjects] Extracting conda package...")
                extract_dir = tmpdir_path / "extracted"
                extract_dir.mkdir()

                # .conda files are zip archives
                import zipfile
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)

                # Now extract the inner tar.zst file
                tar_zst_file = None
                for item in extract_dir.glob("*.tar.zst"):
                    tar_zst_file = item
                    break

                if not tar_zst_file:
                    # Try regular .tar.bz2 format
                    for item in extract_dir.glob("*.tar.bz2"):
                        tar_zst_file = item
                        break

                if not tar_zst_file:
                    raise RuntimeError(f"Could not find tar archive in conda package")

                print(f"[SAM3DObjects] Extracting {tar_zst_file.name}...")
                inner_extract_dir = tmpdir_path / "inner"
                inner_extract_dir.mkdir()

                # Extract tar.zst or tar.bz2
                if tar_zst_file.suffix == ".zst":
                    # Need zstandard decompression
                    try:
                        import zstandard as zstd
                    except ImportError:
                        # Install zstandard if not available (should already be in requirements_env.txt)
                        print("[SAM3DObjects] zstandard not found, installing...")
                        subprocess.run([str(python_exe), "-m", "pip", "install", "zstandard"],
                                     check=True, capture_output=True, text=True)
                        # Use importlib to force fresh import after installation
                        zstd = importlib.import_module("zstandard")

                    with open(tar_zst_file, 'rb') as compressed:
                        dctx = zstd.ZstdDecompressor()
                        with dctx.stream_reader(compressed) as reader:
                            with tarfile.open(fileobj=reader, mode='r|') as tar:
                                tar.extractall(inner_extract_dir)
                else:
                    # Regular tar.bz2
                    with tarfile.open(tar_zst_file, "r:bz2") as tar:
                        tar.extractall(inner_extract_dir)

                # Find CUDA installation directory
                cuda_root = None
                for candidate in [
                    inner_extract_dir / "nvvm",  # Conda packages often have nvvm/ at root
                    inner_extract_dir,
                ]:
                    if (candidate / "bin" / nvcc_exe).exists() or (candidate.parent / "bin" / nvcc_exe).exists():
                        cuda_root = candidate.parent if (candidate.parent / "bin" / nvcc_exe).exists() else candidate
                        break

                if not cuda_root:
                    # CUDA toolkit might be under a subdirectory
                    bin_dirs = list(inner_extract_dir.glob("**/bin"))
                    for bin_dir in bin_dirs:
                        if (bin_dir / nvcc_exe).exists():
                            cuda_root = bin_dir.parent
                            break

                if not cuda_root or not (cuda_root / "bin").exists():
                    raise RuntimeError(f"Could not find CUDA toolkit structure in conda package")

                # Copy CUDA toolkit to venv
                cuda_install_dir = self.env_dir / "cuda"
                if cuda_install_dir.exists():
                    rmtree_windows_robust(cuda_install_dir)

                print(f"[SAM3DObjects] Installing CUDA toolkit to {cuda_install_dir}...")
                shutil.copytree(cuda_root, cuda_install_dir)

                # Verify nvcc exists
                nvcc_path = cuda_install_dir / "bin" / nvcc_exe
                if not nvcc_path.exists():
                    raise RuntimeError(f"nvcc not found after installation at {nvcc_path}")

                # Make nvcc executable (Unix only - Windows doesn't need chmod)
                if system in ["Linux", "Darwin"]:
                    nvcc_path.chmod(0o755)

                # Test nvcc
                test_result = subprocess.run(
                    [str(nvcc_path), "--version"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )

                if test_result.returncode != 0:
                    raise RuntimeError(f"nvcc is not executable: {test_result.stderr}")

                print(f"[SAM3DObjects] CUDA toolkit installed successfully!")
                print(f"[SAM3DObjects] nvcc version: {test_result.stdout.splitlines()[0]}")
                print(f"[SAM3DObjects] CUDA location: {cuda_install_dir}")

        except Exception as e:
            raise RuntimeError(f"Failed to install CUDA toolkit from conda-forge: {e}") from e

    def _install_gcc_from_conda(self, python_exe: Path) -> None:
        """
        Install C++ compiler using micromamba (cross-platform).

        This provides the C++ host compiler that nvcc needs for CUDA JIT compilation.
        - Linux: Installs gxx_linux-64 and sysroot_linux-64
        - Windows: Installs m2w64-toolchain or detects system MSVC
        - macOS: Installs clang_osx-64
        """
        import platform
        system = platform.system()

        print(f"[SAM3DObjects] Installing C++ compiler for {system}...")

        try:
            # Platform-specific compiler packages
            if system == "Windows":
                # On Windows, try to detect and validate system MSVC first
                print("[SAM3DObjects] Checking for system MSVC compiler...")
                cl_path = find_msvc_compiler()

                msvc_works = False
                if cl_path:
                    print(f"[SAM3DObjects] Found MSVC at: {cl_path}")
                    print("[SAM3DObjects] Testing MSVC compilation...")

                    if test_compile_cpp(cl_path):
                        print("[SAM3DObjects] ✓ MSVC compiler works correctly")
                        msvc_works = True
                    else:
                        print("[SAM3DObjects] ✗ MSVC found but test compilation failed")
                        print("[SAM3DObjects] Will install m2w64-toolchain as fallback...")

                if not msvc_works:
                    # Install m2w64 toolchain if no system MSVC
                    print("[SAM3DObjects] No system MSVC found, installing m2w64-toolchain...")
                    _run_subprocess_logged(
                        [
                            str(self.micromamba_exe), "install",
                            "-p", str(self.env_dir),
                            "-c", "conda-forge",
                            "m2w64-gcc",
                            "m2w64-gcc-fortran",
                            "-y"
                        ],
                        self.log_file,
                        "Install m2w64 compiler via micromamba",
                        check=True
                    )

                    # Verify compiler installation
                    gcc_candidates = [
                        self.env_dir / "Library" / "mingw-w64" / "bin" / "gcc.exe",
                        self.env_dir / "Library" / "bin" / "gcc.exe",
                    ]

                    gcc_path = None
                    for candidate in gcc_candidates:
                        if candidate.exists():
                            gcc_path = candidate
                            break

                    if not gcc_path:
                        print("[SAM3DObjects] Warning: m2w64 gcc not found, relying on system compiler")
                    else:
                        print(f"[SAM3DObjects] m2w64 gcc installed: {gcc_path}")
                else:
                    print("[SAM3DObjects] Using system MSVC compiler")

            elif system == "Linux":
                # Linux: Use existing GCC installation logic
                _run_subprocess_logged(
                    [
                        str(self.micromamba_exe), "install",
                        "-p", str(self.env_dir),
                        "-c", "conda-forge",
                        "gxx_linux-64",  # Compiler package
                        "sysroot_linux-64",  # System libraries needed by compiler
                        "-y"
                    ],
                    self.log_file,
                    "Install g++ compiler via micromamba",
                    check=True
                )

                # Verify g++ was installed
                gxx_candidates = [
                    self.env_dir / "bin" / "x86_64-conda-linux-gnu-g++",
                    self.env_dir / "bin" / "g++",
                ]

                gxx_path = None
                for candidate in gxx_candidates:
                    if candidate.exists():
                        gxx_path = candidate
                        break

                if not gxx_path:
                    raise RuntimeError(
                        f"g++ not found after micromamba installation. "
                        f"Checked: {[str(c) for c in gxx_candidates]}"
                    )

                # Test g++
                test_result = subprocess.run(
                    [str(gxx_path), "--version"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )

                if test_result.returncode != 0:
                    raise RuntimeError(f"g++ is not executable: {test_result.stderr}")

                print(f"[SAM3DObjects] g++ compiler installed successfully!")
                print(f"[SAM3DObjects] g++ version: {test_result.stdout.splitlines()[0]}")
                print(f"[SAM3DObjects] g++ location: {gxx_path}")

            elif system == "Darwin":
                # macOS: Install clang
                _run_subprocess_logged(
                    [
                        str(self.micromamba_exe), "install",
                        "-p", str(self.env_dir),
                        "-c", "conda-forge",
                        "clang_osx-64",
                        "clangxx_osx-64",
                        "-y"
                    ],
                    self.log_file,
                    "Install clang compiler via micromamba",
                    check=True
                )
                print("[SAM3DObjects] clang compiler installed successfully!")

        except Exception as e:
            raise RuntimeError(f"Failed to install C++ compiler: {e}") from e

    def setup_environment(self) -> None:
        """Complete environment setup process."""
        print("[SAM3DObjects] Starting installation...")
        print(f"[SAM3DObjects] Full logs will be saved to: {self.log_file}")

        # Step 1: Create venv
        self.create_environment()

        # Step 2: Install all dependencies
        self.install_dependencies()

        # Step 3: Verify
        if self.is_environment_ready():
            print("[SAM3DObjects] Installation complete!")
            print(f"[SAM3DObjects] Full logs: {self.log_file}")
        else:
            raise RuntimeError("Environment setup completed but verification failed")

    # ========================================================================
    # COMPILER DETECTION FUNCTIONS (COMMENTED OUT FOR NOW)
    # ========================================================================
    # These functions detect system compilers to avoid downloading them.
    # They are currently disabled to ensure the fallback download path works.
    # Uncomment and integrate these once the download path is verified.
    # ========================================================================

    # def detect_cxx_compiler(self) -> Optional[Path]:
    #     """
    #     Detect C++ compiler on the system (cross-platform).
    #
    #     Returns:
    #         Path to g++/clang++/cl.exe if found and compatible, None otherwise
    #     """
    #     import shutil
    #
    #     system = platform.system()
    #     compiler_candidates = []
    #
    #     if system == "Linux":
    #         compiler_candidates = ["g++", "clang++"]
    #     elif system == "Darwin":  # macOS
    #         compiler_candidates = ["clang++", "g++"]
    #     elif system == "Windows":
    #         compiler_candidates = ["cl.exe", "g++.exe", "clang++.exe"]
    #
    #     for compiler in compiler_candidates:
    #         compiler_path = shutil.which(compiler)
    #         if compiler_path:
    #             # Verify it works
    #             try:
    #                 result = subprocess.run(
    #                     [compiler_path, "--version"],
    #                     capture_output=True,
    #                     text=True,
    #                     timeout=5
    #                 )
    #                 if result.returncode == 0:
    #                     # Check if version is compatible with CUDA 12.1
    #                     # CUDA 12.1 supports g++ 7.x-12.x, clang 11-15
    #                     if self._verify_cxx_cuda_compatibility(compiler_path, result.stdout):
    #                         print(f"[SAM3DObjects] Found compatible C++ compiler: {compiler_path}")
    #                         print(f"[SAM3DObjects] Version: {result.stdout.splitlines()[0]}")
    #                         return Path(compiler_path)
    #             except Exception as e:
    #                 print(f"[SAM3DObjects] Compiler {compiler_path} not usable: {e}")
    #                 continue
    #
    #     return None
    #
    # def detect_nvcc(self) -> Optional[Path]:
    #     """
    #     Detect CUDA nvcc compiler on the system (cross-platform).
    #
    #     Returns:
    #         Path to nvcc if found and compatible with CUDA 12.x, None otherwise
    #     """
    #     import shutil
    #
    #     # Check PATH first
    #     nvcc_path = shutil.which("nvcc")
    #     if nvcc_path:
    #         if self._verify_nvcc_version(Path(nvcc_path)):
    #             return Path(nvcc_path)
    #
    #     # Check common CUDA installation directories
    #     system = platform.system()
    #     cuda_paths = []
    #
    #     if system == "Linux":
    #         cuda_paths = [
    #             "/usr/local/cuda/bin/nvcc",
    #             "/usr/local/cuda-12.1/bin/nvcc",
    #             "/usr/local/cuda-12/bin/nvcc",
    #         ]
    #     elif system == "Windows":
    #         cuda_paths = [
    #             r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin\nvcc.exe",
    #             r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin\nvcc.exe",
    #         ]
    #     elif system == "Darwin":  # macOS
    #         cuda_paths = [
    #             "/Developer/NVIDIA/CUDA-12.1/bin/nvcc",
    #         ]
    #
    #     for cuda_path in cuda_paths:
    #         cuda_path_obj = Path(cuda_path)
    #         if cuda_path_obj.exists():
    #             if self._verify_nvcc_version(cuda_path_obj):
    #                 return cuda_path_obj
    #
    #     return None
    #
    # def _verify_cxx_cuda_compatibility(self, compiler_path: Path, version_output: str) -> bool:
    #     """
    #     Verify C++ compiler is compatible with CUDA 12.1.
    #
    #     CUDA 12.1 compatibility:
    #     - g++: 7.x through 12.x
    #     - clang: 11.x through 15.x
    #     - MSVC: 2019 or 2022
    #     """
    #     version_lower = version_output.lower()
    #
    #     # Extract version number
    #     import re
    #     version_match = re.search(r'(\d+)\.(\d+)\.(\d+)', version_output)
    #     if not version_match:
    #         return False
    #
    #     major = int(version_match.group(1))
    #     minor = int(version_match.group(2))
    #
    #     # Check g++
    #     if "g++" in compiler_path.name.lower():
    #         # g++ 7.x - 12.x supported
    #         if 7 <= major <= 12:
    #             return True
    #         print(f"[SAM3DObjects] g++ version {major}.{minor} not supported by CUDA 12.1 (need 7-12)")
    #         return False
    #
    #     # Check clang
    #     if "clang" in compiler_path.name.lower():
    #         # clang 11.x - 15.x supported
    #         if 11 <= major <= 15:
    #             return True
    #         print(f"[SAM3DObjects] clang version {major}.{minor} not supported by CUDA 12.1 (need 11-15)")
    #         return False
    #
    #     # Check MSVC (cl.exe)
    #     if "cl.exe" in compiler_path.name.lower():
    #         # MSVC 2019 (19.2x) or 2022 (19.3x+) supported
    #         if major == 19 and minor >= 20:
    #             return True
    #         print(f"[SAM3DObjects] MSVC version {major}.{minor} not supported by CUDA 12.1 (need 2019+)")
    #         return False
    #
    #     return False
    #
    # def _verify_nvcc_version(self, nvcc_path: Path) -> bool:
    #     """
    #     Verify nvcc is compatible (CUDA 12.x).
    #
    #     Returns:
    #         True if nvcc is CUDA 12.x, False otherwise
    #     """
    #     try:
    #         result = subprocess.run(
    #             [str(nvcc_path), "--version"],
    #             capture_output=True,
    #             text=True,
    #             timeout=5
    #         )
    #
    #         if result.returncode == 0:
    #             # Parse CUDA version from nvcc output
    #             import re
    #             version_match = re.search(r'release (\d+)\.(\d+)', result.stdout)
    #             if version_match:
    #                 major = int(version_match.group(1))
    #                 minor = int(version_match.group(2))
    #
    #                 # Accept CUDA 12.x
    #                 if major == 12:
    #                     print(f"[SAM3DObjects] Found compatible nvcc: {nvcc_path}")
    #                     print(f"[SAM3DObjects] CUDA version: {major}.{minor}")
    #                     return True
    #                 else:
    #                     print(f"[SAM3DObjects] nvcc CUDA version {major}.{minor} not compatible (need 12.x)")
    #                     return False
    #
    #     except Exception as e:
    #         print(f"[SAM3DObjects] Could not verify nvcc at {nvcc_path}: {e}")
    #
    #     return False
    #
    # def _setup_compiler_environment(self, cxx_compiler: Optional[Path], nvcc: Optional[Path]) -> dict:
    #     """
    #     Setup environment variables for detected compilers.
    #
    #     Args:
    #         cxx_compiler: Path to C++ compiler (g++/clang++/cl.exe)
    #         nvcc: Path to nvcc compiler
    #
    #     Returns:
    #         Dictionary of environment variables to set
    #     """
    #     env = {}
    #
    #     if cxx_compiler:
    #         env["CUDAHOSTCXX"] = str(cxx_compiler)
    #         # Add compiler directory to PATH
    #         compiler_dir = cxx_compiler.parent
    #         current_path = os.environ.get("PATH", "")
    #         env["PATH"] = f"{compiler_dir}{os.pathsep}{current_path}"
    #
    #     if nvcc:
    #         cuda_home = nvcc.parent.parent  # nvcc is in bin/, CUDA_HOME is parent
    #         env["CUDA_HOME"] = str(cuda_home)
    #         env["CUDA_PATH"] = str(cuda_home)
    #
    #     return env
    #
    # # END OF COMMENTED COMPILER DETECTION FUNCTIONS
