"""
CUDA toolkit and C++ compiler installers.
"""

import importlib
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from .base import Installer
from ..config import CUDA_TOOLKIT_URLS


class CudaToolkitInstaller(Installer):
    """
    Install CUDA toolkit (nvcc compiler + headers) for JIT compilation.

    Tries PyPI first (Tier 1), falls back to conda-forge extraction (Tier 2).
    """

    @property
    def name(self) -> str:
        return "CUDA Toolkit"

    def is_installed(self) -> bool:
        """Check if nvcc is available."""
        nvcc_exe = self.platform.get_nvcc_exe_name()

        # Check in venv
        nvcc_paths = list(self.env_dir.glob(f"**/{nvcc_exe}"))
        if nvcc_paths:
            nvcc_path = nvcc_paths[0]
            try:
                result = subprocess.run(
                    [str(nvcc_path), "--version"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                return result.returncode == 0
            except (subprocess.SubprocessError, OSError):
                pass

        return False

    def install(self) -> bool:
        """Install CUDA toolkit."""
        self.logger.info("Installing CUDA toolkit for JIT compilation...")

        # Try PyPI first (faster, cleaner)
        if self._install_from_pypi():
            return True

        # Fallback to conda-forge extraction
        self.logger.info("PyPI cuda-toolkit incomplete, using conda-forge extraction...")
        return self._install_from_conda()

    def _install_from_pypi(self) -> bool:
        """Install CUDA toolkit from PyPI (Tier 1)."""
        self.logger.info("Attempting to install CUDA toolkit from PyPI...")

        try:
            self.run_pip(
                ["install", "cuda-toolkit[nvcc,cudart,crt]"],
                step_name="Install cuda-toolkit from PyPI",
                check=True
            )

            # Verify nvcc exists
            nvcc_exe = self.platform.get_nvcc_exe_name()
            nvcc_paths = list(self.env_dir.glob(f"**/{nvcc_exe}"))

            if nvcc_paths:
                nvcc_path = nvcc_paths[0]
                result = subprocess.run(
                    [str(nvcc_path), "--version"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )

                if result.returncode == 0:
                    self.logger.success(f"CUDA toolkit from PyPI: {result.stdout.splitlines()[0]}")
                    return True

            self.logger.warning("PyPI cuda-toolkit package incomplete (no nvcc)")
            return False

        except subprocess.CalledProcessError:
            self.logger.warning("PyPI cuda-toolkit installation failed")
            return False

    def _install_from_conda(self) -> bool:
        """Install CUDA toolkit from conda-forge (Tier 2)."""
        cuda_url = CUDA_TOOLKIT_URLS.get(self.platform.name.capitalize())
        if not cuda_url:
            self.logger.error(f"No CUDA toolkit available for {self.platform.name}")
            return False

        self.logger.info("Downloading CUDA toolkit from conda-forge...")

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                archive_path = tmpdir_path / "cudatoolkit-dev.conda"

                # Download conda package
                urllib.request.urlretrieve(cuda_url, archive_path)

                # Extract .conda file (it's a zip)
                self.logger.info("Extracting conda package...")
                extract_dir = tmpdir_path / "extracted"
                extract_dir.mkdir()

                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)

                # Find inner tar archive
                tar_file = None
                for pattern in ["*.tar.zst", "*.tar.bz2"]:
                    for item in extract_dir.glob(pattern):
                        tar_file = item
                        break
                    if tar_file:
                        break

                if not tar_file:
                    raise RuntimeError("Could not find tar archive in conda package")

                # Extract inner archive
                inner_extract_dir = tmpdir_path / "inner"
                inner_extract_dir.mkdir()

                if tar_file.suffix == ".zst":
                    self._extract_tar_zst(tar_file, inner_extract_dir)
                else:
                    with tarfile.open(tar_file, "r:bz2") as tar:
                        tar.extractall(inner_extract_dir)

                # Find CUDA root directory
                nvcc_exe = self.platform.get_nvcc_exe_name()
                cuda_root = self._find_cuda_root(inner_extract_dir, nvcc_exe)

                if not cuda_root:
                    raise RuntimeError("Could not find CUDA toolkit structure in conda package")

                # Copy to venv
                cuda_install_dir = self.env_dir / "cuda"
                if cuda_install_dir.exists():
                    self.platform.rmtree_robust(cuda_install_dir)

                self.logger.info(f"Installing CUDA toolkit to {cuda_install_dir}...")
                shutil.copytree(cuda_root, cuda_install_dir)

                # Make nvcc executable
                nvcc_path = cuda_install_dir / "bin" / nvcc_exe
                if nvcc_path.exists():
                    self.platform.make_executable(nvcc_path)

                # Verify
                result = subprocess.run(
                    [str(nvcc_path), "--version"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )

                if result.returncode != 0:
                    raise RuntimeError(f"nvcc is not executable: {result.stderr}")

                self.logger.success(f"CUDA toolkit: {result.stdout.splitlines()[0]}")
                return True

        except Exception as e:
            self.logger.error(f"Failed to install CUDA toolkit: {e}")
            return False

    def _extract_tar_zst(self, tar_file: Path, dest: Path):
        """Extract tar.zst file."""
        try:
            import zstandard as zstd
        except ImportError:
            # Install zstandard
            self.logger.info("Installing zstandard for extraction...")
            subprocess.run(
                [str(self._paths.python), "-m", "pip", "install", "zstandard"],
                check=True, capture_output=True, text=True
            )
            zstd = importlib.import_module("zstandard")

        with open(tar_file, 'rb') as compressed:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(compressed) as reader:
                with tarfile.open(fileobj=reader, mode='r|') as tar:
                    tar.extractall(dest)

    def _find_cuda_root(self, search_dir: Path, nvcc_exe: str) -> Path:
        """Find the CUDA root directory containing bin/nvcc."""
        # Check common structures
        for candidate in [search_dir / "nvvm", search_dir]:
            if (candidate / "bin" / nvcc_exe).exists():
                return candidate
            if (candidate.parent / "bin" / nvcc_exe).exists():
                return candidate.parent

        # Search for bin directory containing nvcc
        for bin_dir in search_dir.glob("**/bin"):
            if (bin_dir / nvcc_exe).exists():
                return bin_dir.parent

        return None


class CompilerInstaller(Installer):
    """
    Install C++ compiler for CUDA JIT compilation.

    nvcc needs a C++ host compiler to compile the C++ portions of code.
    - Linux: g++ (via gxx_linux-64)
    - Windows: MSVC or m2w64-gcc
    - macOS: clang
    """

    @property
    def name(self) -> str:
        return "C++ Compiler"

    def is_installed(self) -> bool:
        """Check if a suitable compiler is available."""
        compiler_info = self.platform.detect_cxx_compiler()
        if compiler_info and compiler_info.is_compatible:
            return True

        # Check in venv
        if self.platform.name == 'linux':
            gxx_paths = [
                self.env_dir / "bin" / "x86_64-conda-linux-gnu-g++",
                self.env_dir / "bin" / "g++",
            ]
            for path in gxx_paths:
                if path.exists():
                    return True

        return False

    def install(self) -> bool:
        """Install C++ compiler."""
        self.logger.info(f"Installing C++ compiler for {self.platform.name}...")

        try:
            if self.platform.name == 'windows':
                return self._install_windows_compiler()
            elif self.platform.name == 'linux':
                return self._install_linux_compiler()
            elif self.platform.name == 'darwin':
                return self._install_macos_compiler()
            else:
                self.logger.error(f"Unknown platform: {self.platform.name}")
                return False

        except Exception as e:
            self.logger.error(f"Failed to install compiler: {e}")
            return False

    def _install_windows_compiler(self) -> bool:
        """Install compiler on Windows."""
        # Check for system MSVC first
        compiler_info = self.platform.detect_cxx_compiler()
        if compiler_info and compiler_info.is_compatible:
            self.logger.success(f"Using system {compiler_info.name}: {compiler_info.version}")
            return True

        # Install m2w64 toolchain
        self.logger.info("No system MSVC found, installing m2w64-toolchain...")
        self.run_micromamba(
            [
                "install",
                "-p", str(self.env_dir),
                "-c", "conda-forge",
                *self.platform.get_conda_compiler_packages(),
                "-y"
            ],
            step_name="Install m2w64 compiler",
            check=True
        )

        # Verify
        gcc_candidates = [
            self.env_dir / "Library" / "mingw-w64" / "bin" / "gcc.exe",
            self.env_dir / "Library" / "bin" / "gcc.exe",
        ]

        for candidate in gcc_candidates:
            if candidate.exists():
                self.logger.success(f"m2w64 gcc installed: {candidate}")
                return True

        self.logger.warning("m2w64 gcc not found, relying on system compiler")
        return True

    def _install_linux_compiler(self) -> bool:
        """Install compiler on Linux."""
        self.run_micromamba(
            [
                "install",
                "-p", str(self.env_dir),
                "-c", "conda-forge",
                *self.platform.get_conda_compiler_packages(),
                "-y"
            ],
            step_name="Install g++ compiler",
            check=True
        )

        # Verify
        gxx_candidates = [
            self.env_dir / "bin" / "x86_64-conda-linux-gnu-g++",
            self.env_dir / "bin" / "g++",
        ]

        for candidate in gxx_candidates:
            if candidate.exists():
                result = subprocess.run(
                    [str(candidate), "--version"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    self.logger.success(f"g++: {result.stdout.splitlines()[0]}")
                    return True

        self.logger.error("g++ not found after installation")
        return False

    def _install_macos_compiler(self) -> bool:
        """Install compiler on macOS."""
        self.run_micromamba(
            [
                "install",
                "-p", str(self.env_dir),
                "-c", "conda-forge",
                *self.platform.get_conda_compiler_packages(),
                "-y"
            ],
            step_name="Install clang compiler",
            check=True
        )

        self.logger.success("clang compiler installed")
        return True
