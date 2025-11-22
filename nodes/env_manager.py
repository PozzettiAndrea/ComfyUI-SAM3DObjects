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


class SAM3DEnvironmentManager:
    """Manages the isolated venv environment for SAM3D."""

    def __init__(self, node_root: Path):
        """
        Initialize environment manager.

        Args:
            node_root: Root directory of the ComfyUI-SAM3DObjects node
        """
        self.node_root = Path(node_root)
        self.env_dir = self.node_root / "_env"
        self.log_file = self.node_root / "install.log"

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

    def create_environment(self) -> None:
        """Create the isolated Python virtual environment."""
        if self.env_dir.exists():
            if self.is_environment_ready():
                print("[SAM3DObjects] Environment already exists, skipping creation")
                return
            else:
                print("[SAM3DObjects] Recreating incomplete environment")

        print("[SAM3DObjects] Creating Python virtual environment...")

        # Find a compatible Python version (3.10-3.12)
        # PyTorch 2.5.1 doesn't support Python 3.13 yet
        python_candidates = ["python3.10", "python3.11", "python3.12"]
        python_exe = None

        for candidate in python_candidates:
            try:
                result = subprocess.run(
                    [candidate, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    python_exe = candidate
                    print(f"[SAM3DObjects] Using {candidate} for venv")
                    break
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue

        if not python_exe:
            raise RuntimeError(
                "Could not find compatible Python version (3.10-3.12). "
                "PyTorch 2.5.1 requires Python 3.10, 3.11, or 3.12."
            )

        try:
            # Create venv using compatible Python version
            subprocess.run(
                [python_exe, "-m", "venv", str(self.env_dir)],
                check=True,
                capture_output=True,
                text=True
            )
            print("[SAM3DObjects] Virtual environment created")

        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to create virtual environment: {e}\n"
                f"stdout: {e.stdout}\n"
                f"stderr: {e.stderr}"
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

        # Step 3: Install PyTorch with specific CUDA version FIRST
        print("[SAM3DObjects] Installing PyTorch 2.5.1 with CUDA 12.1...")
        print("[SAM3DObjects] This ensures binary compatibility...")

        uv_exe = self.env_dir / "bin" / "uv" if platform.system() != "Windows" else self.env_dir / "Scripts" / "uv.exe"

        _run_subprocess_logged(
            [
                str(python_exe), "-m", "uv", "pip", "install",
                "torch==2.5.1",
                "torchvision==0.20.1",
                "--index-url", "https://download.pytorch.org/whl/cu121"
            ],
            self.log_file,
            "Install PyTorch with CUDA 12.1",
            check=True
        )

        # Step 4: Install all other dependencies
        print("[SAM3DObjects] Installing remaining dependencies...")
        print("[SAM3DObjects] (PyTorch version will be locked to 2.5.1)")

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
        # about PyTorch version, but we intentionally pin it to 2.5.1 for PyTorch3D compatibility

        # Install dependencies but with constraints to keep PyTorch at 2.5.1
        constraints_file = self.node_root / "_pytorch_constraints.txt"
        with open(constraints_file, 'w') as f:
            f.write("torch==2.5.1\n")
            f.write("torchvision==0.20.1\n")

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

        # Step 5: Install kaolin (NVIDIA library with special wheel location)
        print("[SAM3DObjects] Installing Kaolin...")
        _run_subprocess_logged(
            [
                str(python_exe), "-m", "pip", "install",
                "kaolin==0.17.0",
                "-f", "https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
            ],
            self.log_file,
            "Install kaolin from NVIDIA S3",
            check=True
        )

        # Step 6: Install pytorch3d (download prebuilt from conda-forge)
        print("[SAM3DObjects] Installing PyTorch3D...")
        print("[SAM3DObjects] Downloading prebuilt binary from conda-forge...")

        self._install_pytorch3d_from_conda(python_exe)

        # Step 7: Install CUDA toolkit (nvcc compiler + headers) for JIT compilation
        # gsplat and other packages need nvcc for CUDA extension compilation
        print("[SAM3DObjects] Installing CUDA toolkit for JIT compilation...")

        # Try Tier 1 first (PyPI), fallback to Tier 2 (conda-forge extraction)
        if not self._install_cuda_toolkit_pypi(python_exe):
            print("[SAM3DObjects] PyPI cuda-toolkit incomplete, using conda-forge extraction...")
            self._install_cuda_toolkit_from_conda(python_exe)

        # Step 8: Install g++ compiler for CUDA JIT compilation
        # nvcc needs g++ as the host compiler to compile C++ code
        print("[SAM3DObjects] Installing g++ compiler for CUDA JIT compilation...")
        self._install_gcc_from_conda(python_exe)

        print(f"[SAM3DObjects] All dependencies installed! (Full logs: {self.log_file})")

    def _install_pytorch3d_from_conda(self, python_exe: Path) -> None:
        """Download and install prebuilt PyTorch3D from conda-forge."""
        import urllib.request
        import tarfile
        import tempfile
        import shutil

        # PyTorch3D conda package URL for Python 3.10, PyTorch 2.5.1, CUDA 12.1
        # From: https://anaconda.org/pytorch3d/pytorch3d
        pytorch3d_url = "https://conda.anaconda.org/pytorch3d/linux-64/pytorch3d-0.7.7-py310_cu121_pyt251.tar.bz2"

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                archive_path = tmpdir_path / "pytorch3d.tar.bz2"

                # Download conda package
                print("[SAM3DObjects] Downloading pytorch3d conda package...")
                urllib.request.urlretrieve(pytorch3d_url, archive_path)

                # Extract tar.bz2
                print("[SAM3DObjects] Extracting conda package...")
                extract_dir = tmpdir_path / "extracted"
                extract_dir.mkdir()

                with tarfile.open(archive_path, "r:bz2") as tar:
                    tar.extractall(extract_dir)

                # Find the pytorch3d directory in site-packages
                conda_site_packages = extract_dir / "lib" / "python3.10" / "site-packages"

                if not conda_site_packages.exists():
                    raise RuntimeError(f"Could not find site-packages in conda package at {conda_site_packages}")

                # Get our venv's site-packages directory
                if platform.system() == "Windows":
                    venv_site_packages = self.env_dir / "Lib" / "site-packages"
                else:
                    venv_site_packages = self.env_dir / "lib" / "python3.10" / "site-packages"

                # Copy pytorch3d directory
                pytorch3d_src = conda_site_packages / "pytorch3d"
                pytorch3d_dst = venv_site_packages / "pytorch3d"

                if not pytorch3d_src.exists():
                    raise RuntimeError(f"pytorch3d not found in conda package at {pytorch3d_src}")

                print(f"[SAM3DObjects] Copying prebuilt binaries to {venv_site_packages}...")
                if pytorch3d_dst.exists():
                    shutil.rmtree(pytorch3d_dst)
                shutil.copytree(pytorch3d_src, pytorch3d_dst)

                # Also copy any .dist-info directories
                for item in conda_site_packages.glob("pytorch3d*.dist-info"):
                    dst_item = venv_site_packages / item.name
                    if dst_item.exists():
                        shutil.rmtree(dst_item)
                    shutil.copytree(item, dst_item)

                print("[SAM3DObjects] PyTorch3D installed successfully from prebuilt binaries!")

        except Exception as e:
            print(f"[SAM3DObjects] Failed to install prebuilt pytorch3d: {e}")
            print("[SAM3DObjects] Falling back to building from source...")

            # Fallback: build from source
            _run_subprocess_logged(
                [
                    str(python_exe), "-m", "pip", "install",
                    "wheel", "setuptools", "ninja"
                ],
                self.log_file,
                "Install build tools for pytorch3d",
                check=True
            )
            _run_subprocess_logged(
                [
                    str(python_exe), "-m", "pip", "install",
                    "--no-build-isolation",
                    "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.7"
                ],
                self.log_file,
                "Build and install pytorch3d from source",
                check=True
            )

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

            # Verify nvcc exists in the venv
            result = subprocess.run(
                ["find", str(self.env_dir), "-name", "nvcc", "-type", "f"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.stdout.strip():
                nvcc_path = result.stdout.strip().split('\n')[0]  # Take first match
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

        # CUDA toolkit dev package from conda-forge for CUDA 12.1
        # From: https://anaconda.org/conda-forge/cudatoolkit-dev
        cuda_toolkit_url = "https://conda.anaconda.org/conda-forge/linux-64/cudatoolkit-dev-12.1.0-h4b99516_3.conda"

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
                        # Install zstandard if not available
                        subprocess.run([str(python_exe), "-m", "pip", "install", "zstandard"],
                                     check=True, capture_output=True, text=True)
                        import zstandard as zstd

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
                    if (candidate / "bin" / "nvcc").exists() or (candidate.parent / "bin" / "nvcc").exists():
                        cuda_root = candidate.parent if (candidate.parent / "bin" / "nvcc").exists() else candidate
                        break

                if not cuda_root:
                    # CUDA toolkit might be under a subdirectory
                    bin_dirs = list(inner_extract_dir.glob("**/bin"))
                    for bin_dir in bin_dirs:
                        if (bin_dir / "nvcc").exists():
                            cuda_root = bin_dir.parent
                            break

                if not cuda_root or not (cuda_root / "bin").exists():
                    raise RuntimeError(f"Could not find CUDA toolkit structure in conda package")

                # Copy CUDA toolkit to venv
                cuda_install_dir = self.env_dir / "cuda"
                if cuda_install_dir.exists():
                    shutil.rmtree(cuda_install_dir)

                print(f"[SAM3DObjects] Installing CUDA toolkit to {cuda_install_dir}...")
                shutil.copytree(cuda_root, cuda_install_dir)

                # Verify nvcc exists
                nvcc_path = cuda_install_dir / "bin" / "nvcc"
                if not nvcc_path.exists():
                    raise RuntimeError(f"nvcc not found after installation at {nvcc_path}")

                # Make nvcc executable
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
        Download and extract g++ compiler from conda-forge.

        This provides the C++ host compiler that nvcc needs for CUDA JIT compilation.
        """
        import urllib.request
        import tarfile
        import tempfile
        import shutil

        # g++ compiler package from conda-forge for Linux x86_64
        # From: https://anaconda.org/conda-forge/gxx_linux-64
        gcc_url = "https://conda.anaconda.org/conda-forge/linux-64/gxx_linux-64-13.3.0-h6834431_5.conda"

        print("[SAM3DObjects] Downloading g++ compiler from conda-forge...")

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                archive_path = tmpdir_path / "gxx.conda"

                # Download conda package
                urllib.request.urlretrieve(gcc_url, archive_path)

                # Extract .conda file (zip containing tar.zst)
                print("[SAM3DObjects] Extracting g++ package...")
                extract_dir = tmpdir_path / "extracted"
                extract_dir.mkdir()

                import zipfile
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)

                # Extract inner tar.zst
                tar_zst_file = None
                for item in extract_dir.glob("*.tar.zst"):
                    tar_zst_file = item
                    break

                if not tar_zst_file:
                    # Try .tar.bz2
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
                    try:
                        import zstandard as zstd
                    except ImportError:
                        subprocess.run([str(python_exe), "-m", "pip", "install", "zstandard"],
                                     check=True, capture_output=True, text=True)
                        import zstandard as zstd

                    with open(tar_zst_file, 'rb') as compressed:
                        dctx = zstd.ZstdDecompressor()
                        with dctx.stream_reader(compressed) as reader:
                            with tarfile.open(fileobj=reader, mode='r|') as tar:
                                tar.extractall(inner_extract_dir)
                else:
                    with tarfile.open(tar_zst_file, "r:bz2") as tar:
                        tar.extractall(inner_extract_dir)

                # Copy gcc to venv
                gcc_install_dir = self.env_dir / "gcc"
                if gcc_install_dir.exists():
                    shutil.rmtree(gcc_install_dir)

                print(f"[SAM3DObjects] Installing g++ to {gcc_install_dir}...")
                shutil.copytree(inner_extract_dir, gcc_install_dir)

                # Verify g++ exists
                gxx_path = gcc_install_dir / "bin" / "x86_64-conda-linux-gnu-g++"
                if not gxx_path.exists():
                    raise RuntimeError(f"g++ not found after installation at {gxx_path}")

                # Make executable
                gxx_path.chmod(0o755)

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
                print(f"[SAM3DObjects] g++ location: {gcc_install_dir}")

        except Exception as e:
            raise RuntimeError(f"Failed to install g++ from conda-forge: {e}") from e

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
