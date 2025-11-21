"""
Environment manager for SAM3D isolated micromamba environment.

This module handles creation, management, and validation of the isolated
micromamba environment for SAM3D inference.
"""

import os
import sys
import subprocess
import platform
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
    """Manages the isolated micromamba environment for SAM3D."""

    def __init__(self, node_root: Path):
        """
        Initialize environment manager.

        Args:
            node_root: Root directory of the ComfyUI-SAM3DObjects node
        """
        self.node_root = Path(node_root)
        self.env_dir = self.node_root / "_env"
        self.micromamba_dir = self.node_root / "_micromamba"
        self.micromamba_bin = self._get_micromamba_path()
        self.log_file = self.node_root / "install.log"

    def _get_micromamba_path(self) -> Path:
        """Get path to micromamba binary based on platform."""
        if platform.system() == "Windows":
            return self.micromamba_dir / "micromamba.exe"
        else:
            return self.micromamba_dir / "bin" / "micromamba"

    def get_python_executable(self) -> Path:
        """Get path to Python executable in isolated environment."""
        if platform.system() == "Windows":
            return self.env_dir / "python.exe"
        else:
            return self.env_dir / "bin" / "python"

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

    def install_micromamba(self) -> None:
        """Download and install micromamba if not present."""
        if self.micromamba_bin.exists():
            print("[SAM3DObjects] Micromamba already installed")
            return

        print("[SAM3DObjects] Installing micromamba...")
        self.micromamba_dir.mkdir(parents=True, exist_ok=True)

        # Determine download URL based on platform
        system = platform.system()
        machine = platform.machine().lower()

        if system == "Linux":
            if "aarch64" in machine or "arm64" in machine:
                url = "https://micro.mamba.pm/api/micromamba/linux-aarch64/latest"
            else:
                url = "https://micro.mamba.pm/api/micromamba/linux-64/latest"
        elif system == "Darwin":
            if "arm64" in machine:
                url = "https://micro.mamba.pm/api/micromamba/osx-arm64/latest"
            else:
                url = "https://micro.mamba.pm/api/micromamba/osx-64/latest"
        elif system == "Windows":
            url = "https://micro.mamba.pm/api/micromamba/win-64/latest"
        else:
            raise RuntimeError(f"Unsupported platform: {system}")

        # Download micromamba
        print("[SAM3DObjects] Downloading micromamba...")
        try:
            import urllib.request
            import tarfile
            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                archive_path = Path(tmpdir) / "micromamba.tar.bz2"

                urllib.request.urlretrieve(url, archive_path)

                # Extract
                with tarfile.open(archive_path, "r:bz2") as tar:
                    tar.extractall(self.micromamba_dir)

                # Make executable on Unix
                if system != "Windows":
                    os.chmod(self.micromamba_bin, 0o755)

                print("[SAM3DObjects] Micromamba installed")

        except Exception as e:
            raise RuntimeError(f"Failed to install micromamba: {e}") from e

    def create_environment(self, python_version: str = "3.10") -> None:
        """
        Create the isolated micromamba environment.

        Args:
            python_version: Python version to install
        """
        if self.env_dir.exists():
            if self.is_environment_ready():
                print("[SAM3DObjects] Environment already exists, skipping creation")
                return
            else:
                print("[SAM3DObjects] Recreating incomplete environment")

        print("[SAM3DObjects] Creating base environment...")

        # Create environment with Python
        env = os.environ.copy()
        env["MAMBA_ROOT_PREFIX"] = str(self.micromamba_dir)

        try:
            subprocess.run(
                [
                    str(self.micromamba_bin),
                    "create",
                    "-p", str(self.env_dir),
                    f"python={python_version}",
                    "-c", "conda-forge",
                    "-y",
                ],
                env=env,
                check=True,
                capture_output=True,
                text=True
            )
            print("[SAM3DObjects] Base environment created")

        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to create environment: {e}\n"
                f"stdout: {e.stdout}\n"
                f"stderr: {e.stderr}"
            ) from e

    def install_pytorch_and_dependencies(self) -> None:
        """Install dependencies: conda for torch, pip builds pytorch3d from source."""
        python_exe = self.get_python_executable()

        # Step 1: Install conda packages from environment.yml (pytorch with CUDA, no pytorch3d)
        print("[SAM3DObjects] Installing Python environment (this may take several minutes)...")
        env_yml = self.node_root / "environment.yml"
        if not env_yml.exists():
            raise RuntimeError(f"environment.yml not found: {env_yml}")

        env = os.environ.copy()
        env["MAMBA_ROOT_PREFIX"] = str(self.micromamba_dir)

        _run_subprocess_logged(
            [
                str(self.micromamba_bin),
                "install",
                "-p", str(self.env_dir),
                "-f", str(env_yml),
                "-y",
                "--quiet"
            ],
            self.log_file,
            "Install PyTorch and CUDA dependencies",
            env=env,
            check=True
        )

        # Step 2: Install pip packages from requirements.txt (pytorch3d separately at end)
        requirements_file = self.node_root / "requirements.txt"
        if requirements_file.exists():
            print("[SAM3DObjects] Installing pip dependencies...")

            # Read requirements, separate pytorch3d to install last
            with open(requirements_file, 'r') as f:
                requirements = []
                pytorch3d_req = None
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if 'pytorch3d' in line.lower():
                            pytorch3d_req = line
                        else:
                            requirements.append(line)

            # Install everything except pytorch3d first
            if requirements:
                _run_subprocess_logged(
                    [str(python_exe), "-m", "pip", "install", "--quiet", "--no-progress"] + requirements,
                    self.log_file,
                    "Install pip dependencies",
                    check=True
                )

            # Now install pytorch3d from source (it needs torch to be importable during build)
            if pytorch3d_req:
                print("[SAM3DObjects] Building pytorch3d from source (this may take 10-15 minutes)...")
                # Set environment variables to help nvcc use g++ as the CUDA host compiler
                build_env = os.environ.copy()
                build_env['CC'] = '/usr/bin/gcc'
                build_env['CXX'] = '/usr/bin/g++'
                build_env['CUDAHOSTCXX'] = '/usr/bin/g++'  # Force nvcc to use g++ for C++ code
                build_env['PATH'] = '/usr/bin:' + build_env.get('PATH', '')
                _run_subprocess_logged(
                    [str(python_exe), "-m", "pip", "install", pytorch3d_req, "--no-build-isolation", "--quiet"],
                    self.log_file,
                    "Build pytorch3d from source",
                    check=True,
                    env=build_env
                )

        print(f"[SAM3DObjects] All dependencies installed! (Full logs: {self.log_file})")

    def setup_environment(self) -> None:
        """Complete environment setup process."""
        print("[SAM3DObjects] Starting installation...")
        print(f"[SAM3DObjects] Full logs will be saved to: {self.log_file}")

        # Step 1: Install micromamba
        self.install_micromamba()

        # Step 2: Create environment
        self.create_environment()

        # Step 3: Install dependencies
        self.install_pytorch_and_dependencies()

        # Step 4: Verify
        if self.is_environment_ready():
            print("[SAM3DObjects] Installation complete!")
            print(f"[SAM3DObjects] Full logs: {self.log_file}")
        else:
            raise RuntimeError("Environment setup completed but verification failed")
