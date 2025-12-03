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
        try:
            result = subprocess.run(
                [str(python_exe), "-c", "import torch, pytorch3d, sam3d_objects"],
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
        print(f"[SAM3DObjects] Downloading from {url}...")
        try:
            import urllib.request
            import tarfile
            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                archive_path = Path(tmpdir) / "micromamba.tar.bz2"

                urllib.request.urlretrieve(url, archive_path)

                # Extract
                print("[SAM3DObjects] Extracting micromamba...")
                with tarfile.open(archive_path, "r:bz2") as tar:
                    tar.extractall(self.micromamba_dir)

                # Make executable on Unix
                if system != "Windows":
                    os.chmod(self.micromamba_bin, 0o755)

                print("[SAM3DObjects] Micromamba installed successfully!")

        except Exception as e:
            raise RuntimeError(f"Failed to install micromamba: {e}") from e

    def create_environment(self, python_version: str = "3.10") -> None:
        """
        Create the isolated micromamba environment.

        Args:
            python_version: Python version to install
        """
        if self.env_dir.exists():
            print(f"[SAM3DObjects] Environment directory exists: {self.env_dir}")
            if self.is_environment_ready():
                print("[SAM3DObjects] Environment is ready, skipping creation")
                return
            else:
                print("[SAM3DObjects] Environment exists but incomplete, will recreate")

        print(f"[SAM3DObjects] Creating isolated environment: {self.env_dir}")

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
            print("[SAM3DObjects] Base environment created successfully!")

        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to create environment: {e}\n"
                f"stdout: {e.stdout}\n"
                f"stderr: {e.stderr}"
            ) from e

    def install_pytorch_and_dependencies(self) -> None:
        """Install PyTorch, pytorch3d, and SAM3D dependencies."""
        print("[SAM3DObjects] Installing PyTorch and dependencies in isolated environment...")

        python_exe = self.get_python_executable()
        if not python_exe.exists():
            raise RuntimeError(f"Python not found in environment: {python_exe}")

        # Install PyTorch and pytorch3d together via conda to ensure compatibility
        print("[SAM3DObjects] Installing PyTorch and pytorch3d via conda...")
        env = os.environ.copy()
        env["MAMBA_ROOT_PREFIX"] = str(self.micromamba_dir)

        subprocess.run(
            [
                str(self.micromamba_bin),
                "install",
                "-p", str(self.env_dir),
                "-c", "pytorch",
                "-c", "fvcore",
                "-c", "iopath",
                "-c", "conda-forge",
                "pytorch",
                "torchvision",
                "pytorch3d",
                "-y"
            ],
            env=env,
            check=True,
            capture_output=False
        )

        # Install SAM3D requirements (skip pytorch3d since we installed it via conda)
        requirements_file = self.node_root / "requirements.txt"
        if requirements_file.exists():
            print("[SAM3DObjects] Installing SAM3D dependencies...")

            # Read requirements and filter out pytorch3d (already installed via conda)
            with open(requirements_file, 'r') as f:
                requirements = []
                for line in f:
                    line = line.strip()
                    # Skip empty lines, comments, and pytorch3d
                    if line and not line.startswith('#') and 'pytorch3d' not in line.lower():
                        requirements.append(line)

            # Install each requirement
            if requirements:
                subprocess.run(
                    [str(python_exe), "-m", "pip", "install"] + requirements + [
                        "--upgrade-strategy", "only-if-needed"
                    ],
                    check=True,
                    capture_output=False
                )

        print("[SAM3DObjects] All dependencies installed successfully!")

    def setup_environment(self) -> None:
        """Complete environment setup process."""
        print("[SAM3DObjects] ========================================")
        print("[SAM3DObjects] Setting up isolated environment")
        print("[SAM3DObjects] ========================================")

        # Step 1: Install micromamba
        self.install_micromamba()

        # Step 2: Create environment
        self.create_environment()

        # Step 3: Install dependencies
        self.install_pytorch_and_dependencies()

        # Step 4: Verify
        if self.is_environment_ready():
            print("[SAM3DObjects] ========================================")
            print("[SAM3DObjects] Environment setup complete!")
            print(f"[SAM3DObjects] Python: {self.get_python_executable()}")
            print("[SAM3DObjects] ========================================")
        else:
            raise RuntimeError("Environment setup completed but verification failed")
