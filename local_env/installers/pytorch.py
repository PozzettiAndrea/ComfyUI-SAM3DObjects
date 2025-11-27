"""
PyTorch, torchvision, and PyTorch3D installers.
"""

import subprocess
from pathlib import Path

from .base import Installer


class PyTorchInstaller(Installer):
    """
    Install PyTorch, torchvision, and PyTorch3D via micromamba.

    Uses micromamba to install from conda channels, ensuring
    CUDA compatibility between torch, torchvision, and pytorch3d.
    """

    @property
    def name(self) -> str:
        return "PyTorch + PyTorch3D"

    def is_installed(self) -> bool:
        """Check if PyTorch and PyTorch3D are installed with CUDA."""
        result = self.run_python(
            "import torch, pytorch3d; "
            "print(f'PyTorch {torch.__version__}, PyTorch3D {pytorch3d.__version__}, CUDA {torch.cuda.is_available()}')"
        )
        if result.returncode != 0:
            return False
        # Verify CUDA is available
        return 'True' in result.stdout

    def install(self) -> bool:
        """Install PyTorch + PyTorch3D via micromamba."""
        self.logger.info(f"Installing PyTorch {self.config.pytorch_version} + PyTorch3D {self.config.pytorch3d_version}...")
        self.logger.info(f"(Using PyTorch {self.config.pytorch_version} - latest with PyTorch3D prebuilt support)")

        try:
            # Install PyTorch + PyTorch3D together via micromamba
            # This ensures CUDA version compatibility
            self.run_micromamba(
                [
                    "install",
                    "-p", str(self.env_dir),
                    "-c", "pytorch",       # PRIMARY: PyTorch official channel
                    "-c", "pytorch3d",     # PyTorch3D channel
                    "-c", "nvidia",        # NVIDIA CUDA packages
                    "-c", "fvcore",        # PyTorch3D dependency
                    "-c", "conda-forge",   # Fallback
                    f"pytorch=={self.config.pytorch_version}",
                    f"pytorch-cuda={self.config.cuda_version}",
                    f"torchvision=={self.config.torchvision_version}",
                    f"pytorch3d=={self.config.pytorch3d_version}",
                    "-y"
                ],
                step_name=f"Install PyTorch {self.config.pytorch_version} + PyTorch3D via micromamba",
                check=True
            )

            # Verify installation
            result = self.run_python(
                "import torch, pytorch3d; "
                f"print(f'PyTorch {{torch.__version__}}, PyTorch3D {{pytorch3d.__version__}}')"
            )

            if result.returncode == 0:
                self.logger.success(f"Verified: {result.stdout.strip()}")
                return True
            else:
                raise RuntimeError(f"Verification failed: {result.stderr}")

        except subprocess.CalledProcessError as e:
            self.logger.error(f"PyTorch installation failed: {e}")
            return False


class PipDependenciesInstaller(Installer):
    """
    Install pip dependencies from requirements file.

    Installs packages while protecting PyTorch version from being upgraded.
    """

    def __init__(self, *args, requirements_file: Path, **kwargs):
        super().__init__(*args, **kwargs)
        self.requirements_file = requirements_file

    @property
    def name(self) -> str:
        return "Pip Dependencies"

    def is_installed(self) -> bool:
        # Always run to ensure all dependencies are present
        return False

    def install(self) -> bool:
        """Install pip dependencies with PyTorch constraints."""
        self.logger.info("Installing pip dependencies...")

        if not self.requirements_file.exists():
            self.logger.error(f"Requirements file not found: {self.requirements_file}")
            return False

        try:
            # Step 1: Upgrade pip
            self.logger.info("Upgrading pip...")
            self.run_pip(["install", "--upgrade", "pip"], step_name="Upgrade pip", check=True)

            # Step 2: Install uv for faster package installation
            self.logger.info("Installing uv package manager...")
            self.run_pip(["install", "uv"], step_name="Install uv", check=True)

            # Step 3: Separate git dependencies from regular packages
            # uv has issues finding git on Windows, so we handle git deps separately
            regular_reqs = []
            git_reqs = []

            with open(self.requirements_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if line.startswith('git+') or 'github.com' in line:
                        git_reqs.append(line)
                    else:
                        regular_reqs.append(line)

            # Create temporary requirements file without git deps
            temp_reqs_file = self.env_dir.parent / "_requirements_no_git.txt"
            with open(temp_reqs_file, 'w') as f:
                f.write('\n'.join(regular_reqs))

            # Step 4: Install regular packages without deps first to avoid PyTorch upgrade
            self.logger.info("Installing packages (protecting PyTorch version)...")
            self.run_uv_pip(
                [
                    "install",
                    "--no-deps",
                    "-r", str(temp_reqs_file)
                ],
                step_name="Install packages (no deps)",
                check=True
            )

            # Step 5: Install with constraints to keep PyTorch at specific version
            constraints_file = self.env_dir.parent / "_pytorch_constraints.txt"
            with open(constraints_file, 'w') as f:
                f.write(f"torch=={self.config.pytorch_version}\n")
                f.write(f"torchvision=={self.config.torchvision_version}\n")

            # Check if we're on Windows (need to handle Long Path issues)
            import platform
            is_windows = platform.system() == 'Windows'

            if is_windows:
                # On Windows, we need to avoid installing ipywidgets/jupyterlab due to Long Path issues
                # Strategy: install all packages without deps first, then install deps selectively

                # First pass: install all top-level packages without dependencies
                self.logger.info("Installing packages without dependencies (avoiding Long Path issues)...")
                self.run_pip(
                    [
                        "install",
                        "-r", str(temp_reqs_file),
                        "--no-deps",
                        "--no-cache-dir",
                    ],
                    step_name="Install packages (no deps)",
                    check=True
                )

                # Second pass: install dependencies but exclude problematic packages
                # We'll use pip's --dry-run to see what deps are needed, then filter
                self.logger.info("Resolving and installing dependencies (excluding jupyterlab)...")

                # Install most dependencies via a separate pip call that excludes problematic packages
                # Use a constraint file to block ipywidgets and jupyterlab
                exclude_file = self.env_dir.parent / "_exclude_packages.txt"
                with open(exclude_file, 'w') as f:
                    # These packages cause Windows Long Path issues via jupyterlab
                    f.write("# Blocked packages - cause Windows Long Path issues\n")
                    f.write("jupyterlab<0.0.1\n")
                    f.write("jupyterlab-widgets<0.0.1\n")
                    f.write("widgetsnbextension<0.0.1\n")
                    f.write("jupyter-server<0.0.1\n")
                    f.write("notebook<0.0.1\n")
                    f.write("ipywidgets<0.0.1\n")  # Optional pyvista dep for Jupyter
                    f.write("trame<0.0.1\n")  # Optional pyvista dep for Jupyter
                    # NOTE: pyvista and vtk are now ALLOWED (needed by pymeshfix)

                # Try to install with exclusions. If packages conflict, that's OK - we'll handle manually
                try:
                    self.run_pip(
                        [
                            "install",
                            "-r", str(temp_reqs_file),
                            "-c", str(constraints_file),
                            "-c", str(exclude_file),
                            "--no-cache-dir",
                        ],
                        step_name="Install dependencies (excluding jupyterlab)",
                        check=True
                    )
                except subprocess.CalledProcessError:
                    # Some packages may have hard deps on blocked packages
                    # Fall back to installing one-by-one
                    self.logger.warning("Batch install failed, installing packages individually...")
                    for req in regular_reqs:
                        try:
                            self.run_pip(
                                [
                                    "install", req,
                                    "-c", str(constraints_file),
                                    "-c", str(exclude_file),
                                    "--no-cache-dir",
                                ],
                                step_name=f"Install {req[:30]}",
                                check=False  # Don't fail on individual packages
                            )
                        except subprocess.CalledProcessError:
                            self.logger.warning(f"Skipped {req} (incompatible with Long Path workaround)")

                # Clean up exclude file
                if exclude_file.exists():
                    exclude_file.unlink()
            else:
                # On Linux/macOS, install normally
                self.run_pip(
                    [
                        "install",
                        "-r", str(temp_reqs_file),
                        "-c", str(constraints_file),
                        "--upgrade",
                        "--no-cache-dir",
                    ],
                    step_name="Install dependencies with constraints",
                    check=True
                )

            # Step 6: Install git dependencies
            # On Windows, the venv may not have git in PATH, so we need to ensure
            # the parent environment's PATH is available
            if git_reqs:
                self.logger.info(f"Installing {len(git_reqs)} git dependencies...")
                import os
                import shutil

                # Check if git is available
                git_path = shutil.which('git')
                if not git_path:
                    self.logger.warning("Git not found in PATH. Skipping git dependencies.")
                    self.logger.warning("Install Git and add it to PATH to enable git dependencies.")
                else:
                    # Create environment with git available
                    env = os.environ.copy()
                    git_dir = str(Path(git_path).parent)
                    if git_dir not in env.get('PATH', ''):
                        env['PATH'] = git_dir + os.pathsep + env.get('PATH', '')

                    for git_req in git_reqs:
                        self.logger.info(f"Installing {git_req.split('/')[-1].split('@')[0]}...")
                        # Run uv pip with git in PATH (faster than pip)
                        result = self.run_uv_pip(
                            ["install", git_req, "--no-cache-dir"],
                            step_name=f"Install git dep: {git_req[:50]}",
                            check=False,
                            env=env
                        )
                        if result.returncode != 0:
                            self.logger.warning(f"Failed to install {git_req}: {result.stderr}")

            # Clean up
            if constraints_file.exists():
                constraints_file.unlink()
            if temp_reqs_file.exists():
                temp_reqs_file.unlink()

            self.logger.success("Pip dependencies installed")
            return True

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Pip dependencies installation failed: {e}")
            return False
