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
            # Note: uv is already installed by PyTorchPipInstaller

            # Step 1: Separate git dependencies from regular packages
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

            # Step 2: Create constraints file to protect PyTorch version
            constraints_file = self.env_dir.parent / "_pytorch_constraints.txt"
            with open(constraints_file, 'w') as f:
                f.write(f"torch=={self.config.pytorch_version}\n")
                f.write(f"torchvision=={self.config.torchvision_version}\n")

            # Check if we're on Windows (need to handle Long Path issues)
            import platform
            is_windows = platform.system() == 'Windows'

            if is_windows:
                # On Windows, we need to avoid installing jupyterlab due to Long Path issues
                # Create exclusion file for problematic packages
                exclude_file = self.env_dir.parent / "_exclude_packages.txt"
                with open(exclude_file, 'w') as f:
                    # Only block packages that actually cause Windows Long Path issues
                    f.write("# Blocked packages - cause Windows Long Path issues\n")
                    f.write("jupyterlab<0.0.1\n")  # Main culprit for long paths
                    f.write("notebook<0.0.1\n")  # Also has long paths
                    f.write("trame<0.0.1\n")  # Optional pyvista dep, pulls in large deps
                    # NOTE: ipywidgets, jupyterlab-widgets, widgetsnbextension are ALLOWED
                    # (needed by open3d, and they don't cause long path issues themselves)
                    # NOTE: pyvista and vtk are ALLOWED (needed by pymeshfix)

                # Install with uv (much faster than pip)
                self.logger.info("Installing packages with uv (excluding jupyterlab)...")
                try:
                    self.run_uv_pip(
                        [
                            "install",
                            "-r", str(temp_reqs_file),
                            "-c", str(constraints_file),
                            "-c", str(exclude_file),
                        ],
                        step_name="Install dependencies via uv",
                        check=True
                    )
                except subprocess.CalledProcessError:
                    # Some packages may have hard deps on blocked packages
                    # Fall back to installing one-by-one
                    self.logger.warning("Batch install failed, installing packages individually...")
                    for req in regular_reqs:
                        try:
                            self.run_uv_pip(
                                [
                                    "install", req,
                                    "-c", str(constraints_file),
                                    "-c", str(exclude_file),
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
                # On Linux/macOS, install normally with uv
                self.run_uv_pip(
                    [
                        "install",
                        "-r", str(temp_reqs_file),
                        "-c", str(constraints_file),
                    ],
                    step_name="Install dependencies via uv",
                    check=True
                )

            # Step 3: Install git dependencies
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

            self.logger.success("Dependencies installed via uv")
            return True

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Pip dependencies installation failed: {e}")
            return False
