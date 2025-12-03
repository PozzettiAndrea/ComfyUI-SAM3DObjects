"""
Specialized package installers: gsplat, nvdiffrast.

These packages require special handling due to:
- Custom wheel indices
- Platform-specific prebuilt wheels
- Invalid version formats requiring manual extraction
"""

import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from .base import Installer
from ..config import NVDIFFRAST_WHEEL_URLS, get_gsplat_index_url
from ..utils import validate_url


class GsplatInstaller(Installer):
    """Install gsplat from prebuilt wheels."""

    @property
    def name(self) -> str:
        return "gsplat"

    def is_installed(self) -> bool:
        return self.verify_import("gsplat")

    def install(self) -> bool:
        """Install gsplat from prebuilt wheel index."""
        self.logger.info(f"Installing gsplat >= {self.config.gsplat_version}...")

        # Install gsplat dependencies first (rich is required but not auto-installed from prebuilt wheel)
        self.logger.info("Installing gsplat dependencies...")
        try:
            self.run_pip(
                ["install", "rich>=12", "jaxtyping"],
                step_name="Install gsplat dependencies",
                check=True
            )
        except subprocess.CalledProcessError:
            self.logger.warning("Failed to install some gsplat dependencies")

        index_url = get_gsplat_index_url(
            self.config.pytorch_version,
            self.config.cuda_version
        )

        try:
            self.run_pip(
                [
                    "install",
                    f"gsplat>={self.config.gsplat_version}",
                    "--index-url", index_url,
                ],
                step_name="Install gsplat prebuilt wheel",
                check=True
            )

            if self.verify_import("gsplat"):
                self.logger.success("gsplat installed")
                return True
            else:
                self.logger.error("gsplat import failed after installation")
                return False

        except subprocess.CalledProcessError as e:
            self.logger.error(f"gsplat installation failed: {e}")
            return False


class NvdiffrastInstaller(Installer):
    """
    Install nvdiffrast from prebuilt wheels.

    Note: The wheels have invalid version format, so we extract and
    install manually rather than using pip directly.
    """

    @property
    def name(self) -> str:
        return "nvdiffrast"

    def is_installed(self) -> bool:
        return self.verify_import("nvdiffrast")

    def install(self) -> bool:
        """Install nvdiffrast."""
        # Try local wheel first (bundled with the custom node)
        # env_dir is _env, so parent is the node root
        node_root = self.env_dir.parent
        local_wheel = node_root / "local_env_settings" / "nvdiffrast-0.3.5-py3-none-any.whl"
        if local_wheel.exists():
            return self._install_from_local_wheel(local_wheel)

        # Fall back to remote wheel
        wheel_url = NVDIFFRAST_WHEEL_URLS.get(self.platform.name.capitalize())

        if wheel_url:
            return self._install_from_wheel(wheel_url)
        else:
            return self._install_from_source()

    def _install_from_local_wheel(self, wheel_path: Path) -> bool:
        """Install nvdiffrast from local wheel."""
        self.logger.info(f"Installing nvdiffrast from local wheel: {wheel_path.name}...")

        try:
            self.run_pip(
                ["install", str(wheel_path), "--force-reinstall"],
                step_name="Install nvdiffrast from local wheel",
                check=True
            )

            if self.verify_import("nvdiffrast"):
                self.logger.success("nvdiffrast installed from local wheel")
                return True
            else:
                self.logger.error("nvdiffrast import failed after installation")
                return False

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Local wheel installation failed: {e}")
            return False

    def _install_from_wheel(self, url: str) -> bool:
        """Install nvdiffrast from prebuilt wheel."""
        self.logger.info("Installing nvdiffrast from prebuilt wheel...")

        # Validate URL first
        self.logger.info("Validating wheel URL...")
        validation = validate_url(url, timeout=10)

        if not validation['valid']:
            self.logger.error(f"Wheel URL not accessible: {validation['error']}")
            self.logger.info("Falling back to source installation...")
            return self._install_from_source()

        # Log file size if available
        if validation['content_length']:
            size_mb = int(validation['content_length']) / (1024 * 1024)
            self.logger.info(f"Downloading wheel ({size_mb:.1f} MB)...")
        else:
            self.logger.info("Downloading wheel...")

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                wheel_path = tmpdir_path / "nvdiffrast.whl"

                # Download
                urllib.request.urlretrieve(url, wheel_path)

                # Extract (wheel is a zip file)
                extract_dir = tmpdir_path / "extracted"
                extract_dir.mkdir()

                with zipfile.ZipFile(wheel_path, 'r') as zf:
                    zf.extractall(extract_dir)

                # Copy package to site-packages
                site_packages = self._paths.site_packages
                nvdiffrast_src = extract_dir / "nvdiffrast"

                if nvdiffrast_src.exists():
                    nvdiffrast_dest = site_packages / "nvdiffrast"
                    if nvdiffrast_dest.exists():
                        self.platform.rmtree_robust(nvdiffrast_dest)
                    shutil.copytree(nvdiffrast_src, nvdiffrast_dest)
                    self.logger.info(f"Installed nvdiffrast package to {nvdiffrast_dest}")

                # Copy compiled extensions
                lib_ext = self.platform.shared_lib_extension
                plugin_count = 0

                for lib_file in extract_dir.glob(f"*{lib_ext}"):
                    shutil.copy2(lib_file, site_packages / lib_file.name)
                    self.logger.info(f"Installed plugin: {lib_file.name}")
                    plugin_count += 1

                # Windows: also check for .pyd files
                if self.platform.name == 'windows':
                    for pyd_file in extract_dir.glob("*.pyd"):
                        shutil.copy2(pyd_file, site_packages / pyd_file.name)
                        self.logger.info(f"Installed plugin: {pyd_file.name}")
                        plugin_count += 1

            if self.verify_import("nvdiffrast"):
                self.logger.success(f"nvdiffrast installed ({plugin_count} plugin(s))")
                return True
            else:
                self.logger.error("nvdiffrast import failed after installation")
                return False

        except Exception as e:
            self.logger.error(f"Wheel installation failed: {e}")
            self.logger.info("Falling back to source installation...")
            return self._install_from_source()

    def _install_from_source(self) -> bool:
        """Install nvdiffrast from source (requires compiler)."""
        self.logger.warning(f"No prebuilt wheel for {self.platform.name}, installing from source...")
        self.logger.warning("This requires a working C++ compiler and may take several minutes")

        try:
            self.run_pip(
                ["install", "nvdiffrast"],
                step_name="Install nvdiffrast from source",
                check=True
            )

            if self.verify_import("nvdiffrast"):
                self.logger.success("nvdiffrast installed from source")
                return True
            else:
                self.logger.error("nvdiffrast import failed after source installation")
                return False

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Source installation failed: {e}")
            return False
