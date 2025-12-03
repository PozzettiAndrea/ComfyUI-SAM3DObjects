"""
Package installers for SAM3D dependencies.

Each installer handles a specific package or group of related packages.
"""

from .base import Installer
from .micromamba import MicromambaInstaller
from .pytorch import PyTorchInstaller
from .cuda import CudaToolkitInstaller, CompilerInstaller
from .specialized import GsplatInstaller, NvdiffrastInstaller

__all__ = [
    'Installer',
    'MicromambaInstaller',
    'PyTorchInstaller',
    'CudaToolkitInstaller',
    'CompilerInstaller',
    'GsplatInstaller',
    'NvdiffrastInstaller',
]
