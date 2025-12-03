"""
Package installers for SAM3D dependencies.

Each installer handles a specific package or group of related packages.
"""

from .base import Installer
from .micromamba import MicromambaInstaller
from .venv import VenvInstaller
from .pytorch import PyTorchInstaller
from .pytorch_pip import PyTorchPipInstaller
from .cuda import CudaToolkitInstaller, CompilerInstaller
from .specialized import GsplatInstaller, NvdiffrastInstaller

__all__ = [
    'Installer',
    'MicromambaInstaller',
    'VenvInstaller',
    'PyTorchInstaller',
    'PyTorchPipInstaller',
    'CudaToolkitInstaller',
    'CompilerInstaller',
    'GsplatInstaller',
    'NvdiffrastInstaller',
]
