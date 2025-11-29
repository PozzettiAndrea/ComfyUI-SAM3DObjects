"""
Package installers for SAM3D dependencies.

Each installer handles a specific package or group of related packages.
"""

from .base import Installer
from .venv import VenvInstaller
from .pytorch import PipDependenciesInstaller
from .pytorch_pip import PyTorchPipInstaller
from .specialized import GsplatInstaller, NvdiffrastInstaller

__all__ = [
    'Installer',
    'VenvInstaller',
    'PipDependenciesInstaller',
    'PyTorchPipInstaller',
    'GsplatInstaller',
    'NvdiffrastInstaller',
]
