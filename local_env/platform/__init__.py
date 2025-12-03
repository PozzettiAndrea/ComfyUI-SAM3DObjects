"""
Platform detection and provider factory.

This module provides platform-agnostic abstractions for OS-specific operations.
"""

import platform as platform_module
from typing import Type, Optional

from .base import PlatformProvider

# Lazy imports to avoid loading all platform modules
_PROVIDER_MODULES = {
    'Linux': ('linux', 'LinuxPlatformProvider'),
    'Windows': ('windows', 'WindowsPlatformProvider'),
    'Darwin': ('darwin', 'DarwinPlatformProvider'),
}

_cached_provider: Optional[PlatformProvider] = None


def get_platform() -> PlatformProvider:
    """
    Get the platform provider for the current system.

    Returns:
        PlatformProvider instance for the current OS

    Raises:
        RuntimeError: If the platform is not supported
    """
    global _cached_provider

    if _cached_provider is None:
        system = platform_module.system()

        if system not in _PROVIDER_MODULES:
            raise RuntimeError(f"Unsupported platform: {system}")

        module_name, class_name = _PROVIDER_MODULES[system]

        # Dynamic import to avoid loading unused platform modules
        import importlib
        module = importlib.import_module(f'.{module_name}', package=__package__)
        provider_class = getattr(module, class_name)

        _cached_provider = provider_class()

    return _cached_provider


def reset_platform_cache():
    """Reset the cached platform provider (mainly for testing)."""
    global _cached_provider
    _cached_provider = None


__all__ = ['get_platform', 'PlatformProvider', 'reset_platform_cache']
