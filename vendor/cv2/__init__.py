# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
cv2 shim module with graceful degradation.

This module provides a drop-in replacement for OpenCV (cv2) that:
1. Tries to use native cv2 first (best performance)
2. Falls back to scipy/PIL/skimage implementations if cv2 fails

This ensures cross-platform compatibility, especially on Windows where
pip opencv-python wheels often have DLL load failures.

Usage:
    # This module intercepts 'import cv2' when vendor/ is in PYTHONPATH
    import cv2
    img = cv2.imread("image.jpg")
"""

import sys
import os

# Detect if we're being imported recursively
_LOADING_MARKER = '_cv2_shim_is_loading'
if _LOADING_MARKER in os.environ:
    raise ImportError("Recursive cv2 shim import detected")
os.environ[_LOADING_MARKER] = '1'

try:
    # Configuration
    _USE_NATIVE_CV2 = False
    _NATIVE_CV2 = None

    # Try to import the real cv2 from site-packages
    # We need to temporarily remove our vendor path to avoid recursion
    def _try_native_cv2():
        """Attempt to import native cv2 from site-packages."""
        global _USE_NATIVE_CV2, _NATIVE_CV2

        import importlib.util

        # Find the real cv2 in site-packages (not our shim)
        vendor_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Save original sys.path
        original_path = sys.path.copy()

        try:
            # Remove vendor paths from sys.path temporarily
            filtered_path = [p for p in sys.path if vendor_path not in p]
            sys.path = filtered_path

            # Try to find cv2 in the filtered path
            spec = importlib.util.find_spec("cv2")

            if spec is not None and spec.origin is not None:
                # Check it's not our shim
                if 'vendor' not in spec.origin:
                    # Try to load the module
                    _NATIVE_CV2 = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(_NATIVE_CV2)
                    _USE_NATIVE_CV2 = True
                    return True

        except ImportError as e:
            # Native cv2 failed to import (e.g., DLL load error on Windows)
            pass
        except Exception as e:
            # Any other error
            pass
        finally:
            # Restore original sys.path
            sys.path = original_path

        return False

    # Try native cv2
    _try_native_cv2()

    if _USE_NATIVE_CV2 and _NATIVE_CV2 is not None:
        # Use native cv2 - import everything from it
        from loguru import logger
        logger.debug("cv2 shim: Using native OpenCV")

        # Copy all attributes from native cv2
        for attr in dir(_NATIVE_CV2):
            if not attr.startswith('_'):
                globals()[attr] = getattr(_NATIVE_CV2, attr)

        # Also copy special attributes
        __version__ = getattr(_NATIVE_CV2, '__version__', 'native')
        __file__ = getattr(_NATIVE_CV2, '__file__', __file__)

    else:
        # Use our shim implementations
        from loguru import logger
        logger.debug("cv2 shim: Using scipy/PIL/skimage fallback implementations")

        # Import all from shim
        from .shim import *
        from .constants import *

        # Set version to indicate shim
        __version__ = 'shim-1.0.0'

finally:
    # Clean up marker
    if _LOADING_MARKER in os.environ:
        del os.environ[_LOADING_MARKER]
