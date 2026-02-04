# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Image operations with optional cv2 support and scipy/PIL/skimage fallbacks.

This module provides drop-in replacements for OpenCV functions.
If cv2 is available, it will be used. Otherwise, falls back to
scipy, PIL, and scikit-image backends. This avoids Windows DLL
issues with opencv-python while allowing A/B comparison for debugging.

Usage:
    from sam3d_objects.pipeline.image_operations import erode, dilate, HAS_CV2

    # Check if cv2 is being used
    if HAS_CV2:
        print("Using OpenCV")
    else:
        print("Using scipy/PIL/skimage fallbacks")
"""

import numpy as np
from loguru import logger

# Try to import cv2 (optional - may not be installed or may have DLL issues)
try:
    import cv2
    HAS_CV2 = True
    logger.debug("image_operations: cv2 available, using OpenCV backend")
except ImportError:
    HAS_CV2 = False
    logger.debug("image_operations: cv2 not available, using scipy/PIL/skimage fallbacks")

# Fallback imports (always available)
from scipy.ndimage import binary_erosion, binary_dilation, grey_erosion, grey_dilation
from PIL import Image, ImageDraw, ImageFont
from skimage.restoration import inpaint_biharmonic


# =============================================================================
# Morphological operations (replaces cv2.erode, cv2.dilate)
# =============================================================================

def erode(mask, kernel, iterations=1, **kwargs):
    """
    Erode a mask.

    Uses cv2.erode if available, otherwise scipy.ndimage.

    Args:
        mask: numpy array (uint8 or float)
        kernel: structuring element (numpy array)
        iterations: number of times to apply erosion
        **kwargs: additional arguments passed to cv2.erode

    Returns:
        Eroded mask with same dtype as input
    """
    if HAS_CV2:
        return cv2.erode(mask, kernel, iterations=iterations, **kwargs)

    # Fallback: scipy.ndimage
    original_dtype = mask.dtype
    result = mask.copy()

    # Use binary erosion for binary masks, grey erosion for others
    if mask.max() <= 1:
        # Binary mask
        for _ in range(iterations):
            result = binary_erosion(result, structure=kernel).astype(original_dtype)
    else:
        # Grayscale mask
        for _ in range(iterations):
            result = grey_erosion(result, footprint=kernel).astype(original_dtype)

    return result


def dilate(mask, kernel, iterations=1, **kwargs):
    """
    Dilate a mask.

    Uses cv2.dilate if available, otherwise scipy.ndimage.

    Args:
        mask: numpy array (uint8 or float)
        kernel: structuring element (numpy array)
        iterations: number of times to apply dilation
        **kwargs: additional arguments passed to cv2.dilate

    Returns:
        Dilated mask with same dtype as input
    """
    if HAS_CV2:
        return cv2.dilate(mask, kernel, iterations=iterations, **kwargs)

    # Fallback: scipy.ndimage
    original_dtype = mask.dtype
    result = mask.copy()

    # Use binary dilation for binary masks, grey dilation for others
    if mask.max() <= 1:
        # Binary mask
        for _ in range(iterations):
            result = binary_dilation(result, structure=kernel).astype(original_dtype)
    else:
        # Grayscale mask
        for _ in range(iterations):
            result = grey_dilation(result, footprint=kernel).astype(original_dtype)

    return result


# =============================================================================
# Text rendering (replaces cv2.putText, cv2.getTextSize)
# =============================================================================

def get_text_size(text, font_scale=1, thickness=1):
    """
    Get the size of text when rendered.

    Uses cv2.getTextSize if available, otherwise PIL.

    Args:
        text: string to measure
        font_scale: scale factor (cv2 style, ~1-2 for normal text)
        thickness: line thickness

    Returns:
        (width, height) tuple
    """
    if HAS_CV2:
        return cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]

    # Fallback: PIL
    # Approximate cv2 font sizes: font_scale=2 is roughly 40px height
    font_size = int(font_scale * 20)

    try:
        # Try to use a truetype font for better sizing
        font = ImageFont.truetype("arial.ttf", font_size)
    except (OSError, IOError):
        # Fall back to default font
        font = ImageFont.load_default()

    # Create dummy image to get text size
    dummy_img = Image.new('RGB', (1, 1))
    draw = ImageDraw.Draw(dummy_img)
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]

    return (width, height)


def put_text(image, text, org, font_scale=1, color=(255, 255, 255), thickness=1, line_type=None):
    """
    Draw text on an image.

    Uses cv2.putText if available, otherwise PIL.

    Args:
        image: numpy array (H, W, 3) uint8
        text: string to draw
        org: (x, y) position for text origin (bottom-left in cv2)
        font_scale: scale factor
        color: (R, G, B) tuple or (B, G, R) for cv2
        thickness: line thickness
        line_type: cv2 line type (ignored in PIL fallback)

    Returns:
        Image with text drawn on it
    """
    if HAS_CV2:
        line_type_arg = line_type if line_type is not None else cv2.LINE_AA
        return cv2.putText(image.copy(), text, org, cv2.FONT_HERSHEY_SIMPLEX,
                          font_scale, color, thickness, line_type_arg)

    # Fallback: PIL
    # Convert numpy to PIL
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)

    # Approximate cv2 font sizes
    font_size = int(font_scale * 20)

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()

    # cv2 uses bottom-left origin, PIL uses top-left
    # Adjust y coordinate
    text_height = get_text_size(text, font_scale, thickness)[1]
    x, y = org
    y = y - text_height  # Adjust for bottom-left origin

    # Draw text
    draw.text((x, y), text, fill=color, font=font)

    return np.array(img_pil)


# =============================================================================
# Image inpainting (replaces cv2.inpaint)
# =============================================================================

def inpaint(image, mask, inpaint_radius=3, flags=None):
    """
    Inpaint an image.

    Uses cv2.inpaint if available, otherwise scikit-image biharmonic.

    Args:
        image: numpy array (H, W, 3) or (H, W) uint8
        mask: numpy array (H, W) uint8 where non-zero indicates inpainting region
        inpaint_radius: radius of neighborhood
        flags: cv2.INPAINT_TELEA or cv2.INPAINT_NS (default: TELEA)

    Returns:
        Inpainted image as uint8
    """
    if HAS_CV2:
        flags_arg = flags if flags is not None else cv2.INPAINT_TELEA
        return cv2.inpaint(image, mask, inpaint_radius, flags_arg)

    # Fallback: scikit-image biharmonic
    # Normalize image to float [0, 1]
    img_float = image.astype(np.float64) / 255.0

    # Convert mask to boolean
    mask_bool = mask.astype(bool)

    # Apply biharmonic inpainting
    if image.ndim == 3:
        result = inpaint_biharmonic(img_float, mask_bool, channel_axis=-1)
    else:
        result = inpaint_biharmonic(img_float, mask_bool)

    # Convert back to uint8
    return np.clip(result * 255, 0, 255).astype(np.uint8)
