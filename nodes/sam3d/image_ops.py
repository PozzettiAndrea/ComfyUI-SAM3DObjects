# Copyright (c) Meta Platforms, Inc. and affiliates.
# Consolidated from sam3d_objects/pipeline/image_operations.py
"""
Image operations with optional cv2 support and scipy/PIL/skimage fallbacks.
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
# Morphological operations
# =============================================================================

def erode(mask, kernel, iterations=1, **kwargs):
    if HAS_CV2:
        return cv2.erode(mask, kernel, iterations=iterations, **kwargs)
    original_dtype = mask.dtype
    result = mask.copy()
    if mask.max() <= 1:
        for _ in range(iterations):
            result = binary_erosion(result, structure=kernel).astype(original_dtype)
    else:
        for _ in range(iterations):
            result = grey_erosion(result, footprint=kernel).astype(original_dtype)
    return result


def dilate(mask, kernel, iterations=1, **kwargs):
    if HAS_CV2:
        return cv2.dilate(mask, kernel, iterations=iterations, **kwargs)
    original_dtype = mask.dtype
    result = mask.copy()
    if mask.max() <= 1:
        for _ in range(iterations):
            result = binary_dilation(result, structure=kernel).astype(original_dtype)
    else:
        for _ in range(iterations):
            result = grey_dilation(result, footprint=kernel).astype(original_dtype)
    return result


# =============================================================================
# Text rendering
# =============================================================================

def get_text_size(text, font_scale=1, thickness=1):
    if HAS_CV2:
        return cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    font_size = int(font_scale * 20)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()
    dummy_img = Image.new('RGB', (1, 1))
    draw = ImageDraw.Draw(dummy_img)
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return (width, height)


def put_text(image, text, org, font_scale=1, color=(255, 255, 255), thickness=1, line_type=None):
    if HAS_CV2:
        line_type_arg = line_type if line_type is not None else cv2.LINE_AA
        return cv2.putText(image.copy(), text, org, cv2.FONT_HERSHEY_SIMPLEX,
                          font_scale, color, thickness, line_type_arg)
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    font_size = int(font_scale * 20)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()
    text_height = get_text_size(text, font_scale, thickness)[1]
    x, y = org
    y = y - text_height
    draw.text((x, y), text, fill=color, font=font)
    return np.array(img_pil)


# =============================================================================
# Image inpainting
# =============================================================================

def inpaint(image, mask, inpaint_radius=3, flags=None):
    if HAS_CV2:
        flags_arg = flags if flags is not None else cv2.INPAINT_TELEA
        return cv2.inpaint(image, mask, inpaint_radius, flags_arg)
    img_float = image.astype(np.float64) / 255.0
    mask_bool = mask.astype(bool)
    if image.ndim == 3:
        result = inpaint_biharmonic(img_float, mask_bool, channel_axis=-1)
    else:
        result = inpaint_biharmonic(img_float, mask_bool)
    return np.clip(result * 255, 0, 255).astype(np.uint8)
