# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
cv2 shim implementations using scipy, PIL, and scikit-image.
Provides drop-in replacements for OpenCV functions used by MoGe.
"""

import io
import numpy as np
from PIL import Image
from scipy.ndimage import (
    uniform_filter,
    grey_dilation,
    grey_erosion,
    map_coordinates,
)
from skimage.transform import resize as skimage_resize
from skimage.restoration import inpaint_biharmonic

from .constants import *


def resize(src, dsize, dst=None, fx=0, fy=0, interpolation=INTER_LINEAR):
    """
    Resize an image.

    Args:
        src: Input image (numpy array)
        dsize: (width, height) tuple. If (0, 0), uses fx and fy.
        dst: Not used (cv2 compatibility)
        fx: Scale factor along horizontal axis
        fy: Scale factor along vertical axis
        interpolation: Interpolation method (INTER_*)

    Returns:
        Resized image as numpy array
    """
    if dsize is not None and dsize != (0, 0):
        new_width, new_height = dsize
        output_shape = (new_height, new_width)
    else:
        h, w = src.shape[:2]
        new_height = int(h * fy) if fy > 0 else h
        new_width = int(w * fx) if fx > 0 else w
        output_shape = (new_height, new_width)

    if len(src.shape) == 3:
        output_shape = output_shape + (src.shape[2],)

    # Map cv2 interpolation to skimage order
    order_map = {
        INTER_NEAREST: 0,
        INTER_LINEAR: 1,
        INTER_CUBIC: 3,
        INTER_LANCZOS4: 3,  # skimage doesn't have lanczos, use cubic
        INTER_AREA: 1,  # area interpolation approximated by linear
    }
    order = order_map.get(interpolation, 1)

    # Use skimage resize
    # preserve_range=True keeps the original dtype range
    result = skimage_resize(
        src,
        output_shape,
        order=order,
        preserve_range=True,
        anti_aliasing=(order > 0),
    )

    return result.astype(src.dtype)


def remap(src, map1, map2, interpolation, borderMode=BORDER_CONSTANT, borderValue=0):
    """
    Apply a generic geometric transformation using a mapping.

    Args:
        src: Input image
        map1: X coordinates (or combined XY if map2 is None)
        map2: Y coordinates (or None)
        interpolation: Interpolation method
        borderMode: Border handling mode
        borderValue: Value for border pixels

    Returns:
        Remapped image
    """
    if map2 is None:
        # map1 is (H, W, 2) combined XY map
        map_x = map1[..., 0]
        map_y = map1[..., 1]
    else:
        map_x = map1
        map_y = map2

    # Convert to coordinates format for scipy
    # scipy expects (row, col) = (y, x)
    coords = np.array([map_y.ravel(), map_x.ravel()])

    # Handle multi-channel images
    if len(src.shape) == 3:
        result = np.zeros(map_x.shape + (src.shape[2],), dtype=src.dtype)
        for c in range(src.shape[2]):
            order = 1 if interpolation == INTER_LINEAR else 0
            result[..., c] = map_coordinates(
                src[..., c],
                coords,
                order=order,
                mode='constant',
                cval=borderValue if np.isscalar(borderValue) else borderValue[c] if c < len(borderValue) else 0,
            ).reshape(map_x.shape)
    else:
        order = 1 if interpolation == INTER_LINEAR else 0
        result = map_coordinates(
            src,
            coords,
            order=order,
            mode='constant',
            cval=borderValue,
        ).reshape(map_x.shape)

    return result.astype(src.dtype)


def dilate(src, kernel, dst=None, anchor=None, iterations=1, borderType=BORDER_CONSTANT, borderValue=None):
    """
    Dilate an image using a structuring element.

    Args:
        src: Input image
        kernel: Structuring element
        iterations: Number of times dilation is applied

    Returns:
        Dilated image
    """
    result = src.copy()
    for _ in range(iterations):
        if src.max() <= 1 and src.dtype in (bool, np.bool_):
            # Binary dilation
            from scipy.ndimage import binary_dilation
            result = binary_dilation(result, structure=kernel).astype(src.dtype)
        else:
            result = grey_dilation(result, footprint=kernel)
    return result.astype(src.dtype)


def erode(src, kernel, dst=None, anchor=None, iterations=1, borderType=BORDER_CONSTANT, borderValue=None):
    """
    Erode an image using a structuring element.

    Args:
        src: Input image
        kernel: Structuring element
        iterations: Number of times erosion is applied

    Returns:
        Eroded image
    """
    result = src.copy()
    for _ in range(iterations):
        if src.max() <= 1 and src.dtype in (bool, np.bool_):
            # Binary erosion
            from scipy.ndimage import binary_erosion
            result = binary_erosion(result, structure=kernel).astype(src.dtype)
        else:
            result = grey_erosion(result, footprint=kernel)
    return result.astype(src.dtype)


def blur(src, ksize, dst=None, anchor=None, borderType=BORDER_DEFAULT):
    """
    Apply a normalized box filter (blur).

    Args:
        src: Input image
        ksize: (width, height) kernel size tuple

    Returns:
        Blurred image
    """
    if isinstance(ksize, (list, tuple)):
        size = (ksize[1], ksize[0])  # (h, w) for scipy
    else:
        size = (ksize, ksize)

    if len(src.shape) == 3:
        result = np.zeros_like(src)
        for c in range(src.shape[2]):
            result[..., c] = uniform_filter(src[..., c].astype(float), size=size)
        return result.astype(src.dtype)
    else:
        return uniform_filter(src.astype(float), size=size).astype(src.dtype)


def getStructuringElement(shape, ksize, anchor=None):
    """
    Create a structuring element of the specified shape and size.

    Args:
        shape: Element shape (MORPH_RECT, MORPH_CROSS, MORPH_ELLIPSE)
        ksize: (width, height) size tuple

    Returns:
        Structuring element as numpy array
    """
    if isinstance(ksize, (list, tuple)):
        width, height = ksize
    else:
        width = height = ksize

    if shape == MORPH_RECT:
        return np.ones((height, width), dtype=np.uint8)

    elif shape == MORPH_CROSS:
        element = np.zeros((height, width), dtype=np.uint8)
        cx, cy = width // 2, height // 2
        element[cy, :] = 1
        element[:, cx] = 1
        return element

    elif shape == MORPH_ELLIPSE:
        from skimage.morphology import disk
        # disk creates a circular element, approximate ellipse
        radius = min(width, height) // 2
        elem = disk(radius)
        # Resize to exact dimensions if needed
        if elem.shape != (height, width):
            # Pad or crop to exact size
            result = np.zeros((height, width), dtype=np.uint8)
            eh, ew = elem.shape
            start_y = max(0, (height - eh) // 2)
            start_x = max(0, (width - ew) // 2)
            end_y = min(height, start_y + eh)
            end_x = min(width, start_x + ew)
            src_start_y = max(0, (eh - height) // 2)
            src_start_x = max(0, (ew - width) // 2)
            result[start_y:end_y, start_x:end_x] = elem[
                src_start_y:src_start_y + (end_y - start_y),
                src_start_x:src_start_x + (end_x - start_x)
            ]
            return result
        return elem.astype(np.uint8)

    else:
        # Default to rectangle
        return np.ones((height, width), dtype=np.uint8)


def imread(filename, flags=IMREAD_COLOR):
    """
    Load an image from a file.

    Args:
        filename: Path to the image file
        flags: Read mode (IMREAD_COLOR, IMREAD_GRAYSCALE, IMREAD_UNCHANGED)

    Returns:
        Image as numpy array in BGR format (cv2 convention)
    """
    try:
        img = Image.open(filename)

        if flags == IMREAD_GRAYSCALE:
            img = img.convert('L')
            return np.array(img)

        elif flags == IMREAD_UNCHANGED:
            # Keep alpha if present
            arr = np.array(img)
            if len(arr.shape) == 3 and arr.shape[2] >= 3:
                # Convert RGB(A) to BGR(A)
                if arr.shape[2] == 3:
                    arr = arr[:, :, ::-1]
                elif arr.shape[2] == 4:
                    arr = np.concatenate([arr[:, :, 2::-1], arr[:, :, 3:4]], axis=2)
            return arr

        else:  # IMREAD_COLOR
            img = img.convert('RGB')
            arr = np.array(img)
            # Convert RGB to BGR (cv2 convention)
            return arr[:, :, ::-1]

    except Exception:
        return None


def imwrite(filename, img, params=None):
    """
    Save an image to a file.

    Args:
        filename: Path to save the image
        img: Image array (BGR format for color)
        params: Optional encoding parameters

    Returns:
        True on success, False on failure
    """
    try:
        if len(img.shape) == 3:
            if img.shape[2] == 3:
                # Convert BGR to RGB
                img_rgb = img[:, :, ::-1]
            elif img.shape[2] == 4:
                # Convert BGRA to RGBA
                img_rgb = np.concatenate([img[:, :, 2::-1], img[:, :, 3:4]], axis=2)
            else:
                img_rgb = img
            pil_img = Image.fromarray(img_rgb.astype(np.uint8))
        else:
            pil_img = Image.fromarray(img.astype(np.uint8))

        # Handle quality parameters
        save_kwargs = {}
        if params is not None:
            for i in range(0, len(params), 2):
                if params[i] == IMWRITE_JPEG_QUALITY:
                    save_kwargs['quality'] = params[i + 1]
                elif params[i] == IMWRITE_PNG_COMPRESSION:
                    save_kwargs['compress_level'] = params[i + 1]

        pil_img.save(filename, **save_kwargs)
        return True

    except Exception:
        return False


def cvtColor(src, code):
    """
    Convert image from one color space to another.

    Args:
        src: Input image
        code: Color conversion code (COLOR_*)

    Returns:
        Converted image
    """
    if code == COLOR_BGR2RGB or code == COLOR_RGB2BGR:
        if len(src.shape) == 3 and src.shape[2] >= 3:
            return src[:, :, ::-1].copy()
        return src.copy()

    elif code == COLOR_BGR2GRAY:
        if len(src.shape) == 3:
            # BGR to Gray: 0.114*B + 0.587*G + 0.299*R
            return (0.114 * src[:, :, 0] + 0.587 * src[:, :, 1] + 0.299 * src[:, :, 2]).astype(src.dtype)
        return src.copy()

    elif code == COLOR_RGB2GRAY:
        if len(src.shape) == 3:
            # RGB to Gray: 0.299*R + 0.587*G + 0.114*B
            return (0.299 * src[:, :, 0] + 0.587 * src[:, :, 1] + 0.114 * src[:, :, 2]).astype(src.dtype)
        return src.copy()

    elif code == COLOR_GRAY2BGR or code == COLOR_GRAY2RGB:
        if len(src.shape) == 2:
            return np.stack([src, src, src], axis=2)
        return src.copy()

    elif code == COLOR_BGRA2BGR or code == COLOR_RGBA2RGB:
        if len(src.shape) == 3 and src.shape[2] == 4:
            return src[:, :, :3].copy()
        return src.copy()

    elif code == COLOR_BGR2BGRA or code == COLOR_RGB2RGBA:
        if len(src.shape) == 3 and src.shape[2] == 3:
            alpha = np.full(src.shape[:2] + (1,), 255, dtype=src.dtype)
            return np.concatenate([src, alpha], axis=2)
        return src.copy()

    else:
        # Unknown conversion, return copy
        return src.copy()


def imencode(ext, img, params=None):
    """
    Encode an image into a memory buffer.

    Args:
        ext: File extension (e.g., '.jpg', '.png')
        img: Image array (BGR format)
        params: Optional encoding parameters

    Returns:
        (success, buffer) tuple
    """
    try:
        ext = ext.lower().lstrip('.')

        if len(img.shape) == 3:
            if img.shape[2] == 3:
                img_rgb = img[:, :, ::-1]
            elif img.shape[2] == 4:
                img_rgb = np.concatenate([img[:, :, 2::-1], img[:, :, 3:4]], axis=2)
            else:
                img_rgb = img
            pil_img = Image.fromarray(img_rgb.astype(np.uint8))
        else:
            pil_img = Image.fromarray(img.astype(np.uint8))

        buffer = io.BytesIO()

        # Map extension to format
        format_map = {
            'jpg': 'JPEG',
            'jpeg': 'JPEG',
            'png': 'PNG',
            'bmp': 'BMP',
            'tif': 'TIFF',
            'tiff': 'TIFF',
            'webp': 'WEBP',
        }
        fmt = format_map.get(ext, 'PNG')

        save_kwargs = {}
        if params is not None:
            for i in range(0, len(params), 2):
                if params[i] == IMWRITE_JPEG_QUALITY:
                    save_kwargs['quality'] = params[i + 1]
                elif params[i] == IMWRITE_PNG_COMPRESSION:
                    save_kwargs['compress_level'] = params[i + 1]

        pil_img.save(buffer, format=fmt, **save_kwargs)
        return True, np.frombuffer(buffer.getvalue(), dtype=np.uint8)

    except Exception:
        return False, None


def imdecode(buf, flags=IMREAD_COLOR):
    """
    Decode an image from a memory buffer.

    Args:
        buf: Buffer containing encoded image data
        flags: Read mode

    Returns:
        Decoded image as numpy array
    """
    try:
        if isinstance(buf, np.ndarray):
            buf = buf.tobytes()
        elif hasattr(buf, 'read'):
            buf = buf.read()

        img = Image.open(io.BytesIO(buf))

        if flags == IMREAD_GRAYSCALE:
            img = img.convert('L')
            return np.array(img)

        elif flags == IMREAD_UNCHANGED:
            arr = np.array(img)
            if len(arr.shape) == 3 and arr.shape[2] >= 3:
                if arr.shape[2] == 3:
                    arr = arr[:, :, ::-1]
                elif arr.shape[2] == 4:
                    arr = np.concatenate([arr[:, :, 2::-1], arr[:, :, 3:4]], axis=2)
            return arr

        else:  # IMREAD_COLOR
            img = img.convert('RGB')
            arr = np.array(img)
            return arr[:, :, ::-1]

    except Exception:
        return None


def inpaint(src, inpaintMask, inpaintRadius, flags=INPAINT_TELEA):
    """
    Restore the selected region in an image using inpainting.

    Args:
        src: Input image (8-bit 1-channel or 3-channel)
        inpaintMask: Inpainting mask (8-bit 1-channel, non-zero pixels indicate area to inpaint)
        inpaintRadius: Radius of neighborhood (not used in biharmonic method)
        flags: Inpainting method (INPAINT_NS or INPAINT_TELEA)

    Returns:
        Inpainted image
    """
    # Convert to float for skimage
    img_float = src.astype(np.float64) / 255.0

    # Convert mask to boolean
    mask_bool = inpaintMask.astype(bool)

    # Apply biharmonic inpainting
    if len(src.shape) == 3:
        result = inpaint_biharmonic(img_float, mask_bool, channel_axis=-1)
    else:
        result = inpaint_biharmonic(img_float, mask_bool)

    # Convert back to uint8
    return np.clip(result * 255, 0, 255).astype(np.uint8)


def getTextSize(text, fontFace, fontScale, thickness):
    """
    Calculate the size of a text string.

    Args:
        text: Input text
        fontFace: Font type
        fontScale: Font scale factor
        thickness: Line thickness

    Returns:
        ((width, height), baseline) tuple
    """
    # Approximate text size based on font scale
    # cv2 fontScale=1 is roughly 20px height
    char_width = int(fontScale * 12)
    char_height = int(fontScale * 20)

    width = len(text) * char_width
    height = char_height
    baseline = int(fontScale * 5)

    return (width, height), baseline


def putText(img, text, org, fontFace, fontScale, color, thickness=1, lineType=LINE_8, bottomLeftOrigin=False):
    """
    Draw a text string on an image.

    Args:
        img: Input image
        text: Text string
        org: (x, y) position of text origin
        fontFace: Font type
        fontScale: Font scale factor
        color: Text color (BGR)
        thickness: Line thickness
        lineType: Line type
        bottomLeftOrigin: If True, origin is bottom-left

    Returns:
        Image with text (modified in place)
    """
    from PIL import ImageDraw, ImageFont

    # Convert to PIL
    if len(img.shape) == 3:
        if img.shape[2] == 3:
            pil_img = Image.fromarray(img[:, :, ::-1])  # BGR to RGB
        else:
            pil_img = Image.fromarray(img)
    else:
        pil_img = Image.fromarray(img)

    draw = ImageDraw.Draw(pil_img)

    # Approximate font size
    font_size = int(fontScale * 20)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()

    # Adjust position for bottom-left origin
    x, y = org
    if not bottomLeftOrigin:
        # cv2 uses bottom-left by default for text
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_height = text_bbox[3] - text_bbox[1]
        y = y - text_height

    # Convert BGR color to RGB
    if isinstance(color, (list, tuple)) and len(color) >= 3:
        color_rgb = (color[2], color[1], color[0])
    else:
        color_rgb = color

    draw.text((x, y), text, fill=color_rgb, font=font)

    # Convert back
    result = np.array(pil_img)
    if len(img.shape) == 3 and img.shape[2] == 3:
        result = result[:, :, ::-1]  # RGB to BGR

    # Modify in place
    img[:] = result
    return img
