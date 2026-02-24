"""
Consolidated transforms module for SAM3D.

Contains image/mask preprocessing, 3D transforms, pose target converters,
and point remapping utilities.

Consolidated from:
- sam3d_objects/model/backbone/dit/embedder/point_remapper.py
- sam3d_objects/data/dataset/tdfy/img_processing.py
- sam3d_objects/data/dataset/tdfy/transforms_3d.py
- sam3d_objects/data/dataset/tdfy/img_and_mask_transforms.py
- sam3d_objects/data/dataset/tdfy/preprocessor.py
- sam3d_objects/data/dataset/tdfy/pose_target.py
"""

import math
import random
import warnings
from collections import namedtuple
from dataclasses import dataclass, asdict, field
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as tv_transforms
import torchvision.transforms.functional as TF
from torchvision import transforms
from torchvision.transforms import functional as tv_F
from loguru import logger
from pytorch3d.transforms import (
    Rotate,
    Translate,
    Scale,
    Transform3d,
    quaternion_to_matrix,
    axis_angle_to_quaternion,
    matrix_to_quaternion,
)

from .data_utils import expand_as_right, tree_tensor_map


# =============================================================================
# Point Remapper (from point_remapper.py)
# =============================================================================

class PointRemapper(nn.Module):
    """Handles remapping of 3D point coordinates and their inverse transformations."""

    VALID_TYPES = ["linear", "sinh", "exp", "sinh_exp", "exp_disparity"]

    def __init__(self, remap_type: str = "exp"):
        super().__init__()
        self.remap_type = remap_type

        if remap_type not in self.VALID_TYPES:
            raise ValueError(
                f"Invalid remap type: {remap_type}. Must be one of {self.VALID_TYPES}"
            )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Apply remapping to point coordinates."""
        if self.remap_type == "linear":
            return points

        elif self.remap_type == "sinh":
            return torch.asinh(points)

        elif self.remap_type == "exp":
            xy_scaled, z_exp = points.split([2, 1], dim=-1)
            # Use log1p for better numerical stability near zero
            z = torch.log1p(z_exp)
            xy = xy_scaled / (1 + z_exp)
            return torch.cat([xy, z], dim=-1)
        
        elif self.remap_type == "exp_disparity":
            xy_scaled, z_exp = points.split([2, 1], dim=-1)
            xy = xy_scaled / z_exp
            z = torch.log(z_exp)
            return torch.cat([xy, z], dim=-1)

        elif self.remap_type == "sinh_exp":
            xy_sinh, z_exp = points.split([2, 1], dim=-1)
            xy = torch.asinh(xy_sinh)
            z = torch.log(z_exp.clamp(min=1e-8))
            return torch.cat([xy, z], dim=-1)

        else:
            raise ValueError(f"Unknown remap type: {self.remap_type}")

    def inverse(self, points: torch.Tensor) -> torch.Tensor:
        """Apply inverse remapping to recover original point coordinates."""
        if self.remap_type == "linear":
            return points

        elif self.remap_type == "sinh":
            return torch.sinh(points)

        elif self.remap_type == "exp":
            xy, z = points.split([2, 1], dim=-1)
            # Inverse of log1p is expm1(z) = exp(z) - 1
            z_exp = torch.expm1(z)
            # Inverse of xy/(1+z_exp) is xy*(1+z_exp)
            return torch.cat([xy * (1 + z_exp), z_exp], dim=-1)

        elif self.remap_type == "exp_disparity":
            xy, z = points.split([2, 1], dim=-1)
            z_exp = torch.exp(z)
            return torch.cat([xy * z_exp, z_exp], dim=-1)

        elif self.remap_type == "sinh_exp":
            xy, z = points.split([2, 1], dim=-1)
            return torch.cat([torch.sinh(xy), torch.exp(z)], dim=-1)

        else:
            raise ValueError(f"Unknown remap type: {self.remap_type}")

    def extra_repr(self) -> str:
        return f"remap_type='{self.remap_type}'"


# =============================================================================
# Image Processing (from img_processing.py)
# =============================================================================

class RandomResizedCrop(transforms.RandomResizedCrop):
    """
    RandomResizedCrop for matching TF/TPU implementation: no for-loop is used.
    This may lead to results different with torchvision's version.
    Following BYOL's TF code:
    https://github.com/deepmind/deepmind-research/blob/master/byol/utils/dataset.py#L206
    """

    @staticmethod
    def get_params(img, scale, ratio):
        width, height = tv_F._get_image_size(img)
        area = height * width

        target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
        log_ratio = torch.log(torch.tensor(ratio))
        aspect_ratio = torch.exp(
            torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
        ).item()

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        w = min(w, width)
        h = min(h, height)

        i = torch.randint(0, height - h + 1, size=(1,)).item()
        j = torch.randint(0, width - w + 1, size=(1,)).item()

        return i, j, h, w


# following PT3D CO3D data to pad image
def pad_to_square(image, value=0):
    _, _, h, w = image.shape  # Assuming image is in (B, C, H, W) format
    if h == w:
        return image  # The image is already square

    # Calculate the padding
    diff = abs(h - w)
    pad2 = diff

    # Pad the image to make it square
    if h > w:
        padding = (0, pad2, 0, 0)  # Pad width (left, right, top, bottom)
    else:
        padding = (0, 0, 0, pad2)  # Pad height
    # Apply padding
    padded_image = torch.nn.functional.pad(image, padding, mode="constant", value=value)
    return padded_image


def preprocess_img(
    x,
    mask=None,
    img_target_shape=224,
    mask_target_shape=256,
    normalize=False,
):
    if x.shape[1] != x.shape[2]:
        x = pad_to_square(x)
    if mask is not None and mask.shape[1] != mask.shape[2]:
        mask = pad_to_square(mask)
    if x.shape[2] != img_target_shape:
        x = F.interpolate(
            x,
            size=(img_target_shape, img_target_shape),
            # scale_factor=float(img_target_shape)/x.shape[2],
            mode="bilinear",
        )
    if mask is not None and mask.shape[2] != mask_target_shape:
        if mask is not None:
            mask = F.interpolate(
                mask,
                size=(mask_target_shape, mask_target_shape),
                # scale_factor=float(mask_target_shape)/mask.shape[2],
                mode="nearest",
            )
    if normalize:
        imgs_normed = resnet_img_normalization(x)
    else:
        imgs_normed = x
    return imgs_normed, mask


def resnet_img_normalization(x):
    resnet_mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).reshape(
        (3, 1, 1)
    )
    resnet_std = torch.tensor([0.229, 0.224, 0.225], device=x.device).reshape((3, 1, 1))
    if x.ndim == 4:
        resnet_mean = resnet_mean[None]
        resnet_std = resnet_std[None]
    x = (x - resnet_mean) / resnet_std
    return x


# pad image to be centered for unprojecting depth
def pad_to_square_centered(image, value=0, pointmap=None):
    h, w = image.shape[-2], image.shape[-1]  # Assuming image is in (B, C, H, W) format
    if h == w:
        if pointmap is not None:
            return image, pointmap
        return image  # The image is already square

    # Calculate the padding
    diff = abs(h - w)
    pad1 = diff // 2
    pad2 = diff - pad1

    # Pad the image to make it square
    if h > w:
        padding = (pad1, pad2, 0, 0)  # Pad width (left, right, top, bottom)
    else:
        padding = (0, 0, pad1, pad2)  # Pad height
    # Apply padding to image
    padded_image = F.pad(image, padding, mode="constant", value=value)

    # Apply padding to pointmap if provided
    if pointmap is not None:
        # Pad pointmap using torch functional with NaN fill value
        padded_pointmap = F.pad(pointmap, padding, mode="constant", value=float("nan"))

        return padded_image, padded_pointmap
    return padded_image


def crop_img_to_obj(mask, context_size):
    nonzeros = torch.nonzero(mask)
    if len(nonzeros) > 0:
        r_max, c_max = nonzeros.max(dim=0)[0]
        r_min, c_min = nonzeros.min(dim=0)[0]
        box_h = max(1, r_max - r_min)
        box_w = max(1, c_max - c_min)
        left = max(0, c_min - int(box_w * context_size))
        right = min(mask.shape[-1], c_max + int(box_w * context_size))
        top = max(0, r_min - int(box_h * context_size))
        bot = min(mask.shape[-2], r_max + int(box_h * context_size))
        return left, right, top, bot
    return None, None, None, None


def random_pad(img, mask=None, max_ratio=0.0, pointmap=None):
    max_size = int(max(img.shape) * max_ratio)
    padding = tuple([random.randint(0, max_size) for _ in range(4)])
    img = F.pad(img, padding)
    if mask is not None:
        mask = F.pad(mask, padding)

    if pointmap is not None:
        pointmap = F.pad(pointmap, padding, mode="constant", value=float("nan"))
        return img, mask, pointmap
    return img, mask


def get_img_color_augmentation(
    color_jit_prob=0.5,
    gaussian_blur_prob=0.1,
):
    transform = transforms.Compose(
        [
            # (a) Random Color Jitter
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                    )
                ],
                p=color_jit_prob,
            ),
            # (b) Randomly apply GaussianBlur
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))],
                p=gaussian_blur_prob,
            ),
        ]
    )
    return transform


# =============================================================================
# 3D Transforms (from transforms_3d.py)
# =============================================================================

DecomposedTransform = namedtuple(
    "DecomposedTransform", ["scale", "rotation", "translation"]
)


def compose_transform(
    scale: torch.Tensor, rotation: torch.Tensor, translation: torch.Tensor
) -> Transform3d:
    """
    Args:
        scale: (..., 3) tensor of scale factors
        rotation: (..., 3, 3) tensor of rotation matrices
        translation: (..., 3) tensor of translation vectors
    """
    tfm = Transform3d(dtype=scale.dtype, device=scale.device)
    return tfm.scale(scale).rotate(rotation).translate(translation)


def decompose_transform(transform: Transform3d) -> DecomposedTransform:
    """
    Returns:
        scale: (..., 3) tensor of scale factors
        rotation: (..., 3, 3) tensor of rotation matrices
        translation: (..., 3) tensor of translation vectors
    """
    matrices = transform.get_matrix()
    scale = torch.norm(matrices[:, :3, :3], dim=-1)
    rotation = matrices[:, :3, :3] / scale.unsqueeze(-1)  # Normalize rotation matrix
    translation = matrices[:, 3, :3]  # Extract translation vector
    return DecomposedTransform(scale, rotation, translation)


def get_rotation_about_x_axis(angle: float = math.pi / 2) -> torch.Tensor:
    axis = torch.tensor([1.0, 0.0, 0.0])
    axis_angle = axis * angle
    return axis_angle_to_quaternion(axis_angle)


# =============================================================================
# Image and Mask Transforms (from img_and_mask_transforms.py)
# =============================================================================

def UNNORMALIZE(mean, std):
    mean = torch.tensor(mean).reshape((3, 1, 1))
    std = torch.tensor(std).reshape((3, 1, 1))

    def unnormalize_img(img):
        assert img.ndim == 3 and img.shape[0] == 3

        return img * std.to(img.device) + mean.to(img.device)

    return unnormalize_img


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


IMAGENET_NORMALIZATION = tv_transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
IMAGENET_UNNORMALIZATION = UNNORMALIZE(IMAGENET_MEAN, IMAGENET_STD)


class BoundingBoxError(Exception):
    pass


def check_bounding_box(bbox_w, bbox_h):
    if bbox_w < 2 or bbox_h < 2:
        raise BoundingBoxError("Bounding box dimensions must be at least 2x2.")


class RGBAImageProcessor:
    def __init__(
        self,
        resize_and_make_square_kwargs: Optional[Dict] = None,
        object_crop_kwargs: Optional[Dict] = None,
        remove_background: bool = False,
        imagenet_normalization: bool = False,
    ):
        self.remove_background = remove_background
        self.resize_and_pad_kwargs = resize_and_make_square_kwargs
        self.object_crop_kwargs = object_crop_kwargs
        self.imagenet_normalization = imagenet_normalization
        if resize_and_make_square_kwargs is not None:
            self.transforms = resize_and_make_square(**resize_and_make_square_kwargs)

    def __call__(
        self, image: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mask is None:
            assert (
                image.shape[0] == 4
            ), f"Requires 4 channels (RGB + alpha), got {image.shape[0]=}"
            image, mask = split_rgba(image)
        else:
            assert (
                image.shape[0] == 3
            ), f"Requires 3 channels (RGB), got {image.shape[0]=}"
            assert mask.dim() == 2, f"Requires 2D mask, got {mask.dim()=}"

        if not self.object_crop_kwargs in [None, False]:
            image, mask = crop_around_mask_with_padding(
                image, mask, **self.object_crop_kwargs
            )

        if self.remove_background:
            image, mask = rembg(image, mask)

        image = self.transforms["img_transform"](image)
        mask = self.transforms["mask_transform"](mask.unsqueeze(0))

        if self.imagenet_normalization:
            image = IMAGENET_NORMALIZATION(image)
        return image, mask


def load_rgb(fpath: str) -> torch.Tensor:
    """
    Load a RGB(A) image from a file path.
    """
    image = plt.imread(fpath)  # Why use matplotlib?
    if image.dtype == "uint8":
        image = image / 255
        image = image.astype(np.float32)
    image = torch.from_numpy(image)
    image = image.permute(2, 0, 1).contiguous()
    return image


def concat_rgba(
    rgb_image: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Create a 4-channel RGBA image from a 3-channel RGB image and a mask.
    """
    assert rgb_image.dim() == 3, f"{rgb_image.shape=}"
    assert mask.dim() == 2, f"{mask.shape=}"
    assert rgb_image.shape[0] == 3, f"{rgb_image.shape[0]=}"
    assert rgb_image.shape[1:] == mask.shape, f"{rgb_image.shape[1:]=} != {mask.shape=}"
    return torch.cat((rgb_image, mask[None, ...]), dim=0)


def split_rgba(rgba_image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Split a 4-channel RGBA image into a 3-channel RGB image and a 1-channel mask.

    Args:
        rgba_image: A 4-channel RGBA image.

    Returns:
        A tuple of (rgb_image, mask).
    """
    assert rgba_image.dim() == 3, f"{rgba_image.shape=}"
    assert rgba_image.shape[0] == 4, f"{rgba_image.shape[0]=}"
    return rgba_image[:3], rgba_image[3]


def get_mask(
    rgb_image: torch.Tensor,
    depth_image: torch.Tensor,
    mask_source: str,
) -> torch.Tensor:
    """
    Extract a mask from either the alpha channel of an RGB image or a depth image.

    Args:
        rgb_image: Tensor of shape (B, C, H, W) or (C, H, W) where C >= 4 if using alpha channel
        depth_image: Tensor of shape (B, 1, H, W) or (1, H, W) containing depth information
        mask_source: Source of the mask, either "ALPHA_CHANNEL" or "DEPTH"

    Returns:
        mask: Tensor of shape (B, 1, H, W) or (1, H, W) containing the extracted mask
    """
    # Handle unbatched inputs (add batch dimension if needed)
    is_batched = len(rgb_image.shape) == 4

    if not is_batched:
        rgb_image = rgb_image.unsqueeze(0)
        if depth_image is not None:
            depth_image = depth_image.unsqueeze(0)

    if mask_source == "ALPHA_CHANNEL":
        if rgb_image.shape[1] != 4:
            logger.warning(f"No ALPHA CHANNEL for the image, cannot read mask.")
            mask = None
        else:
            mask = rgb_image[:, 3:4, :, :]
    elif mask_source == "DEPTH":
        mask = depth_image
    else:
        raise ValueError(f"Invalid mask source: {mask_source}")

    # Remove batch dimension if input was unbatched
    if not is_batched:
        mask = mask.squeeze(0)

    return mask


def rembg(image, mask, pointmap=None):
    """
    Remove the background from an image using a mask.
    For pointmaps, sets background regions to NaN.

    This function follows the standard transform pattern:
    - If called with (image, mask), returns (image, mask)
    - If called with (image, mask, pointmap), returns (image, mask, pointmap)
    """
    masked_image = image * mask

    if pointmap is not None:
        masked_pointmap = torch.where(mask > 0, pointmap, torch.nan)
        return masked_image, mask, masked_pointmap

    return masked_image, mask


def resize_and_make_square(
    img_size: int,
    make_square: bool | str = False,
):
    """
    Create image and mask transforms based on configuration.

    Returns:
        dict: {"img_transform": img_transform, "mask_transform": mask_transform}
    """
    if isinstance(make_square, str):
        make_square = make_square.lower()
    assert make_square in ["pad", "crop", False]
    pre_resize_transform = tv_transforms.Lambda(lambda x: x)
    post_resize_transform = tv_transforms.Lambda(lambda x: x)
    if make_square == "pad":
        pre_resize_transform = pad_to_square_centered
    elif make_square == "crop":
        post_resize_transform = tv_transforms.CenterCrop(img_size)

    img_resize = tv_transforms.Resize(img_size)
    mask_resize = tv_transforms.Resize(
        img_size,
        interpolation=tv_transforms.InterpolationMode.BILINEAR,
    )

    img_transform = tv_transforms.Compose(
        [
            pre_resize_transform,
            img_resize,
            post_resize_transform,
        ]
    )

    mask_transform = tv_transforms.Compose(
        [
            pre_resize_transform,
            mask_resize,
            post_resize_transform,
        ]
    )

    return {
        "img_transform": img_transform,
        "mask_transform": mask_transform,
    }


def crop_around_mask_with_random_box_size_factor(
    loaded_image: torch.Tensor,
    mask: torch.Tensor,
    random_box_size_factor: float = 1.0,
    pointmap: Optional[torch.Tensor] = None,
) -> np.ndarray:
    return crop_around_mask_with_padding(
        loaded_image,
        mask,
        box_size_factor=1.0 + random.uniform(0, 1) * random_box_size_factor,
        padding_factor=0.0,
        pointmap=pointmap,
    )


def crop_around_mask_with_padding(
    loaded_image: torch.Tensor,
    mask: torch.Tensor,
    box_size_factor: float = 1.6,
    padding_factor: float = 0.1,
    pointmap: Optional[torch.Tensor] = None,
) -> np.ndarray:
    # cast to ensure the function can be called normally
    cast_mask = False
    if mask.dim() == 3:
        assert mask.shape[0] == 1, "cannot take mask with channel dimension not 1"
        mask = mask[0]
        cast_mask = True
    loaded_image = concat_rgba(loaded_image, mask)

    bbox = compute_mask_bbox(mask, box_size_factor)
    loaded_image = torchvision.transforms.functional.crop(
        loaded_image, bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]
    )

    # Crop pointmap if provided
    if pointmap is not None:
        pointmap = torchvision.transforms.functional.crop(
            pointmap, bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]
        )

    C, H, W = loaded_image.shape
    max_dim = max(H, W)  # Get the larger dimension

    # Step 1: Pad to square shape
    pad_h = (max_dim - H) // 2
    pad_w = (max_dim - W) // 2
    pad_h_extra = (max_dim - H) - pad_h  # To ensure even padding
    pad_w_extra = (max_dim - W) - pad_w

    loaded_image = torch.nn.functional.pad(
        loaded_image, (pad_w, pad_w_extra, pad_h, pad_h_extra), mode="constant", value=0
    )
    if pointmap is not None:
        pointmap = torch.nn.functional.pad(
            pointmap,
            (pad_w, pad_w_extra, pad_h, pad_h_extra),
            mode="constant",
            value=float("nan"),
        )

    # Step 2: Extend by 10% on each side; idk but this seems to have better results overall
    if padding_factor > 0:
        extend_size = int(max_dim * padding_factor)  # 10% extension on each side
        loaded_image = torch.nn.functional.pad(
            loaded_image,
            (extend_size, extend_size, extend_size, extend_size),
            mode="constant",
            value=0,
        )

        if pointmap is not None:
            pointmap = torch.nn.functional.pad(
                pointmap,
                (extend_size, extend_size, extend_size, extend_size),
                mode="constant",
                value=float("nan"),
            )

    rgb_image, mask = split_rgba(loaded_image)
    if cast_mask:
        mask = mask[None]

    if pointmap is not None:
        return rgb_image, mask, pointmap
    return rgb_image, mask


def compute_mask_bbox(
    mask: torch.Tensor, box_size_factor: float = 1.0
) -> tuple[float, float, float, float]:
    """
    Compute a bounding box around a binary mask with optional size adjustment.

    Args:
        mask: A 2D binary tensor where non-zero values represent the object of interest.
        box_size_factor: Factor to scale the bounding box size. Values > 1.0 create a larger box.
            Default is 1.0 (tight bounding box).

    Returns:
        A tuple of (x1, y1, x2, y2) coordinates representing the bounding box,
        where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.

    Raises:
        ValueError: If mask is not a torch.Tensor or not a 2D tensor.
    """
    if not isinstance(mask, torch.Tensor):
        raise ValueError("Mask must be a torch.Tensor")
    if not mask.dim() == 2:
        raise ValueError("Mask must be a 2D tensor")
    bbox_indices = torch.nonzero(mask)
    if bbox_indices.numel() == 0:
        # Handle empty mask case
        return (0, 0, 0, 0)

    y_indices = bbox_indices[:, 0]
    x_indices = bbox_indices[:, 1]

    min_x = torch.min(x_indices).item()
    min_y = torch.min(y_indices).item()
    max_x = torch.max(x_indices).item()
    max_y = torch.max(y_indices).item()

    bbox = (min_x, min_y, max_x, max_y)

    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2

    bbox_w, bbox_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    check_bounding_box(bbox_w, bbox_h)

    size = max(bbox_w, bbox_h, 2)
    size = int(size * box_size_factor)

    bbox = (
        int(center_x - size // 2),
        int(center_y - size // 2),
        int(center_x + size // 2),
        int(center_y + size // 2),
    )
    # bbox = tuple(map(int, bbox))
    return bbox


def crop_and_pad(image, bbox):
    """
    Crop an image using a bounding box and pad with zeros if out of bounds.

    Args:
        image (torch.Tensor): CxHxW image.
        bbox (tuple): (x1, y1, x2, y2) bounding box.

    Returns:
        torch.Tensor: Cropped and zero-padded image.
    """
    C, H, W = image.shape
    x1, y1, x2, y2 = bbox

    # Ensure coordinates are integers
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # Compute cropping coordinates
    x1_pad, y1_pad = max(0, -x1), max(0, -y1)
    x2_pad, y2_pad = max(0, x2 - W), max(0, y2 - H)

    # Compute valid region in the original image
    x1_crop, y1_crop = max(0, x1), max(0, y1)
    x2_crop, y2_crop = min(W, x2), min(H, y2)

    # Extract the valid part
    cropped = image[:, y1_crop:y2_crop, x1_crop:x2_crop]

    # Create a zero-padded output
    padded = torch.zeros((C, y2 - y1, x2 - x1), dtype=image.dtype)

    # Place the cropped image into the zero-padded array
    padded[
        :, y1_pad : y1_pad + cropped.shape[1], x1_pad : x1_pad + cropped.shape[2]
    ] = cropped

    return padded


def resize_all_to_same_size(
    rgb_image: torch.Tensor,
    mask: torch.Tensor,
    pointmap: Optional[torch.Tensor] = None,
    target_size: Optional[tuple[int, int]] = None,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Resize RGB image, mask, and pointmap to the same size.
    
    This is crucial when pointmaps have different resolution than RGB images,
    which must be done BEFORE any cropping operations.
    
    Args:
        rgb_image: RGB image tensor of shape (C, H, W)
        mask: Mask tensor of shape (H, W) or (1, H, W)
        pointmap: Optional pointmap tensor of shape (C_p, H_p, W_p)
        target_size: Target size as (H, W). If None, uses RGB image size.
        
    Returns:
        Tuple of (resized_rgb, resized_mask, resized_pointmap)
    """
    squeeze_mask = (mask.dim() == 2) 
    if squeeze_mask:
        mask = mask.unsqueeze(0)
    
    if target_size is None:
        target_size = (rgb_image.shape[1], rgb_image.shape[2])  # (H, W)
    
    rgb_needs_resize = (rgb_image.shape[1], rgb_image.shape[2]) != target_size
    if rgb_needs_resize:
        rgb_image = torchvision.transforms.functional.resize(
            rgb_image, target_size, interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        mask = torchvision.transforms.functional.resize(
            mask, target_size, interpolation=torchvision.transforms.InterpolationMode.NEAREST
        )
    
    if pointmap is not None:
        pointmap_size = (pointmap.shape[1], pointmap.shape[2])
        if pointmap_size != target_size:
            # Handle NaN values in pointmap during resizing
            # Direct resize would propagate NaN values, so we need special handling
            nan_mask = torch.isnan(pointmap).any(dim=0)
            pointmap_clean = torch.where(torch.isnan(pointmap), torch.zeros_like(pointmap), pointmap)
            pointmap_resized = torchvision.transforms.functional.resize(
                pointmap_clean, target_size, interpolation=torchvision.transforms.InterpolationMode.BILINEAR
            )
            
            # Resize the nan mask to identify which regions should remain invalid
            nan_mask_resized = torchvision.transforms.functional.resize(
                nan_mask.unsqueeze(0).float(), target_size, 
                interpolation=torchvision.transforms.InterpolationMode.NEAREST
            ).squeeze(0) > 0.5
            
            # Restore NaN values in regions that were originally invalid
            pointmap = torch.where(
                nan_mask_resized.unsqueeze(0).expand_as(pointmap_resized),
                torch.full_like(pointmap_resized, float('nan')),
                pointmap_resized
            )
    
    if squeeze_mask:
        mask = mask.squeeze(0)
    
    if pointmap is not None:
        return rgb_image, mask, pointmap
    return rgb_image, mask


SSINormalizedPointmap = namedtuple("SSINormalizedPointmap", ["pointmap", "scale", "shift"])
class SSIPointmapNormalizer:

    def normalize(self, pointmap: torch.Tensor, mask: torch.Tensor,
        scale: Optional[torch.Tensor] = None, shift: Optional[torch.Tensor] = None,
    ) -> SSINormalizedPointmap:
        if scale is None or shift is None:
            normalized_pointmap, scale, shift = normalize_pointmap_ssi(pointmap)
        else:
            assert scale.shape == (3,) and shift.shape == (3,), "scale and shift must be in (3,) format"
            normalized_pointmap = _apply_metric_to_ssi(pointmap, scale, shift)
        return SSINormalizedPointmap(normalized_pointmap, scale, shift)
    
    def denormalize(self, pointmap: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        pointmap = _apply_metric_to_ssi(pointmap, scale, shift, apply_inverse=True)
        return pointmap



class ObjectCentricSSI(SSIPointmapNormalizer):
    def __init__(self,
        use_scene_scale: bool = True,
        quantile_drop_threshold: float = 0.1,
        clip_beyond_scale: Optional[float] = None,
        # scale_factor: float = 3.8076, # e^(1.337); empirical mean of R3+Artist train
        scale_factor: float = 1.0, # e^(1.337); empirical mean of R3+Artist train
        allow_scale_and_shift_override: bool = False,
        raise_on_no_valid_points: bool = False,
    ):
        self.use_scene_scale = use_scene_scale
        self.quantile_drop_threshold = quantile_drop_threshold
        self.clip_beyond_scale = clip_beyond_scale
        self.scale_factor = scale_factor
        self.allow_scale_and_shift_override = allow_scale_and_shift_override
        self.raise_on_no_valid_points = raise_on_no_valid_points

    def _compute_scale_and_shift(self, pointmap: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pointmap_size = (pointmap.shape[1], pointmap.shape[2])

        
        mask_resized = torchvision.transforms.functional.resize(
            mask, pointmap_size,
            interpolation=torchvision.transforms.InterpolationMode.NEAREST
        ).squeeze(0)

        pointmap_flat = pointmap.reshape(3, -1)
        # Get valid points from the mask
        mask_bool = mask_resized.reshape(-1) > 0.5
        mask_points = pointmap_flat[:, mask_bool]

        if mask_points.isfinite().max() == 0:
            if self.raise_on_no_valid_points:
                raise ValueError(f"No valid points found in mask")
            logger.warning(f"No valid points found in mask; setting scale to {self.scale_factor} and shift to 0")
            return torch.ones_like(pointmap_flat[:,0]) * self.scale_factor, torch.zeros_like(pointmap_flat[:,0])

        # Compute median for shift
        shift = mask_points.nanmedian(dim=-1).values
        # logger.info(f"{pointmap.shape=} {mask_resized.shape=} {shift.shape=}")


        if self.use_scene_scale == True:
            # Normalize by the scene scale
            points_centered = pointmap_flat - shift.unsqueeze(-1)
            max_dims = points_centered.abs().max(dim=0).values
            scale = max_dims.nanmedian(dim=-1).values
        elif self.use_scene_scale == False:
            # Normalize by the object scale
            shifted_mask_points = mask_points - shift.unsqueeze(-1)
            norm = shifted_mask_points.norm(dim=0)
            quantiles = torch.nanquantile(norm,
                torch.tensor([self.quantile_drop_threshold, 1. - self.quantile_drop_threshold],
                device=shifted_mask_points.device),
                dim=-1)
            scale = (quantiles[1] - quantiles[0]).max(dim=-1).values * 2.0
        elif self.use_scene_scale.upper() == "OBJECT_NORM_MEDIAN":
            # Normalize by the object scale
            shifted_mask_points = mask_points - shift.unsqueeze(-1)
            norm = shifted_mask_points.norm(dim=0)
            scale = norm.nanmedian(dim=-1).values
        else:
            raise ValueError(f"Invalid use_scene_scale: {self.use_scene_scale}")
        scale = scale.expand_as(shift) # per-dim scaling
        scale = scale * self.scale_factor
        return scale, shift
    
    def normalize(self, pointmap: torch.Tensor, mask: torch.Tensor,
        scale: Optional[torch.Tensor] = None, shift: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 1. resize mask to size of pointmap using nearest interpolation
        # 2. get mask points: pointmap[mask > 0.5]
        # 3. shift = mask_points.median() # xyz
        # 4. scale = # filter. If no points, then
        # logger.info(f"{pointmap.shape=} {mask.shape=}")
        assert pointmap.shape[0] == 3, "pointmap must be in (3, H, W) format"
        pointmap_size = (pointmap.shape[1], pointmap.shape[2])

        _scale, _shift = self._compute_scale_and_shift(pointmap, mask)
        if scale is not None and self.allow_scale_and_shift_override:
            _scale = scale
        if shift is not None and self.allow_scale_and_shift_override:
            _shift = shift
        return_scale, return_shift = _scale, _shift

        # Apply normalization
        pointmap_normalized = _apply_metric_to_ssi(pointmap, return_scale, return_shift)
        
        if self.clip_beyond_scale is not None and self.clip_beyond_scale > 0:
            new_norm = pointmap_normalized.norm(dim=0)
            pointmap_normalized = torch.where(
                new_norm > self.clip_beyond_scale,
                torch.full_like(pointmap_normalized, float('nan')),
                pointmap_normalized
            )

        return SSINormalizedPointmap(pointmap_normalized, return_scale, return_shift)


class ObjectApparentSizeSSI(SSIPointmapNormalizer):
    def __init__(self,
            clip_beyond_scale: Optional[float] = None,
            use_scene_scale: bool = True, 
            scale_factor: float = 1.0, # e^(1.337); empirical mean of R3+Artist train
        ):
        self.clip_beyond_scale = clip_beyond_scale
        self.use_scene_scale = use_scene_scale
        self.scale_factor = scale_factor

    def _get_scale_and_shift(self, pointmap: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pointmap_size = (pointmap.shape[1], pointmap.shape[2])
        pointmap_flat = pointmap.reshape(3, -1)

        if not self.use_scene_scale:
            # Get valid points from the mask
            mask_resized = torchvision.transforms.functional.resize(
                mask, pointmap_size,
                interpolation=torchvision.transforms.InterpolationMode.NEAREST
            ).squeeze(0)
            mask_bool = mask_resized.reshape(-1) > 0.5
            pointmap_flat = pointmap_flat[:, mask_bool]

        # Median z-distance
        median_z = pointmap_flat[-1, ...].nanmedian().unsqueeze(0)
        scale = median_z.expand(3) * self.scale_factor
        shift = torch.zeros_like(scale)
        # logger.info(f'median z = {median_z}')
        return scale, shift

    def normalize(self,
        pointmap: torch.Tensor,
        mask: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        shift: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert pointmap.shape[0] == 3, "pointmap must be in (3, H, W) format"
        pointmap_size = (pointmap.shape[1], pointmap.shape[2])

        if scale is None or shift is None:
            scale, shift = self._get_scale_and_shift(pointmap, mask)
        else:
            assert scale.shape == (3,) and shift.shape == (3,), "scale and shift must be in (3,) format"

        # Apply normalization and clip
        pointmap_normalized = _apply_metric_to_ssi(pointmap, scale, shift)
        # logger.info(f"{pointmap_normalized.shape=}")
        
        if self.clip_beyond_scale is not None and self.clip_beyond_scale > 0:
            pointmap_normalized = torch.where(
                pointmap_normalized[-1, ...] > self.clip_beyond_scale,
                torch.full_like(pointmap_normalized, float('nan')),
                pointmap_normalized
            )
        
        # return pointmap_normalized, scale, shift
        return SSINormalizedPointmap(pointmap_normalized, scale, shift)


class NormalizedDisparitySpaceSSI(SSIPointmapNormalizer):
    def __init__(self,
        clip_beyond_scale: Optional[float] = None,
        use_scene_scale: bool = True,
        log_disparity_shift: float = 0.0,
    ):
        self.clip_beyond_scale = clip_beyond_scale
        self.use_scene_scale = use_scene_scale
        self.point_remapper = PointRemapper(remap_type="exp_disparity")
        self.log_disparity_shift = log_disparity_shift

    def normalize(self, pointmap: torch.Tensor, mask: torch.Tensor,
        scale: Optional[torch.Tensor] = None, shift: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert pointmap.shape[0] == 3, "pointmap must be in (3, H, W) format"


        disparity_space_pointmap = self.point_remapper.forward(pointmap.permute(1, 2, 0)).permute(2, 0, 1)
        if scale is None or shift is None:
            scale, shift = self._get_scale_and_shift(disparity_space_pointmap, mask)
        else:
            assert scale.shape == (3,) and shift.shape == (3,), "scale and shift must be in (3,) format"

        # pointmap_normalized = pointmap.clone().detach()
        pointmap_normalized = _apply_metric_to_ssi(disparity_space_pointmap, scale, shift)
        # logger.info(f"{pointmap_normalized.shape=}")
        
        if self.clip_beyond_scale is not None and self.clip_beyond_scale > 0:
            pointmap_normalized = torch.where(
                pointmap_normalized[2, ...].abs() > self.clip_beyond_scale,
                torch.full_like(pointmap_normalized, float('nan')),
                pointmap_normalized
            )
        
        # return pointmap_normalized, scale, shift
        return SSINormalizedPointmap(pointmap_normalized, scale, shift)
    
    def denormalize(self, pointmap: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        pointmap = _apply_metric_to_ssi(pointmap, scale, shift, apply_inverse=True)
        pointmap = self.point_remapper.inverse(pointmap.permute(1, 2, 0)).permute(2, 0, 1)
        return pointmap

    def _get_scale_and_shift(self, pointmap: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pointmap_size = (pointmap.shape[1], pointmap.shape[2])
        mask_resized = torchvision.transforms.functional.resize(
            mask, pointmap_size,
            interpolation=torchvision.transforms.InterpolationMode.NEAREST
        ).squeeze(0)

        pointmap_flat = pointmap.reshape(3, -1)
        if self.use_scene_scale:
            median_z = pointmap_flat[-1, ...].nanmedian().unsqueeze(0)
            shift = torch.zeros_like(median_z.expand(3))
            shift[-1, ...] = median_z[0] + self.log_disparity_shift
        else:
            # Get valid points from the mask (shift, x/z, y/z, log(z))
            mask_bool = mask_resized.reshape(-1) > 0.5
            pointmap_flat = pointmap_flat[:, mask_bool]
            shift = pointmap_flat.nanmedian(dim=-1).values

        scale = torch.ones_like(shift)
        # logger.info(f'median z = {median_z}')
        return scale, shift

def normalize_pointmap_ssi(pointmap: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalize pointmap using Scale-Shift Invariant (SSI) normalization.
    
    Args:
        pointmap: Pointmap tensor of shape (H, W, 3) or (3, H, W)
        
    Returns:
        Tuple of (normalized_pointmap, scale, shift)
    """
    
    # Convert to (H, W, 3) if needed for get_scale_and_shift
    if pointmap.shape[0] == 3:
        pointmap_hw3 = pointmap.permute(1, 2, 0)
        original_format = 'chw'
    else:
        pointmap_hw3 = pointmap
        original_format = 'hwc'
    
    # Get scale and shift using existing method
    scale, shift = ScaleShiftInvariant.get_scale_and_shift(pointmap_hw3)
    
    pointmap_normalized = _apply_metric_to_ssi(pointmap, scale, shift)
    return pointmap_normalized, scale, shift

def _apply_metric_to_ssi(pointmap: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, apply_inverse: bool = False) -> torch.Tensor:
    """
    Normalize pointmap using Scale-Shift Invariant (SSI) normalization.
    
    Args:
        pointmap: Pointmap tensor of shape (H, W, 3) or (3, H, W)
        
    Returns:
        Tuple of (normalized_pointmap, scale, shift)
    """
    
    # Convert to (H, W, 3) if needed for get_scale_and_shift
    if pointmap.shape[0] == 3:
        pointmap_hw3 = pointmap.permute(1, 2, 0)
        original_format = 'chw'
    else:
        pointmap_hw3 = pointmap
        original_format = 'hwc'
    
    # Apply normalization
    ssi_to_metric = ScaleShiftInvariant.ssi_to_metric(scale, shift)
    metric_to_ssi = ssi_to_metric.inverse()
    transform_to_apply = metric_to_ssi

    if apply_inverse:
        transform_to_apply = ssi_to_metric

    pointmap_flat = pointmap_hw3.reshape(-1, 3)
    pointmap_normalized = transform_to_apply.transform_points(pointmap_flat)
    
    # Reshape back to original format
    if original_format == 'chw':
        pointmap_normalized = pointmap_normalized.reshape(pointmap.shape[1], pointmap.shape[2], 3).permute(2, 0, 1)
    else:
        pointmap_normalized = pointmap_normalized.reshape(pointmap_hw3.shape)
    
    return pointmap_normalized


def perturb_mask_translation(
    image: torch.Tensor,
    mask: torch.Tensor,
    max_px_delta: int = 5,
):
    """
    Applies data augmentation to the mask by randomly translating the mask.

    Args:
        image: (C, H, W) float32 [0, 1] tensor.
        mask: (1, H, W) float32 [0, 1] tensor.
        max_px_delta: The maximum number of pixels we will randomly shift by in each 2D direction.
    """
    dx = random.randint(-max_px_delta, max_px_delta)
    dy = random.randint(-max_px_delta, max_px_delta)

    mask = mask.squeeze(0)
    mask = torch.roll(mask, shifts=(dy, dx), dims=(0, 1))
    
    # Zero out wrapped regions
    if dy > 0:
        mask[:dy, :] = 0
    elif dy < 0:
        mask[dy:, :] = 0
    if dx > 0:
        mask[:, :dx] = 0
    elif dx < 0:
        mask[:, dx:] = 0
    
    mask = mask.unsqueeze(0)
    return image, mask


def perturb_mask_boundary(
    image: torch.Tensor,
    mask: torch.Tensor,
    kernel_range: tuple[int, int] = (2, 5),
    p_erode: float = 0.1,
    p_dilate: float = 0.8,
    **kwargs,
):
    """
    Applies data augmentation to the mask by randomly eroding or dilating the mask.

    Args:
        image: (C, H, W) float32 [0, 1] tensor.
        mask: (1, H, W) float32 [0, 1] tensor.
        kernel_range: Range of kernel sizes to sample from.
        p_erode: Probability of erosion.
        p_dilate: Probability of dilation.
        kwargs: Kwargs for the erode/dilate function.
    """
    from .image_ops import erode, dilate

    C, H, W = image.shape
    assert mask.shape == (1, H, W)
    assert mask.dtype == torch.float32
    assert torch.all((mask == 0) | (mask == 1)), "Mask must be binary (0 or 1)"

    p_none = 1.0 - p_erode - p_dilate
    assert 0 <= p_none <= 1, "Probabilities must sum to 1 and be valid."

    # Sample operation.
    op = random.choices(["erode", "dilate", "none"], weights=[p_erode, p_dilate, p_none], k=1)[0]
    
    if op == "none":
        pass
    else:
        # Sample kernel size
        ksize = random.randint(*kernel_range)
        kernel = np.ones((ksize, ksize), np.uint8)

        mask = mask.squeeze().cpu().numpy().astype(np.uint8)  # (H, W)

        if op == "erode":
            mask = erode(mask, kernel, **kwargs)
        elif op == "dilate":
            mask = dilate(mask, kernel, **kwargs)
        else:
            raise NotImplementedError

        mask = torch.from_numpy(mask).float()[None]  # (1, H, W)

    return image, mask


def resolution_blur(
    image: torch.Tensor,
    mask: torch.Tensor,
    scale_range=(0.05, 0.95),
    interpolation_down=tv_transforms.InterpolationMode.BICUBIC,
    interpolation_up=tv_transforms.InterpolationMode.BICUBIC,
):
    """
    Blur the input image by applying upsample(downsample(x)).

    Args:
        image (torch.Tensor): Image tensor of shape (C, H, W), float32, with values in [0, 1].
        mask (torch.Tensor): Mask tensor of shape (1, H, W), float32, with values in [0, 1]. The mask is returned unchanged.
        scale_range: Tuple of (min_scale, max_scale) for downsampling.
        interpolation_down: Interpolation mode for downsampling.
        interpolation_up: Interpolation mode for upsampling.
    """
    C, H, W = image.shape
    scale = random.uniform(*scale_range)
    new_H, new_W = max(1, int(H * scale)), max(1, int(W * scale))

    # Downsample
    image = TF.resize(image, size=[new_H, new_W], interpolation=interpolation_down)
    
    # Upsample back to original size
    image = TF.resize(image, size=[H, W], interpolation=interpolation_up)

    return image, mask


def gaussian_blur(
    image: torch.Tensor,
    mask: torch.Tensor,
    kernel_range: tuple[int, int] = (3, 15),
    sigma_range: tuple[int, int] = (0.1, 4.0),
):
    """
    Apply gaussian blur to the input image.

    Args:
        image (torch.Tensor): Image tensor of shape (C, H, W), float32, with values in [0, 1].
        mask (torch.Tensor): Mask tensor of shape (1, H, W), float32, with values in [0, 1]. The mask is returned unchanged.
        kernel_range (tuple): Range of odd kernel sizes to sample from for the Gaussian blur (min, max).
        sigma_range (tuple): Range of sigma values (standard deviation) to sample from for the Gaussian kernel (min, max).
    """
    kernel_size = random.choice([k for k in range(kernel_range[0], kernel_range[1]+1) if k % 2 == 1])
    sigma = random.uniform(*sigma_range)
    pad = kernel_size // 2

    # Step 1: Pad the image
    image = F.pad(image.unsqueeze(0), (pad, pad, pad, pad), mode='replicate')
    
    # Step 2: Apply gaussian blur
    image = TF.gaussian_blur(image, kernel_size=[kernel_size, kernel_size], sigma=sigma)
    
    # Step 3: Unpad to get back to original size
    image = image[:, :, pad:-pad, pad:-pad]
    
    return image.squeeze(0), mask


def apply_blur_augmentation(
    image: torch.Tensor,
    mask: torch.Tensor,
    p_resolution: float = 0.33,
    p_gaussian: float = 0.33,
    gaussian_kwargs: dict = None,
    resolution_kwargs: dict = None,
):
    """Apply blur augmentation with configurable parameters"""
    
    # Handle None defaults BEFORE unpacking
    if gaussian_kwargs is None:
        gaussian_kwargs = {}
    if resolution_kwargs is None:
        resolution_kwargs = {}
    
    p_none = 1.0 - p_gaussian - p_resolution
    assert 0 <= p_none <= 1, "Probabilities must sum to 1 and be valid."
    
    operation = random.choices(
        ["gaussian", "resolution", "none"], 
        weights=[p_gaussian, p_resolution, p_none], 
        k=1
    )[0]
    
    if operation == "gaussian":
        return gaussian_blur(image, mask, **gaussian_kwargs)
    elif operation == "resolution":
        return resolution_blur(image, mask, **resolution_kwargs)
    elif operation == "none":
        return image, mask
    else:
        raise NotImplementedError


# =============================================================================
# Preprocessor (from preprocessor.py)
# =============================================================================

@dataclass
class PreProcessor:
    """
    Preprocessor configuration for image, mask, and pointmap transforms.

    Transform application order:
    1. Pointmap normalization (if normalize_pointmap=True)
    2. Joint transforms (img_mask_pointmap_joint_transform or img_mask_joint_transform)
    3. Individual transforms (img_transform, mask_transform, pointmap_transform)

    For backward compatibility, img_mask_joint_transform is preserved. When both
    img_mask_pointmap_joint_transform and img_mask_joint_transform are present,
    img_mask_pointmap_joint_transform takes priority.
    """

    img_transform: Callable = (None,)
    mask_transform: Callable = (None,)
    img_mask_joint_transform: list[Callable] = (None,)
    rgb_img_mask_joint_transform: list[Callable] = (None,)

    # New fields for pointmap support
    pointmap_transform: Callable = (None,)
    img_mask_pointmap_joint_transform: list[Callable] = (None,)
    
    # Pointmap normalization option
    normalize_pointmap: bool = False
    pointmap_normalizer: Optional[Callable] = None
    rgb_pointmap_normalizer: Optional[Callable] = None

    def __post_init__(self):
        if self.pointmap_normalizer is None:
            self.pointmap_normalizer = SSIPointmapNormalizer()
            if self.normalize_pointmap == False:
                warnings.warn("normalize_pointmap is also set to False, which means we will return the moments but not normalize the pointmap. This supports old unnormalized pointmap models, but this is dangerous behavior.", DeprecationWarning, stacklevel=2)

        if self.rgb_pointmap_normalizer is None:
            logger.warning("No rgb pointmap normalizer provided, using scale + shift ")
            self.rgb_pointmap_normalizer = self.pointmap_normalizer


    def _normalize_pointmap(
        self, pointmap: torch.Tensor,
        mask: torch.Tensor,
        pointmap_normalizer: Callable,
        scale: Optional[torch.Tensor] = None,
        shift: Optional[torch.Tensor] = None,
    ):
        if pointmap is None:
            return pointmap, None, None

        if self.normalize_pointmap == False:
            # old behavior: Pose is normalized to the pointmap center, but pointmap is not
            _, pointmap_scale, pointmap_shift = pointmap_normalizer.normalize(pointmap, mask)
            return pointmap, pointmap_scale, pointmap_shift
        
        if scale is not None or shift is not None:
            return pointmap_normalizer.normalize(pointmap, mask, scale, shift)
            
        return pointmap_normalizer.normalize(pointmap, mask)

    def _process_image_mask_pointmap_mess(
        self, rgb_image, rgb_image_mask, pointmap=None
    ):
        """Extended version that handles pointmaps"""
 
        # Apply pointmap normalization if enabled
        pointmap_for_crop, pointmap_scale, pointmap_shift = self._normalize_pointmap(
            pointmap, rgb_image_mask, self.pointmap_normalizer
        )

        # Apply transforms to the original full rgb image and mask.
        rgb_image, rgb_image_mask = self._preprocess_rgb_image_mask(rgb_image, rgb_image_mask)

        # These two are typically used for getting cropped images of the object
        #   : first apply joint transforms
        processed_rgb_image, processed_mask, processed_pointmap = (
            self._preprocess_image_mask_pointmap(rgb_image, rgb_image_mask, pointmap_for_crop)
        )
        #   : then apply individual transforms on top of the joint transforms
        processed_rgb_image = self._apply_transform(
            processed_rgb_image, self.img_transform
        )
        processed_mask = self._apply_transform(processed_mask, self.mask_transform)
        if processed_pointmap is not None:
            processed_pointmap = self._apply_transform(
                processed_pointmap, self.pointmap_transform
            )

        # This version is typically the full version of the image
        #   : apply individual transforms only
        rgb_image = self._apply_transform(rgb_image, self.img_transform)
        rgb_image_mask = self._apply_transform(rgb_image_mask, self.mask_transform)
        
        rgb_pointmap, rgb_pointmap_scale, rgb_pointmap_shift = self._normalize_pointmap(
            pointmap, rgb_image_mask, self.rgb_pointmap_normalizer, pointmap_scale, pointmap_shift
        )

        if rgb_pointmap is not None:
            rgb_pointmap = self._apply_transform(rgb_pointmap, self.pointmap_transform)

        result = {
            "mask": processed_mask,
            "image": processed_rgb_image,
            "rgb_image": rgb_image,
            "rgb_image_mask": rgb_image_mask,
        }

        # Add pointmap results if available
        if processed_pointmap is not None:
            result.update(
                {
                    "pointmap": processed_pointmap,
                    "rgb_pointmap": rgb_pointmap,
                }
            )
            
        # Add normalization parameters if normalization was applied
        if pointmap_scale is not None and pointmap_shift is not None:
            result.update(
                {
                    "pointmap_scale": pointmap_scale,
                    "pointmap_shift": pointmap_shift,
                    "rgb_pointmap_scale": rgb_pointmap_scale,
                    "rgb_pointmap_shift": rgb_pointmap_shift,
                }
            )

        return result

    def _process_image_and_mask_mess(self, rgb_image, rgb_image_mask):
        """Original method - calls extended version without pointmap"""
        return self._process_image_mask_pointmap_mess(rgb_image, rgb_image_mask, None)

    def _preprocess_rgb_image_mask(self, rgb_image: torch.Tensor, rgb_image_mask: torch.Tensor):
        """Apply joint transforms to rgb_image and rgb_image_mask."""
        if (
            self.rgb_img_mask_joint_transform != (None,)
            and self.rgb_img_mask_joint_transform is not None
        ):
            for trans in self.rgb_img_mask_joint_transform:
                rgb_image, rgb_image_mask = trans(rgb_image, rgb_image_mask)
        return rgb_image, rgb_image_mask

    def _preprocess_image_mask_pointmap(self, rgb_image, mask_image, pointmap=None):
        """Apply joint transforms with priority: triple transforms > dual transforms."""
        # Priority: img_mask_pointmap_joint_transform when pointmap is provided
        if (
            self.img_mask_pointmap_joint_transform != (None,)
            and self.img_mask_pointmap_joint_transform is not None
            and pointmap is not None
        ):
            for trans in self.img_mask_pointmap_joint_transform:
                rgb_image, mask_image, pointmap = trans(
                    rgb_image, mask_image, pointmap=pointmap
                )
            return rgb_image, mask_image, pointmap

        # Fallback: img_mask_joint_transform (existing behavior)
        elif (
            self.img_mask_joint_transform != (None,)
            and self.img_mask_joint_transform is not None
        ):
            for trans in self.img_mask_joint_transform:
                rgb_image, mask_image = trans(rgb_image, mask_image)
            return rgb_image, mask_image, pointmap

        return rgb_image, mask_image, pointmap

    def _preprocess_image_and_mask(self, rgb_image, mask_image):
        """Backward compatibility wrapper - only applies dual transforms"""
        rgb_image, mask_image, _ = self._preprocess_image_mask_pointmap(
            rgb_image, mask_image, None
        )
        return rgb_image, mask_image

    # keep here for backward compatibility
    def _preprocess_image_and_mask_inference(self, rgb_image, mask_image):
        warnings.warn(
            "The _preprocess_image_and_mask_inference is deprecated! Please use _preprocess_image_and_mask",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return self._preprocess_image_and_mask(rgb_image, mask_image)

    def _apply_transform(self, input: torch.Tensor, transform):
        if input is not None and transform is not None and transform != (None,):
            input = transform(input)

        return input

# =============================================================================
# Pose Target (from pose_target.py)
# =============================================================================

@dataclass
class InstancePose:
    """
    Stores the pose of an object.
    Also, stores some information about the scene that was used to normalize the pose.
    """

    instance_scale_l2c: torch.Tensor
    instance_position_l2c: torch.Tensor
    instance_quaternion_l2c: torch.Tensor
    scene_scale: torch.Tensor
    scene_shift: torch.Tensor

    @classmethod
    def _broadcast_postcompose(
        cls,
        scale: torch.Tensor,
        rotation: torch.Tensor,
        translation: torch.Tensor,
        transform_to_postcompose: Transform3d,
    ) -> Transform3d:
        """
        Assumes scale, rotation, translation are of shape:
            B, K, C
            ---
            B: batch size
            K: number of objects
            C: number of channels

        Takes a transform where
            get_matrix() has shape (B, 3, 3)

        Returns pose.compose(transform_to_postcompose)
        """
        scale_c = scale.shape[-1]
        ndim_orig = scale.ndim
        if ndim_orig == 3:
            b, k, _ = scale.shape
        elif ndim_orig == 2:
            b = scale.shape[0]
            k = 1
        elif ndim_orig == 1:
            b = 1
            k = 1
        else:
            raise ValueError(f"Invalid scale shape: {scale.shape}")

        # Create transform of shape (B * K)
        wide = {"scale": scale, "rotation": rotation, "translation": translation}
        shapes_orig = {k: v.shape for k, v in wide.items()}
        long = tree_tensor_map(lambda x: x.reshape(b * k, x.shape[-1]), wide)
        long["rotation"] = quaternion_to_matrix(long["rotation"])
        if scale_c == 1:
            long["scale"] = long["scale"].expand(b * k, 3)

        composed = compose_transform(**long)

        # Apply transform to shape (B * K)
        pc_transform = transform_to_postcompose.get_matrix()
        pc_transform = pc_transform.repeat(k, 1, 1)
        stacked_pc_transform = Transform3d(matrix=pc_transform)
        assert stacked_pc_transform.get_matrix().shape == composed.get_matrix().shape
        postcomposed = composed.compose(stacked_pc_transform)

        # Decompose transform to shape (B, K, C)
        scale, rotation, translation = decompose_transform(postcomposed)
        rotation = matrix_to_quaternion(rotation)
        pc_long = {"scale": scale, "rotation": rotation, "translation": translation}
        pc_wide = tree_tensor_map(lambda x: x.reshape(b, k, x.shape[-1]), pc_long)
        if scale_c == 1:
            pc_wide["scale"] = pc_wide["scale"][..., 0].unsqueeze(-1)
        for k, shape in shapes_orig.items():
            pc_wide[k] = pc_wide[k].reshape(*shape)
        return pc_wide["scale"], pc_wide["rotation"], pc_wide["translation"]


@dataclass
class PoseTarget:
    x_instance_scale: torch.Tensor
    x_instance_rotation: torch.Tensor
    x_instance_translation: torch.Tensor
    x_scene_scale: torch.Tensor
    x_scene_center: torch.Tensor
    x_translation_scale: torch.Tensor
    pose_target_convention: str = field(default="unknown")


@dataclass
class InvariantPoseTarget:
    """
    This is the canonical representation of pose targets, used for computing metrics.
        instance_pose <-> invariant_pose_targets <-> all other pose_target_conventions

    Background:
    ---
    We want to estimate a transformation T: R^3 -> R^3 despite scene scale ambiguity.

    The transformation taking object points to scene points is defined as
        T(x) = s · R(q) · x + t
        where:
            - x is a point in the object coordinate frame,
            - q is a unit quaternion representing rotation,
            - s is the object-to-scene scale, and
            - t is the translation.

    However, there is an inherent scale ambiguity in the scene, denoted as s_scene;
    This ambiguity introduces irreducible error that complicates both evaluation and training.

    To decouple the scene scale from the invariant quantities, we define:
        T(x)  = s_scene · |t_rel| [ s_tilde · R(q) · x + t_unit ]
        where we define
            t_rel = t / s_scene
            s_rel = s / s_scene
            s_tilde = s_rel / |t_rel|
            t_unit = t_rel / |t_rel|

    During training, you would predict (q, s_tilde, t_unit), leaving s_scene separate.


    Hand-wavy error analysis:
    ---
    1. Naive (coupled) estimate:
       T(x) = s_scene [ s_rel · R(q) · x + t_rel ]

       We can define:
           U = ln(s_rel)
           V = ln(|t_rel|)
       so that the error is governed by Var(U + V).

    2. In the decoupled case, we have:
       T(x) = s_scene · |t_rel| [ s_tilde · R(q) · x + t_unit ]
            = s_scene · |t_rel| [ (s_rel / |t_rel|) R(q) · x + t_unit ]
       Then ln(s_tilde) = ln(s_rel) - ln(|t_rel|) = U - V, and the error is
       Var(U - V) = Var(U) + Var(V) - 2Cov(U, V).

    """

    # These are invariant
    q: torch.Tensor
    t_unit: torch.Tensor
    s_scene: torch.Tensor
    t_scene_center: Optional[torch.Tensor] = None
    t_rel_norm: Optional[torch.Tensor] = None
    s_tilde: Optional[torch.Tensor] = None
    s_rel: Optional[torch.Tensor] = None

    def __post_init__(self):
        # Check that fields that are required always have values.
        if self.q is None:
            raise ValueError("Field 'q' (quaternion) must be provided.")
        if self.s_scene is None:
            raise ValueError("Field 's_scene' must be provided.")
        if self.s_rel is None:
            if self.s_tilde is not None:
                self.s_rel = self.s_tilde * self.t_rel_norm
            else:
                raise ValueError("Field 's_rel' or 's_tilde' must be provided.")
        if self.t_unit is None:
            raise ValueError("Field 't_unit' must be provided.")

        if self.t_scene_center is None:
            self.t_scene_center = torch.zeros_like(self.t_unit)

        # There is a simple relationship between s_tilde and t_rel_norm:
        #    s_tilde = s_rel / t_rel_norm
        #
        # If one of these is missing and the other is provided, we can compute the missing field.
        if self.s_tilde is None and self.t_rel_norm is not None:
            self.s_tilde = self.s_rel / self.t_rel_norm
        elif self.t_rel_norm is None and self.s_tilde is not None:
            self.t_rel_norm = self.s_rel / self.s_tilde

        # If both are provided, we check for consistency.
        if self.s_tilde is not None and self.t_rel_norm is not None:
            computed_s_tilde = self.s_rel / self.t_rel_norm
            # If the provided s_tilde deviates from what is computed, update it.
            if not torch.allclose(self.s_tilde, computed_s_tilde, atol=1e-6):
                logger.warning(
                    f"s_tilde and t_rel_norm are provided, but they are not consistent. "
                    f"Updating s_tilde to {computed_s_tilde}."
                )
                self.s_tilde = computed_s_tilde

        self._validate_fields()

    def _validate_fields(self):
        for field in self.__dict__:
            if self.__dict__[field] is None:
                raise ValueError(f"Field '{field}' must be provided.")


    @staticmethod
    def from_instance_pose(instance_pose: InstancePose) -> "InvariantPoseTarget":
        q = instance_pose.instance_quaternion_l2c
        s_obj_to_scene = instance_pose.instance_scale_l2c      # (..., 1) or (..., 3)
        t_obj_to_scene = instance_pose.instance_position_l2c   # (..., 3)
        s_scene = instance_pose.scene_scale                    # (..., 1) or scalar-broadcastable
        t_scene_center = instance_pose.scene_shift             # (..., 3)

        # Normalize to scene scale (per the derivation)
        if not ( s_obj_to_scene.ndim == (s_scene.ndim + 1)):
            raise ValueError(f"s_scene should be ND [...,3] and s_obj_to_scene should be (N+1)D [...,K,3], but got {s_scene.shape=} {s_obj_to_scene.shape=}")
        if not (t_obj_to_scene.ndim == (s_scene.ndim + 1)):
            raise ValueError(f"t_scene_center should be ND [B,3] and t_obj_to_scene should be (N+1)D [B,K,3], but got {t_scene_center.shape=} {t_obj_to_scene.shape=}")
        s_scene_exp = s_scene.unsqueeze(-2)

        s_rel = s_obj_to_scene / s_scene_exp
        t_rel = t_obj_to_scene / s_scene_exp

        # Robust norms
        eps = 1e-8
        t_rel_norm = t_rel.norm(dim=-1, keepdim=True).clamp_min(eps)

        s_tilde = s_rel / t_rel_norm
        t_unit = t_rel / t_rel_norm

        return InvariantPoseTarget(
            q=q,
            s_scene=s_scene,
            t_scene_center=t_scene_center,
            s_rel=s_rel,
            s_tilde=s_tilde,
            t_unit=t_unit,
            t_rel_norm=t_rel_norm,
        )


    @staticmethod
    def to_instance_pose(invariant_targets: "InvariantPoseTarget") -> InstancePose:
        # scale factor per the derivation: s_scene * |t_rel|
        # Normalize to scene scale (per the derivation)
        t_rel_norm_ndim = invariant_targets.t_rel_norm.ndim
        if not (invariant_targets.s_scene.ndim == (t_rel_norm_ndim - 1)) :
            raise ValueError(f"s_scene should be ND [...,3] and t_rel_norm should be (N+1)D [...,K,3], but got {invariant_targets.s_scene.shape=} {invariant_targets.t_rel_norm.shape=}")

        scale = invariant_targets.s_scene.unsqueeze(-2) * invariant_targets.t_rel_norm
        return InstancePose(
            instance_scale_l2c=invariant_targets.s_tilde * scale,
            instance_position_l2c=invariant_targets.t_unit * scale,
            instance_quaternion_l2c=invariant_targets.q,
            scene_scale=invariant_targets.s_scene,
            scene_shift=invariant_targets.t_scene_center,
        )


class PoseTargetConvention:
    """
    Converts pose_targets <-> instance_pose <-> invariant_pose_targets
    """

    pose_target_convention: str

    @classmethod
    def from_invariant(cls, invariant_targets: InvariantPoseTarget) -> PoseTarget:
        raise NotImplementedError("Implement this in a subclass")

    @classmethod
    def to_invariant(cls, instance_pose: InstancePose) -> InvariantPoseTarget:
        raise NotImplementedError("Implement this in a subclass")

    @classmethod
    def from_instance_pose(cls, instance_pose: InstancePose) -> PoseTarget:
        invariant_targets = InvariantPoseTarget.from_instance_pose(instance_pose)
        return cls.from_invariant(invariant_targets)

    @classmethod
    def to_instance_pose(cls, pose_target: PoseTarget) -> InstancePose:
        invariant_targets = cls.to_invariant(pose_target)
        return InvariantPoseTarget.to_instance_pose(invariant_targets)


class ScaleShiftInvariant(PoseTargetConvention):
    """

    Midas eq. (6): https://arxiv.org/pdf/1907.01341v3
    But for pointmaps (see MoGe): https://arxiv.org/pdf/2410.19115
    """

    pose_target_convention: str = "ScaleShiftInvariant"
    scale_mean = torch.tensor([1.0232692956924438, 1.0232691764831543, 1.0232692956924438]).to(torch.float32)
    scale_std = torch.tensor([1.3773751258850098, 1.3773752450942993, 1.3773750066757202]).to(torch.float32)
    translation_mean = torch.tensor([0.003191213821992278, 0.017236359417438507, 0.9401122331619263]).to(torch.float32)
    translation_std = torch.tensor([1.341888666152954, 0.7665449380874634, 3.175130605697632]).to(torch.float32)

    @classmethod
    def from_instance_pose(cls, instance_pose: InstancePose, normalize: bool = False) -> PoseTarget:
        metric_to_ssi = cls.ssi_to_metric(
            instance_pose.scene_scale, instance_pose.scene_shift
        ).inverse()

        ssi_scale, ssi_rotation, ssi_translation = InstancePose._broadcast_postcompose(
            scale=instance_pose.instance_scale_l2c,
            rotation=instance_pose.instance_quaternion_l2c,
            translation=instance_pose.instance_position_l2c,
            transform_to_postcompose=metric_to_ssi,
        )
        # logger.info(f"{normalize=} {ssi_scale.shape=} {ssi_rotation.shape=} {ssi_translation.shape=}")
        if normalize:
            device = ssi_scale.device
            ssi_scale = (ssi_scale - cls.scale_mean.to(device)) / cls.scale_std.to(device)
            ssi_translation = (ssi_translation - cls.translation_mean.to(device)) / cls.translation_std.to(device)

        return PoseTarget(
            x_instance_scale=ssi_scale,
            x_instance_rotation=ssi_rotation,
            x_instance_translation=ssi_translation,
            x_scene_scale=instance_pose.scene_scale,
            x_scene_center=instance_pose.scene_shift,
            x_translation_scale=torch.ones_like(ssi_scale)[..., 0].unsqueeze(-1),
            pose_target_convention=cls.pose_target_convention,
        )

    @classmethod
    def to_instance_pose(cls, pose_target: PoseTarget, normalize: bool = False) -> InstancePose:
        scene_scale = pose_target.x_scene_scale
        scene_shift = pose_target.x_scene_center
        ssi_to_metric = cls.ssi_to_metric(scene_scale, scene_shift)

        if normalize:
            device = pose_target.x_instance_scale.device
            pose_target.x_instance_scale = pose_target.x_instance_scale * cls.scale_std.to(device) + cls.scale_mean.to(device)
            pose_target.x_instance_translation = pose_target.x_instance_translation * cls.translation_std.to(device) + cls.translation_mean.to(device)

        ins_scale, ins_rotation, ins_translation = InstancePose._broadcast_postcompose(
            scale=pose_target.x_instance_scale,
            rotation=pose_target.x_instance_rotation,
            translation=pose_target.x_instance_translation,
            transform_to_postcompose=ssi_to_metric,
        )

        return InstancePose(
            instance_scale_l2c=ins_scale,
            instance_position_l2c=ins_translation,
            instance_quaternion_l2c=ins_rotation,
            scene_scale=scene_scale,
            scene_shift=scene_shift,
        )

    @classmethod
    def to_invariant(cls, pose_target: PoseTarget, normalize: bool = False) -> InvariantPoseTarget:
        instance_pose = cls.to_instance_pose(pose_target, normalize=normalize)
        return InvariantPoseTarget.from_instance_pose(instance_pose)

    @classmethod
    def from_invariant(cls, invariant_targets: InvariantPoseTarget, normalize: bool = False) -> PoseTarget:
        instance_pose = InvariantPoseTarget.to_instance_pose(invariant_targets)
        return cls.from_instance_pose(instance_pose, normalize=normalize)

    @classmethod
    def get_scale_and_shift(cls, pointmap):
        shift_z = pointmap[..., -1].nanmedian().unsqueeze(0)
        shift = torch.zeros_like(shift_z.expand(1, 3))
        shift[..., -1] = shift_z

        shifted_pointmap = pointmap - shift
        scale = shifted_pointmap.abs().nanmean().to(shift.device)

        shift = shift.reshape(3)
        scale = scale.expand(3)

        return scale, shift

    @staticmethod
    def ssi_to_metric(scale: torch.Tensor, shift: torch.Tensor):
        if scale.ndim == 1:
            scale = scale.unsqueeze(0)
        if shift.ndim == 1:
            shift = shift.unsqueeze(0)
        return Transform3d().scale(scale).translate(shift).to(shift.device)


class ScaleShiftInvariantWTranslationScale(PoseTargetConvention):
    """

    Midas eq. (6): https://arxiv.org/pdf/1907.01341v3
    But for pointmaps (see MoGe): https://arxiv.org/pdf/2410.19115
    """

    pose_target_convention: str = "ScaleShiftInvariantWTranslationScale"
    scale_mean = torch.tensor([1.0232692956924438, 1.0232691764831543, 1.0232692956924438]).to(torch.float32)
    scale_std = torch.tensor([1.3773751258850098, 1.3773752450942993, 1.3773750066757202]).to(torch.float32)
    translation_mean = torch.tensor([0.003191213821992278, 0.017236359417438507, 0.9401122331619263]).to(torch.float32)
    translation_std = torch.tensor([1.341888666152954, 0.7665449380874634, 3.175130605697632]).to(torch.float32)

    @classmethod
    def from_instance_pose(cls, instance_pose: InstancePose, normalize: bool = False) -> PoseTarget:
        metric_to_ssi = cls.ssi_to_metric(
            instance_pose.scene_scale, instance_pose.scene_shift
        ).inverse()

        ssi_scale, ssi_rotation, ssi_translation = InstancePose._broadcast_postcompose(
            scale=instance_pose.instance_scale_l2c,
            rotation=instance_pose.instance_quaternion_l2c,
            translation=instance_pose.instance_position_l2c,
            transform_to_postcompose=metric_to_ssi,
        )

        ssi_translation_scale = ssi_translation.norm(dim=-1, keepdim=True)
        ssi_translation_unit = ssi_translation / ssi_translation_scale.clamp_min(1e-7)

        return PoseTarget(
            x_instance_scale=ssi_scale,
            x_instance_rotation=ssi_rotation,
            x_instance_translation=ssi_translation_unit,
            x_scene_scale=instance_pose.scene_scale,
            x_scene_center=instance_pose.scene_shift,
            x_translation_scale=ssi_translation_scale,
            pose_target_convention=cls.pose_target_convention,
        )

    @classmethod
    def to_instance_pose(cls, pose_target: PoseTarget, normalize: bool = False) -> InstancePose:
        scene_scale = pose_target.x_scene_scale
        scene_shift = pose_target.x_scene_center
        ssi_to_metric = cls.ssi_to_metric(scene_scale, scene_shift)

        ins_translation_unit = pose_target.x_instance_translation / pose_target.x_instance_translation.norm(dim=-1, keepdim=True)
        ins_translation = ins_translation_unit * pose_target.x_translation_scale


        ins_scale, ins_rotation, ins_translation = InstancePose._broadcast_postcompose(
            scale=pose_target.x_instance_scale,
            rotation=pose_target.x_instance_rotation,
            translation=ins_translation,
            transform_to_postcompose=ssi_to_metric,
        )


        return InstancePose(
            instance_scale_l2c=ins_scale,
            instance_position_l2c=ins_translation,
            instance_quaternion_l2c=ins_rotation,
            scene_scale=scene_scale,
            scene_shift=scene_shift,
        )

    @classmethod
    def to_invariant(cls, pose_target: PoseTarget) -> InvariantPoseTarget:
        instance_pose = cls.to_instance_pose(pose_target)
        return InvariantPoseTarget.from_instance_pose(instance_pose)

    @classmethod
    def from_invariant(cls, invariant_targets: InvariantPoseTarget) -> PoseTarget:
        instance_pose = InvariantPoseTarget.to_instance_pose(invariant_targets)
        return cls.from_instance_pose(instance_pose)

    @classmethod
    def get_scale_and_shift(cls, pointmap):
        shift_z = pointmap[..., -1].nanmedian().unsqueeze(0)
        shift = torch.zeros_like(shift_z.expand(1, 3))
        shift[..., -1] = shift_z

        shifted_pointmap = pointmap - shift
        scale = shifted_pointmap.abs().nanmean().to(shift.device)

        shift = shift.reshape(3)
        scale = scale.expand(3)

        return scale, shift

    @staticmethod
    def ssi_to_metric(scale: torch.Tensor, shift: torch.Tensor):
        if scale.ndim == 1:
            scale = scale.unsqueeze(0)
        if shift.ndim == 1:
            shift = shift.unsqueeze(0)
        return Transform3d().scale(scale).translate(shift).to(shift.device)


class DisparitySpace(PoseTargetConvention):
    pose_target_convention: str = "DisparitySpace"
    
    @classmethod
    def from_instance_pose(cls, instance_pose: InstancePose, normalize: bool = False) -> PoseTarget:
 
        # x_instance_scale = orig_scale / scene_scale
        # x_instance_translation = [x/z, y/z, 0]  / scene_scale
        # x_translation_scale = z  / scene_scale
        assert torch.allclose(instance_pose.scene_scale, torch.ones_like(instance_pose.scene_scale))

        if not instance_pose.scene_shift.ndim == instance_pose.instance_position_l2c.ndim - 1:
            raise ValueError(f"scene_shift must be (N+1)D and instance_position_l2c must be (N+1)D, but got {instance_pose.scene_shift.ndim} and {instance_pose.instance_position_l2c.ndim}")
        shift_xy, shift_z_log = instance_pose.scene_shift.unsqueeze(-2).split([2, 1], dim=-1)


        pose_xy, pose_z = instance_pose.instance_position_l2c.split([2, 1], dim=-1)
        # Handle batch dimensions properly
        if shift_xy.ndim < pose_xy.ndim:
            shift_xy = shift_xy.unsqueeze(-2)
        pose_xy_scaled = pose_xy / pose_z - shift_xy

        pose_z_scaled_log = torch.log(pose_z) - shift_z_log
        x_instance_scale_log = torch.log(instance_pose.instance_scale_l2c) - torch.log(pose_z)

        x_instance_translation = torch.cat([pose_xy_scaled, torch.zeros_like(pose_z)], dim=-1)
        x_translation_scale = torch.exp(pose_z_scaled_log)
        x_instance_scale = torch.exp(x_instance_scale_log)



        return PoseTarget(
            x_instance_scale=x_instance_scale,
            x_instance_translation=x_instance_translation,
            x_instance_rotation=instance_pose.instance_quaternion_l2c,
            x_scene_scale=instance_pose.scene_scale,
            x_scene_center=instance_pose.scene_shift,
            x_translation_scale=x_translation_scale,
            pose_target_convention=cls.pose_target_convention,
        )

    @classmethod
    def to_instance_pose(cls, pose_target: PoseTarget, normalize: bool = False) -> InstancePose:
        scene_scale = pose_target.x_scene_scale
        scene_shift = pose_target.x_scene_center

        if not pose_target.x_scene_center.ndim == pose_target.x_instance_translation.ndim - 1:
            raise ValueError(f"x_scene_center must be (N+1)D and x_instance_translation must be (N+1)D, but got {pose_target.x_scene_center.ndim} and {pose_target.x_instance_translation.ndim}")
        shift_xy, shift_z_log = pose_target.x_scene_center.unsqueeze(-2).split([2, 1], dim=-1)
        scene_z_scale = torch.exp(shift_z_log)
 
        z = pose_target.x_translation_scale
        ins_translation = pose_target.x_instance_translation.clone()
        ins_translation[...,2] = 1.0
        ins_translation[...,:2] = ins_translation[...,:2] + shift_xy
        ins_translation = ins_translation * z * scene_z_scale

        ins_scale = pose_target.x_instance_scale * z * scene_z_scale

        return InstancePose(
            instance_scale_l2c=ins_scale * scene_scale,
            instance_position_l2c=ins_translation * scene_scale,
            instance_quaternion_l2c=pose_target.x_instance_rotation,
            scene_scale=scene_scale,
            scene_shift=scene_shift,
        )

    @classmethod
    def to_invariant(cls, pose_target: PoseTarget, normalize: bool = False) -> InvariantPoseTarget:
        instance_pose = cls.to_instance_pose(pose_target, normalize=normalize)
        return InvariantPoseTarget.from_instance_pose(instance_pose)

    @classmethod
    def from_invariant(cls, invariant_targets: InvariantPoseTarget, normalize: bool = False) -> PoseTarget:
        instance_pose = InvariantPoseTarget.to_instance_pose(invariant_targets)
        return cls.from_instance_pose(instance_pose, normalize=normalize)



class NormalizedSceneScale(PoseTargetConvention):
    """
    x_instance_scale and x_translation_scale are normalized to x_scene_scale
    """

    pose_target_convention: str = "NormalizedSceneScale"

    @classmethod
    def from_invariant(cls, invariant_targets: InvariantPoseTarget):
        translation = invariant_targets.t_unit * invariant_targets.t_rel_norm
        return PoseTarget(
            x_instance_scale=invariant_targets.s_rel,
            x_instance_rotation=invariant_targets.q,
            x_instance_translation=translation,
            x_scene_scale=invariant_targets.s_scene,
            x_scene_center=invariant_targets.t_scene_center,
            x_translation_scale=torch.ones_like(invariant_targets.t_rel_norm),
            pose_target_convention=cls.pose_target_convention,
        )

    @classmethod
    def to_invariant(cls, pose_target: PoseTarget):
        t_rel_norm = torch.norm(
            pose_target.x_instance_translation, dim=-1, keepdim=True
        )
        return InvariantPoseTarget(
            s_scene=pose_target.x_scene_scale,
            s_rel=pose_target.x_instance_scale,
            q=pose_target.x_instance_rotation,
            t_unit=pose_target.x_instance_translation / t_rel_norm,
            t_rel_norm=t_rel_norm,
            t_scene_center=pose_target.x_scene_center,
        )


class Naive(PoseTargetConvention):
    pose_target_convention: str = "Naive"

    @classmethod
    def from_invariant(cls, invariant_targets: InvariantPoseTarget):
        s_scene = invariant_targets.s_rel * invariant_targets.s_scene
        t_scene = invariant_targets.t_unit * invariant_targets.t_rel_norm
        return PoseTarget(
            x_instance_scale=s_scene,
            x_instance_rotation=invariant_targets.q,
            x_instance_translation=t_scene,
            x_scene_scale=invariant_targets.s_scene,
            x_scene_center=invariant_targets.t_scene_center,
            x_translation_scale=torch.ones_like(invariant_targets.t_rel_norm),
            pose_target_convention=cls.pose_target_convention,
        )

    @classmethod
    def to_invariant(cls, pose_target: PoseTarget):
        s_scene = pose_target.x_scene_scale
        t_rel_norm = torch.norm(
            pose_target.x_instance_translation, dim=-1, keepdim=True
        )
        return InvariantPoseTarget(
            s_scene=s_scene,
            t_scene_center=pose_target.x_scene_center,
            s_rel=pose_target.x_instance_scale / s_scene,
            q=pose_target.x_instance_rotation,
            t_unit=pose_target.x_instance_translation / t_rel_norm,
            t_rel_norm=t_rel_norm,
        )


class NormalizedSceneScaleAndTranslation(PoseTargetConvention):
    """
    x_instance_scale and x_translation_scale are normalized to x_scene_scale
    x_instance_translation is unit
    """

    pose_target_convention: str = "NormalizedSceneScaleAndTranslation"

    @classmethod
    def from_invariant(cls, invariant_targets: InvariantPoseTarget):
        return PoseTarget(
            x_instance_scale=invariant_targets.s_rel,
            x_instance_rotation=invariant_targets.q,
            x_instance_translation=invariant_targets.t_unit,
            x_scene_scale=invariant_targets.s_scene,
            x_scene_center=invariant_targets.t_scene_center,
            x_translation_scale=invariant_targets.t_rel_norm,
            pose_target_convention=cls.pose_target_convention,
        )

    @classmethod
    def to_invariant(cls, pose_target: PoseTarget):
        return InvariantPoseTarget(
            s_scene=pose_target.x_scene_scale,
            t_scene_center=pose_target.x_scene_center,
            s_rel=pose_target.x_instance_scale,
            q=pose_target.x_instance_rotation,
            t_unit=pose_target.x_instance_translation,
            t_rel_norm=pose_target.x_translation_scale,
        )


class ApparentSize(PoseTargetConvention):
    pose_target_convention: str = "ApparentSize"

    @classmethod
    def from_invariant(cls, invariant_targets: InvariantPoseTarget):
        return PoseTarget(
            x_instance_scale=invariant_targets.s_tilde,
            x_instance_rotation=invariant_targets.q,
            x_instance_translation=invariant_targets.t_unit,
            x_scene_scale=invariant_targets.s_scene,
            x_scene_center=invariant_targets.t_scene_center,
            x_translation_scale=invariant_targets.t_rel_norm,
            pose_target_convention=cls.pose_target_convention,
        )

    @classmethod
    def to_invariant(cls, pose_target: PoseTarget):
        return InvariantPoseTarget(
            s_scene=pose_target.x_scene_scale,
            t_scene_center=pose_target.x_scene_center,
            s_tilde=pose_target.x_instance_scale,
            q=pose_target.x_instance_rotation,
            t_unit=pose_target.x_instance_translation,
            t_rel_norm=pose_target.x_translation_scale,
        )


class Identity(PoseTargetConvention):
    """
    Identity convention - no transformation applied.
    Direct passthrough mapping between instance pose and pose target values.
    This preserves all values including scene_scale and scene_shift.
    """
    
    pose_target_convention: str = "Identity"
    
    @classmethod
    def from_instance_pose(cls, instance_pose: InstancePose) -> PoseTarget:
        return PoseTarget(
            x_instance_scale=instance_pose.instance_scale_l2c,
            x_instance_rotation=instance_pose.instance_quaternion_l2c,
            x_instance_translation=instance_pose.instance_position_l2c,
            x_scene_scale=instance_pose.scene_scale,
            x_scene_center=instance_pose.scene_shift,
            x_translation_scale=torch.ones_like(instance_pose.instance_scale_l2c)[..., 0].unsqueeze(-1),
            pose_target_convention=cls.pose_target_convention,
        )
    
    @classmethod
    def to_instance_pose(cls, pose_target: PoseTarget) -> InstancePose:
        return InstancePose(
            instance_scale_l2c=pose_target.x_instance_scale,
            instance_position_l2c=pose_target.x_instance_translation,
            instance_quaternion_l2c=pose_target.x_instance_rotation,
            scene_scale=pose_target.x_scene_scale,
            scene_shift=pose_target.x_scene_center,
        )
    
    @classmethod
    def to_invariant(cls, pose_target: PoseTarget) -> InvariantPoseTarget:
        instance_pose = cls.to_instance_pose(pose_target)
        return InvariantPoseTarget.from_instance_pose(instance_pose)
    
    @classmethod
    def from_invariant(cls, invariant_targets: InvariantPoseTarget) -> PoseTarget:
        instance_pose = InvariantPoseTarget.to_instance_pose(invariant_targets)
        return cls.from_instance_pose(instance_pose)


class PoseTargetConverter:
    @staticmethod
    def pose_target_to_instance_pose(pose_target: PoseTarget, normalize: bool = False) -> InstancePose:
        _convention_class = globals()[pose_target.pose_target_convention]
        if _convention_class == ScaleShiftInvariant:
            return _convention_class.to_instance_pose(pose_target, normalize=normalize)
        else:
            return _convention_class.to_instance_pose(pose_target)

    @staticmethod
    def instance_pose_to_pose_target(
        instance_pose: InstancePose, pose_target_convention: str, normalize: bool = False
    ) -> PoseTarget:
        _convention_class = globals()[pose_target_convention]
        if _convention_class == ScaleShiftInvariant:
            return _convention_class.from_instance_pose(instance_pose, normalize=normalize)
        else:
            return _convention_class.from_instance_pose(instance_pose)

    @staticmethod
    def dicts_instance_pose_to_pose_target(
        pose_target_convention: str,
        **kwargs,
    ):
        instance_pose = InstancePose(**kwargs)
        pose_target = PoseTargetConverter.instance_pose_to_pose_target(
            instance_pose, pose_target_convention
        )
        return asdict(pose_target)

    @staticmethod
    def dicts_pose_target_to_instance_pose(
        **kwargs,
    ):
        pose_target_convention = kwargs.get("pose_target_convention")
        _convention_class = globals()[pose_target_convention]
        assert (
            _convention_class.pose_target_convention == pose_target_convention
        ), f"Normalization name mismatch: {_convention_class.pose_target_convention} != {pose_target_convention}"

        normalize = kwargs.pop("normalize", False)
        pose_target = PoseTarget(**kwargs)
        instance_pose = PoseTargetConverter.pose_target_to_instance_pose(pose_target, normalize)
        return asdict(instance_pose)


class LogScaleShiftNormalizer:
    def __init__(self, shift_log: torch.Tensor = 0.0, scale_log: torch.Tensor = 1.0):
        self.shift_log = shift_log
        self.scale_log = scale_log

    def normalize(self, value: torch.Tensor):
        return torch.log(value) - self.shift_log / self.scale_log

    def denormalize(self, value: torch.Tensor):
        return torch.exp(value * self.scale_log + self.shift_log)