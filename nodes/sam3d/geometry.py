# Copyright (c) Meta Platforms, Inc. and affiliates.
# Consolidated from moge/utils/geometry_torch.py and moge/utils/geometry_numpy.py
from typing import *
from functools import partial
import math
from collections import namedtuple

import numpy as np
from scipy.signal import fftconvolve
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.types
import utils3d

try:
    import cv2 as _cv2
except ImportError:
    _cv2 = None


# ===========================================================================
# NumPy geometry utilities (from geometry_numpy.py)
# ===========================================================================

def weighted_mean_numpy(x: np.ndarray, w: np.ndarray = None, axis: Union[int, Tuple[int,...]] = None, keepdims: bool = False, eps: float = 1e-7) -> np.ndarray:
    if w is None:
        return np.mean(x, axis=axis)
    else:
        w = w.astype(x.dtype)
        return (x * w).mean(axis=axis) / np.clip(w.mean(axis=axis), eps, None)


def harmonic_mean_numpy(x: np.ndarray, w: np.ndarray = None, axis: Union[int, Tuple[int,...]] = None, keepdims: bool = False, eps: float = 1e-7) -> np.ndarray:
    if w is None:
        return 1 / (1 / np.clip(x, eps, None)).mean(axis=axis)
    else:
        w = w.astype(x.dtype)
        return 1 / (weighted_mean_numpy(1 / (x + eps), w, axis=axis, keepdims=keepdims, eps=eps) + eps)


def normalized_view_plane_uv_numpy(width: int, height: int, aspect_ratio: float = None, dtype: np.dtype = np.float32) -> np.ndarray:
    if aspect_ratio is None:
        aspect_ratio = width / height
    span_x = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5
    span_y = 1 / (1 + aspect_ratio ** 2) ** 0.5
    u = np.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, width, dtype=dtype)
    v = np.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height, height, dtype=dtype)
    u, v = np.meshgrid(u, v, indexing='xy')
    uv = np.stack([u, v], axis=-1)
    return uv


def focal_to_fov_numpy(focal: np.ndarray):
    return 2 * np.arctan(0.5 / focal)


def fov_to_focal_numpy(fov: np.ndarray):
    return 0.5 / np.tan(fov / 2)


def intrinsics_to_fov_numpy(intrinsics: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    fov_x = focal_to_fov_numpy(intrinsics[..., 0, 0])
    fov_y = focal_to_fov_numpy(intrinsics[..., 1, 1])
    return fov_x, fov_y


def point_map_to_depth_legacy_numpy(points: np.ndarray):
    height, width = points.shape[-3:-1]
    diagonal = (height ** 2 + width ** 2) ** 0.5
    uv = normalized_view_plane_uv_numpy(width, height, dtype=points.dtype)
    _, uv = np.broadcast_arrays(points[..., :2], uv)
    b = (uv * points[..., 2:]).reshape(*points.shape[:-3], -1)
    A = np.stack([points[..., :2], -uv], axis=-1).reshape(*points.shape[:-3], -1, 2)
    M = A.swapaxes(-2, -1) @ A
    solution = (np.linalg.inv(M + 1e-6 * np.eye(2)) @ (A.swapaxes(-2, -1) @ b[..., None])).squeeze(-1)
    focal, shift = solution
    depth = points[..., 2] + shift[..., None, None]
    fov_x = np.arctan(width / diagonal / focal) * 2
    fov_y = np.arctan(height / diagonal / focal) * 2
    return depth, fov_x, fov_y, shift


def solve_optimal_focal_shift(uv: np.ndarray, xyz: np.ndarray):
    "Solve `min |focal * xy / (z + shift) - uv|` with respect to shift and focal"
    from scipy.optimize import least_squares
    uv, xy, z = uv.reshape(-1, 2), xyz[..., :2].reshape(-1, 2), xyz[..., 2].reshape(-1)

    def fn(uv, xy, z, shift):
        xy_proj = xy / (z + shift)[:, None]
        f = (xy_proj * uv).sum() / np.square(xy_proj).sum()
        err = (f * xy_proj - uv).ravel()
        return err

    solution = least_squares(partial(fn, uv, xy, z), x0=0, ftol=1e-3, method='lm')
    optim_shift = solution['x'].squeeze().astype(np.float32)
    xy_proj = xy / (z + optim_shift)[:, None]
    optim_focal = (xy_proj * uv).sum() / np.square(xy_proj).sum()
    return optim_shift, optim_focal


def solve_optimal_shift(uv: np.ndarray, xyz: np.ndarray, focal: float):
    "Solve `min |focal * xy / (z + shift) - uv|` with respect to shift"
    from scipy.optimize import least_squares
    uv, xy, z = uv.reshape(-1, 2), xyz[..., :2].reshape(-1, 2), xyz[..., 2].reshape(-1)

    def fn(uv, xy, z, shift):
        xy_proj = xy / (z + shift)[:, None]
        err = (focal * xy_proj - uv).ravel()
        return err

    solution = least_squares(partial(fn, uv, xy, z), x0=0, ftol=1e-3, method='lm')
    optim_shift = solution['x'].squeeze().astype(np.float32)
    return optim_shift


def recover_focal_shift_numpy(points: np.ndarray, mask: np.ndarray = None, focal: float = None, downsample_size: Tuple[int, int] = (64, 64)):
    assert points.shape[-1] == 3, "Points should (H, W, 3)"
    height, width = points.shape[-3], points.shape[-2]

    uv = normalized_view_plane_uv_numpy(width=width, height=height)

    if mask is None:
        assert _cv2 is not None, "cv2 required for recover_focal_shift_numpy without mask"
        points_lr = _cv2.resize(points, downsample_size, interpolation=_cv2.INTER_LINEAR).reshape(-1, 3)
        uv_lr = _cv2.resize(uv, downsample_size, interpolation=_cv2.INTER_LINEAR).reshape(-1, 2)
    else:
        (points_lr, uv_lr), mask_lr = mask_aware_nearest_resize_numpy((points, uv), mask, downsample_size)

    if points_lr.size < 2:
        return 1., 0.

    if focal is None:
        focal, shift = solve_optimal_focal_shift(uv_lr, points_lr)
    else:
        shift = solve_optimal_shift(uv_lr, points_lr, focal)

    return focal, shift


def mask_aware_nearest_resize_numpy(
    inputs: Union[np.ndarray, Tuple[np.ndarray, ...], None],
    mask: np.ndarray,
    size: Tuple[int, int],
    return_index: bool = False
) -> Tuple[Union[np.ndarray, Tuple[np.ndarray, ...], None], np.ndarray, Tuple[np.ndarray, ...]]:
    height, width = mask.shape[-2:]
    target_width, target_height = size
    filter_h_f, filter_w_f = max(1, height / target_height), max(1, width / target_width)
    filter_h_i, filter_w_i = math.ceil(filter_h_f), math.ceil(filter_w_f)
    filter_size = filter_h_i * filter_w_i
    padding_h, padding_w = filter_h_i // 2 + 1, filter_w_i // 2 + 1

    uv = utils3d.numpy.image_pixel_center(width=width, height=height, dtype=np.float32)
    indices = np.arange(height * width, dtype=np.int32).reshape(height, width)
    padded_uv = np.full((height + 2 * padding_h, width + 2 * padding_w, 2), 0, dtype=np.float32)
    padded_uv[padding_h:padding_h + height, padding_w:padding_w + width] = uv
    padded_mask = np.full((*mask.shape[:-2], height + 2 * padding_h, width + 2 * padding_w), False, dtype=bool)
    padded_mask[..., padding_h:padding_h + height, padding_w:padding_w + width] = mask
    padded_indices = np.full((height + 2 * padding_h, width + 2 * padding_w), 0, dtype=np.int32)
    padded_indices[padding_h:padding_h + height, padding_w:padding_w + width] = indices
    windowed_uv = utils3d.numpy.sliding_window_2d(padded_uv, (filter_h_i, filter_w_i), 1, axis=(0, 1))
    windowed_mask = utils3d.numpy.sliding_window_2d(padded_mask, (filter_h_i, filter_w_i), 1, axis=(-2, -1))
    windowed_indices = utils3d.numpy.sliding_window_2d(padded_indices, (filter_h_i, filter_w_i), 1, axis=(0, 1))

    target_centers = utils3d.numpy.image_uv(width=target_width, height=target_height, dtype=np.float32) * np.array([width, height], dtype=np.float32)
    target_lefttop = target_centers - np.array((filter_w_f / 2, filter_h_f / 2), dtype=np.float32)
    target_window = np.round(target_lefttop).astype(np.int32) + np.array((padding_w, padding_h), dtype=np.int32)

    target_window_centers = windowed_uv[target_window[..., 1], target_window[..., 0], :, :, :].reshape(target_height, target_width, 2, filter_size)
    target_window_mask = windowed_mask[..., target_window[..., 1], target_window[..., 0], :, :].reshape(*mask.shape[:-2], target_height, target_width, filter_size)
    target_window_indices = windowed_indices[target_window[..., 1], target_window[..., 0], :, :].reshape(*([-1] * (mask.ndim - 2)), target_height, target_width, filter_size)

    dist = np.square(target_window_centers - target_centers[..., None])
    dist = dist[..., 0, :] + dist[..., 1, :]
    dist = np.where(target_window_mask, dist, np.inf)
    nearest_in_window = np.argmin(dist, axis=-1, keepdims=True)
    nearest_idx = np.take_along_axis(target_window_indices, nearest_in_window, axis=-1).squeeze(-1)
    nearest_i, nearest_j = nearest_idx // width, nearest_idx % width
    target_mask = np.any(target_window_mask, axis=-1)
    batch_indices = [np.arange(n).reshape([1] * i + [n] + [1] * (mask.ndim - i - 1)) for i, n in enumerate(mask.shape[:-2])]

    index = (*batch_indices, nearest_i, nearest_j)

    if inputs is None:
        outputs = None
    elif isinstance(inputs, np.ndarray):
        outputs = inputs[index]
    elif isinstance(inputs, Sequence):
        outputs = tuple(x[index] for x in inputs)
    else:
        raise ValueError(f'Invalid input type: {type(inputs)}')

    if return_index:
        return outputs, target_mask, index
    else:
        return outputs, target_mask


def mask_aware_area_resize_numpy(image: np.ndarray, mask: np.ndarray, target_width: int, target_height: int) -> Tuple[Tuple[np.ndarray, ...], np.ndarray]:
    height, width = mask.shape[-2:]
    if image.shape[-2:] == (height, width):
        omit_channel_dim = True
    else:
        omit_channel_dim = False
    if omit_channel_dim:
        image = image[..., None]
    image = np.where(mask[..., None], image, 0)

    filter_h_f, filter_w_f = max(1, height / target_height), max(1, width / target_width)
    filter_h_i, filter_w_i = math.ceil(filter_h_f) + 1, math.ceil(filter_w_f) + 1
    filter_size = filter_h_i * filter_w_i
    padding_h, padding_w = filter_h_i // 2 + 1, filter_w_i // 2 + 1

    uv = utils3d.numpy.image_pixel_center(width=width, height=height, dtype=np.float32)
    indices = np.arange(height * width, dtype=np.int32).reshape(height, width)
    padded_uv = np.full((height + 2 * padding_h, width + 2 * padding_w, 2), 0, dtype=np.float32)
    padded_uv[padding_h:padding_h + height, padding_w:padding_w + width] = uv
    padded_mask = np.full((*mask.shape[:-2], height + 2 * padding_h, width + 2 * padding_w), False, dtype=bool)
    padded_mask[..., padding_h:padding_h + height, padding_w:padding_w + width] = mask
    padded_indices = np.full((height + 2 * padding_h, width + 2 * padding_w), 0, dtype=np.int32)
    padded_indices[padding_h:padding_h + height, padding_w:padding_w + width] = indices
    windowed_uv = utils3d.numpy.sliding_window_2d(padded_uv, (filter_h_i, filter_w_i), 1, axis=(0, 1))
    windowed_mask = utils3d.numpy.sliding_window_2d(padded_mask, (filter_h_i, filter_w_i), 1, axis=(-2, -1))
    windowed_indices = utils3d.numpy.sliding_window_2d(padded_indices, (filter_h_i, filter_w_i), 1, axis=(0, 1))

    target_center = utils3d.numpy.image_uv(width=target_width, height=target_height, dtype=np.float32) * np.array([width, height], dtype=np.float32)
    target_lefttop = target_center - np.array((filter_w_f / 2, filter_h_f / 2), dtype=np.float32)
    target_bottomright = target_center + np.array((filter_w_f / 2, filter_h_f / 2), dtype=np.float32)
    target_window = np.floor(target_lefttop).astype(np.int32) + np.array((padding_w, padding_h), dtype=np.int32)

    target_window_centers = windowed_uv[target_window[..., 1], target_window[..., 0], :, :, :].reshape(target_height, target_width, 2, filter_size)
    target_window_mask = windowed_mask[..., target_window[..., 1], target_window[..., 0], :, :].reshape(*mask.shape[:-2], target_height, target_width, filter_size)
    target_window_indices = windowed_indices[target_window[..., 1], target_window[..., 0], :, :].reshape(target_height, target_width, filter_size)

    target_window_lefttop = np.maximum(target_window_centers - 0.5, target_lefttop[..., None])
    target_window_bottomright = np.minimum(target_window_centers + 0.5, target_bottomright[..., None])
    target_window_area = (target_window_bottomright - target_window_lefttop).clip(0, None)
    target_window_area = np.where(target_window_mask, target_window_area[..., 0, :] * target_window_area[..., 1, :], 0)

    target_window_image = image.reshape(*image.shape[:-3], height * width, -1)[..., target_window_indices, :].swapaxes(-2, -1)
    target_mask = np.sum(target_window_area, axis=-1) >= 0.25
    target_image = weighted_mean_numpy(target_window_image, target_window_area[..., None, :], axis=-1)

    if omit_channel_dim:
        target_image = target_image[..., 0]

    return target_image, target_mask


def norm3d(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.square(x[..., 0]) + np.square(x[..., 1]) + np.square(x[..., 2]))


def depth_occlusion_edge_numpy(depth: np.ndarray, mask: np.ndarray, kernel_size: int = 3, tol: float = 0.1):
    disp = np.where(mask, 1 / depth, 0)
    disp_pad = np.pad(disp, (kernel_size // 2, kernel_size // 2), constant_values=0)
    mask_pad = np.pad(mask, (kernel_size // 2, kernel_size // 2), constant_values=False)
    disp_window = utils3d.numpy.sliding_window_2d(disp_pad, (kernel_size, kernel_size), 1, axis=(-2, -1))
    mask_window = utils3d.numpy.sliding_window_2d(mask_pad, (kernel_size, kernel_size), 1, axis=(-2, -1))
    disp_mean = weighted_mean_numpy(disp_window, mask_window, axis=(-2, -1))
    fg_edge_mask = mask & (disp > (1 + tol) * disp_mean)
    bg_edge_mask = mask & (disp_mean > (1 + tol) * disp)
    return fg_edge_mask, bg_edge_mask


def disk_kernel(radius: int) -> np.ndarray:
    L = np.arange(-radius, radius + 1)
    X, Y = np.meshgrid(L, L)
    kernel = ((X**2 + Y**2) <= radius**2).astype(np.float32)
    kernel /= np.sum(kernel)
    return kernel


def disk_blur(image: np.ndarray, radius: int) -> np.ndarray:
    if radius == 0:
        return image
    kernel = disk_kernel(radius)
    if image.ndim == 2:
        blurred = fftconvolve(image, kernel, mode='same')
    elif image.ndim == 3:
        channels = []
        for i in range(image.shape[2]):
            blurred_channel = fftconvolve(image[..., i], kernel, mode='same')
            channels.append(blurred_channel)
        blurred = np.stack(channels, axis=-1)
    else:
        raise ValueError("Image must be 2D or 3D.")
    return blurred


def depth_of_field(
    img: np.ndarray,
    disp: np.ndarray,
    focus_disp: float,
    max_blur_radius: int = 10,
) -> np.ndarray:
    assert _cv2 is not None, "cv2 required for depth_of_field"
    max_disp = np.max(disp)
    disp = disp / max_disp
    focus_disp = focus_disp / max_disp
    dilated_disp = []
    for radius in range(max_blur_radius + 1):
        dilated_disp.append(_cv2.dilate(disp, _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1)), iterations=1))
    blur_radii = np.clip(abs(disp - focus_disp) * max_blur_radius, 0, max_blur_radius).astype(np.int32)
    for radius in range(max_blur_radius + 1):
        dialted_blur_radii = np.clip(abs(dilated_disp[radius] - focus_disp) * max_blur_radius, 0, max_blur_radius).astype(np.int32)
        mask = (dialted_blur_radii >= radius) & (dialted_blur_radii >= blur_radii) & (dilated_disp[radius] > disp)
        blur_radii[mask] = dialted_blur_radii[mask]
    blur_radii = np.clip(blur_radii, 0, max_blur_radius)
    blur_radii = _cv2.blur(blur_radii, (5, 5))
    unique_radii = np.unique(blur_radii)
    precomputed = {}
    for radius in range(max_blur_radius + 1):
        if radius not in unique_radii:
            continue
        precomputed[radius] = disk_blur(img, radius)
    output = np.zeros_like(img)
    for r in unique_radii:
        mask = blur_radii == r
        output[mask] = precomputed[r][mask]
    return output


# ===========================================================================
# Torch geometry utilities (from geometry_torch.py)
# ===========================================================================

def weighted_mean(x: torch.Tensor, w: torch.Tensor = None, dim: Union[int, torch.Size] = None, keepdim: bool = False, eps: float = 1e-7) -> torch.Tensor:
    if w is None:
        return x.mean(dim=dim, keepdim=keepdim)
    else:
        w = w.to(x.dtype)
        return (x * w).mean(dim=dim, keepdim=keepdim) / w.mean(dim=dim, keepdim=keepdim).add(eps)


def harmonic_mean(x: torch.Tensor, w: torch.Tensor = None, dim: Union[int, torch.Size] = None, keepdim: bool = False, eps: float = 1e-7) -> torch.Tensor:
    if w is None:
        return x.add(eps).reciprocal().mean(dim=dim, keepdim=keepdim).reciprocal()
    else:
        w = w.to(x.dtype)
        return weighted_mean(x.add(eps).reciprocal(), w, dim=dim, keepdim=keepdim, eps=eps).add(eps).reciprocal()


def geometric_mean(x: torch.Tensor, w: torch.Tensor = None, dim: Union[int, torch.Size] = None, keepdim: bool = False, eps: float = 1e-7) -> torch.Tensor:
    if w is None:
        return x.add(eps).log().mean(dim=dim).exp()
    else:
        w = w.to(x.dtype)
        return weighted_mean(x.add(eps).log(), w, dim=dim, keepdim=keepdim, eps=eps).exp()


def normalized_view_plane_uv(width: int, height: int, aspect_ratio: float = None, dtype: torch.dtype = None, device: torch.device = None) -> torch.Tensor:
    if aspect_ratio is None:
        aspect_ratio = width / height
    span_x = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5
    span_y = 1 / (1 + aspect_ratio ** 2) ** 0.5
    u = torch.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, width, dtype=dtype, device=device)
    v = torch.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height, height, dtype=dtype, device=device)
    u, v = torch.meshgrid(u, v, indexing='xy')
    uv = torch.stack([u, v], dim=-1)
    return uv


def gaussian_blur_2d(input: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    kernel = torch.exp(-(torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=input.dtype, device=input.device) ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    kernel = (kernel[:, None] * kernel[None, :]).reshape(1, 1, kernel_size, kernel_size)
    input = F.pad(input, (kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2), mode='replicate')
    input = F.conv2d(input, kernel, groups=input.shape[1])
    return input


def focal_to_fov(focal: torch.Tensor):
    return 2 * torch.atan(0.5 / focal)


def fov_to_focal(fov: torch.Tensor):
    return 0.5 / torch.tan(fov / 2)


def angle_diff_vec3(v1: torch.Tensor, v2: torch.Tensor, eps: float = 1e-12):
    return torch.atan2(torch.cross(v1, v2, dim=-1).norm(dim=-1) + eps, (v1 * v2).sum(dim=-1))


def intrinsics_to_fov(intrinsics: torch.Tensor):
    focal_x = intrinsics[..., 0, 0]
    focal_y = intrinsics[..., 1, 1]
    return 2 * torch.atan(0.5 / focal_x), 2 * torch.atan(0.5 / focal_y)


def point_map_to_depth_legacy(points: torch.Tensor):
    height, width = points.shape[-3:-1]
    diagonal = (height ** 2 + width ** 2) ** 0.5
    uv = normalized_view_plane_uv(width, height, dtype=points.dtype, device=points.device)
    b = (uv * points[..., 2:]).flatten(-3, -1)
    A = torch.stack([points[..., :2], -uv.expand_as(points[..., :2])], dim=-1).flatten(-4, -2)
    M = A.transpose(-2, -1) @ A
    solution = (torch.inverse(M + 1e-6 * torch.eye(2).to(A)) @ (A.transpose(-2, -1) @ b[..., None])).squeeze(-1)
    focal, shift = solution.unbind(-1)
    depth = points[..., 2] + shift[..., None, None]
    fov_x = torch.atan(width / diagonal / focal) * 2
    fov_y = torch.atan(height / diagonal / focal) * 2
    return depth, fov_x, fov_y, shift


def view_plane_uv_to_focal(uv: torch.Tensor):
    normed_uv = normalized_view_plane_uv(width=uv.shape[-2], height=uv.shape[-3], device=uv.device, dtype=uv.dtype)
    focal = (uv * normed_uv).sum() / uv.square().sum().add(1e-12)
    return focal


def recover_focal_shift(points: torch.Tensor, mask: torch.Tensor = None, focal: torch.Tensor = None, downsample_size: Tuple[int, int] = (64, 64)):
    shape = points.shape
    height, width = points.shape[-3], points.shape[-2]

    points = points.reshape(-1, *shape[-3:])
    mask = None if mask is None else mask.reshape(-1, *shape[-3:-1])
    focal = focal.reshape(-1) if focal is not None else None
    uv = normalized_view_plane_uv(width, height, dtype=points.dtype, device=points.device)

    points_lr = F.interpolate(points.permute(0, 3, 1, 2), downsample_size, mode='nearest').permute(0, 2, 3, 1)
    uv_lr = F.interpolate(uv.unsqueeze(0).permute(0, 3, 1, 2), downsample_size, mode='nearest').squeeze(0).permute(1, 2, 0)
    mask_lr = None if mask is None else F.interpolate(mask.to(torch.float32).unsqueeze(1), downsample_size, mode='nearest').squeeze(1) > 0

    uv_lr_np = uv_lr.cpu().numpy()
    points_lr_np = points_lr.detach().cpu().numpy()
    focal_np = focal.cpu().numpy() if focal is not None else None
    mask_lr_np = None if mask is None else mask_lr.cpu().numpy()
    optim_shift, optim_focal = [], []
    for i in range(points.shape[0]):
        points_lr_i_np = points_lr_np[i] if mask is None else points_lr_np[i][mask_lr_np[i]]
        uv_lr_i_np = uv_lr_np if mask is None else uv_lr_np[mask_lr_np[i]]
        if uv_lr_i_np.shape[0] < 2:
            optim_focal.append(1)
            optim_shift.append(0)
            continue
        if focal is None:
            optim_shift_i, optim_focal_i = solve_optimal_focal_shift(uv_lr_i_np, points_lr_i_np)
            optim_focal.append(float(optim_focal_i))
        else:
            optim_shift_i = solve_optimal_shift(uv_lr_i_np, points_lr_i_np, focal_np[i])
        optim_shift.append(float(optim_shift_i))
    optim_shift = torch.tensor(optim_shift, device=points.device, dtype=points.dtype).reshape(shape[:-3])

    if focal is None:
        optim_focal = torch.tensor(optim_focal, device=points.device, dtype=points.dtype).reshape(shape[:-3])
    else:
        optim_focal = focal.reshape(shape[:-3])

    return optim_focal, optim_shift


def mask_aware_nearest_resize(
    inputs: Union[torch.Tensor, Sequence[torch.Tensor], None],
    mask: torch.BoolTensor,
    size: Tuple[int, int],
    return_index: bool = False
) -> Tuple[Union[torch.Tensor, Sequence[torch.Tensor], None], torch.BoolTensor, Tuple[torch.LongTensor, ...]]:
    height, width = mask.shape[-2:]
    target_width, target_height = size
    device = mask.device
    filter_h_f, filter_w_f = max(1, height / target_height), max(1, width / target_width)
    filter_h_i, filter_w_i = math.ceil(filter_h_f), math.ceil(filter_w_f)
    filter_size = filter_h_i * filter_w_i
    padding_h, padding_w = filter_h_i // 2 + 1, filter_w_i // 2 + 1

    uv = utils3d.torch.image_pixel_center(width=width, height=height, dtype=torch.float32, device=device)
    indices = torch.arange(height * width, dtype=torch.long, device=device).reshape(height, width)
    padded_uv = torch.full((height + 2 * padding_h, width + 2 * padding_w, 2), 0, dtype=torch.float32, device=device)
    padded_uv[padding_h:padding_h + height, padding_w:padding_w + width] = uv
    padded_mask = torch.full((*mask.shape[:-2], height + 2 * padding_h, width + 2 * padding_w), False, dtype=torch.bool, device=device)
    padded_mask[..., padding_h:padding_h + height, padding_w:padding_w + width] = mask
    padded_indices = torch.full((height + 2 * padding_h, width + 2 * padding_w), 0, dtype=torch.long, device=device)
    padded_indices[padding_h:padding_h + height, padding_w:padding_w + width] = indices
    windowed_uv = utils3d.torch.sliding_window_2d(padded_uv, (filter_h_i, filter_w_i), 1, dim=(0, 1))
    windowed_mask = utils3d.torch.sliding_window_2d(padded_mask, (filter_h_i, filter_w_i), 1, dim=(-2, -1))
    windowed_indices = utils3d.torch.sliding_window_2d(padded_indices, (filter_h_i, filter_w_i), 1, dim=(0, 1))

    target_uv = utils3d.torch.image_uv(width=target_width, height=target_height, dtype=torch.float32, device=device) * torch.tensor([width, height], dtype=torch.float32, device=device)
    target_lefttop = target_uv - torch.tensor((filter_w_f / 2, filter_h_f / 2), dtype=torch.float32, device=device)
    target_window = torch.round(target_lefttop).long() + torch.tensor((padding_w, padding_h), dtype=torch.long, device=device)

    target_window_uv = windowed_uv[target_window[..., 1], target_window[..., 0], :, :, :].reshape(target_height, target_width, 2, filter_size)
    target_window_mask = windowed_mask[..., target_window[..., 1], target_window[..., 0], :, :].reshape(*mask.shape[:-2], target_height, target_width, filter_size)
    target_window_indices = windowed_indices[target_window[..., 1], target_window[..., 0], :, :].reshape(target_height, target_width, filter_size)
    target_window_indices = target_window_indices.expand_as(target_window_mask)

    dist = torch.where(target_window_mask, torch.norm(target_window_uv - target_uv[..., None], dim=-2), torch.inf)
    nearest = torch.argmin(dist, dim=-1, keepdim=True)
    nearest_idx = torch.gather(target_window_indices, index=nearest, dim=-1).squeeze(-1)
    target_mask = torch.any(target_window_mask, dim=-1)
    nearest_i, nearest_j = nearest_idx // width, nearest_idx % width
    batch_indices = [torch.arange(n, device=device).reshape([1] * i + [n] + [1] * (mask.dim() - i - 1)) for i, n in enumerate(mask.shape[:-2])]

    index = (*batch_indices, nearest_i, nearest_j)

    if inputs is None:
        outputs = None
    elif isinstance(inputs, torch.Tensor):
        outputs = inputs[index]
    elif isinstance(inputs, Sequence):
        outputs = tuple(x[index] for x in inputs)
    else:
        raise ValueError(f'Invalid input type: {type(inputs)}')

    if return_index:
        return outputs, target_mask, index
    else:
        return outputs, target_mask


def theshold_depth_change(depth: torch.Tensor, mask: torch.Tensor, pooler: Literal['min', 'max'], rtol: float = 0.2, kernel_size: int = 3):
    *batch_shape, height, width = depth.shape
    depth = depth.reshape(-1, 1, height, width)
    mask = mask.reshape(-1, 1, height, width)
    if pooler == 'max':
        pooled_depth = F.max_pool2d(torch.where(mask, depth, -torch.inf), kernel_size, stride=1, padding=kernel_size // 2)
        output_mask = pooled_depth > depth * (1 + rtol)
    elif pooler == 'min':
        pooled_depth = -F.max_pool2d(-torch.where(mask, depth, torch.inf), kernel_size, stride=1, padding=kernel_size // 2)
        output_mask = pooled_depth < depth * (1 - rtol)
    else:
        raise ValueError(f'Unsupported pooler: {pooler}')
    output_mask = output_mask.reshape(*batch_shape, height, width)
    return output_mask


def depth_occlusion_edge(depth: torch.FloatTensor, mask: torch.BoolTensor, kernel_size: int = 3, tol: float = 0.1):
    device, dtype = depth.device, depth.dtype
    disp = torch.where(mask, 1 / depth, 0)
    disp_pad = F.pad(disp, (kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2), value=0)
    mask_pad = F.pad(mask, (kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2), value=False)
    disp_window = utils3d.torch.sliding_window_2d(disp_pad, (kernel_size, kernel_size), 1, dim=(-2, -1))
    mask_window = utils3d.torch.sliding_window_2d(mask_pad, (kernel_size, kernel_size), 1, dim=(-2, -1))
    disp_mean = weighted_mean(disp_window, mask_window, dim=(-2, -1))
    fg_edge_mask = mask & (disp / disp_mean > 1 + tol)
    bg_edge_mask = mask & (disp_mean / disp > 1 + tol)
    fg_edge_mask = fg_edge_mask & F.max_pool2d(bg_edge_mask.float(), kernel_size + 2, stride=1, padding=kernel_size // 2 + 1).bool()
    bg_edge_mask = bg_edge_mask & F.max_pool2d(fg_edge_mask.float(), kernel_size + 2, stride=1, padding=kernel_size // 2 + 1).bool()
    return fg_edge_mask, bg_edge_mask


def dilate_with_mask(input: torch.Tensor, mask: torch.BoolTensor, filter: Literal['min', 'max', 'mean', 'median'] = 'mean', iterations: int = 1) -> torch.Tensor:
    kernel = torch.tensor([[False, True, False], [True, True, True], [False, True, False]], device=input.device, dtype=torch.bool)
    for _ in range(iterations):
        input_window = utils3d.torch.sliding_window_2d(F.pad(input, (1, 1, 1, 1), mode='constant', value=0), window_size=3, stride=1, dim=(-2, -1))
        mask_window = kernel & utils3d.torch.sliding_window_2d(F.pad(mask, (1, 1, 1, 1), mode='constant', value=False), window_size=3, stride=1, dim=(-2, -1))
        if filter == 'min':
            input = torch.where(mask, input, torch.where(mask_window, input_window, torch.inf).min(dim=(-2, -1)).values)
        elif filter == 'max':
            input = torch.where(mask, input, torch.where(mask_window, input_window, -torch.inf).max(dim=(-2, -1)).values)
        elif filter == 'mean':
            input = torch.where(mask, input, torch.where(mask_window, input_window, torch.nan).nanmean(dim=(-2, -1)))
        elif filter == 'median':
            input = torch.where(mask, input, torch.where(mask_window, input_window, torch.nan).flatten(-2).nanmedian(dim=-1).values)
        mask = mask_window.any(dim=(-2, -1))
    return input, mask
