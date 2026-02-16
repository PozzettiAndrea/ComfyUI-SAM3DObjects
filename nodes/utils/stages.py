"""
Pipeline stages for SAM3D inference.

This module contains all pipeline stages:
- Stage 1: Sparse structure generation
- Stage 2: SLAT generation
- Stage 3: Gaussian/Mesh decoding
- Texture baking

Each function loads its own models directly - no shared state.
"""

import gc
import logging
import sys
import base64
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger("sam3dobjects")

# Add vendor/ to sys.path BEFORE any sam3d_objects imports
# This ensures all imports resolve through the same path
_VENDOR_PATH = str(Path(__file__).parent.parent / "vendor")
if _VENDOR_PATH not in sys.path:
    sys.path.insert(0, _VENDOR_PATH)

# Pre-import spconv and modules that use it BEFORE hydra runs
# This prevents double-import crashes (pybind11 can only init once)
import spconv  # noqa: E402
import sam3d_objects.model.backbone.tdfy_dit.modules.sparse  # noqa: E402
import sam3d_objects.model.backbone.tdfy_dit.models  # noqa: E402

import numpy as np
import torch

# Patch spconv to support bfloat16 (missing in spconv <=2.3.8)
try:
    from cumm import tensorview as tv
    from spconv.pytorch import cppcore
    if torch.bfloat16 not in cppcore._TORCH_DTYPE_TO_TV:
        cppcore._TORCH_DTYPE_TO_TV[torch.bfloat16] = tv.bfloat16
        cppcore._TV_DTYPE_TO_TORCH[tv.bfloat16] = torch.bfloat16
        log.info("Patched spconv with bfloat16 support")
except Exception:
    pass  # Non-critical; bf16 just won't be available
import comfy.model_management

from .helpers import preprocess_image_lazy, save_output_to_disk

# Module-level model cache: key -> nn.Module (on GPU)
# comfy-env's SubprocessModelPatcher auto-detects GPU models and handles VRAM eviction.
_MODEL_CACHE = {}


# =============================================================================
# Config / Model Loading Helpers
# =============================================================================

def _load_config(config_path: str):
    """Load pipeline.yaml and return config + checkpoint_dir."""
    from omegaconf import OmegaConf
    config = OmegaConf.load(config_path)
    checkpoint_dir = Path(config_path).parent
    return config, checkpoint_dir


def _resolve_dtype(precision: str) -> torch.dtype:
    """Convert precision string to torch dtype.

    Args:
        precision: One of "bf16", "fp16", "fp32".
    """
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return torch.float32


def auto_detect_attn_backend() -> str:
    """Auto-detect best attention backend based on installed packages.

    Priority: flash_attn > xformers > sdpa (always available).
    Safe to call before vendor imports (only does __import__ checks).
    """
    for module in ["flash_attn", "xformers"]:
        try:
            __import__(module)
            log.info("Auto-detected attention backend: %s", module)
            return module
        except ImportError:
            continue
    log.info("Auto-detected attention backend: sdpa (no flash_attn or xformers found)")
    return "sdpa"


def _get_or_load_generator(config_path: str, generator_type: str, precision: str = "fp16"):
    """
    Load a generator model (ss_generator or slat_generator).

    Args:
        config_path: Path to pipeline.yaml
        generator_type: 'ss' or 'slat'
        precision: "bf16", "fp16", or "fp32"

    Returns:
        The generator model (on GPU)
    """
    cache_key = f"generator:{generator_type}"
    if cache_key in _MODEL_CACHE:
        log.debug("Using cached %s_generator", generator_type)
        return _MODEL_CACHE[cache_key]

    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    from sam3d_objects.model.io import load_model_from_checkpoint, filter_and_remove_prefix_state_dict_fn, try_instantiate_on_meta

    config, checkpoint_dir = _load_config(config_path)

    gen_config_path = checkpoint_dir / config[f"{generator_type}_generator_config_path"]
    gen_ckpt_path = checkpoint_dir / config[f"{generator_type}_generator_ckpt_path"]

    gen_config = OmegaConf.load(gen_config_path)
    model_config = gen_config["module"]["generator"]["backbone"]

    load_device = comfy.model_management.get_torch_device()
    model, use_assign = try_instantiate_on_meta(instantiate, model_config)
    model = load_model_from_checkpoint(
        model,
        str(gen_ckpt_path),
        strict=False,
        device=str(load_device),
        freeze=True,
        eval=True,
        state_dict_key="state_dict",
        state_dict_fn=filter_and_remove_prefix_state_dict_fn("_base_models.generator."),
        assign=use_assign,
    )

    model = model.to(_resolve_dtype(precision))

    _MODEL_CACHE[cache_key] = model
    return model


def _get_or_load_decoder(config_path: str, decoder_type: str, precision: str = "fp16"):
    """
    Load a decoder model (ss_decoder, slat_decoder_gs, slat_decoder_mesh).

    Args:
        config_path: Path to pipeline.yaml
        decoder_type: 'ss', 'slat_decoder_gs', 'slat_decoder_mesh'
        precision: "bf16", "fp16", or "fp32"

    Returns:
        The decoder model (on GPU)
    """
    cache_key = f"decoder:{decoder_type}"
    if cache_key in _MODEL_CACHE:
        log.debug("Using cached %s", decoder_type)
        return _MODEL_CACHE[cache_key]

    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    from sam3d_objects.model.io import load_model_from_checkpoint, remove_prefix_state_dict_fn, try_instantiate_on_meta

    config, checkpoint_dir = _load_config(config_path)

    if decoder_type == 'ss':
        dec_config_path = checkpoint_dir / config.ss_decoder_config_path
        dec_ckpt_path = checkpoint_dir / config.ss_decoder_ckpt_path
    else:
        dec_config_path = checkpoint_dir / config[f"{decoder_type}_config_path"]
        dec_ckpt_path = checkpoint_dir / config[f"{decoder_type}_ckpt_path"]

    dec_config = OmegaConf.load(dec_config_path)

    load_device = comfy.model_management.get_torch_device()
    model, use_assign = try_instantiate_on_meta(instantiate, dec_config)
    model = load_model_from_checkpoint(
        model,
        str(dec_ckpt_path),
        strict=False,
        device=str(load_device),
        freeze=True,
        eval=True,
        state_dict_key=None,  # Decoder checkpoints have weights at root level
        state_dict_fn=remove_prefix_state_dict_fn("module."),
        assign=use_assign,
    )

    model = model.to(_resolve_dtype(precision))

    _MODEL_CACHE[cache_key] = model
    return model


def _get_or_load_condition_embedder(config_path: str, embedder_type: str, precision: str = "fp16"):
    """
    Load a condition embedder (ss or slat).

    Args:
        config_path: Path to pipeline.yaml
        embedder_type: 'ss' or 'slat'
        precision: "bf16", "fp16", or "fp32"

    Returns:
        The embedder model (on GPU)
    """
    cache_key = f"embedder:{embedder_type}"
    if cache_key in _MODEL_CACHE:
        log.debug("Using cached %s_embedder", embedder_type)
        return _MODEL_CACHE[cache_key]

    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    from sam3d_objects.model.io import load_model_from_checkpoint, filter_and_remove_prefix_state_dict_fn, try_instantiate_on_meta

    config, checkpoint_dir = _load_config(config_path)

    gen_config_path = checkpoint_dir / config[f"{embedder_type}_generator_config_path"]
    gen_ckpt_path = checkpoint_dir / config[f"{embedder_type}_generator_ckpt_path"]

    gen_config = OmegaConf.load(gen_config_path)
    embedder_config = gen_config.module.condition_embedder.backbone

    load_device = comfy.model_management.get_torch_device()
    embedder, use_assign = try_instantiate_on_meta(instantiate, embedder_config)
    embedder = load_model_from_checkpoint(
        embedder,
        str(gen_ckpt_path),
        strict=False,
        device=str(load_device),
        freeze=True,
        eval=True,
        state_dict_key="state_dict",
        state_dict_fn=filter_and_remove_prefix_state_dict_fn("_base_models.condition_embedder."),
        assign=use_assign,
    )

    _reinit_dino_buffers(embedder)
    _share_dino_backbones(embedder)
    embedder = embedder.to(_resolve_dtype(precision))

    _MODEL_CACHE[cache_key] = embedder
    return embedder


def _get_preprocessor(config_path: str, preprocessor_type: str):
    """Get preprocessor from config."""
    from hydra.utils import instantiate

    config, _ = _load_config(config_path)
    preprocessor_config = config.get(f'{preprocessor_type}_preprocessor')

    if preprocessor_config:
        return instantiate(preprocessor_config)
    else:
        from sam3d_objects.pipeline import preprocess_utils
        return preprocess_utils.get_default_preprocessor()


def _unload(*models):
    """Delete models and free VRAM."""
    for m in models:
        if m is not None:
            del m
    gc.collect()
    torch.cuda.empty_cache()




def _reinit_dino_buffers(model):
    """Reinitialize Dino mean/std buffers after meta-device loading.

    Meta-device init creates these as empty tensors. They need correct
    ImageNet normalization values for proper preprocessing.
    """
    for module in model.modules():
        if type(module).__name__ == 'Dino' and hasattr(module, 'mean'):
            device = module.mean.device
            dtype = module.mean.dtype
            module.mean = torch.as_tensor([[0.485, 0.456, 0.406]]).view(-1, 1, 1).to(device=device, dtype=dtype)
            module.std = torch.as_tensor([[0.229, 0.224, 0.225]]).view(-1, 1, 1).to(device=device, dtype=dtype)


def _share_dino_backbones(model):
    """Share frozen DINOv2 backbone between Dino instances. Saves ~1.1GB VRAM.

    The EmbedderFuser has two Dino modules (one for image, one for mask)
    with identical frozen backbones loaded from the same pretrained weights.
    """
    if not hasattr(model, 'module_list'):
        return

    from sam3d_objects.model.backbone.dit.embedder.dino import Dino
    dino_modules = [m for m in model.module_list if isinstance(m, Dino)]

    if len(dino_modules) > 1:
        shared = dino_modules[0].backbone
        for dino in dino_modules[1:]:
            dino.backbone = shared
        log.info("Shared DINOv2 backbone across %d Dino instances (saved ~1.1GB)", len(dino_modules))
        gc.collect()
        torch.cuda.empty_cache()


def clear_model_cache():
    """Clear all cached models and free memory."""
    global _MODEL_CACHE
    _MODEL_CACHE.clear()
    gc.collect()
    torch.cuda.empty_cache()


# =============================================================================
# Pose Transformation Helpers
# =============================================================================

def _apply_pose_to_gaussian(gaussian, pose_data: Dict, device="cuda"):
    """
    Apply pose transformation (rotation, translation, scale) to a Gaussian object.
    """
    from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion, quaternion_multiply, Transform3d

    rotation = pose_data.get("rotation")
    translation = pose_data.get("translation")
    scale = pose_data.get("scale")

    if rotation is None or translation is None or scale is None:
        log.warning("Missing pose data, skipping Gaussian pose application")
        return gaussian

    # Convert to tensors
    if hasattr(rotation, 'cpu'):
        rotation = rotation.cpu()
    rotation = torch.tensor(rotation, dtype=torch.float32, device=device).squeeze()

    if hasattr(translation, 'cpu'):
        translation = translation.cpu()
    translation = torch.tensor(translation, dtype=torch.float32, device=device).squeeze()

    if hasattr(scale, 'cpu'):
        scale = scale.cpu()
    scale = torch.tensor(scale, dtype=torch.float32, device=device).squeeze()

    # Ensure correct shapes
    if rotation.dim() == 1:
        rotation = rotation.unsqueeze(0)
    if translation.dim() == 1:
        translation = translation.unsqueeze(0)
    if scale.dim() == 0:
        scale = scale.unsqueeze(0).expand(3)
    elif scale.dim() == 1 and scale.shape[0] == 1:
        scale = scale.expand(3)
    scale_val = scale.mean()

    # Build Transform3d: scale -> rotate -> translate
    rot_matrix = quaternion_to_matrix(rotation)
    tfm = (
        Transform3d(device=device)
        .scale(scale_val.expand(3)[None])
        .rotate(rot_matrix)
        .translate(translation)
    )

    # 1. Transform positions
    positions = gaussian.get_xyz
    positions_world = tfm.transform_points(positions.unsqueeze(0)).squeeze(0)
    gaussian.from_xyz(positions_world)

    # 2. Apply scale to Gaussian scaling (in log-space)
    log_scale = torch.log(scale_val).expand_as(gaussian._scaling)
    gaussian._scaling = gaussian._scaling + log_scale

    # 3. Compose rotations
    tfm_matrix = tfm.get_matrix()[0]
    rotation_matrix = tfm_matrix[:3, :3]
    scale_factors = rotation_matrix.norm(dim=0)
    pure_rotation_matrix = rotation_matrix / scale_factors[None, :]
    pose_rotation_quat = matrix_to_quaternion(pure_rotation_matrix[None])

    current_rotations = gaussian.get_rotation
    new_rotations = quaternion_multiply(pose_rotation_quat, current_rotations)
    gaussian.from_rotation(new_rotations)

    log.info(f"Applied pose to Gaussian: scale={scale_val:.4f}, trans={translation.squeeze().tolist()}")
    return gaussian


def _apply_pose_to_vertices(vertices: np.ndarray, pose_data: Dict) -> np.ndarray:
    """
    Apply pose transformation (rotation, translation, scale) to vertices.
    """
    from pytorch3d.transforms import quaternion_to_matrix

    rotation = pose_data.get("rotation")
    translation = pose_data.get("translation")
    scale = pose_data.get("scale")

    if rotation is None or translation is None or scale is None:
        log.warning("Missing pose data, skipping pose application")
        return vertices

    # Convert to numpy arrays
    if hasattr(rotation, 'cpu'):
        rotation = rotation.cpu().numpy()
    rotation = np.array(rotation).squeeze()

    if hasattr(translation, 'cpu'):
        translation = translation.cpu().numpy()
    translation = np.array(translation).squeeze()

    if hasattr(scale, 'cpu'):
        scale = scale.cpu().numpy()
    scale = np.array(scale).squeeze()

    # Convert quaternion (wxyz) to rotation matrix
    quat_tensor = torch.from_numpy(rotation.astype(np.float32)).unsqueeze(0)
    rot_matrix = quaternion_to_matrix(quat_tensor).squeeze(0).numpy()

    # Apply scale (use mean for uniform scaling)
    scale_val = scale.mean() if scale.ndim > 0 else float(scale)

    # Apply transformation: (R * S * v) + t
    vertices_transformed = (vertices @ (rot_matrix.T * scale_val)) + translation
    return vertices_transformed


# =============================================================================
# Stage 1: Sparse Structure Generation
# =============================================================================

def run_stage1(
    config_path: str,
    image,
    mask,
    pointmap,
    seed: int = 42,
    inference_steps: int = 25,
    cfg_strength: float = 7.0,
    cfg_strength_pm: float = 0.0,
    output_dir: str = None,
    memory: str = "cpu_offload",
    precision: str = "fp16",
) -> Dict[str, Any]:
    """
    Run Stage 1 (sparse structure generation).

    Models loaded: ss_generator (~6.7 GB), ss_condition_embedder (~1.2 GB), ss_decoder (~150 MB)
    Peak VRAM: ~8-9 GB
    """
    from sam3d_objects.pipeline.inference_utils import (
        downsample_sparse_structure,
        prune_sparse_structure,
        get_pose_decoder,
    )

    log.info("Running Stage 1 (sparse gen)")

    config, checkpoint_dir = _load_config(config_path)
    dtype = _resolve_dtype(precision)

    # Set seed
    torch.manual_seed(seed)

    # Convert image/mask to numpy
    image_np = np.array(image)
    mask_np = np.array(mask) if mask is not None else None

    # Ensure RGBA format - USE MASK AS ALPHA CHANNEL for proper cropping
    if image_np.ndim == 2:
        image_np = np.stack([image_np] * 3 + [np.full_like(image_np, 255)], axis=-1)
    elif image_np.shape[-1] == 3:
        if mask_np is not None:
            if mask_np.dtype == np.float32 or mask_np.dtype == np.float64:
                alpha = (mask_np * 255).astype(np.uint8)
            else:
                alpha = mask_np.astype(np.uint8)
            if alpha.ndim == 2:
                alpha = alpha[:, :, np.newaxis]
        else:
            alpha = np.full((image_np.shape[0], image_np.shape[1], 1), 255, dtype=np.uint8)
        image_np = np.concatenate([image_np, alpha], axis=-1)

    # Get preprocessor and preprocess image
    log.info("Preprocessing image...")
    ss_preprocessor = _get_preprocessor(config_path, 'ss')

    # Convert pointmap to tensor for preprocessing
    if isinstance(pointmap, np.ndarray):
        pointmap_tensor = torch.from_numpy(pointmap).float()
    else:
        pointmap_tensor = pointmap

    # Preprocess
    ss_input_dict = preprocess_image_lazy(image_np, mask_np, ss_preprocessor, pointmap=pointmap_tensor)

    # Save debug image
    debug_image_path = None
    if output_dir:
        try:
            from PIL import Image as PILImage
            debug_img = ss_input_dict.get("image")
            if debug_img is not None:
                if debug_img.dim() == 4:
                    debug_img = debug_img[0]
                debug_img_np = debug_img.cpu().permute(1, 2, 0).numpy()
                debug_img_np = (debug_img_np * 255).clip(0, 255).astype(np.uint8)
                debug_pil = PILImage.fromarray(debug_img_np)
                debug_image_path = str(Path(output_dir) / "debug_preprocessed_stage1.png")
                debug_pil.save(debug_image_path)
                log.debug("Saved debug image: %s", debug_image_path)
        except Exception as e:
            log.warning("Failed to save debug image: %s", e)

    # Store pointmap scale/shift for pose decoding
    pointmap_scale = ss_input_dict.get("pointmap_scale", None)
    pointmap_shift = ss_input_dict.get("pointmap_shift", None)

    # Load models (cached on GPU; comfy-env handles VRAM eviction)
    log.info("Loading Stage 1 models...")
    ss_generator = _get_or_load_generator(config_path, 'ss', precision)
    ss_decoder = _get_or_load_decoder(config_path, 'ss', precision)
    ss_embedder = _get_or_load_condition_embedder(config_path, 'ss', precision)

    # Configure generator (match original override_ss_generator_cfg_config)
    ss_generator.no_shortcut = True
    ss_generator.reverse_fn.strength = cfg_strength
    ss_generator.reverse_fn.strength_pm = cfg_strength_pm
    ss_generator.inference_steps = inference_steps
    # Critical settings from original that were missing:
    ss_generator.reverse_fn.interval = getattr(config, 'ss_cfg_interval', [0, 500])
    ss_generator.rescale_t = getattr(config, 'ss_rescale_t', 3)
    ss_generator.reverse_fn.backbone.condition_embedder.normalize_images = True
    ss_generator.reverse_fn.unconditional_handling = "add_flag"

    log.info("Running sparse structure generation...")

    downsample_ss_dist = getattr(config, 'downsample_ss_dist', 0)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=dtype):
            image_tensor = ss_input_dict["image"]
            bs = image_tensor.shape[0]

            # Check if MM-DiT architecture
            has_latent_mapping = hasattr(ss_generator.reverse_fn.backbone, "latent_mapping")

            if has_latent_mapping:
                latent_shape_dict = {
                    k: (bs,) + (v.pos_emb.shape[0], v.input_layer.in_features)
                    for k, v in ss_generator.reverse_fn.backbone.latent_mapping.items()
                }
            else:
                latent_shape_dict = (bs,) + (4096, 8)

            # Get condition input mapping from config
            ss_condition_input_mapping = getattr(config, 'ss_condition_input_mapping', ['image'])

            # Get condition embeddings
            condition_args = [ss_input_dict[k] for k in ss_condition_input_mapping if k in ss_input_dict]
            condition_kwargs = {k: v for k, v in ss_input_dict.items() if k not in ss_condition_input_mapping}

            # Embed conditions
            if ss_embedder is not None and len(condition_args) > 0:
                tokens = ss_embedder(*condition_args, **condition_kwargs)
                condition_args = (tokens,)
                condition_kwargs = {}
            elif ss_embedder is not None:
                tokens = ss_embedder(**condition_kwargs)
                condition_args = (tokens,)
                condition_kwargs = {}

            # Run generator
            return_dict = ss_generator(
                latent_shape_dict,
                image_tensor.device,
                *condition_args,
                **condition_kwargs,
            )

            if not has_latent_mapping:
                return_dict = {"shape": return_dict}

            shape_latent = return_dict["shape"]

            # Decode sparse structure
            ss = ss_decoder(
                shape_latent.permute(0, 2, 1)
                .contiguous()
                .view(shape_latent.shape[0], 8, 16, 16, 16)
            )
            coords = torch.argwhere(ss > 0)[:, [0, 2, 3, 4]].int()

            # Downsample output
            return_dict["coords_original"] = coords
            original_shape = coords.shape

            if downsample_ss_dist > 0:
                coords = prune_sparse_structure(
                    coords,
                    max_neighbor_axes_dist=downsample_ss_dist,
                )

            coords, downsample_factor = downsample_sparse_structure(coords)
            log.debug("Downsampled coords from %d to %d", original_shape[0], coords.shape[0])

            return_dict["coords"] = coords
            return_dict["downsample_factor"] = downsample_factor

    # Apply pose decoding
    log.info("Decoding pose...")
    pose_decoder_name = getattr(config, 'pose_decoder_name', 'default')
    pose_decoder = get_pose_decoder(pose_decoder_name)
    pose_result = pose_decoder(
        return_dict,
        scene_scale=pointmap_scale,
        scene_shift=pointmap_shift,
    )
    return_dict.update(pose_result)

    # Rescale after downsampling
    return_dict["scale"] = return_dict["scale"] * return_dict["downsample_factor"]

    # Add voxel coordinates
    return_dict["voxel"] = return_dict["coords"][:, 1:] / 64 - 0.5

    log.info("Stage 1 complete")

    # Save sparse structure
    if output_dir:
        save_dir = Path(output_dir)
    else:
        import tempfile
        save_dir = Path(tempfile.mkdtemp())
    save_dir.mkdir(parents=True, exist_ok=True)

    sparse_path = save_dir / "sparse_structure.pt"
    torch.save(return_dict, sparse_path)
    log.info("Saved sparse structure: %s", sparse_path)

    saved_output = {
        "output_dir": str(save_dir),
        "files": {"sparse_structure": str(sparse_path)},
        "metadata": {}
    }
    if debug_image_path:
        saved_output["files"]["debug_image"] = debug_image_path

    # Extract pose data for direct access
    rotation = return_dict.get("rotation")
    translation = return_dict.get("translation")
    scale = return_dict.get("scale")

    # Convert tensors to lists for JSON serialization
    if rotation is not None and hasattr(rotation, 'tolist'):
        rotation = rotation.cpu().tolist() if hasattr(rotation, 'cpu') else rotation.tolist()
    if translation is not None and hasattr(translation, 'tolist'):
        translation = translation.cpu().tolist() if hasattr(translation, 'cpu') else translation.tolist()
    if scale is not None and hasattr(scale, 'tolist'):
        scale = scale.cpu().tolist() if hasattr(scale, 'cpu') else scale.tolist()

    return {
        "status": "success",
        "stage1_mode": True,
        "output": saved_output,
        "rotation": rotation,
        "translation": translation,
        "scale": scale,
        "debug_image": debug_image_path,
    }


# =============================================================================
# Stage 2: SLAT Generation
# =============================================================================

def run_stage2(
    config_path: str,
    image,
    mask,
    stage1_output: Dict,
    seed: int = 42,
    inference_steps: int = 25,
    cfg_strength: float = 5.0,
    output_dir: str = None,
    memory: str = "cpu_offload",
    precision: str = "fp16",
) -> Dict[str, Any]:
    """
    Run Stage 2 (SLAT generation).

    Models loaded: slat_generator (~4.9 GB), slat_condition_embedder (~1.2 GB)
    Peak VRAM: ~6-7 GB
    """
    from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp

    log.info("Running Stage 2 (SLAT gen)")

    config, checkpoint_dir = _load_config(config_path)
    dtype = _resolve_dtype(precision)

    # Set seed
    torch.manual_seed(seed)

    # Convert image/mask to numpy
    image_np = np.array(image)
    mask_np = np.array(mask) if mask is not None else None

    # Ensure RGBA format
    if image_np.ndim == 2:
        image_np = np.stack([image_np] * 3 + [np.full_like(image_np, 255)], axis=-1)
    elif image_np.shape[-1] == 3:
        if mask_np is not None:
            if mask_np.dtype == np.float32 or mask_np.dtype == np.float64:
                alpha = (mask_np * 255).astype(np.uint8)
            else:
                alpha = mask_np.astype(np.uint8)
            if alpha.ndim == 2:
                alpha = alpha[:, :, np.newaxis]
        else:
            alpha = np.full((image_np.shape[0], image_np.shape[1], 1), 255, dtype=np.uint8)
        image_np = np.concatenate([image_np, alpha], axis=-1)

    # Get preprocessor and preprocess image
    log.info("Preprocessing image for SLAT...")
    slat_preprocessor = _get_preprocessor(config_path, 'slat')
    slat_input_dict = preprocess_image_lazy(image_np, mask_np, slat_preprocessor)

    # Get coords from stage1_output
    coords = stage1_output.get("coords")
    if isinstance(coords, str):
        coords = pickle.loads(base64.b64decode(coords))
    if isinstance(coords, np.ndarray):
        coords = torch.from_numpy(coords).int()
    coords = coords.cuda()

    # Load models (cached on GPU; comfy-env handles VRAM eviction)
    log.info("Loading Stage 2 models...")
    slat_generator = _get_or_load_generator(config_path, 'slat', precision)
    slat_embedder = _get_or_load_condition_embedder(config_path, 'slat', precision)

    # Configure generator (match original override_slat_generator_cfg_config)
    slat_generator.no_shortcut = True
    # Read cfg_strength from config if available (config may override the default)
    slat_cfg = getattr(config, 'slat_cfg_strength', cfg_strength)
    slat_generator.reverse_fn.strength = slat_cfg
    slat_generator.inference_steps = inference_steps
    # Critical settings from original that were missing:
    slat_generator.reverse_fn.interval = getattr(config, 'slat_cfg_interval', [0, 500])
    slat_generator.rescale_t = getattr(config, 'slat_rescale_t', 3)

    log.info("Running SLAT generation...")

    # Get SLAT normalization stats
    slat_mean = torch.tensor(getattr(config, 'slat_mean', [0.0] * 8)).cuda()
    slat_std = torch.tensor(getattr(config, 'slat_std', [1.0] * 8)).cuda()

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=dtype):
            image_tensor = slat_input_dict["image"]
            DEVICE = image_tensor.device
            latent_shape = (image_tensor.shape[0],) + (coords.shape[0], 8)

            # Get condition input mapping from config
            slat_condition_input_mapping = getattr(config, 'slat_condition_input_mapping', ['image'])

            # Get condition embeddings
            condition_args = [slat_input_dict[k] for k in slat_condition_input_mapping if k in slat_input_dict]
            condition_kwargs = {k: v for k, v in slat_input_dict.items() if k not in slat_condition_input_mapping}

            # Embed conditions
            if slat_embedder is not None and len(condition_args) > 0:
                tokens = slat_embedder(*condition_args, **condition_kwargs)
                condition_args = (tokens,)
                condition_kwargs = {}
            elif slat_embedder is not None:
                tokens = slat_embedder(**condition_kwargs)
                condition_args = (tokens,)
                condition_kwargs = {}

            # Add coords to condition args
            condition_args = condition_args + (coords.cpu().numpy(),)

            # Run generator
            slat_feats = slat_generator(
                latent_shape, DEVICE, *condition_args, **condition_kwargs
            )

            # Create SparseTensor
            slat = sp.SparseTensor(
                coords=coords,
                feats=slat_feats[0],
            ).to(DEVICE)

            # Apply mean/std normalization
            slat = slat * slat_std + slat_mean

    log.info("Stage 2 complete")

    # Build output dict with SLAT for saving
    output_dict = {
        "slat": slat,
        "stage1_data": stage1_output,
    }

    # Save output to disk
    if output_dir:
        saved_output = save_output_to_disk(output_dict, Path(output_dir))
    else:
        import tempfile
        temp_dir = Path(tempfile.mkdtemp())
        slat_path = temp_dir / "slat.pt"
        torch.save(output_dict, slat_path)
        saved_output = {
            "output_dir": str(temp_dir),
            "files": {"slat": str(slat_path)},
            "metadata": {}
        }

    return {
        "status": "success",
        "stage2_mode": True,
        "output": saved_output,
    }


# =============================================================================
# Stage 3: Gaussian/Mesh Decoding
# =============================================================================

def run_decode(
    config_path: str,
    slat_data: Dict,
    decode_format: str = "gaussian",
    output_dir: str = None,
    with_postprocess: bool = False,
    simplify: float = 0.95,
    up_axis: str = "Y-up (standard)",
    world_coordinates: bool = False,
    memory: str = "cpu_offload",
    precision: str = "fp16",
) -> Dict[str, Any]:
    """
    Run Stage 3 (Gaussian or Mesh decoding).

    Models loaded: slat_decoder_gs (~170 MB) or slat_decoder_mesh (~364 MB)
    Peak VRAM: ~1-2 GB
    """
    import trimesh
    from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp

    log.info("Running decode (%s)", decode_format)

    # Extract slat - handle different input formats
    slat = None

    if isinstance(slat_data, sp.SparseTensor):
        slat = slat_data
    elif isinstance(slat_data, dict) and "slat" in slat_data:
        slat_inner = slat_data["slat"]
        if isinstance(slat_inner, sp.SparseTensor):
            slat = slat_inner
        elif isinstance(slat_inner, str):
            slat_inner = pickle.loads(base64.b64decode(slat_inner))
            if isinstance(slat_inner, sp.SparseTensor):
                slat = slat_inner
            else:
                coords = slat_inner["coords"]
                feats = slat_inner["feats"]
        elif isinstance(slat_inner, dict):
            coords = slat_inner["coords"]
            feats = slat_inner["feats"]
    elif isinstance(slat_data, dict):
        coords = slat_data.get("coords")
        feats = slat_data.get("feats")

    # If we don't have slat yet, reconstruct from coords/feats
    if slat is None:
        if isinstance(coords, str):
            coords = pickle.loads(base64.b64decode(coords))
        if isinstance(feats, str):
            feats = pickle.loads(base64.b64decode(feats))

        coords = torch.from_numpy(coords).int().cuda() if isinstance(coords, np.ndarray) else coords.int().cuda()
        feats = torch.from_numpy(feats).cuda() if isinstance(feats, np.ndarray) else feats.cuda()
        slat = sp.SparseTensor(coords=coords, feats=feats)
    else:
        slat = slat.cuda()

    # Load decoder
    if decode_format == "gaussian":
        decoder_name = 'slat_decoder_gs'
    else:
        decoder_name = 'slat_decoder_mesh'

    log.info("Loading decoder (%s)...", decoder_name)
    decoder = _get_or_load_decoder(config_path, decoder_name, precision)

    log.info("Running decoder...")

    with torch.no_grad():
        output = decoder(slat)

    log.info("Decode complete")

    # Determine output directory
    if output_dir:
        save_dir = Path(output_dir)
    else:
        import tempfile
        save_dir = Path(tempfile.mkdtemp())
    save_dir.mkdir(parents=True, exist_ok=True)

    saved_files = {"output_dir": str(save_dir), "files": {}}

    if decode_format == "gaussian":
        gaussian = output[0] if isinstance(output, (list, tuple)) else output

        # Apply pose transformation
        pose_data = None
        if isinstance(slat_data, dict) and "stage1_data" in slat_data:
            stage1 = slat_data["stage1_data"]
            if isinstance(stage1, dict):
                pose_data = {
                    "rotation": stage1.get("rotation"),
                    "translation": stage1.get("translation"),
                    "scale": stage1.get("scale"),
                }

        if world_coordinates and pose_data is not None and pose_data.get("rotation") is not None:
            log.info("Applying pose transformation to Gaussian...")
            gaussian = _apply_pose_to_gaussian(gaussian, pose_data)
        elif not world_coordinates:
            log.info("Skipping pose (world_coordinates=False)")

        ply_path = save_dir / "gaussian.ply"
        try:
            transform = None
            if up_axis == "Y-up (standard)":
                transform = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
            gaussian.save_ply(str(ply_path), transform=transform)
            saved_files["files"]["ply"] = str(ply_path)
            log.info("Saved Gaussian PLY: %s", ply_path)
        except Exception as e:
            log.warning("Failed to save Gaussian PLY: %s", e)

        return {
            "status": "success",
            "stage2_mode": True,
            "output": saved_files,
            "file_output": saved_files,
        }
    else:
        # Mesh output
        mesh = output[0] if isinstance(output, (list, tuple)) else output

        try:
            vertices = mesh.vertices.cpu().numpy() if hasattr(mesh.vertices, 'cpu') else mesh.vertices
            faces = mesh.faces.cpu().numpy() if hasattr(mesh.faces, 'cpu') else mesh.faces

            # Get vertex colors BEFORE postprocessing
            original_vertex_colors = None
            if hasattr(mesh, 'vertex_attrs') and mesh.vertex_attrs is not None:
                if isinstance(mesh.vertex_attrs, torch.Tensor):
                    attrs = mesh.vertex_attrs.cpu().numpy()
                    original_vertex_colors = attrs[:, :3] if attrs.shape[-1] >= 3 else attrs
                elif isinstance(mesh.vertex_attrs, dict) and 'color' in mesh.vertex_attrs:
                    vc = mesh.vertex_attrs['color']
                    if hasattr(vc, 'cpu'):
                        original_vertex_colors = vc.cpu().numpy()
                    else:
                        original_vertex_colors = vc

            # Apply postprocessing if requested
            if with_postprocess:
                log.info("Applying mesh postprocessing (simplify=%s)...", simplify)
                from sam3d_objects.model.backbone.tdfy_dit.utils.postprocessing_utils import postprocess_mesh
                vertices, faces = postprocess_mesh(
                    vertices,
                    faces,
                    simplify=True,
                    simplify_ratio=simplify,
                    fill_holes=True,
                    verbose=True,
                )
                log.info("Postprocessing complete: %d vertices, %d faces", len(vertices), len(faces))
                original_vertex_colors = None

            # Process vertex colors if available
            vertex_colors = None
            if original_vertex_colors is not None:
                vc = original_vertex_colors
                if vc.max() <= 1.0:
                    vc = (vc * 255).astype(np.uint8)
                if vc.shape[-1] == 3:
                    alpha = np.full((vc.shape[0], 1), 255, dtype=np.uint8)
                    vc = np.concatenate([vc, alpha], axis=-1)
                vertex_colors = vc

            # Apply pose transformation
            pose_data = None
            if isinstance(slat_data, dict) and "stage1_data" in slat_data:
                stage1 = slat_data["stage1_data"]
                if isinstance(stage1, dict):
                    pose_data = {
                        "rotation": stage1.get("rotation"),
                        "translation": stage1.get("translation"),
                        "scale": stage1.get("scale"),
                    }

            if world_coordinates and pose_data is not None and pose_data.get("rotation") is not None:
                log.info("Applying pose transformation to world coordinates...")
                vertices = _apply_pose_to_vertices(vertices, pose_data)
            elif not world_coordinates:
                log.info("Skipping pose (world_coordinates=False)")
            else:
                log.info("No pose data found, mesh will be in normalized coordinates")

            # Transform from Z-up to Y-up if requested
            if up_axis == "Y-up (standard)":
                z_to_y_up = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
                vertices = vertices @ z_to_y_up

            # Create trimesh with vertex colors
            trimesh_mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                vertex_colors=vertex_colors,
                process=False
            )

            # Save GLB
            glb_path = save_dir / "mesh.glb"
            glb_bytes = trimesh_mesh.export(file_type="glb")
            with open(glb_path, 'wb') as f:
                f.write(glb_bytes)
            saved_files["files"]["glb"] = str(glb_path)
            log.info("Saved Mesh GLB: %s", glb_path)

        except Exception as e:
            log.error("Failed to save Mesh GLB: %s", e)
            import traceback
            traceback.print_exc(file=sys.stderr)
            mesh_path = save_dir / "mesh.pt"
            mesh_data = {
                "vertices": mesh.vertices.cpu() if hasattr(mesh.vertices, 'cpu') else mesh.vertices,
                "faces": mesh.faces.cpu() if hasattr(mesh.faces, 'cpu') else mesh.faces,
            }
            torch.save(mesh_data, mesh_path)
            saved_files["files"]["mesh_pt"] = str(mesh_path)

        return {
            "status": "success",
            "stage2_mode": True,
            "output": saved_files,
            "file_output": saved_files,
        }


# =============================================================================
# Texture Baking
# =============================================================================

def run_texture_bake_direct(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run texture baking directly without loading any models.

    Loads Gaussian from PLY and Mesh from GLB, then calls to_glb() for texture baking.
    """
    import trimesh

    log.info("Running direct texture baking")

    # Extract parameters
    ply_path = request["ply_path"]
    glb_path = request["glb_path"]
    output_dir = request["output_dir"]
    texture_mode = request.get("texture_mode", "opt")
    texture_size = request.get("texture_size", 1024)
    rendering_engine = request.get("rendering_engine", "nvdiffrast")

    log.info("PLY: %s", ply_path)
    log.info("GLB: %s", glb_path)
    log.info("Mode: %s, Size: %d", texture_mode, texture_size)

    from sam3d_objects.model.backbone.tdfy_dit.representations.gaussian import Gaussian
    from sam3d_objects.model.backbone.tdfy_dit.representations.mesh.cube2mesh import MeshExtractResult
    from sam3d_objects.model.backbone.tdfy_dit.utils.postprocessing_utils import to_glb

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Gaussian from PLY
    log.info("Loading Gaussian from PLY...")
    gaussian = Gaussian(
        aabb=[-1, -1, -1, 2, 2, 2],
        sh_degree=0,
        device=device
    )
    gaussian.load_ply(ply_path)
    log.info("Loaded Gaussian with %d points", gaussian._xyz.shape[0])

    # Load Mesh from GLB
    log.info("Loading Mesh from GLB...")
    loaded = trimesh.load(glb_path)

    if isinstance(loaded, trimesh.Scene):
        meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise RuntimeError("No mesh geometries found in GLB")
        trimesh_mesh = meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)
    else:
        trimesh_mesh = loaded

    log.info("Loaded mesh with %d vertices", len(trimesh_mesh.vertices))

    # Convert to MeshExtractResult format
    vertices_np = np.array(trimesh_mesh.vertices)
    vertices_tensor = torch.tensor(vertices_np, dtype=torch.float32, device=device)
    faces_tensor = torch.tensor(np.array(trimesh_mesh.faces), dtype=torch.long, device=device)

    if trimesh_mesh.visual is not None and hasattr(trimesh_mesh.visual, 'vertex_colors') and trimesh_mesh.visual.vertex_colors is not None:
        vertex_colors = np.array(trimesh_mesh.visual.vertex_colors)[:, :3] / 255.0
    else:
        vertex_colors = np.ones((len(trimesh_mesh.vertices), 3), dtype=np.float32)

    vertex_attrs_tensor = torch.tensor(vertex_colors, dtype=torch.float32, device=device)

    mesh = MeshExtractResult(
        vertices=vertices_tensor,
        faces=faces_tensor,
        vertex_attrs=vertex_attrs_tensor,
        res=64
    )

    # Run texture baking
    log.info("Running texture baking...")
    result_mesh = to_glb(
        gaussian,
        mesh,
        simplify=0,
        fill_holes=False,
        texture_size=texture_size,
        verbose=False,
        with_mesh_postprocess=False,
        with_texture_baking=True,
        rendering_engine=rendering_engine,
        texture_mode=texture_mode,
    )

    # Undo to_glb's Z->Y transform
    undo_transform = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    result_mesh.vertices = result_mesh.vertices @ undo_transform

    # Save textured GLB
    output_path = Path(output_dir) / "mesh_textured.glb"
    glb_bytes = result_mesh.export(file_type="glb")
    with open(output_path, 'wb') as f:
        f.write(glb_bytes)

    log.info("Saved textured GLB: %s", output_path)

    # Cleanup
    _unload(gaussian, mesh)

    return {
        "status": "success",
        "output": {
            "glb_path": str(output_path),
        }
    }


# =============================================================================
# Legacy Aliases (for backward compatibility during transition)
# =============================================================================

def run_stage1_lazy(lazy_manager, *args, **kwargs):
    """Legacy wrapper - extracts config_path from lazy_manager."""
    return run_stage1(lazy_manager.config_path, *args, **kwargs)

def run_stage2_lazy(lazy_manager, *args, **kwargs):
    """Legacy wrapper - extracts config_path from lazy_manager."""
    return run_stage2(lazy_manager.config_path, *args, **kwargs)

def run_decode_lazy(lazy_manager, *args, **kwargs):
    """Legacy wrapper - extracts config_path from lazy_manager."""
    return run_decode(lazy_manager.config_path, *args, **kwargs)

def run_depth_only(model_manager, image, unload_after=True, depth_backend="moge2"):
    """Legacy wrapper - depth is now handled directly in the node."""
    raise NotImplementedError("run_depth_only is deprecated. Load MoGe directly in the node.")
