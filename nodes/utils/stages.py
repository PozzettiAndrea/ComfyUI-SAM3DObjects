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
import sys
import base64
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

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

from .helpers import preprocess_image_lazy, save_output_to_disk
from . import model_cache


# =============================================================================
# Config / Model Loading Helpers
# =============================================================================

def _load_config(config_path: str):
    """Load pipeline.yaml and return config + checkpoint_dir."""
    from omegaconf import OmegaConf
    config = OmegaConf.load(config_path)
    checkpoint_dir = Path(config_path).parent
    return config, checkpoint_dir


def _get_dtype(config):
    """Get torch dtype from config."""
    dtype_str = getattr(config, 'dtype', 'float16')
    if dtype_str == 'bfloat16':
        return torch.bfloat16
    elif dtype_str == 'float16':
        return torch.float16
    return torch.float32


def _load_generator(config_path: str, generator_type: str):
    """
    Load a generator model (ss_generator or slat_generator).

    Args:
        config_path: Path to pipeline.yaml
        generator_type: 'ss' or 'slat'

    Returns:
        Loaded generator model on GPU
    """
    cache_key = f"generator:{generator_type}"
    cached = model_cache.try_load(cache_key)
    if cached is not None:
        print(f"[Worker] Using cached {generator_type}_generator", file=sys.stderr)
        return cached

    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    from sam3d_objects.model.io import load_model_from_checkpoint, filter_and_remove_prefix_state_dict_fn, try_instantiate_on_meta

    config, checkpoint_dir = _load_config(config_path)

    gen_config_path = checkpoint_dir / config[f"{generator_type}_generator_config_path"]
    gen_ckpt_path = checkpoint_dir / config[f"{generator_type}_generator_ckpt_path"]

    gen_config = OmegaConf.load(gen_config_path)
    model_config = gen_config["module"]["generator"]["backbone"]

    model, use_assign = try_instantiate_on_meta(instantiate, model_config)
    model = load_model_from_checkpoint(
        model,
        str(gen_ckpt_path),
        strict=False,
        device="cpu",
        freeze=True,
        eval=True,
        state_dict_key="state_dict",
        state_dict_fn=filter_and_remove_prefix_state_dict_fn("_base_models.generator."),
        assign=use_assign,
    )

    model = model.cuda().to(_get_dtype(config))
    return model


def _load_decoder(config_path: str, decoder_type: str):
    """
    Load a decoder model (ss_decoder, slat_decoder_gs, slat_decoder_mesh).

    Args:
        config_path: Path to pipeline.yaml
        decoder_type: 'ss', 'slat_decoder_gs', 'slat_decoder_mesh'

    Returns:
        Loaded decoder model on GPU
    """
    cache_key = f"decoder:{decoder_type}"
    cached = model_cache.try_load(cache_key)
    if cached is not None:
        print(f"[Worker] Using cached {decoder_type}", file=sys.stderr)
        return cached

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

    model, use_assign = try_instantiate_on_meta(instantiate, dec_config)
    model = load_model_from_checkpoint(
        model,
        str(dec_ckpt_path),
        strict=False,
        device="cpu",
        freeze=True,
        eval=True,
        state_dict_key=None,  # Decoder checkpoints have weights at root level
        state_dict_fn=remove_prefix_state_dict_fn("module."),
        assign=use_assign,
    )

    model = model.cuda()
    return model


def _load_condition_embedder(config_path: str, embedder_type: str):
    """
    Load a condition embedder (ss or slat).

    Args:
        config_path: Path to pipeline.yaml
        embedder_type: 'ss' or 'slat'

    Returns:
        Loaded embedder on GPU
    """
    cache_key = f"embedder:{embedder_type}"
    cached = model_cache.try_load(cache_key)
    if cached is not None:
        print(f"[Worker] Using cached {embedder_type}_embedder", file=sys.stderr)
        return cached

    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    from sam3d_objects.model.io import load_model_from_checkpoint, filter_and_remove_prefix_state_dict_fn, try_instantiate_on_meta

    config, checkpoint_dir = _load_config(config_path)

    gen_config_path = checkpoint_dir / config[f"{embedder_type}_generator_config_path"]
    gen_ckpt_path = checkpoint_dir / config[f"{embedder_type}_generator_ckpt_path"]

    gen_config = OmegaConf.load(gen_config_path)
    embedder_config = gen_config.module.condition_embedder.backbone

    embedder, use_assign = try_instantiate_on_meta(instantiate, embedder_config)
    embedder = load_model_from_checkpoint(
        embedder,
        str(gen_ckpt_path),
        strict=False,
        device="cpu",
        freeze=True,
        eval=True,
        state_dict_key="state_dict",
        state_dict_fn=filter_and_remove_prefix_state_dict_fn("_base_models.condition_embedder."),
        assign=use_assign,
    )

    embedder = embedder.cuda().to(_get_dtype(config))
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


def _offload_models(memory_mode, **named_models):
    """Offload models according to memory strategy.

    Args:
        memory_mode: 'cache_gpu', 'cpu_offload', or 'delete'
        **named_models: cache_key=model pairs
    """
    for key, model in named_models.items():
        if model is not None:
            model_cache.offload(key, model, memory_mode)
    if memory_mode != "cache_gpu":
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
        print(f"[Worker] Warning: Missing pose data, skipping Gaussian pose application", file=sys.stderr)
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

    print(f"[Worker] Applied pose to Gaussian: scale={scale_val:.4f}, trans={translation.squeeze().tolist()}", file=sys.stderr)
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
        print(f"[Worker] Warning: Missing pose data, skipping pose application", file=sys.stderr)
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

    print(f"[Worker] Running Stage 1 (sparse gen)", file=sys.stderr)

    config, checkpoint_dir = _load_config(config_path)
    dtype = _get_dtype(config)

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
    print(f"[Worker] Preprocessing image...", file=sys.stderr)
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
                print(f"[Worker] Saved debug image: {debug_image_path}", file=sys.stderr)
        except Exception as e:
            print(f"[Worker] Failed to save debug image: {e}", file=sys.stderr)

    # Store pointmap scale/shift for pose decoding
    pointmap_scale = ss_input_dict.get("pointmap_scale", None)
    pointmap_shift = ss_input_dict.get("pointmap_shift", None)

    # Load models
    print(f"[Worker] Loading Stage 1 models...", file=sys.stderr)
    ss_generator = _load_generator(config_path, 'ss')
    ss_decoder = _load_decoder(config_path, 'ss')
    ss_embedder = _load_condition_embedder(config_path, 'ss')

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

    print(f"[Worker] Running sparse structure generation...", file=sys.stderr)

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
            print(f"[Worker] Downsampled coords from {original_shape[0]} to {coords.shape[0]}", file=sys.stderr)

            return_dict["coords"] = coords
            return_dict["downsample_factor"] = downsample_factor

    # Apply pose decoding
    print(f"[Worker] Decoding pose...", file=sys.stderr)
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

    # Offload models
    print(f"[Worker] Offloading Stage 1 models (mode={memory})...", file=sys.stderr)
    _offload_models(
        memory,
        **{"generator:ss": ss_generator, "decoder:ss": ss_decoder, "embedder:ss": ss_embedder},
    )

    print(f"[Worker] Stage 1 complete", file=sys.stderr)

    # Save sparse structure
    if output_dir:
        save_dir = Path(output_dir)
    else:
        import tempfile
        save_dir = Path(tempfile.mkdtemp())
    save_dir.mkdir(parents=True, exist_ok=True)

    sparse_path = save_dir / "sparse_structure.pt"
    torch.save(return_dict, sparse_path)
    print(f"[Worker] Saved sparse structure: {sparse_path}", file=sys.stderr)

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
) -> Dict[str, Any]:
    """
    Run Stage 2 (SLAT generation).

    Models loaded: slat_generator (~4.9 GB), slat_condition_embedder (~1.2 GB)
    Peak VRAM: ~6-7 GB
    """
    from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp

    print(f"[Worker] Running Stage 2 (SLAT gen)", file=sys.stderr)

    config, checkpoint_dir = _load_config(config_path)
    dtype = _get_dtype(config)

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
    print(f"[Worker] Preprocessing image for SLAT...", file=sys.stderr)
    slat_preprocessor = _get_preprocessor(config_path, 'slat')
    slat_input_dict = preprocess_image_lazy(image_np, mask_np, slat_preprocessor)

    # Get coords from stage1_output
    coords = stage1_output.get("coords")
    if isinstance(coords, str):
        coords = pickle.loads(base64.b64decode(coords))
    if isinstance(coords, np.ndarray):
        coords = torch.from_numpy(coords).int()
    coords = coords.cuda()

    # Load models
    print(f"[Worker] Loading Stage 2 models...", file=sys.stderr)
    slat_generator = _load_generator(config_path, 'slat')
    slat_embedder = _load_condition_embedder(config_path, 'slat')

    # Configure generator (match original override_slat_generator_cfg_config)
    slat_generator.no_shortcut = True
    # Read cfg_strength from config if available (config may override the default)
    slat_cfg = getattr(config, 'slat_cfg_strength', cfg_strength)
    slat_generator.reverse_fn.strength = slat_cfg
    slat_generator.inference_steps = inference_steps
    # Critical settings from original that were missing:
    slat_generator.reverse_fn.interval = getattr(config, 'slat_cfg_interval', [0, 500])
    slat_generator.rescale_t = getattr(config, 'slat_rescale_t', 3)

    print(f"[Worker] Running SLAT generation...", file=sys.stderr)

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

    # Offload models
    print(f"[Worker] Offloading Stage 2 models (mode={memory})...", file=sys.stderr)
    _offload_models(
        memory,
        **{"generator:slat": slat_generator, "embedder:slat": slat_embedder},
    )

    print(f"[Worker] Stage 2 complete", file=sys.stderr)

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
) -> Dict[str, Any]:
    """
    Run Stage 3 (Gaussian or Mesh decoding).

    Models loaded: slat_decoder_gs (~170 MB) or slat_decoder_mesh (~364 MB)
    Peak VRAM: ~1-2 GB
    """
    import trimesh
    from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp

    print(f"[Worker] Running decode ({decode_format})", file=sys.stderr)

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

    print(f"[Worker] Loading decoder ({decoder_name})...", file=sys.stderr)
    decoder = _load_decoder(config_path, decoder_name)

    print(f"[Worker] Running decoder...", file=sys.stderr)

    with torch.no_grad():
        output = decoder(slat)

    # Offload decoder
    print(f"[Worker] Offloading decoder (mode={memory})...", file=sys.stderr)
    _offload_models(memory, **{f"decoder:{decoder_name}": decoder})

    print(f"[Worker] Decode complete", file=sys.stderr)

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
            print(f"[Worker] Applying pose transformation to Gaussian...", file=sys.stderr)
            gaussian = _apply_pose_to_gaussian(gaussian, pose_data)
        elif not world_coordinates:
            print(f"[Worker] Skipping pose (world_coordinates=False)", file=sys.stderr)

        ply_path = save_dir / "gaussian.ply"
        try:
            transform = None
            if up_axis == "Y-up (standard)":
                transform = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
            gaussian.save_ply(str(ply_path), transform=transform)
            saved_files["files"]["ply"] = str(ply_path)
            print(f"[Worker] Saved Gaussian PLY: {ply_path}", file=sys.stderr)
        except Exception as e:
            print(f"[Worker] Warning: Failed to save Gaussian PLY: {e}", file=sys.stderr)

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
                print(f"[Worker] Applying mesh postprocessing (simplify={simplify})...", file=sys.stderr)
                from sam3d_objects.model.backbone.tdfy_dit.utils.postprocessing_utils import postprocess_mesh
                vertices, faces = postprocess_mesh(
                    vertices,
                    faces,
                    simplify=True,
                    simplify_ratio=simplify,
                    fill_holes=True,
                    verbose=True,
                )
                print(f"[Worker] Postprocessing complete: {len(vertices)} vertices, {len(faces)} faces", file=sys.stderr)
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
                print(f"[Worker] Applying pose transformation to world coordinates...", file=sys.stderr)
                vertices = _apply_pose_to_vertices(vertices, pose_data)
            elif not world_coordinates:
                print(f"[Worker] Skipping pose (world_coordinates=False)", file=sys.stderr)
            else:
                print(f"[Worker] No pose data found, mesh will be in normalized coordinates", file=sys.stderr)

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
            print(f"[Worker] Saved Mesh GLB: {glb_path}", file=sys.stderr)

        except Exception as e:
            print(f"[Worker] Warning: Failed to save Mesh GLB: {e}", file=sys.stderr)
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

    print("[Worker] Running direct texture baking", file=sys.stderr)

    # Extract parameters
    ply_path = request["ply_path"]
    glb_path = request["glb_path"]
    output_dir = request["output_dir"]
    texture_mode = request.get("texture_mode", "opt")
    texture_size = request.get("texture_size", 1024)
    rendering_engine = request.get("rendering_engine", "nvdiffrast")

    print(f"[Worker] PLY: {ply_path}", file=sys.stderr)
    print(f"[Worker] GLB: {glb_path}", file=sys.stderr)
    print(f"[Worker] Mode: {texture_mode}, Size: {texture_size}", file=sys.stderr)

    from sam3d_objects.model.backbone.tdfy_dit.representations.gaussian import Gaussian
    from sam3d_objects.model.backbone.tdfy_dit.representations.mesh.cube2mesh import MeshExtractResult
    from sam3d_objects.model.backbone.tdfy_dit.utils.postprocessing_utils import to_glb

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load Gaussian from PLY
    print(f"[Worker] Loading Gaussian from PLY...", file=sys.stderr)
    gaussian = Gaussian(
        aabb=[-1, -1, -1, 2, 2, 2],
        sh_degree=0,
        device=device
    )
    gaussian.load_ply(ply_path)
    print(f"[Worker] Loaded Gaussian with {gaussian._xyz.shape[0]} points", file=sys.stderr)

    # Load Mesh from GLB
    print(f"[Worker] Loading Mesh from GLB...", file=sys.stderr)
    loaded = trimesh.load(glb_path)

    if isinstance(loaded, trimesh.Scene):
        meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise RuntimeError("No mesh geometries found in GLB")
        trimesh_mesh = meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)
    else:
        trimesh_mesh = loaded

    print(f"[Worker] Loaded mesh with {len(trimesh_mesh.vertices)} vertices", file=sys.stderr)

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
    print(f"[Worker] Running texture baking...", file=sys.stderr)
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

    print(f"[Worker] Saved textured GLB: {output_path}", file=sys.stderr)

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
