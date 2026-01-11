"""
Pipeline stages for SAM3D inference with lazy loading.

This module contains:
- Stage 1: Sparse structure generation
- Stage 2: SLAT generation
- Stage 3: Gaussian/Mesh decoding
"""

import sys
import base64
import pickle
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import torch

from preprocessing import preprocess_image_lazy
from utils import save_output_to_disk


def _apply_pose_to_gaussian(gaussian, pose_data: Dict, device="cuda"):
    """
    Apply pose transformation (rotation, translation, scale) to a Gaussian object.

    Transforms the Gaussian's internal tensors (positions, rotations, scales)
    to world coordinates using the pose computed during Stage 1.

    Based on get_gs_transformed() from original SAM3D code.

    Args:
        gaussian: Gaussian object from decoder
        pose_data: Dict with 'rotation' (wxyz quaternion), 'translation', 'scale'
        device: Device for tensor operations

    Returns:
        Transformed Gaussian object (modified in place)
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
        rotation = rotation.unsqueeze(0)  # (1, 4)
    if translation.dim() == 1:
        translation = translation.unsqueeze(0)  # (1, 3)
    if scale.dim() == 0:
        scale = scale.unsqueeze(0).expand(3)  # (3,)
    elif scale.dim() == 1 and scale.shape[0] == 1:
        scale = scale.expand(3)
    scale_val = scale.mean()

    # Build Transform3d: scale -> rotate -> translate
    rot_matrix = quaternion_to_matrix(rotation)  # (1, 3, 3)
    tfm = (
        Transform3d(device=device)
        .scale(scale_val.expand(3)[None])
        .rotate(rot_matrix)
        .translate(translation)
    )

    # 1. Transform positions
    positions = gaussian.get_xyz  # (N, 3)
    positions_world = tfm.transform_points(positions.unsqueeze(0)).squeeze(0)
    gaussian.from_xyz(positions_world)

    # 2. Apply scale to Gaussian scaling (in log-space)
    # _scaling stores log(scale), so we add log(scale_factor)
    log_scale = torch.log(scale_val).expand_as(gaussian._scaling)
    gaussian._scaling = gaussian._scaling + log_scale

    # 3. Compose rotations: new_rotation = pose_rotation * current_rotation
    # Extract pure rotation from transform matrix (remove scale)
    tfm_matrix = tfm.get_matrix()[0]  # (4, 4)
    rotation_matrix = tfm_matrix[:3, :3]
    scale_factors = rotation_matrix.norm(dim=0)
    pure_rotation_matrix = rotation_matrix / scale_factors[None, :]
    pose_rotation_quat = matrix_to_quaternion(pure_rotation_matrix[None])  # (1, 4)

    current_rotations = gaussian.get_rotation  # (N, 4)
    new_rotations = quaternion_multiply(pose_rotation_quat, current_rotations)
    gaussian.from_rotation(new_rotations)

    print(f"[Worker] Applied pose to Gaussian: scale={scale_val:.4f}, trans={translation.squeeze().tolist()}", file=sys.stderr)

    return gaussian


def _apply_pose_to_vertices(vertices: np.ndarray, pose_data: Dict) -> np.ndarray:
    """
    Apply pose transformation (rotation, translation, scale) to vertices.

    Transforms normalized mesh vertices to world coordinates using the pose
    computed during Stage 1 (sparse structure generation).

    Args:
        vertices: Mesh vertices in normalized Z-up space, shape (N, 3)
        pose_data: Dict with 'rotation' (wxyz quaternion), 'translation', 'scale'

    Returns:
        Transformed vertices in world coordinates (still Z-up)
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
    # vertices @ rot_matrix.T applies rotation (equivalent to R @ v for each vertex)
    vertices_transformed = (vertices @ (rot_matrix.T * scale_val)) + translation

    return vertices_transformed


def run_stage1_lazy(
    lazy_manager,
    image,
    mask,
    pointmap,
    seed: int = 42,
    inference_steps: int = 25,
    cfg_strength: float = 7.0,
    cfg_strength_pm: float = 0.0,
    unload_after: bool = True,
    output_dir: str = None
) -> Dict[str, Any]:
    """
    Run Stage 1 (sparse structure generation) using lazy loading.

    Models loaded: ss_generator (~6.7 GB), ss_condition_embedder (~1.2 GB), ss_decoder (~150 MB)
    Peak VRAM: ~8-9 GB

    Args:
        lazy_manager: LazyModelManager instance
        image: PIL Image
        mask: PIL Image (mask)
        pointmap: numpy array (H, W, 3) from depth estimation
        seed: Random seed
        inference_steps: Number of inference steps
        cfg_strength: CFG strength
        cfg_strength_pm: Pointmap guidance strength (how much depth influences structure)
        unload_after: Whether to unload models after use
        output_dir: Directory to save output files

    Returns:
        Dict with stage1 output file paths and pose data
    """
    from sam3d_objects.pipeline.inference_utils import (
        downsample_sparse_structure,
        prune_sparse_structure,
    )

    print(f"[Worker] Running Stage 1 (sparse gen) with lazy loading", file=sys.stderr)

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
            # Use mask as alpha channel - this enables proper bbox cropping
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
    ss_preprocessor = lazy_manager.get_preprocessor('ss')

    # Convert pointmap to tensor for preprocessing
    if isinstance(pointmap, np.ndarray):
        pointmap_tensor = torch.from_numpy(pointmap).float()
    else:
        pointmap_tensor = pointmap

    # Preprocess
    ss_input_dict = preprocess_image_lazy(image_np, mask_np, ss_preprocessor, pointmap=pointmap_tensor)

    # Save debug image showing what was actually passed to the model
    debug_image_path = None
    if output_dir:
        try:
            from PIL import Image as PILImage
            # Get the CROPPED preprocessed image (CHW format) - this is what DINO sees
            # "image" = cropped + transformed to 518x518 (what model receives)
            # "rgb_image" = full image just resized (NOT what model receives)
            debug_img = ss_input_dict.get("image")  # Must be "image", not "rgb_image"
            if debug_img is not None:
                # Remove batch dim if present, convert CHW->HWC
                if debug_img.dim() == 4:
                    debug_img = debug_img[0]
                debug_img_np = debug_img.cpu().permute(1, 2, 0).numpy()
                print(f"[Worker] Debug image (what DINO sees): {debug_img_np.shape}", file=sys.stderr)
                # Denormalize and convert to uint8
                debug_img_np = (debug_img_np * 255).clip(0, 255).astype(np.uint8)
                debug_pil = PILImage.fromarray(debug_img_np)
                debug_image_path = str(Path(output_dir) / "debug_preprocessed_stage1.png")
                debug_pil.save(debug_image_path)
                print(f"[Worker] Saved debug image: {debug_image_path} (size: {debug_pil.size})", file=sys.stderr)
        except Exception as e:
            print(f"[Worker] Failed to save debug image: {e}", file=sys.stderr)

    # Store pointmap scale/shift for pose decoding
    pointmap_scale = ss_input_dict.get("pointmap_scale", None)
    pointmap_shift = ss_input_dict.get("pointmap_shift", None)

    # Load models
    print(f"[Worker] Loading Stage 1 models...", file=sys.stderr)
    ss_generator = lazy_manager.load_model('ss_generator')
    ss_decoder = lazy_manager.load_model('ss_decoder')
    ss_embedder = lazy_manager.load_condition_embedder('ss')

    # Configure generator
    ss_generator.no_shortcut = True
    ss_generator.reverse_fn.strength = cfg_strength
    ss_generator.reverse_fn.strength_pm = cfg_strength_pm
    ss_generator.inference_steps = inference_steps

    print(f"[Worker] Running sparse structure generation...", file=sys.stderr)

    dtype = lazy_manager._get_dtype()
    downsample_ss_dist = lazy_manager.get_config_value('downsample_ss_dist', 0)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=dtype):
            # Get condition embeddings
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
            ss_condition_input_mapping = lazy_manager.get_config_value('ss_condition_input_mapping', ['image'])

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
    pose_decoder = lazy_manager.get_pose_decoder()
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

    # Unload models if requested
    if unload_after:
        print(f"[Worker] Unloading Stage 1 models...", file=sys.stderr)
        lazy_manager.unload_model('ss_generator')
        lazy_manager.unload_model('ss_decoder')
        lazy_manager.unload_condition_embedder('ss')

    print(f"[Worker] Stage 1 complete", file=sys.stderr)

    # Save sparse structure directly (don't use save_output_to_disk which has detection logic)
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


def run_stage2_lazy(
    lazy_manager,
    image,
    mask,
    stage1_output: Dict,
    seed: int = 42,
    inference_steps: int = 25,
    cfg_strength: float = 5.0,
    unload_after: bool = True,
    output_dir: str = None
) -> Dict[str, Any]:
    """
    Run Stage 2 (SLAT generation) using lazy loading.

    Models loaded: slat_generator (~4.9 GB), slat_condition_embedder (~1.2 GB)
    Peak VRAM: ~6-7 GB

    Args:
        lazy_manager: LazyModelManager instance
        image: PIL Image
        mask: PIL Image (mask)
        stage1_output: Dict with coords from Stage 1
        seed: Random seed
        inference_steps: Number of inference steps
        cfg_strength: CFG strength
        unload_after: Whether to unload models after use
        output_dir: Directory to save output files

    Returns:
        Dict with SLAT file path
    """
    from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp

    print(f"[Worker] Running Stage 2 (SLAT gen) with lazy loading", file=sys.stderr)

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
            # Use mask as alpha channel - this enables proper bbox cropping
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
    slat_preprocessor = lazy_manager.get_preprocessor('slat')
    slat_input_dict = preprocess_image_lazy(image_np, mask_np, slat_preprocessor)

    # Get coords from stage1_output (may be base64-encoded from lazy stage1)
    coords = stage1_output.get("coords")
    if isinstance(coords, str):
        coords = pickle.loads(base64.b64decode(coords))
    if isinstance(coords, np.ndarray):
        coords = torch.from_numpy(coords).int()
    coords = coords.cuda()

    # Load models
    print(f"[Worker] Loading Stage 2 models...", file=sys.stderr)
    slat_generator = lazy_manager.load_model('slat_generator')
    slat_embedder = lazy_manager.load_condition_embedder('slat')

    # Configure generator
    slat_generator.no_shortcut = True
    slat_generator.reverse_fn.strength = cfg_strength
    slat_generator.inference_steps = inference_steps

    print(f"[Worker] Running SLAT generation...", file=sys.stderr)

    dtype = lazy_manager._get_dtype()
    slat_mean, slat_std = lazy_manager.get_slat_stats()
    slat_mean = slat_mean.cuda()
    slat_std = slat_std.cuda()

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=dtype):
            image_tensor = slat_input_dict["image"]
            DEVICE = image_tensor.device
            latent_shape = (image_tensor.shape[0],) + (coords.shape[0], 8)

            # Get condition input mapping from config
            slat_condition_input_mapping = lazy_manager.get_config_value('slat_condition_input_mapping', ['image'])

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

    # Unload models if requested
    if unload_after:
        print(f"[Worker] Unloading Stage 2 models...", file=sys.stderr)
        lazy_manager.unload_model('slat_generator')
        lazy_manager.unload_condition_embedder('slat')

    print(f"[Worker] Stage 2 complete", file=sys.stderr)

    # Build output dict with SLAT for saving
    output_dict = {
        "slat": slat,
        "stage1_data": stage1_output,
    }

    # Save output to disk (consistent with non-lazy path)
    if output_dir:
        saved_output = save_output_to_disk(output_dict, Path(output_dir))
    else:
        # Fallback to temp directory
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


def run_decode_lazy(
    lazy_manager,
    slat_data: Dict,
    decode_format: str = "gaussian",
    unload_after: bool = True,
    output_dir: str = None,
    with_postprocess: bool = False,
    simplify: float = 0.95,
    up_axis: str = "Y-up (standard)"
) -> Dict[str, Any]:
    """
    Run Stage 3 (Gaussian or Mesh decoding) using lazy loading.

    Models loaded: slat_decoder_gs (~170 MB) or slat_decoder_mesh (~364 MB)
    Peak VRAM: ~1-2 GB

    Args:
        lazy_manager: LazyModelManager instance
        slat_data: Dict with slat from Stage 2 (may be base64-encoded)
        decode_format: "gaussian" or "mesh"
        unload_after: Whether to unload models after use
        output_dir: Directory to save output files
        with_postprocess: Apply mesh simplification + hole filling (mesh only)
        simplify: Mesh simplification ratio (only used when with_postprocess=True)

    Returns:
        Dict with file paths (ply_path for gaussian, glb_path for mesh)
    """
    import trimesh
    from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp

    print(f"[Worker] Running decode ({decode_format}) with lazy loading", file=sys.stderr)

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

    print(f"[Worker] Loading decoder: {decoder_name}...", file=sys.stderr)
    decoder = lazy_manager.load_model(decoder_name)

    print(f"[Worker] Running decoder...", file=sys.stderr)

    with torch.no_grad():
        output = decoder(slat)

    # Unload decoder if requested
    if unload_after:
        print(f"[Worker] Unloading decoder...", file=sys.stderr)
        lazy_manager.unload_model(decoder_name)

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

        # Apply pose transformation to get world coordinates (still in Z-up space)
        # Pose data comes from Stage 1 (sparse structure generation)
        pose_data = None
        if isinstance(slat_data, dict) and "stage1_data" in slat_data:
            stage1 = slat_data["stage1_data"]
            if isinstance(stage1, dict):
                pose_data = {
                    "rotation": stage1.get("rotation"),
                    "translation": stage1.get("translation"),
                    "scale": stage1.get("scale"),
                }

        if pose_data is not None and pose_data.get("rotation") is not None:
            print(f"[Worker] Applying pose transformation to Gaussian...", file=sys.stderr)
            gaussian = _apply_pose_to_gaussian(gaussian, pose_data)

        ply_path = save_dir / "gaussian.ply"
        try:
            # Compute transform matrix if Y-up is requested
            # This is applied AFTER pose (pose is in Z-up space, then convert to Y-up for output)
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
                    original_vertex_colors = mesh.vertex_attrs.cpu().numpy()
                elif isinstance(mesh.vertex_attrs, dict) and 'color' in mesh.vertex_attrs:
                    vc = mesh.vertex_attrs['color']
                    if hasattr(vc, 'cpu'):
                        original_vertex_colors = vc.cpu().numpy()
                    else:
                        original_vertex_colors = vc

            # Apply postprocessing if requested
            if with_postprocess:
                print(f"[Worker] Applying mesh postprocessing (simplify={simplify})...", file=sys.stderr)
                print(f"[Worker] Note: Vertex colors will be interpolated after simplification", file=sys.stderr)
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

            # Apply pose transformation to get world coordinates (still in Z-up space)
            # Pose data comes from Stage 1 (sparse structure generation)
            pose_data = None
            if isinstance(slat_data, dict) and "stage1_data" in slat_data:
                stage1 = slat_data["stage1_data"]
                if isinstance(stage1, dict):
                    pose_data = {
                        "rotation": stage1.get("rotation"),
                        "translation": stage1.get("translation"),
                        "scale": stage1.get("scale"),
                    }

            if pose_data is not None and pose_data.get("rotation") is not None:
                print(f"[Worker] Applying pose transformation to world coordinates...", file=sys.stderr)
                vertices = _apply_pose_to_vertices(vertices, pose_data)
            else:
                print(f"[Worker] No pose data found, mesh will be in normalized coordinates", file=sys.stderr)

            # Transform from Z-up to Y-up if requested (GLB standard)
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


def run_generate_slat(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run combined Stage 1 + Stage 2 (SLAT generation) with lazy loading.

    This is the main entry point for the SAM3DGenerateSLAT node.
    Combines sparse structure generation and SLAT generation into one operation
    with internal caching support.

    Args:
        request: Dict containing:
            - image: Base64-encoded PIL image
            - mask: Base64-encoded numpy mask
            - pointmap_path: Path to pointmap.pt from depth estimation
            - output_dir: Directory to save outputs
            - seed: Random seed
            - stage1_steps: Inference steps for Stage 1
            - stage1_cfg: CFG strength for Stage 1
            - stage2_steps: Inference steps for Stage 2
            - stage2_cfg: CFG strength for Stage 2
            - skip_stage1: If True, load existing sparse_structure.pt

    Returns:
        Dict with status and slat_path
    """
    import os
    import io
    import base64
    import pickle
    import traceback
    from PIL import Image

    from lazy_manager import get_model_manager
    from preprocessing import load_pointmap_from_file

    try:
        # Extract parameters
        image_b64 = request["image"]
        mask_b64 = request["mask"]
        pointmap_path = request["pointmap_path"]
        output_dir = request["output_dir"]
        seed = request.get("seed", 42)
        stage1_steps = request.get("stage1_steps", 12)
        stage1_cfg = request.get("stage1_cfg", 7.5)
        stage1_cfg_pm = request.get("stage1_cfg_pm", 0.0)
        stage2_steps = request.get("stage2_steps", 12)
        stage2_cfg = request.get("stage2_cfg", 5.0)
        skip_stage1 = request.get("skip_stage1", False)
        use_distillation = request.get("use_distillation", False)

        if use_distillation:
            print(f"[Worker] WARNING: Distillation not yet supported in lazy loading mode. Using standard models.", file=sys.stderr)

        # Deserialize image
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Deserialize mask
        mask = pickle.loads(base64.b64decode(mask_b64))
        mask_pil = Image.fromarray(mask)

        # Load pointmap
        print(f"[Worker] Loading pointmap from: {pointmap_path}", file=sys.stderr)
        pointmap = load_pointmap_from_file(pointmap_path)
        print(f"[Worker] Pointmap shape: {pointmap.shape}", file=sys.stderr)

        # Get config path from request (passed from LoadSAM3DModel via generator)
        config_path = request.get("config_path")
        if not config_path:
            raise ValueError("config_path not provided in request")

        print(f"[Worker] Config path: {config_path}", file=sys.stderr)

        # Get lazy manager
        lazy_manager = get_model_manager(str(config_path), compile=False)

        # Stage 1: Sparse structure generation
        stage1_output = None
        sparse_path = os.path.join(output_dir, "sparse_structure.pt")
        debug_image_path = None

        if skip_stage1 and os.path.exists(sparse_path):
            print(f"[Worker] Skipping Stage 1 - loading cached sparse structure", file=sys.stderr)
            stage1_output = torch.load(sparse_path, weights_only=False)
            # Check for cached debug image
            cached_debug = os.path.join(output_dir, "debug_preprocessed_stage1.png")
            if os.path.exists(cached_debug):
                debug_image_path = cached_debug
        else:
            print(f"[Worker] Running Stage 1 (sparse gen)...", file=sys.stderr)
            result = run_stage1_lazy(
                lazy_manager,
                image,
                mask_pil,
                pointmap,
                seed=seed,
                inference_steps=stage1_steps,
                cfg_strength=stage1_cfg,
                cfg_strength_pm=stage1_cfg_pm,
                unload_after=True,
                output_dir=output_dir
            )

            if result.get("status") != "success":
                return result

            # Capture debug image path from Stage 1
            debug_image_path = result.get("debug_image")

            # Load the saved sparse structure for Stage 2
            stage1_output = torch.load(sparse_path, weights_only=False)

        # Stage 2: SLAT generation
        print(f"[Worker] Running Stage 2 (SLAT gen)...", file=sys.stderr)
        result = run_stage2_lazy(
            lazy_manager,
            image,
            mask_pil,
            stage1_output,
            seed=seed,
            inference_steps=stage2_steps,
            cfg_strength=stage2_cfg,
            unload_after=True,
            output_dir=output_dir
        )

        if result.get("status") != "success":
            return result

        # Extract SLAT path from result
        slat_path = None
        if "output" in result and "files" in result["output"]:
            slat_path = result["output"]["files"].get("slat")

        if not slat_path:
            slat_path = os.path.join(output_dir, "slat.pt")

        print(f"[Worker] SLAT generation complete: {slat_path}", file=sys.stderr)

        # Load pose data from sparse_structure.pt
        rotation = None
        translation = None
        scale = None
        if stage1_output is not None:
            rotation = stage1_output.get("rotation")
            translation = stage1_output.get("translation")
            scale = stage1_output.get("scale")

            # Convert tensors to lists for JSON serialization
            if rotation is not None and hasattr(rotation, 'tolist'):
                rotation = rotation.cpu().tolist() if hasattr(rotation, 'cpu') else rotation.tolist()
            if translation is not None and hasattr(translation, 'tolist'):
                translation = translation.cpu().tolist() if hasattr(translation, 'cpu') else translation.tolist()
            if scale is not None and hasattr(scale, 'tolist'):
                scale = scale.cpu().tolist() if hasattr(scale, 'cpu') else scale.tolist()

            print(f"[Worker] Extracted pose data from Stage 1", file=sys.stderr)

        result = {
            "status": "success",
            "slat_path": slat_path,
            "files": {
                "slat": slat_path,
                "sparse_structure": sparse_path,
            },
            "output_dir": output_dir,
            "rotation": rotation,
            "translation": translation,
            "scale": scale,
        }

        # Include debug image if available
        if debug_image_path:
            result["debug_image"] = debug_image_path
            result["files"]["debug_image"] = debug_image_path

        return result

    except Exception as e:
        print(f"[Worker] Error in generate_slat: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def run_decode(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle decode command - loads only decoder, no image needed.

    This is a dedicated command handler for decoding SLAT to Gaussian or Mesh.
    It only loads the specific decoder model needed, nothing else.

    Args:
        request: Dict containing:
            - config_path: Path to pipeline.yaml
            - slat_path: Path to slat.pt file
            - output_dir: Directory to save output
            - decode_format: "gaussian" or "mesh"
            - with_postprocess: (mesh only) Apply simplification
            - simplify: (mesh only) Simplification ratio

    Returns:
        Dict with status and file paths (ply_path or glb_path)
    """
    import os
    import traceback

    from lazy_manager import get_model_manager

    try:
        # Extract parameters
        config_path = request.get("config_path")
        slat_path = request.get("slat_path")
        output_dir = request.get("output_dir")
        decode_format = request.get("decode_format", "gaussian")
        with_postprocess = request.get("with_postprocess", False)
        simplify = request.get("simplify", 0.95)
        up_axis = request.get("up_axis", "Y-up (standard)")

        if not config_path:
            raise ValueError("config_path not provided in request")
        if not slat_path:
            raise ValueError("slat_path not provided in request")

        print(f"[Worker] Decode command: format={decode_format}", file=sys.stderr)
        print(f"[Worker] Config path: {config_path}", file=sys.stderr)
        print(f"[Worker] SLAT path: {slat_path}", file=sys.stderr)
        print(f"[Worker] Output dir: {output_dir}", file=sys.stderr)

        # Load SLAT data
        slat_data = torch.load(slat_path, weights_only=False)
        print(f"[Worker] Loaded SLAT data", file=sys.stderr)

        # Get lazy manager and run decode
        lazy_manager = get_model_manager(str(config_path), compile=False)

        result = run_decode_lazy(
            lazy_manager,
            slat_data,
            decode_format=decode_format,
            unload_after=True,
            output_dir=output_dir,
            with_postprocess=with_postprocess,
            simplify=simplify,
            up_axis=up_axis
        )

        return {
            "status": "success",
            **result
        }

    except Exception as e:
        print(f"[Worker] Error in decode: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
