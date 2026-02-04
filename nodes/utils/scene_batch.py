"""
Scene batch processing for SAM3D with phase-based model loading.

This module processes multiple masks efficiently by:
1. Loading Stage1 models ONCE, processing ALL masks, then unloading
2. Loading Stage2 models ONCE, processing ALL sparse structures, then unloading
3. Loading MeshDecoder ONCE, processing ALL SLATs, then unloading
4. (Optional) Loading GaussianDecoder ONCE, texture baking ALL meshes, then unloading

Each phase loads its own models directly - no shared lazy manager.
"""

import gc
import os
import sys
import base64
import io
import json
import pickle
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image


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


def _unload(*models):
    """Delete models and free VRAM."""
    for m in models:
        if m is not None:
            del m
    gc.collect()
    torch.cuda.empty_cache()


def run_scene_generate_batch(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process multiple masks through the full pipeline with phase-based model loading.

    Each phase loads models directly, processes all objects, then unloads.
    """
    from .helpers import preprocess_image_lazy, load_pointmap_from_file
    from .stages import run_texture_bake_direct as texture_bake_impl

    # Suppress stdout (used for JSON IPC)
    import os as _os
    _os.environ['TQDM_DISABLE'] = '1'

    _original_stdout_fd = _os.dup(1)
    _devnull_fd = _os.open(_os.devnull, _os.O_WRONLY)
    _os.dup2(_devnull_fd, 1)

    _original_stdout = sys.stdout
    _devnull_file = open(_os.devnull, 'w')
    sys.stdout = _devnull_file

    try:
        # Extract parameters
        image_b64 = request["image"]
        masks_b64 = request["masks"]
        pointmap_path = request["pointmap_path"]
        base_output_dir = request["base_output_dir"]
        config_path = request["config_path"]
        mesh_config_path = request["mesh_config_path"]
        gs_config_path = request.get("gs_config_path")

        seed = request.get("seed", 42)
        stage1_steps = request.get("stage1_steps", 12)
        stage1_cfg = request.get("stage1_cfg", 7.5)
        stage1_cfg_pm = request.get("stage1_cfg_pm", 0.0)
        stage2_steps = request.get("stage2_steps", 12)
        stage2_cfg = request.get("stage2_cfg", 5.0)
        with_postprocess = request.get("with_postprocess", False)
        simplify = request.get("simplify", 0.95)
        add_textures = request.get("add_textures", False)
        texture_mode = request.get("texture_mode", "opt")
        texture_size = request.get("texture_size", 1024)

        num_objects = len(masks_b64)
        print(f"[Worker] Scene batch: Processing {num_objects} object(s)", file=sys.stderr)

        # Deserialize image once
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image)

        # Deserialize all masks
        masks = []
        for mask_b64 in masks_b64:
            mask_np = pickle.loads(base64.b64decode(mask_b64))
            masks.append(mask_np)

        # Load pointmap once
        print(f"[Worker] Loading pointmap from: {pointmap_path}", file=sys.stderr)
        pointmap = load_pointmap_from_file(pointmap_path)
        print(f"[Worker] Pointmap shape: {pointmap.shape}", file=sys.stderr)

        # Create object directories
        object_dirs = []
        for idx in range(num_objects):
            object_dir = os.path.join(base_output_dir, f"object_{idx}")
            os.makedirs(object_dir, exist_ok=True)
            object_dirs.append(object_dir)

        # Initialize results
        object_results = [{"index": i} for i in range(num_objects)]

        # ============================================================
        # PHASE 1: Stage 1 (Sparse Structure) for ALL masks
        # ============================================================
        print(f"\n[Worker] ========== PHASE 1: Stage 1 (Sparse Gen) ==========", file=sys.stderr)
        phase1_start = time.time()

        _run_phase1_stage1(
            config_path, image, masks, pointmap, object_dirs, object_results,
            seed, stage1_steps, stage1_cfg, stage1_cfg_pm
        )

        print(f"[Worker] Phase 1 complete: {time.time() - phase1_start:.0f}s", file=sys.stderr)

        # ============================================================
        # PHASE 2: Stage 2 (SLAT Gen) for ALL sparse structures
        # ============================================================
        print(f"\n[Worker] ========== PHASE 2: Stage 2 (SLAT Gen) ==========", file=sys.stderr)
        phase2_start = time.time()

        slat_paths = _run_phase2_stage2(
            config_path, image, masks, object_dirs, object_results,
            seed, stage2_steps, stage2_cfg
        )

        print(f"[Worker] Phase 2 complete: {time.time() - phase2_start:.0f}s", file=sys.stderr)

        # ============================================================
        # PHASE 3: Mesh Decode for ALL SLATs
        # ============================================================
        print(f"\n[Worker] ========== PHASE 3: Mesh Decode ==========", file=sys.stderr)
        phase3_start = time.time()

        glb_paths = _run_phase3_mesh_decode(
            mesh_config_path, slat_paths, object_dirs, object_results,
            with_postprocess, simplify
        )

        print(f"[Worker] Phase 3 complete: {time.time() - phase3_start:.0f}s", file=sys.stderr)

        # ============================================================
        # PHASE 4 (Optional): Gaussian Decode + Texture Bake
        # ============================================================
        if add_textures and gs_config_path:
            print(f"\n[Worker] ========== PHASE 4: Gaussian + Texture Bake ==========", file=sys.stderr)
            phase4_start = time.time()

            _run_phase4_texture(
                gs_config_path, slat_paths, glb_paths, object_dirs, object_results,
                texture_bake_impl, texture_mode, texture_size
            )

            print(f"[Worker] Phase 4 complete: {time.time() - phase4_start:.0f}s", file=sys.stderr)

        print(f"\n[Worker] Scene batch complete: {num_objects} object(s)", file=sys.stderr)

        return {
            "status": "success",
            "objects": object_results,
            "output_dir": base_output_dir,
            "num_objects": num_objects,
        }

    except Exception as e:
        print(f"[Worker] Error in scene_generate_batch: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
    finally:
        try:
            _os.dup2(_original_stdout_fd, 1)
            _os.close(_original_stdout_fd)
            _os.close(_devnull_fd)
            sys.stdout = _original_stdout
            sys.stdout.flush()
            _devnull_file.close()
        except NameError:
            pass


def _run_phase1_stage1(
    config_path, image, masks, pointmap, object_dirs, object_results,
    seed, stage1_steps, stage1_cfg, stage1_cfg_pm
):
    """Phase 1: Load Stage1 models once, process all masks, unload."""
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    from sam3d_objects.model.io import load_model_from_checkpoint, filter_and_remove_prefix_state_dict_fn
    from sam3d_objects.pipeline.inference_utils import (
        downsample_sparse_structure, prune_sparse_structure, get_pose_decoder
    )

    config, checkpoint_dir = _load_config(config_path)
    dtype = _get_dtype(config)

    # Load Stage 1 models
    print(f"[Worker] Loading Stage 1 models...", file=sys.stderr)

    # Load ss_generator
    gen_config_path = checkpoint_dir / config.ss_generator_config_path
    gen_ckpt_path = checkpoint_dir / config.ss_generator_ckpt_path
    gen_config = OmegaConf.load(gen_config_path)
    ss_generator = instantiate(gen_config["module"]["generator"]["backbone"])
    ss_generator = load_model_from_checkpoint(
        ss_generator, str(gen_ckpt_path), strict=False, device="cpu", freeze=True, eval=True,
        state_dict_key="state_dict",
        state_dict_fn=filter_and_remove_prefix_state_dict_fn("_base_models.generator."),
    ).cuda().to(dtype)

    # Load ss_decoder
    dec_config_path = checkpoint_dir / config.ss_decoder_config_path
    dec_ckpt_path = checkpoint_dir / config.ss_decoder_ckpt_path
    dec_config = OmegaConf.load(dec_config_path)
    from sam3d_objects.model.io import remove_prefix_state_dict_fn
    ss_decoder = instantiate(dec_config)
    ss_decoder = load_model_from_checkpoint(
        ss_decoder, str(dec_ckpt_path), strict=False, device="cpu", freeze=True, eval=True,
        state_dict_fn=remove_prefix_state_dict_fn("module."),
    ).cuda()

    # Load ss_embedder
    embedder_config = gen_config.module.condition_embedder.backbone
    ss_embedder = instantiate(embedder_config)
    ss_embedder = load_model_from_checkpoint(
        ss_embedder, str(gen_ckpt_path), strict=False, device="cpu", freeze=True, eval=True,
        state_dict_key="state_dict",
        state_dict_fn=filter_and_remove_prefix_state_dict_fn("_base_models.condition_embedder."),
    ).cuda().to(dtype)

    # Get preprocessor
    preprocessor_config = config.get('ss_preprocessor')
    if preprocessor_config:
        ss_preprocessor = instantiate(preprocessor_config)
    else:
        from sam3d_objects.pipeline import preprocess_utils
        ss_preprocessor = preprocess_utils.get_default_preprocessor()

    # Get pose decoder
    pose_decoder_name = getattr(config, 'pose_decoder_name', 'default')
    pose_decoder = get_pose_decoder(pose_decoder_name)

    # Configure generator
    ss_generator.no_shortcut = True
    ss_generator.reverse_fn.strength = stage1_cfg
    ss_generator.reverse_fn.strength_pm = stage1_cfg_pm
    ss_generator.inference_steps = stage1_steps

    downsample_ss_dist = getattr(config, 'downsample_ss_dist', 0)
    ss_condition_input_mapping = getattr(config, 'ss_condition_input_mapping', ['image'])

    # Process each mask
    for idx, (mask_np, object_dir) in enumerate(zip(masks, object_dirs)):
        print(f"[Worker] Stage 1 [{idx+1}/{len(masks)}]...", file=sys.stderr)

        sparse_path = os.path.join(object_dir, "sparse_structure.pt")
        metadata_path = os.path.join(object_dir, "stage1_metadata.json")

        # Check cache
        use_cache = False
        if os.path.exists(sparse_path) and os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    cached = json.load(f)
                if (cached.get("seed") == seed and cached.get("steps") == stage1_steps and
                    cached.get("cfg") == stage1_cfg and cached.get("cfg_pm", 0.0) == stage1_cfg_pm):
                    use_cache = True
            except:
                pass

        if use_cache:
            print(f"[Worker] Stage 1 [{idx+1}]: Using cache", file=sys.stderr)
            continue

        torch.manual_seed(seed)
        mask_pil = Image.fromarray(mask_np)
        image_np = np.array(image)

        # Build RGBA with mask as alpha
        if image_np.shape[-1] == 3:
            if mask_np.dtype in (np.float32, np.float64):
                alpha = (mask_np * 255).astype(np.uint8)
            else:
                alpha = mask_np.astype(np.uint8)
            if alpha.ndim == 2:
                alpha = alpha[:, :, np.newaxis]
            image_np = np.concatenate([image_np, alpha], axis=-1)

        # Convert pointmap to tensor
        if isinstance(pointmap, np.ndarray):
            pointmap_tensor = torch.from_numpy(pointmap).float()
        else:
            pointmap_tensor = pointmap

        # Preprocess
        ss_input_dict = preprocess_image_lazy(image_np, mask_np, ss_preprocessor, pointmap=pointmap_tensor)
        pointmap_scale = ss_input_dict.get("pointmap_scale")
        pointmap_shift = ss_input_dict.get("pointmap_shift")

        # Run generation
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=dtype):
                image_tensor = ss_input_dict["image"]
                bs = image_tensor.shape[0]

                has_latent_mapping = hasattr(ss_generator.reverse_fn.backbone, "latent_mapping")
                if has_latent_mapping:
                    latent_shape_dict = {
                        k: (bs,) + (v.pos_emb.shape[0], v.input_layer.in_features)
                        for k, v in ss_generator.reverse_fn.backbone.latent_mapping.items()
                    }
                else:
                    latent_shape_dict = (bs,) + (4096, 8)

                condition_args = [ss_input_dict[k] for k in ss_condition_input_mapping if k in ss_input_dict]
                condition_kwargs = {k: v for k, v in ss_input_dict.items() if k not in ss_condition_input_mapping}

                if ss_embedder is not None and len(condition_args) > 0:
                    tokens = ss_embedder(*condition_args, **condition_kwargs)
                    condition_args = (tokens,)
                    condition_kwargs = {}

                return_dict = ss_generator(latent_shape_dict, image_tensor.device, *condition_args, **condition_kwargs)

                if not has_latent_mapping:
                    return_dict = {"shape": return_dict}

                shape_latent = return_dict["shape"]
                ss = ss_decoder(shape_latent.permute(0, 2, 1).contiguous().view(shape_latent.shape[0], 8, 16, 16, 16))
                coords = torch.argwhere(ss > 0)[:, [0, 2, 3, 4]].int()

                return_dict["coords_original"] = coords
                if downsample_ss_dist > 0:
                    coords = prune_sparse_structure(coords, max_neighbor_axes_dist=downsample_ss_dist)
                coords, downsample_factor = downsample_sparse_structure(coords)
                return_dict["coords"] = coords
                return_dict["downsample_factor"] = downsample_factor

        # Pose decoding
        pose_result = pose_decoder(return_dict, scene_scale=pointmap_scale, scene_shift=pointmap_shift)
        return_dict.update(pose_result)
        return_dict["scale"] = return_dict["scale"] * return_dict["downsample_factor"]
        return_dict["voxel"] = return_dict["coords"][:, 1:] / 64 - 0.5

        # Save
        torch.save(return_dict, sparse_path)
        with open(metadata_path, 'w') as f:
            json.dump({"seed": seed, "steps": stage1_steps, "cfg": stage1_cfg, "cfg_pm": stage1_cfg_pm}, f)

        # Extract pose for results
        rotation = return_dict.get("rotation")
        translation = return_dict.get("translation")
        scale = return_dict.get("scale")
        if rotation is not None and hasattr(rotation, 'tolist'):
            rotation = rotation.cpu().tolist() if hasattr(rotation, 'cpu') else rotation.tolist()
        if translation is not None and hasattr(translation, 'tolist'):
            translation = translation.cpu().tolist() if hasattr(translation, 'cpu') else translation.tolist()
        if scale is not None and hasattr(scale, 'tolist'):
            scale = scale.cpu().tolist() if hasattr(scale, 'cpu') else scale.tolist()

        object_results[idx]["rotation"] = rotation
        object_results[idx]["translation"] = translation
        object_results[idx]["scale"] = scale

    # Unload
    print(f"[Worker] Unloading Stage 1 models...", file=sys.stderr)
    _unload(ss_generator, ss_decoder, ss_embedder)


def _run_phase2_stage2(
    config_path, image, masks, object_dirs, object_results,
    seed, stage2_steps, stage2_cfg
):
    """Phase 2: Load Stage2 models once, process all sparse structures, unload."""
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    from sam3d_objects.model.io import load_model_from_checkpoint, filter_and_remove_prefix_state_dict_fn
    from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp

    config, checkpoint_dir = _load_config(config_path)
    dtype = _get_dtype(config)

    # Load Stage 2 models
    print(f"[Worker] Loading Stage 2 models...", file=sys.stderr)

    gen_config_path = checkpoint_dir / config.slat_generator_config_path
    gen_ckpt_path = checkpoint_dir / config.slat_generator_ckpt_path
    gen_config = OmegaConf.load(gen_config_path)

    slat_generator = instantiate(gen_config["module"]["generator"]["backbone"])
    slat_generator = load_model_from_checkpoint(
        slat_generator, str(gen_ckpt_path), strict=False, device="cpu", freeze=True, eval=True,
        state_dict_key="state_dict",
        state_dict_fn=filter_and_remove_prefix_state_dict_fn("_base_models.generator."),
    ).cuda().to(dtype)

    embedder_config = gen_config.module.condition_embedder.backbone
    slat_embedder = instantiate(embedder_config)
    slat_embedder = load_model_from_checkpoint(
        slat_embedder, str(gen_ckpt_path), strict=False, device="cpu", freeze=True, eval=True,
        state_dict_key="state_dict",
        state_dict_fn=filter_and_remove_prefix_state_dict_fn("_base_models.condition_embedder."),
    ).cuda().to(dtype)

    # Get preprocessor
    preprocessor_config = config.get('slat_preprocessor')
    if preprocessor_config:
        slat_preprocessor = instantiate(preprocessor_config)
    else:
        from sam3d_objects.pipeline import preprocess_utils
        slat_preprocessor = preprocess_utils.get_default_preprocessor()

    # Configure generator
    slat_generator.no_shortcut = True
    slat_generator.reverse_fn.strength = stage2_cfg
    slat_generator.inference_steps = stage2_steps

    slat_mean = torch.tensor(getattr(config, 'slat_mean', [0.0] * 8)).cuda()
    slat_std = torch.tensor(getattr(config, 'slat_std', [1.0] * 8)).cuda()
    slat_condition_input_mapping = getattr(config, 'slat_condition_input_mapping', ['image'])

    slat_paths = []

    for idx, (mask_np, object_dir) in enumerate(zip(masks, object_dirs)):
        print(f"[Worker] Stage 2 [{idx+1}/{len(masks)}]...", file=sys.stderr)

        torch.manual_seed(seed)
        mask_pil = Image.fromarray(mask_np)
        image_np = np.array(image)

        # Build RGBA
        if image_np.shape[-1] == 3:
            if mask_np.dtype in (np.float32, np.float64):
                alpha = (mask_np * 255).astype(np.uint8)
            else:
                alpha = mask_np.astype(np.uint8)
            if alpha.ndim == 2:
                alpha = alpha[:, :, np.newaxis]
            image_np = np.concatenate([image_np, alpha], axis=-1)

        # Load sparse structure
        sparse_path = os.path.join(object_dir, "sparse_structure.pt")
        stage1_output = torch.load(sparse_path, weights_only=False)
        coords = stage1_output.get("coords")
        if isinstance(coords, np.ndarray):
            coords = torch.from_numpy(coords).int()
        coords = coords.cuda()

        # Preprocess
        slat_input_dict = preprocess_image_lazy(image_np, mask_np, slat_preprocessor)

        # Run generation
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=dtype):
                image_tensor = slat_input_dict["image"]
                DEVICE = image_tensor.device
                latent_shape = (image_tensor.shape[0],) + (coords.shape[0], 8)

                condition_args = [slat_input_dict[k] for k in slat_condition_input_mapping if k in slat_input_dict]
                condition_kwargs = {k: v for k, v in slat_input_dict.items() if k not in slat_condition_input_mapping}

                if slat_embedder is not None and len(condition_args) > 0:
                    tokens = slat_embedder(*condition_args, **condition_kwargs)
                    condition_args = (tokens,)
                    condition_kwargs = {}

                condition_args = condition_args + (coords.cpu().numpy(),)
                slat_feats = slat_generator(latent_shape, DEVICE, *condition_args, **condition_kwargs)

                slat = sp.SparseTensor(coords=coords, feats=slat_feats[0]).to(DEVICE)
                slat = slat * slat_std + slat_mean

        # Save
        slat_path = os.path.join(object_dir, "slat.pt")
        torch.save({"slat": slat, "stage1_data": stage1_output}, slat_path)
        slat_paths.append(slat_path)
        object_results[idx]["slat_path"] = slat_path

        # Extract pose if needed
        if "rotation" not in object_results[idx] or object_results[idx]["rotation"] is None:
            rotation = stage1_output.get("rotation")
            translation = stage1_output.get("translation")
            scale = stage1_output.get("scale")
            if rotation is not None and hasattr(rotation, 'tolist'):
                rotation = rotation.cpu().tolist() if hasattr(rotation, 'cpu') else rotation.tolist()
            if translation is not None and hasattr(translation, 'tolist'):
                translation = translation.cpu().tolist() if hasattr(translation, 'cpu') else translation.tolist()
            if scale is not None and hasattr(scale, 'tolist'):
                scale = scale.cpu().tolist() if hasattr(scale, 'cpu') else scale.tolist()
            object_results[idx]["rotation"] = rotation
            object_results[idx]["translation"] = translation
            object_results[idx]["scale"] = scale

    # Unload
    print(f"[Worker] Unloading Stage 2 models...", file=sys.stderr)
    _unload(slat_generator, slat_embedder)

    return slat_paths


def _run_phase3_mesh_decode(
    mesh_config_path, slat_paths, object_dirs, object_results,
    with_postprocess, simplify
):
    """Phase 3: Load mesh decoder once, process all SLATs, unload."""
    import trimesh
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    from sam3d_objects.model.io import load_model_from_checkpoint, remove_prefix_state_dict_fn
    from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp

    config, checkpoint_dir = _load_config(mesh_config_path)

    # Load mesh decoder
    print(f"[Worker] Loading mesh decoder...", file=sys.stderr)
    dec_config_path = checkpoint_dir / config.slat_decoder_mesh_config_path
    dec_ckpt_path = checkpoint_dir / config.slat_decoder_mesh_ckpt_path
    dec_config = OmegaConf.load(dec_config_path)

    decoder = instantiate(dec_config)
    decoder = load_model_from_checkpoint(
        decoder, str(dec_ckpt_path), strict=False, device="cpu", freeze=True, eval=True,
        state_dict_fn=remove_prefix_state_dict_fn("module."),
    ).cuda()

    glb_paths = []

    for idx, (slat_path, object_dir) in enumerate(zip(slat_paths, object_dirs)):
        print(f"[Worker] Mesh decode [{idx+1}/{len(slat_paths)}]...", file=sys.stderr)

        slat_data = torch.load(slat_path, weights_only=False)

        # Extract slat
        slat = slat_data.get("slat")
        if not isinstance(slat, sp.SparseTensor):
            coords = slat.get("coords") if isinstance(slat, dict) else slat_data.get("coords")
            feats = slat.get("feats") if isinstance(slat, dict) else slat_data.get("feats")
            if isinstance(coords, np.ndarray):
                coords = torch.from_numpy(coords).int()
            if isinstance(feats, np.ndarray):
                feats = torch.from_numpy(feats)
            slat = sp.SparseTensor(coords=coords.cuda(), feats=feats.cuda())
        else:
            slat = slat.cuda()

        with torch.no_grad():
            output = decoder(slat)

        mesh = output[0] if isinstance(output, (list, tuple)) else output
        vertices = mesh.vertices.cpu().numpy() if hasattr(mesh.vertices, 'cpu') else mesh.vertices
        faces = mesh.faces.cpu().numpy() if hasattr(mesh.faces, 'cpu') else mesh.faces

        # Get vertex colors
        vertex_colors = None
        if hasattr(mesh, 'vertex_attrs') and mesh.vertex_attrs is not None:
            if isinstance(mesh.vertex_attrs, torch.Tensor):
                attrs = mesh.vertex_attrs.cpu().numpy()
                vc = attrs[:, :3] if attrs.shape[-1] >= 3 else attrs
                if vc.max() <= 1.0:
                    vc = (vc * 255).astype(np.uint8)
                alpha = np.full((vc.shape[0], 1), 255, dtype=np.uint8)
                vertex_colors = np.concatenate([vc, alpha], axis=-1)

        # Postprocess if requested
        if with_postprocess:
            from sam3d_objects.model.backbone.tdfy_dit.utils.postprocessing_utils import postprocess_mesh
            vertices, faces = postprocess_mesh(vertices, faces, simplify=True, simplify_ratio=simplify, fill_holes=True, verbose=False)
            vertex_colors = None

        # Z-up to Y-up transform
        z_to_y_up = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        vertices = vertices @ z_to_y_up

        trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=vertex_colors, process=False)

        glb_path = os.path.join(object_dir, "mesh.glb")
        glb_bytes = trimesh_mesh.export(file_type="glb")
        with open(glb_path, 'wb') as f:
            f.write(glb_bytes)

        glb_paths.append(glb_path)
        object_results[idx]["glb_path"] = glb_path

    # Unload
    print(f"[Worker] Unloading mesh decoder...", file=sys.stderr)
    _unload(decoder)

    return glb_paths


def _run_phase4_texture(
    gs_config_path, slat_paths, glb_paths, object_dirs, object_results,
    texture_bake_impl, texture_mode, texture_size
):
    """Phase 4: Load gaussian decoder once, decode all, then texture bake."""
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    from sam3d_objects.model.io import load_model_from_checkpoint, remove_prefix_state_dict_fn
    from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp

    config, checkpoint_dir = _load_config(gs_config_path)

    # Load gaussian decoder
    print(f"[Worker] Loading gaussian decoder...", file=sys.stderr)
    dec_config_path = checkpoint_dir / config.slat_decoder_gs_config_path
    dec_ckpt_path = checkpoint_dir / config.slat_decoder_gs_ckpt_path
    dec_config = OmegaConf.load(dec_config_path)

    decoder = instantiate(dec_config)
    decoder = load_model_from_checkpoint(
        decoder, str(dec_ckpt_path), strict=False, device="cpu", freeze=True, eval=True,
        state_dict_fn=remove_prefix_state_dict_fn("module."),
    ).cuda()

    ply_paths = []

    # Decode all gaussians
    for idx, (slat_path, object_dir) in enumerate(zip(slat_paths, object_dirs)):
        print(f"[Worker] Gaussian decode [{idx+1}/{len(slat_paths)}]...", file=sys.stderr)

        slat_data = torch.load(slat_path, weights_only=False)
        slat = slat_data.get("slat")
        if not isinstance(slat, sp.SparseTensor):
            coords = slat.get("coords") if isinstance(slat, dict) else slat_data.get("coords")
            feats = slat.get("feats") if isinstance(slat, dict) else slat_data.get("feats")
            if isinstance(coords, np.ndarray):
                coords = torch.from_numpy(coords).int()
            if isinstance(feats, np.ndarray):
                feats = torch.from_numpy(feats)
            slat = sp.SparseTensor(coords=coords.cuda(), feats=feats.cuda())
        else:
            slat = slat.cuda()

        with torch.no_grad():
            output = decoder(slat)

        gaussian = output[0] if isinstance(output, (list, tuple)) else output

        ply_path = os.path.join(object_dir, "gaussian.ply")
        transform = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])  # Z-up to Y-up
        gaussian.save_ply(ply_path, transform=transform)

        ply_paths.append(ply_path)
        object_results[idx]["ply_path"] = ply_path

    # Unload gaussian decoder
    print(f"[Worker] Unloading gaussian decoder...", file=sys.stderr)
    _unload(decoder)

    # Texture bake all
    for idx, (ply_path, glb_path, object_dir) in enumerate(zip(ply_paths, glb_paths, object_dirs)):
        print(f"[Worker] Texture bake [{idx+1}/{len(ply_paths)}]...", file=sys.stderr)
        try:
            bake_result = texture_bake_impl({
                "ply_path": ply_path,
                "glb_path": glb_path,
                "output_dir": object_dir,
                "texture_mode": texture_mode,
                "texture_size": texture_size,
                "rendering_engine": "nvdiffrast",
            })
            if bake_result.get("status") == "success":
                textured_glb = bake_result.get("output", {}).get("glb_path")
                if textured_glb:
                    object_results[idx]["textured_glb_path"] = textured_glb
        except Exception as e:
            print(f"[Worker] Warning: Texture bake failed for object {idx}: {e}", file=sys.stderr)
