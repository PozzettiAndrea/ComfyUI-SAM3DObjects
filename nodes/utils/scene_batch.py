"""
Scene batch processing for SAM3D with phase-based model loading.

This module processes multiple masks efficiently by:
1. Loading Stage1 models ONCE, processing ALL masks, then unloading
2. Loading Stage2 models ONCE, processing ALL sparse structures, then unloading
3. Loading MeshDecoder ONCE, processing ALL SLATs, then unloading
4. (Optional) Loading GaussianDecoder ONCE, texture baking ALL meshes, then unloading

This avoids loading/unloading the same heavy models N times for N objects.
"""

import os
import sys
import base64
import io
import json
import pickle
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image


def run_scene_generate_batch(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process multiple masks through the full pipeline with phase-based model loading.

    Args:
        request: Dict containing:
            - image: Base64-encoded PIL image
            - masks: List of base64-encoded numpy masks
            - pointmap_path: Path to pointmap.pt from depth estimation
            - base_output_dir: Base directory for all outputs
            - config_path: Path to pipeline.yaml (generator)
            - mesh_config_path: Path to mesh decoder config
            - gs_config_path: Path to gaussian decoder config (optional, for textures)
            - seed: Random seed
            - stage1_steps: Inference steps for Stage 1
            - stage1_cfg: CFG strength for Stage 1
            - stage1_cfg_pm: Pointmap CFG strength
            - stage2_steps: Inference steps for Stage 2
            - stage2_cfg: CFG strength for Stage 2
            - with_postprocess: Apply mesh postprocessing
            - simplify: Mesh simplification ratio
            - add_textures: Enable texture baking
            - texture_mode: "opt" or "fast"
            - texture_size: Texture resolution

    Returns:
        Dict with:
            - status: "success" or "error"
            - objects: List of per-object results (glb_path, pose, etc.)
            - output_dir: Base output directory
    """
    from .lazy_manager import get_lazy_manager
    from .helpers import preprocess_image_lazy, load_pointmap_from_file
    from .stages import run_stage1_lazy, run_stage2_lazy, run_decode_lazy
    from .stages import run_texture_bake_direct as texture_bake_impl

    # Suppress ALL stdout output (including C-level from libraries like pymeshfix)
    # This is critical because stdout is used for JSON IPC
    import os as _os
    _os.environ['TQDM_DISABLE'] = '1'

    # Save original file descriptors
    _original_stdout_fd = _os.dup(1)  # Save original stdout fd
    _devnull_fd = _os.open(_os.devnull, _os.O_WRONLY)
    _os.dup2(_devnull_fd, 1)  # Redirect fd 1 (stdout) to /dev/null

    # Also redirect Python's sys.stdout
    _original_stdout = sys.stdout
    _devnull_file = open(_os.devnull, 'w')
    sys.stdout = _devnull_file

    try:
        # Extract parameters
        image_b64 = request["image"]
        masks_b64 = request["masks"]  # List of base64-encoded masks
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
        for i, mask_b64 in enumerate(masks_b64):
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

        # Get lazy manager for generator (Stage 1 + 2)
        print(f"[Worker] Initializing lazy manager with config: {config_path}", file=sys.stderr)
        lazy_manager = get_lazy_manager(str(config_path), compile=False)

        # ============================================================
        # PHASE 1: Stage 1 (Sparse Structure) for ALL masks
        # ============================================================
        print(f"\n[Worker] ========== PHASE 1: Stage 1 (Sparse Gen) ==========", file=sys.stderr)
        phase1_start = time.time()

        # NOTE: We don't accumulate sparse_structures in memory!
        # Each one is saved to disk and loaded on-demand in Stage 2.

        for idx, (mask_np, object_dir) in enumerate(zip(masks, object_dirs)):
            print(f"[Worker] Stage 1 [{idx+1}/{num_objects}]...", file=sys.stderr)

            # Create mask PIL for preprocessing
            mask_pil = Image.fromarray(mask_np)

            # Check for cached sparse structure
            sparse_path = os.path.join(object_dir, "sparse_structure.pt")
            metadata_path = os.path.join(object_dir, "stage1_metadata.json")

            use_cache = False
            if os.path.exists(sparse_path) and os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        cached = json.load(f)
                    if (cached.get("seed") == seed and
                        cached.get("steps") == stage1_steps and
                        cached.get("cfg") == stage1_cfg and
                        cached.get("cfg_pm", 0.0) == stage1_cfg_pm):
                        use_cache = True
                except:
                    pass

            if use_cache:
                print(f"[Worker] Stage 1 [{idx+1}/{num_objects}]: Using cache", file=sys.stderr)
                stage1_output = torch.load(sparse_path, weights_only=False)
            else:
                # Run Stage 1 - models stay loaded between objects
                result = run_stage1_lazy(
                    lazy_manager,
                    image,
                    mask_pil,
                    pointmap,
                    seed=seed,
                    inference_steps=stage1_steps,
                    cfg_strength=stage1_cfg,
                    cfg_strength_pm=stage1_cfg_pm,
                    unload_after=False,  # Keep models loaded!
                    output_dir=object_dir
                )

                if result.get("status") != "success":
                    raise RuntimeError(f"Stage 1 failed for object {idx}: {result.get('error')}")

                # Save metadata
                with open(metadata_path, 'w') as f:
                    json.dump({
                        "seed": seed,
                        "steps": stage1_steps,
                        "cfg": stage1_cfg,
                        "cfg_pm": stage1_cfg_pm,
                    }, f)

                # Load saved sparse structure
                stage1_output = torch.load(sparse_path, weights_only=False)

                # Extract pose data
                object_results[idx]["rotation"] = result.get("rotation")
                object_results[idx]["translation"] = result.get("translation")
                object_results[idx]["scale"] = result.get("scale")

            # Don't keep stage1_output in memory - it's saved to disk at sparse_path
            # and will be loaded on-demand in Stage 2
            del stage1_output

        # Now unload Stage 1 models
        print(f"[Worker] Unloading Stage 1 models...", file=sys.stderr)
        lazy_manager.unload_model('ss_generator')
        lazy_manager.unload_model('ss_decoder')
        lazy_manager.unload_condition_embedder('ss')
        torch.cuda.empty_cache()
        print(f"[Worker] ✓ Phase 1 complete: {time.time() - phase1_start:.0f}s", file=sys.stderr)

        # ============================================================
        # PHASE 2: Stage 2 (SLAT Gen) for ALL sparse structures
        # ============================================================
        print(f"\n[Worker] ========== PHASE 2: Stage 2 (SLAT Gen) ==========", file=sys.stderr)
        phase2_start = time.time()

        slat_paths = []

        for idx, (mask_np, object_dir) in enumerate(zip(masks, object_dirs)):
            print(f"[Worker] Stage 2 [{idx+1}/{num_objects}]...", file=sys.stderr)

            # Create mask PIL for preprocessing
            mask_pil = Image.fromarray(mask_np)

            # Load sparse structure from disk (not kept in memory during Stage 1)
            sparse_path = os.path.join(object_dir, "sparse_structure.pt")
            stage1_output = torch.load(sparse_path, weights_only=False)

            # Run Stage 2 - models stay loaded between objects
            result = run_stage2_lazy(
                lazy_manager,
                image,
                mask_pil,
                stage1_output,
                seed=seed,
                inference_steps=stage2_steps,
                cfg_strength=stage2_cfg,
                unload_after=False,  # Keep models loaded!
                output_dir=object_dir
            )

            if result.get("status") != "success":
                raise RuntimeError(f"Stage 2 failed for object {idx}: {result.get('error')}")

            # Get SLAT path
            slat_path = None
            if "output" in result and "files" in result["output"]:
                slat_path = result["output"]["files"].get("slat")
            if not slat_path:
                slat_path = os.path.join(object_dir, "slat.pt")

            slat_paths.append(slat_path)
            object_results[idx]["slat_path"] = slat_path

            # Save pose data if not already saved from Stage 1
            if "rotation" not in object_results[idx] or object_results[idx]["rotation"] is None:
                # Get pose from sparse structure
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

        # Now unload Stage 2 models
        print(f"[Worker] Unloading Stage 2 models...", file=sys.stderr)
        lazy_manager.unload_model('slat_generator')
        lazy_manager.unload_condition_embedder('slat')
        torch.cuda.empty_cache()
        print(f"[Worker] ✓ Phase 2 complete: {time.time() - phase2_start:.0f}s", file=sys.stderr)

        # ============================================================
        # PHASE 3: Mesh Decode for ALL SLATs
        # ============================================================
        print(f"\n[Worker] ========== PHASE 3: Mesh Decode ==========", file=sys.stderr)
        phase3_start = time.time()

        # Get mesh decoder lazy manager (may use different config)
        mesh_lazy_manager = get_lazy_manager(str(mesh_config_path), compile=False)

        glb_paths = []

        for idx, (slat_path, object_dir) in enumerate(zip(slat_paths, object_dirs)):
            print(f"[Worker] Mesh decode [{idx+1}/{num_objects}]...", file=sys.stderr)

            # Load SLAT data
            slat_data = torch.load(slat_path, weights_only=False)

            # Run decode - models stay loaded between objects
            result = run_decode_lazy(
                mesh_lazy_manager,
                slat_data,
                decode_format="mesh",
                unload_after=False,  # Keep models loaded!
                output_dir=object_dir,
                with_postprocess=with_postprocess,
                simplify=simplify
            )

            if result.get("status") != "success":
                raise RuntimeError(f"Mesh decode failed for object {idx}: {result.get('error')}")

            # Get GLB path
            glb_path = None
            if "output" in result and "files" in result["output"]:
                glb_path = result["output"]["files"].get("glb")
            if "file_output" in result and "files" in result["file_output"]:
                glb_path = result["file_output"]["files"].get("glb")
            if not glb_path:
                glb_path = os.path.join(object_dir, "mesh.glb")

            glb_paths.append(glb_path)
            object_results[idx]["glb_path"] = glb_path

        # Unload mesh decoder
        print(f"[Worker] Unloading mesh decoder...", file=sys.stderr)
        mesh_lazy_manager.unload_model('slat_decoder_mesh')
        torch.cuda.empty_cache()
        print(f"[Worker] ✓ Phase 3 complete: {time.time() - phase3_start:.0f}s", file=sys.stderr)

        # ============================================================
        # PHASE 4 (Optional): Gaussian Decode + Texture Bake for ALL
        # ============================================================
        if add_textures and gs_config_path:
            print(f"\n[Worker] ========== PHASE 4: Gaussian + Texture Bake ==========", file=sys.stderr)
            phase4_start = time.time()

            # Get gaussian decoder lazy manager
            gs_lazy_manager = get_lazy_manager(str(gs_config_path), compile=False)

            ply_paths = []

            # First pass: decode all Gaussians
            for idx, (slat_path, object_dir) in enumerate(zip(slat_paths, object_dirs)):
                print(f"[Worker] Gaussian decode [{idx+1}/{num_objects}]...", file=sys.stderr)

                # Load SLAT data
                slat_data = torch.load(slat_path, weights_only=False)

                # Run decode - models stay loaded
                result = run_decode_lazy(
                    gs_lazy_manager,
                    slat_data,
                    decode_format="gaussian",
                    unload_after=False,  # Keep models loaded!
                    output_dir=object_dir
                )

                if result.get("status") != "success":
                    raise RuntimeError(f"Gaussian decode failed for object {idx}: {result.get('error')}")

                # Get PLY path
                ply_path = None
                if "output" in result and "files" in result["output"]:
                    ply_path = result["output"]["files"].get("ply")
                if "file_output" in result and "files" in result["file_output"]:
                    ply_path = result["file_output"]["files"].get("ply")
                if not ply_path:
                    ply_path = os.path.join(object_dir, "gaussian.ply")

                ply_paths.append(ply_path)
                object_results[idx]["ply_path"] = ply_path

            # Unload gaussian decoder
            print(f"[Worker] Unloading gaussian decoder...", file=sys.stderr)
            gs_lazy_manager.unload_model('slat_decoder_gs')
            torch.cuda.empty_cache()

            # Second pass: texture bake all
            for idx, (ply_path, glb_path, object_dir) in enumerate(zip(ply_paths, glb_paths, object_dirs)):
                print(f"[Worker] Texture bake [{idx+1}/{num_objects}]...", file=sys.stderr)

                try:
                    bake_result = texture_bake_impl({
                        "ply_path": ply_path,
                        "glb_path": glb_path,
                        "output_dir": object_dir,
                        "mode": texture_mode,
                        "texture_size": texture_size,
                        "rendering_engine": "nvdiffrast",
                    })

                    if bake_result.get("status") == "success":
                        textured_glb = bake_result.get("glb_path")
                        if textured_glb:
                            object_results[idx]["textured_glb_path"] = textured_glb
                            print(f"[Worker] Textured GLB: {textured_glb}", file=sys.stderr)
                except Exception as e:
                    print(f"[Worker] Warning: Texture bake failed for object {idx}: {e}", file=sys.stderr)
                    # Continue with other objects

            print(f"[Worker] ✓ Phase 4 complete: {time.time() - phase4_start:.0f}s", file=sys.stderr)

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
        # Restore stdout file descriptors and Python stdout
        try:
            # Restore fd-level stdout first
            _os.dup2(_original_stdout_fd, 1)
            _os.close(_original_stdout_fd)
            _os.close(_devnull_fd)

            # Restore Python's sys.stdout
            sys.stdout = _original_stdout
            sys.stdout.flush()
            _devnull_file.close()
            print(f"[Worker] stdout restored, returning response", file=sys.stderr)
        except NameError:
            pass  # Variables weren't defined
