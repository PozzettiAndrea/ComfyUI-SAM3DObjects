"""
Main inference orchestration for SAM3D.

This module contains the main run_inference function that handles all inference modes.
"""

import sys
import os
import base64
import pickle
import traceback
from pathlib import Path
from typing import Any, Dict
import numpy as np
import torch

from lazy_manager import get_model_manager
from utils import (
    deserialize_image,
    deserialize_mask,
    transform_to_global_coordinates,
    save_output_to_disk,
)
from preprocessing import load_pointmap_from_file
from stages import run_stage1_lazy, run_stage2_lazy, run_decode_lazy
from depth import run_depth_only


def run_inference(request: Dict[str, Any]) -> Dict[str, Any]:
    """Run inference on the given request."""
    try:
        # Check for special modes that don't need full inference setup
        config_path = request.get("config_path")
        compile_model = request.get("compile", False)

        # Handle depth_only mode (MoGe depth estimation only)
        if request.get("depth_only", False):
            image_b64 = request["image"]
            image = deserialize_image(image_b64)
            depth_backend = request.get("depth_backend", "moge2")
            unload_after = not request.get("keep_in_vram", False)

            print(f"[Worker] Running depth estimation (backend={depth_backend})", file=sys.stderr)
            model_manager = get_model_manager(config_path, compile_model)
            return run_depth_only(model_manager, image, unload_after=unload_after, depth_backend=depth_backend)

        # Extract request parameters
        use_cache = request.get("use_cache", False)
        image_b64 = request.get("image")  # May be None for decode-only
        mask_b64 = request.get("mask")    # May be None for decode-only
        seed = request.get("seed", 42)
        stage1_inference_steps = request.get("stage1_inference_steps", 25)
        stage2_inference_steps = request.get("stage2_inference_steps", 25)
        stage1_cfg_strength = request.get("stage1_cfg_strength", 7.0)
        stage2_cfg_strength = request.get("stage2_cfg_strength", 5.0)
        texture_size = request.get("texture_size", 1024)
        simplify = request.get("simplify", 0.95)
        output_dir = request.get("output_dir", "/tmp/sam3d_output")
        stage1_only = request.get("stage1_only", False)
        stage1_output_b64 = request.get("stage1_output", None)
        stage2_only = request.get("stage2_only", False)
        stage2_output_b64 = request.get("stage2_output", None)
        slat_only = request.get("slat_only", False)
        slat_output_b64 = request.get("slat_output", None)
        gaussian_only = request.get("gaussian_only", False)
        mesh_only = request.get("mesh_only", False)
        save_files = request.get("save_files", False)
        with_mesh_postprocess = request.get("with_mesh_postprocess", False)
        with_texture_baking = request.get("with_texture_baking", True)
        use_vertex_color = request.get("use_vertex_color", False)
        use_stage1_distillation = request.get("use_stage1_distillation", False)
        use_stage2_distillation = request.get("use_stage2_distillation", False)
        texture_mode = request.get("texture_mode", "opt")
        rendering_engine = request.get("rendering_engine", "nvdiffrast")
        merge_mask = request.get("merge_mask", True)
        auto_resize_mask = request.get("auto_resize_mask", True)

        # Load pointmap from .pt file if provided
        pointmap = None
        intrinsics = None
        if request.get("pointmap_path") is not None:
            pointmap_path = request.get("pointmap_path")
            print(f"[Worker] Loading pointmap from: {pointmap_path}", file=sys.stderr)
            pointmap = load_pointmap_from_file(pointmap_path)
            print(f"[Worker] Pointmap shape: {pointmap.shape}", file=sys.stderr)
        if request.get("intrinsics") is not None:
            intrinsics_np = pickle.loads(base64.b64decode(request.get("intrinsics")))
            intrinsics = torch.from_numpy(intrinsics_np).cuda() if torch.cuda.is_available() else torch.from_numpy(intrinsics_np)
            print(f"[Worker] Intrinsics shape: {intrinsics.shape}", file=sys.stderr)

        # Check if we're running an individual stage (on-demand loading)
        has_stage1_input = request.get("stage1_output_path") is not None or request.get("stage1_output") is not None
        has_slat_input = request.get("slat_output_path") is not None or request.get("slat_output") is not None
        is_single_stage = (
            (stage1_only and pointmap is not None) or
            (slat_only and has_stage1_input) or
            (gaussian_only and has_slat_input) or
            (mesh_only and has_slat_input)
        )

        # Get model manager (always on-demand loading)
        model_manager = get_model_manager(config_path, compile_model)

        # Only load full pipeline if we're NOT running a single stage
        model = None
        if not is_single_stage:
            try:
                model = model_manager.get_full_pipeline()
            except torch.cuda.OutOfMemoryError as e:
                import gc
                gc.collect()
                torch.cuda.empty_cache()

                error_msg = (
                    f"Out of memory loading SAM3D models. "
                    f"SAM3D requires 32GB+ VRAM for full pipeline. "
                    f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB. "
                    f"Tip: Use individual stage nodes (DepthEstimate, SparseGen, etc.) "
                    f"to run on GPUs with less VRAM."
                )
                raise RuntimeError(error_msg) from e

        # Deserialize inputs (may be None for decode-only operations)
        image = deserialize_image(image_b64) if image_b64 else None
        mask = deserialize_mask(mask_b64) if mask_b64 else None

        # Load stage1_output if provided
        stage1_output = None
        if request.get("stage1_output_path") is not None and os.path.exists(request.get("stage1_output_path")):
            print(f"[Worker] Loading Stage 1 output from: {request.get('stage1_output_path')}", file=sys.stderr)
            stage1_output = torch.load(request.get("stage1_output_path"), weights_only=False)
        elif request.get("stage1_output") is not None:
            stage1_output = pickle.loads(base64.b64decode(request.get("stage1_output")))

        # Deserialize stage2_output if provided
        stage2_output = None
        if stage2_output_b64 is not None:
            stage2_output = pickle.loads(base64.b64decode(stage2_output_b64))

            # Check if this needs combining separate Gaussian and Mesh data
            if isinstance(stage2_output, dict) and stage2_output.get("_needs_combination"):
                print(f"[Worker] Combining separate Gaussian and Mesh data", file=sys.stderr)
                gaussian_b64 = stage2_output["_gaussian_serialized"]
                mesh_b64 = stage2_output["_mesh_serialized"]

                gaussian_dict = pickle.loads(base64.b64decode(gaussian_b64))
                mesh_dict = pickle.loads(base64.b64decode(mesh_b64))

                stage2_output = {
                    "gaussian": gaussian_dict.get("gaussian"),
                    "mesh": mesh_dict.get("mesh"),
                    "stage1_data": mesh_dict.get("stage1_data", gaussian_dict.get("stage1_data", {}))
                }

            elif isinstance(stage2_output, dict) and stage2_output.get("_needs_file_loading"):
                print(f"[Worker] Loading Gaussian and Mesh from file paths", file=sys.stderr)
                glb_path = stage2_output["_glb_path"]
                ply_path = stage2_output["_ply_path"]

                import trimesh
                from sam3d_objects.model.backbone.tdfy_dit.representations.gaussian import Gaussian
                from sam3d_objects.model.backbone.tdfy_dit.representations.mesh.cube2mesh import MeshExtractResult

                device = 'cuda' if torch.cuda.is_available() else 'cpu'

                # Load Gaussian
                gaussian = Gaussian(aabb=[-1, -1, -1, 2, 2, 2], sh_degree=0, device=device)
                gaussian.load_ply(ply_path)

                # Load Mesh
                loaded = trimesh.load(glb_path)
                if isinstance(loaded, trimesh.Scene):
                    meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
                    if not meshes:
                        raise RuntimeError("No mesh geometries found in GLB scene")
                    trimesh_mesh = meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)
                else:
                    trimesh_mesh = loaded

                vertices_tensor = torch.tensor(np.array(trimesh_mesh.vertices), dtype=torch.float32, device=device)
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

                stage2_output = {
                    "gaussian": [gaussian],
                    "mesh": [mesh],
                    "stage1_data": {}
                }

        # Load slat_output if provided
        slat_output = None
        if request.get("slat_output_path") is not None and os.path.exists(request.get("slat_output_path")):
            print(f"[Worker] Loading SLAT output from: {request.get('slat_output_path')}", file=sys.stderr)
            slat_output = torch.load(request.get("slat_output_path"), weights_only=False)
        elif request.get("slat_output") is not None:
            slat_output = pickle.loads(base64.b64decode(request.get("slat_output")))

        print(f"[Worker] Running inference (seed={seed})", file=sys.stderr)
        print(f"[Worker] Stage 1: steps={stage1_inference_steps}, cfg={stage1_cfg_strength}", file=sys.stderr)
        print(f"[Worker] Stage 2: steps={stage2_inference_steps}, cfg={stage2_cfg_strength}", file=sys.stderr)
        print(f"[Worker] Postprocess: texture_size={texture_size}, simplify={simplify}", file=sys.stderr)
        if image is not None:
            print(f"[Worker] Image: mode={image.mode}, size={image.size}", file=sys.stderr)
        if mask is not None:
            print(f"[Worker] Mask: shape={mask.shape}, dtype={mask.dtype}", file=sys.stderr)

        # Ensure mask is uint8 in [0, 255] range
        if mask is not None and mask.dtype != np.uint8:
            if mask.max() <= 1.0:
                mask = (mask * 255).astype(np.uint8)
            else:
                mask = mask.astype(np.uint8)

        # Single-stage operations (on-demand model loading)
        unload_after = not request.get("keep_in_vram", False)

        if is_single_stage:
            print(f"[Worker] Running single stage (on-demand loading)", file=sys.stderr)

            # Stage 1 only
            if stage1_only and pointmap is not None:
                from PIL import Image as PILImage
                mask_pil = PILImage.fromarray(mask)
                return run_stage1_lazy(
                    model_manager, image, mask_pil, pointmap,
                    seed=seed,
                    inference_steps=stage1_inference_steps,
                    cfg_strength=stage1_cfg_strength,
                    unload_after=unload_after,
                    output_dir=output_dir
                )

            # Stage 2 only
            if slat_only and stage1_output is not None:
                from PIL import Image as PILImage
                mask_pil = PILImage.fromarray(mask)
                return run_stage2_lazy(
                    model_manager, image, mask_pil, stage1_output,
                    seed=seed,
                    inference_steps=stage2_inference_steps,
                    cfg_strength=stage2_cfg_strength,
                    unload_after=unload_after,
                    output_dir=output_dir
                )

            # Gaussian decode only
            if gaussian_only and slat_output is not None:
                slat_data = slat_output.get("slat") if isinstance(slat_output, dict) else slat_output
                if isinstance(slat_data, str):
                    slat_data = pickle.loads(base64.b64decode(slat_data))
                return run_decode_lazy(
                    model_manager, slat_data,
                    decode_format="gaussian",
                    unload_after=unload_after,
                    output_dir=output_dir
                )

            # Mesh decode only
            if mesh_only and slat_output is not None:
                slat_data = slat_output.get("slat") if isinstance(slat_output, dict) else slat_output
                if isinstance(slat_data, str):
                    slat_data = pickle.loads(base64.b64decode(slat_data))
                return run_decode_lazy(
                    model_manager, slat_data,
                    decode_format="mesh",
                    unload_after=unload_after,
                    output_dir=output_dir,
                    with_postprocess=with_mesh_postprocess,
                    simplify=simplify
                )

        # Run full pipeline inference
        output = model.run(
            image, mask,
            seed=seed,
            stage1_inference_steps=stage1_inference_steps,
            stage2_inference_steps=stage2_inference_steps,
            stage1_cfg_strength=stage1_cfg_strength,
            stage2_cfg_strength=stage2_cfg_strength,
            simplify=simplify,
            texture_size=texture_size,
            with_mesh_postprocess=with_mesh_postprocess,
            with_texture_baking=with_texture_baking,
            use_vertex_color=use_vertex_color,
            stage1_only=stage1_only,
            stage1_output=stage1_output,
            stage2_only=stage2_only,
            stage2_output=stage2_output,
            slat_only=slat_only,
            slat_output=slat_output,
            gaussian_only=gaussian_only,
            mesh_only=mesh_only,
            save_files=save_files,
            use_cache=use_cache,
            pointmap=pointmap,
            use_stage1_distillation=use_stage1_distillation,
            use_stage2_distillation=use_stage2_distillation,
            texture_mode=texture_mode,
            rendering_engine=rendering_engine,
            merge_mask=merge_mask,
            auto_resize_mask=auto_resize_mask,
        )

        print(f"[Worker] Inference completed", file=sys.stderr)

        # Transform to global coordinates for final outputs
        if not stage1_only and not stage2_only and not slat_only:
            output = transform_to_global_coordinates(output)

        # Handle stage1_only mode
        if stage1_only:
            print(f"[Worker] Stage 1 only - saving to disk", file=sys.stderr)
            saved_output = save_output_to_disk(output, Path(output_dir))

            rotation = output.get("rotation")
            translation = output.get("translation")
            scale = output.get("scale")

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
            }

        # Handle stage2_only mode
        if stage2_only:
            serialized_output = base64.b64encode(pickle.dumps(output)).decode('utf-8')
            return {
                "status": "success",
                "stage2_mode": True,
                "output": serialized_output
            }

        # Handle slat_only mode
        if slat_only:
            saved_output = save_output_to_disk(output, Path(output_dir))
            return {
                "status": "success",
                "stage2_mode": True,
                "output": saved_output
            }

        # Handle gaussian_only and mesh_only modes
        if gaussian_only or mesh_only:
            saved_output = save_output_to_disk(output, Path(output_dir))
            serialized_output = base64.b64encode(pickle.dumps(output)).decode('utf-8')
            return {
                "status": "success",
                "stage2_mode": True,
                "output": serialized_output,
                "file_output": saved_output
            }

        # Normal mode: Save final output
        saved_output = save_output_to_disk(output, Path(output_dir))

        return {
            "status": "success",
            "output": saved_output
        }

    except Exception as e:
        print(f"[Worker] Error during inference: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
