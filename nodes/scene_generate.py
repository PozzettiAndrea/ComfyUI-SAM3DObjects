"""SAM3DSceneGenerate node - phase-based batch processing of masks to 3D objects."""

import logging
import os
import shutil
from typing import Any

import numpy as np
import torch

from comfy_api.latest import io

log = logging.getLogger("sam3dobjects")


class SAM3DSceneGenerate(io.ComfyNode):
    """
    Scene Generation - Batch process multiple masks to 3D objects.

    Takes an image and batch of masks, generates a separate 3D mesh for each mask.
    Uses phase-based processing for efficiency (models loaded once per phase):
      Phase 1: Stage 1 (sparse) for ALL masks
      Phase 2: Stage 2 (SLAT) for ALL masks
      Phase 3: Mesh decode for ALL SLATs
      Phase 4: (optional) Gaussian decode + texture bake for ALL objects

    All objects share the same pointmap from depth estimation.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SAM3DSceneGenerate",
            category="SAM3DObjects",
            description="Batch process multiple masks to 3D objects. Each mask becomes a separate GLB mesh.",
            inputs=[
                io.Custom("SAM3D_MODEL").Input("generator", tooltip="Generator from LoadSAM3DModel"),
                io.Custom("SAM3D_MODEL").Input("slat_decoder_gs", tooltip="Gaussian decoder from LoadSAM3DModel (needed if add_textures=True)"),
                io.Custom("SAM3D_MODEL").Input("slat_decoder_mesh", tooltip="Mesh decoder from LoadSAM3DModel"),
                io.Image.Input("image", tooltip="Input RGB image"),
                io.Mask.Input("masks", tooltip="Batch of masks [N, H, W] - each becomes a 3D object"),
                io.Custom("SAM3D_INTRINSICS").Input("intrinsics", tooltip="Camera intrinsics from SAM3DDepthEstimate"),
                io.Custom("SAM3D_POINTMAP").Input("pointmap", tooltip="Pointmap tensor from SAM3DDepthEstimate"),
                io.Int.Input("seed",
                    default=42,
                    min=0,
                    max=2**31 - 1,
                    control_after_generate="fixed",
                    tooltip="Random seed for generation"
                ),
                io.Int.Input("stage1_steps",
                    default=12,
                    min=1,
                    max=50,
                    tooltip="Inference steps for Stage 1 (sparse structure). 12 = fast, 25 = quality",
                    optional=True,
                ),
                io.Float.Input("stage1_cfg",
                    default=7.5,
                    min=1.0,
                    max=15.0,
                    step=0.5,
                    tooltip="CFG strength for Stage 1",
                    optional=True,
                ),
                io.Float.Input("stage1_cfg_pm",
                    default=0.0,
                    min=0.0,
                    max=10.0,
                    step=0.5,
                    tooltip="Pointmap guidance strength for Stage 1",
                    optional=True,
                ),
                io.Int.Input("stage2_steps",
                    default=12,
                    min=1,
                    max=50,
                    tooltip="Inference steps for Stage 2 (SLAT). 12 = fast, 25 = quality",
                    optional=True,
                ),
                io.Float.Input("stage2_cfg",
                    default=5.0,
                    min=1.0,
                    max=15.0,
                    step=0.5,
                    tooltip="CFG strength for Stage 2",
                    optional=True,
                ),
                io.Boolean.Input("with_postprocess",
                    default=False,
                    tooltip="Apply mesh simplification + hole filling",
                    optional=True,
                ),
                io.Float.Input("simplify",
                    default=0.95,
                    min=0.5,
                    max=0.98,
                    step=0.01,
                    tooltip="Mesh simplification ratio (0.95 = keep 5%)",
                    optional=True,
                ),
                io.Boolean.Input("add_textures",
                    default=False,
                    tooltip="Enable Gaussian decode + texture baking for higher quality output",
                    optional=True,
                ),
                io.Combo.Input("texture_mode", options=["opt", "fast"],
                    default="opt",
                    tooltip="Texture baking mode (only used if add_textures=True)",
                    optional=True,
                ),
                io.Int.Input("texture_size",
                    default=1024,
                    min=512,
                    max=4096,
                    step=512,
                    tooltip="Texture resolution (only used if add_textures=True)",
                    optional=True,
                ),
                io.Int.Input("stage1_batch_size",
                    default=1,
                    min=1,
                    max=16,
                    tooltip="Process N masks through Stage 1 diffusion simultaneously. Higher = faster but more VRAM. 1 = sequential (safe). Try 4-8 on 24GB+.",
                    optional=True,
                ),
            ],
            outputs=[
                io.String.Output(display_name="output_folder", tooltip="Path to folder containing all generated GLB files (object_0/, object_1/, etc.)"),
            ],
        )

    @classmethod
    @torch.no_grad()
    def execute(
        cls,
        generator,
        slat_decoder_gs,
        slat_decoder_mesh,
        image,  # torch.Tensor [B, H, W, C]
        masks,  # torch.Tensor [N, H, W]
        intrinsics,  # numpy array
        pointmap,
        seed: int,
        stage1_steps: int = 12,
        stage1_cfg: float = 7.5,
        stage1_cfg_pm: float = 0.0,
        stage2_steps: int = 12,
        stage2_cfg: float = 5.0,
        with_postprocess: bool = False,
        simplify: float = 0.95,
        add_textures: bool = False,
        texture_mode: str = "opt",
        texture_size: int = 1024,
        stage1_batch_size: int = 1,
        **kwargs,
    ):
        """
        Generate 3D objects for each mask using phase-based batching.

        Models are loaded once per phase and reused for all objects,
        minimising GPU<->CPU model swaps.
        """
        import torch
        from PIL import Image
        from pathlib import Path

        import folder_paths
        import comfy.model_management
        from .utils.stages import run_stage1, run_stage2, run_decode, run_texture_bake_direct
        from .utils.helpers import ensure_decoder_files
        from .utils.vram_log import vram

        # Get batch size from mask tensor [N, H, W]
        if len(masks.shape) == 3:
            batch_size = masks.shape[0]
        else:
            batch_size = 1
            masks = masks.unsqueeze(0)

        log.info("SceneGenerate: Processing %d object(s) with phase-based batching", batch_size)
        vram("SceneGenerate: start")

        # Create output directory
        output_root = folder_paths.get_output_directory()
        existing = []
        for name in os.listdir(output_root):
            if name.startswith("sam3d_scene_") and os.path.isdir(os.path.join(output_root, name)):
                try:
                    existing.append(int(name.split("_")[-1]))
                except ValueError:
                    pass
        next_num = max(existing) + 1 if existing else 1
        base_output_dir = os.path.join(output_root, f"sam3d_scene_{next_num}")
        os.makedirs(base_output_dir, exist_ok=True)

        # Convert ComfyUI IMAGE to PIL
        if image.dim() == 4:
            image_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        else:
            image_np = (image.cpu().numpy() * 255).astype(np.uint8)
        image_pil = Image.fromarray(image_np)

        # Save intrinsics and image for downstream use (pose optimization etc.)
        torch.save(intrinsics, os.path.join(base_output_dir, "intrinsics.pt"))
        image_pil.save(os.path.join(base_output_dir, "image.png"))

        # Get config paths
        config_path = generator["config_path"]
        precision = generator.get("precision", "bf16")
        mesh_config = slat_decoder_mesh["config_path"]

        # Ensure decoder files are downloaded
        ensure_decoder_files(mesh_config, "mesh")
        if add_textures:
            gs_config = slat_decoder_gs["config_path"]
            ensure_decoder_files(gs_config, "gaussian")

        # Save pointmap once (shared across all objects)
        pointmap_path = os.path.join(base_output_dir, "pointmap.pt")
        if not os.path.exists(pointmap_path):
            torch.save({"pointmap": pointmap.clone()}, pointmap_path)

        # Prepare per-object directories and masks
        object_dirs = []
        mask_pils = []
        for idx in range(batch_size):
            object_dir = os.path.join(base_output_dir, f"object_{idx}")
            os.makedirs(object_dir, exist_ok=True)
            object_dirs.append(object_dir)

            mask_np = masks[idx].cpu().numpy()
            if mask_np.dtype in (np.float32, np.float64):
                mask_uint8 = (mask_np * 255).astype(np.uint8)
            else:
                mask_uint8 = mask_np.astype(np.uint8)
            mask_pils.append(Image.fromarray(mask_uint8))

            # Save mask per-object (each is different)
            np.save(os.path.join(object_dir, "mask.npy"), mask_np)

        # ==================================================================
        # PHASE 1: Stage 1 (Sparse Structure) for ALL masks
        # ==================================================================
        vram("SceneGenerate: PHASE 1 start")
        log.info("========== PHASE 1: Stage 1 (Sparse Gen) -- %d objects ==========", batch_size)
        stage1_outputs = []
        for idx, (object_dir, mask_pil) in enumerate(zip(object_dirs, mask_pils)):
            comfy.model_management.throw_exception_if_processing_interrupted()
            log.info("Stage 1 [%d/%d]...", idx + 1, batch_size)
            result = run_stage1(
                config_path,
                image_pil,
                mask_pil,
                pointmap,
                seed=seed,
                inference_steps=stage1_steps,
                cfg_strength=stage1_cfg,
                cfg_strength_pm=stage1_cfg_pm,
                output_dir=object_dir,
                precision=precision,
            )
            # Use in-memory data directly (avoids torch.load round-trip)
            stage1_outputs.append(result["data"])
            # Prevent reserved pool growth between objects
            comfy.model_management.soft_empty_cache()

        # ==================================================================
        # PHASE 2: Stage 2 (SLAT Gen) for ALL sparse structures
        # ==================================================================
        vram("SceneGenerate: PHASE 2 start")
        log.info("========== PHASE 2: Stage 2 (SLAT Gen) -- %d objects ==========", batch_size)
        slat_data_list = []
        for idx, (object_dir, mask_pil, stage1_output) in enumerate(
            zip(object_dirs, mask_pils, stage1_outputs)
        ):
            comfy.model_management.throw_exception_if_processing_interrupted()
            log.info("Stage 2 [%d/%d]...", idx + 1, batch_size)
            result = run_stage2(
                config_path,
                image_pil,
                mask_pil,
                stage1_output,
                seed=seed,
                inference_steps=stage2_steps,
                cfg_strength=stage2_cfg,
                output_dir=object_dir,
                precision=precision,
            )
            # Keep SLAT in memory (avoids 2N torch.load round-trips in Phase 3+4)
            slat_data_list.append(result["data"])

        # Free stage1 outputs (no longer needed)
        del stage1_outputs

        # ==================================================================
        # PHASE 3: Mesh Decode for ALL SLATs
        # ==================================================================
        vram("SceneGenerate: PHASE 3 start")
        log.info("========== PHASE 3: Mesh Decode -- %d objects ==========", batch_size)
        glb_paths = []
        for idx, (object_dir, slat_data) in enumerate(zip(object_dirs, slat_data_list)):
            comfy.model_management.throw_exception_if_processing_interrupted()
            log.info("Mesh decode [%d/%d]...", idx + 1, batch_size)
            result = run_decode(
                mesh_config,
                slat_data=slat_data,
                decode_format="mesh",
                output_dir=object_dir,
                with_postprocess=with_postprocess,
                simplify=simplify,
                up_axis="Y-up (standard)",
                world_coordinates=False,
                precision=precision,
            )
            glb_in_object = result.get("output", {}).get("files", {}).get("glb")
            if glb_in_object and os.path.exists(glb_in_object):
                base_glb = os.path.join(base_output_dir, f"object_{idx}.glb")
                shutil.move(glb_in_object, base_glb)
                glb_paths.append(base_glb)
                log.info("Mesh decode [%d]: %s", idx, base_glb)
            else:
                glb_paths.append(None)
                log.warning("Mesh decode [%d]: no GLB produced", idx)

        # ==================================================================
        # PHASE 4 (Optional): Gaussian Decode + Texture Bake
        # ==================================================================
        if add_textures:
            vram("SceneGenerate: PHASE 4 start")
            log.info("========== PHASE 4: Gaussian + Texture Bake -- %d objects ==========", batch_size)

            # 4a: Gaussian decode all (reuse in-memory SLATs)
            ply_paths = []
            for idx, (object_dir, slat_data) in enumerate(zip(object_dirs, slat_data_list)):
                comfy.model_management.throw_exception_if_processing_interrupted()
                log.info("Gaussian decode [%d/%d]...", idx + 1, batch_size)
                result = run_decode(
                    gs_config,
                    slat_data=slat_data,
                    decode_format="gaussian",
                    output_dir=object_dir,
                    up_axis="Y-up (standard)",
                    world_coordinates=False,
                    precision=precision,
                )
                ply_path = result.get("output", {}).get("files", {}).get("ply")
                ply_paths.append(ply_path)

            # 4b: Texture bake all
            for idx, (object_dir, ply_path, glb_path) in enumerate(
                zip(object_dirs, ply_paths, glb_paths)
            ):
                comfy.model_management.throw_exception_if_processing_interrupted()
                if not ply_path or not glb_path:
                    continue
                log.info("Texture bake [%d/%d]...", idx + 1, batch_size)
                try:
                    bake_result = run_texture_bake_direct({
                        "ply_path": ply_path,
                        "glb_path": glb_path,
                        "output_dir": object_dir,
                        "texture_mode": texture_mode,
                        "texture_size": texture_size,
                        "rendering_engine": "nvdiffrast",
                    })
                    if bake_result.get("status") == "success":
                        textured_glb = bake_result.get("output", {}).get("glb_path")
                        if textured_glb and os.path.exists(textured_glb):
                            base_textured = os.path.join(base_output_dir, f"object_{idx}_textured.glb")
                            shutil.move(textured_glb, base_textured)
                            # Update glb_path to prefer textured version
                            glb_paths[idx] = base_textured
                            log.info("Texture bake [%d]: %s", idx, base_textured)
                except Exception as e:
                    log.warning("Texture bake failed for object %d: %s", idx, e)

        # Free SLAT data
        del slat_data_list

        vram("SceneGenerate: done")
        log.info("SceneGenerate: Completed %d object(s)", batch_size)
        log.info("SceneGenerate: Output folder: %s", base_output_dir)

        return io.NodeOutput(base_output_dir,)
