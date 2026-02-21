"""SAM3DSceneGenerate node - phase-based batch processing of masks to 3D objects."""

import logging
import os
from typing import Any

import numpy as np

log = logging.getLogger("sam3dobjects")


class SAM3DSceneGenerate:
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
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generator": ("SAM3D_MODEL", {"tooltip": "Generator from LoadSAM3DModel"}),
                "slat_decoder_gs": ("SAM3D_MODEL", {"tooltip": "Gaussian decoder from LoadSAM3DModel (needed if add_textures=True)"}),
                "slat_decoder_mesh": ("SAM3D_MODEL", {"tooltip": "Mesh decoder from LoadSAM3DModel"}),
                "image": ("IMAGE", {"tooltip": "Input RGB image"}),
                "masks": ("MASK", {"tooltip": "Batch of masks [N, H, W] - each becomes a 3D object"}),
                "intrinsics": ("SAM3D_INTRINSICS", {"tooltip": "Camera intrinsics from SAM3DDepthEstimate"}),
                "pointmap": ("SAM3D_POINTMAP", {"tooltip": "Pointmap tensor from SAM3DDepthEstimate"}),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 2**31 - 1,
                    "control_after_generate": "fixed",
                    "tooltip": "Random seed for generation"
                }),
            },
            "optional": {
                "stage1_steps": ("INT", {
                    "default": 12,
                    "min": 1,
                    "max": 50,
                    "tooltip": "Inference steps for Stage 1 (sparse structure). 12 = fast, 25 = quality"
                }),
                "stage1_cfg": ("FLOAT", {
                    "default": 7.5,
                    "min": 1.0,
                    "max": 15.0,
                    "step": 0.5,
                    "tooltip": "CFG strength for Stage 1"
                }),
                "stage1_cfg_pm": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.5,
                    "tooltip": "Pointmap guidance strength for Stage 1"
                }),
                "stage2_steps": ("INT", {
                    "default": 12,
                    "min": 1,
                    "max": 50,
                    "tooltip": "Inference steps for Stage 2 (SLAT). 12 = fast, 25 = quality"
                }),
                "stage2_cfg": ("FLOAT", {
                    "default": 5.0,
                    "min": 1.0,
                    "max": 15.0,
                    "step": 0.5,
                    "tooltip": "CFG strength for Stage 2"
                }),
                "with_postprocess": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Apply mesh simplification + hole filling"
                }),
                "simplify": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.5,
                    "max": 0.98,
                    "step": 0.01,
                    "tooltip": "Mesh simplification ratio (0.95 = keep 5%)"
                }),
                "add_textures": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable Gaussian decode + texture baking for higher quality output"
                }),
                "texture_mode": (["opt", "fast"], {
                    "default": "opt",
                    "tooltip": "Texture baking mode (only used if add_textures=True)"
                }),
                "texture_size": ("INT", {
                    "default": 1024,
                    "min": 512,
                    "max": 4096,
                    "step": 512,
                    "tooltip": "Texture resolution (only used if add_textures=True)"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_folder",)
    OUTPUT_TOOLTIPS = (
        "Path to folder containing all generated GLB files (object_0/, object_1/, etc.)",
    )
    FUNCTION = "generate_scene"
    CATEGORY = "SAM3DObjects"
    DESCRIPTION = "Batch process multiple masks to 3D objects. Each mask becomes a separate GLB mesh."

    def generate_scene(
        self,
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
        from .utils.stages import run_stage1, run_stage2, run_decode, run_texture_bake_direct
        from .utils.helpers import ensure_decoder_files

        # Get batch size from mask tensor [N, H, W]
        if len(masks.shape) == 3:
            batch_size = masks.shape[0]
        else:
            batch_size = 1
            masks = masks.unsqueeze(0)

        log.info("SceneGenerate: Processing %d object(s) with phase-based batching", batch_size)

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

            # Save mask and pointmap for pose optimization
            np.save(os.path.join(object_dir, "mask.npy"), mask_np)
            pointmap_path = os.path.join(object_dir, "pointmap.pt")
            if not os.path.exists(pointmap_path):
                torch.save({"pointmap": pointmap}, pointmap_path)

        # ==================================================================
        # PHASE 1: Stage 1 (Sparse Structure) for ALL masks
        # ==================================================================
        log.info("========== PHASE 1: Stage 1 (Sparse Gen) — %d objects ==========", batch_size)
        stage1_outputs = []
        for idx, (object_dir, mask_pil) in enumerate(zip(object_dirs, mask_pils)):
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
            sparse_file = result["output"]["files"]["sparse_structure"]
            stage1_outputs.append(torch.load(sparse_file, weights_only=False))

        # ==================================================================
        # PHASE 2: Stage 2 (SLAT Gen) for ALL sparse structures
        # ==================================================================
        log.info("========== PHASE 2: Stage 2 (SLAT Gen) — %d objects ==========", batch_size)
        slat_files = []
        for idx, (object_dir, mask_pil, stage1_output) in enumerate(
            zip(object_dirs, mask_pils, stage1_outputs)
        ):
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
            slat_files.append(result["output"]["files"]["slat"])

        # ==================================================================
        # PHASE 3: Mesh Decode for ALL SLATs
        # ==================================================================
        log.info("========== PHASE 3: Mesh Decode — %d objects ==========", batch_size)
        glb_paths = []
        for idx, (object_dir, slat_file) in enumerate(zip(object_dirs, slat_files)):
            log.info("Mesh decode [%d/%d]...", idx + 1, batch_size)
            slat_data = torch.load(slat_file, weights_only=False)
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
            glb_path = result.get("output", {}).get("files", {}).get("glb")
            glb_paths.append(glb_path)
            log.info("Mesh decode [%d]: %s", idx, glb_path)

        # ==================================================================
        # PHASE 4 (Optional): Gaussian Decode + Texture Bake
        # ==================================================================
        if add_textures:
            log.info("========== PHASE 4: Gaussian + Texture Bake — %d objects ==========", batch_size)

            # 4a: Gaussian decode all
            ply_paths = []
            for idx, (object_dir, slat_file) in enumerate(zip(object_dirs, slat_files)):
                log.info("Gaussian decode [%d/%d]...", idx + 1, batch_size)
                slat_data = torch.load(slat_file, weights_only=False)
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
                        if textured_glb:
                            log.info("Texture bake [%d]: %s", idx, textured_glb)
                except Exception as e:
                    log.warning("Texture bake failed for object %d: %s", idx, e)

        log.info("SceneGenerate: Completed %d object(s)", batch_size)
        log.info("SceneGenerate: Output folder: %s", base_output_dir)

        return (base_output_dir,)
