"""SAM3DSceneGenerate node - batch process multiple masks to 3D objects."""

import logging
import os
import json
from typing import Any

log = logging.getLogger("sam3dobjects")

class SAM3DSceneGenerate:
    """
    Scene Generation - Batch process multiple masks to 3D objects.

    Takes an image and batch of masks, generates a separate 3D mesh for each mask.
    Integrates the full pipeline: SLAT generation -> Mesh decoding -> (optional) Texture baking.

    Default: Outputs vertex-colored GLB meshes (fast).
    With add_textures=True: Also runs Gaussian decode + texture baking for higher quality.

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
                "use_distillation": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use distilled models for faster generation (less quality)"
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
        use_distillation: bool = False,
    ):
        """
        Generate 3D objects for each mask in the batch.

        This method runs in an isolated subprocess with its own Python environment.

        Uses phase-based batch processing for efficiency:
        - Loads Stage1 models ONCE, processes ALL masks, then unloads
        - Loads Stage2 models ONCE, processes ALL sparse structures, then unloads
        - Loads MeshDecoder ONCE, processes ALL SLATs, then unloads
        - (Optional) Loads GaussianDecoder ONCE, texture bakes ALL meshes, then unloads

        Returns path to output folder containing object_0/, object_1/, etc.
        """
        # These imports happen in the isolated subprocess
        import os
        import io
        import base64
        import pickle
        import torch
        import numpy as np
        from pathlib import Path
        from PIL import Image
        import folder_paths

        from .utils.scene_batch import run_scene_generate_batch

        # Get batch size from mask tensor [N, H, W]
        if len(masks.shape) == 3:
            batch_size = masks.shape[0]
        else:
            batch_size = 1
            masks = masks.unsqueeze(0)

        log.info("SceneGenerate: Processing %d object(s) with phase-based batching", batch_size)
        if add_textures:
            log.info("SceneGenerate: Texture baking enabled (mode=%s, size=%d)", texture_mode, texture_size)

        # Create output directory for scene generation
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

        # Save intrinsics for pose optimization
        intrinsics_path = os.path.join(base_output_dir, "intrinsics.pt")
        torch.save(intrinsics, intrinsics_path)
        log.info("SceneGenerate: Saved intrinsics to %s", intrinsics_path)

        # Save image for pose optimization (needed for render-and-compare)
        image_path = os.path.join(base_output_dir, "image.png")
        image_pil.save(image_path)
        log.info("SceneGenerate: Saved image to %s", image_path)

        # Get config paths from models
        generator_config = generator["config_path"]
        mesh_config = slat_decoder_mesh["config_path"]
        gs_config = slat_decoder_gs["config_path"] if add_textures else None

        # Serialize image to base64
        img_buffer = io.BytesIO()
        image_pil.save(img_buffer, format="PNG")
        image_b64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

        # Convert masks to list and serialize
        masks_b64 = []
        for idx in range(batch_size):
            single_mask = masks[idx]  # [H, W]
            mask_np = single_mask.cpu().numpy()
            mask_b64 = base64.b64encode(pickle.dumps(mask_np)).decode('utf-8')
            masks_b64.append(mask_b64)

            # Create object directory and save mask for pose optimization
            object_dir = os.path.join(base_output_dir, f"object_{idx}")
            os.makedirs(object_dir, exist_ok=True)
            mask_path = os.path.join(object_dir, "mask.npy")
            np.save(mask_path, mask_np)

            # Save pointmap to object directory for pose optimization
            object_pointmap_path = os.path.join(object_dir, "pointmap.pt")
            if not os.path.exists(object_pointmap_path):
                torch.save({"pointmap": pointmap}, object_pointmap_path)

        # Build request for batch processing
        request = {
            "image": image_b64,
            "masks": masks_b64,
            "pointmap": pointmap,
            "base_output_dir": base_output_dir,
            "config_path": generator_config,
            "mesh_config_path": mesh_config,
            "gs_config_path": gs_config,
            "seed": seed,
            "stage1_steps": stage1_steps,
            "stage1_cfg": stage1_cfg,
            "stage1_cfg_pm": stage1_cfg_pm,
            "stage2_steps": stage2_steps,
            "stage2_cfg": stage2_cfg,
            "with_postprocess": with_postprocess,
            "simplify": simplify,
            "add_textures": add_textures,
            "texture_mode": texture_mode,
            "texture_size": texture_size,
        }

        # Run batch processing - models are loaded once per phase
        log.info("SceneGenerate: Starting batch processing...")
        result = run_scene_generate_batch(request)

        if result.get("status") == "error":
            raise RuntimeError(f"Batch scene generation failed: {result.get('error')}")

        # Process results - meshes are now in world coordinates (pose baked in)
        objects = result.get("objects", [])
        for obj_result in objects:
            idx = obj_result.get("index", 0)

            # Log output paths
            if obj_result.get("glb_path"):
                log.info("SceneGenerate [%d]: Mesh -> %s", idx, obj_result['glb_path'])
            if obj_result.get("textured_glb_path"):
                log.info("SceneGenerate [%d]: Textured -> %s", idx, obj_result['textured_glb_path'])

        log.info("SceneGenerate: Completed %d object(s)", batch_size)
        log.info("SceneGenerate: Output folder: %s", base_output_dir)

        return (base_output_dir,)
