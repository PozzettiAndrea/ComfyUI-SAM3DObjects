"""SAM3DGenerateSLAT node - merged Stage 1 + Stage 2 with internal caching."""

import logging
import os
import json
from typing import Any

import torch
from comfy_api.latest import io

log = logging.getLogger("sam3dobjects")

class SAM3DGenerateSLAT(io.ComfyNode):
    """
    Generate SLAT (Structured Latent).

    Combines Stage 1 (sparse structure) and Stage 2 (SLAT generation) into one node.
    Uses internal caching - if Stage 1 was already computed with same params, skips it.

    Lazy loading ensures low VRAM usage:
    - Stage 1 models loaded (~8GB), run, unloaded
    - Stage 2 models loaded (~6GB), run, unloaded
    - Peak VRAM: ~8-9GB (not cumulative)
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SAM3DGenerateSLAT",
            category="SAM3DObjects",
            description="Generate SLAT from image+mask+depth. For batch processing, use SAM3DSceneGenerate.",
            inputs=[
                io.Custom("SAM3D_MODEL").Input("generator", tooltip="Generator from LoadSAM3DModel"),
                io.Image.Input("image", tooltip="Input RGB image"),
                io.Mask.Input("mask", tooltip="Binary mask for object"),
                io.Custom("SAM3D_POINTMAP").Input("pointmap", tooltip="Pointmap tensor from SAM3DDepthEstimate"),
                io.Int.Input("seed",
                    default=42,
                    min=0,
                    max=2**31 - 1,
                    control_after_generate="fixed",
                    tooltip="Random seed for generation"
                ),
                io.Int.Input("stage1_steps",
                    default=25,
                    min=1,
                    max=50,
                    tooltip="Inference steps for Stage 1 (sparse structure). 25 = quality (original default)",
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
                    tooltip="Pointmap guidance strength for Stage 1. Higher = more depth influence on structure",
                    optional=True,
                ),
                io.Int.Input("stage2_steps",
                    default=25,
                    min=1,
                    max=50,
                    tooltip="Inference steps for Stage 2 (SLAT). 25 = quality (original default)",
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
                io.Boolean.Input("use_distillation",
                    default=False,
                    tooltip="Use distilled models for faster generation (less quality)",
                    optional=True,
                ),
            ],
            outputs=[
                io.String.Output(display_name="slat_path", tooltip="Path to SLAT file. Pass to GaussianDecode and MeshDecode"),
                io.Image.Output(display_name="debug_preprocessed", tooltip="Debug: The exact 518x518 cropped image fed to DINO embedder (crop around mask bbox)"),
            ],
        )

    @staticmethod
    def _check_stage1_cache(output_dir: str, seed: int, steps: int, cfg: float, cfg_pm: float) -> bool:
        """Check if Stage 1 output exists with matching params."""
        import os
        import json

        sparse_path = os.path.join(output_dir, "sparse_structure.pt")
        metadata_path = os.path.join(output_dir, "stage1_metadata.json")

        if not os.path.exists(sparse_path):
            return False

        if not os.path.exists(metadata_path):
            return False

        try:
            with open(metadata_path, 'r') as f:
                cached = json.load(f)

            # Check if params match
            if (cached.get("seed") == seed and
                cached.get("steps") == steps and
                cached.get("cfg") == cfg and
                cached.get("cfg_pm", 0.0) == cfg_pm):
                return True
        except Exception as e:
            log.debug("Failed to read SLAT cache metadata: %s", e)

        return False

    @staticmethod
    def _get_next_run_dir(base_output_dir: str) -> str:
        """Find next available sam3d_run_N directory in the output folder."""
        existing = []
        if os.path.exists(base_output_dir):
            for name in os.listdir(base_output_dir):
                if name.startswith("sam3d_run_") and os.path.isdir(os.path.join(base_output_dir, name)):
                    try:
                        num = int(name.split("_")[-1])
                        existing.append(num)
                    except ValueError:
                        pass
        next_num = max(existing) + 1 if existing else 1
        run_dir = os.path.join(base_output_dir, f"sam3d_run_{next_num}")
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    @staticmethod
    def _save_stage1_metadata(output_dir: str, seed: int, steps: int, cfg: float, cfg_pm: float):
        """Save Stage 1 params for cache validation."""
        import json

        metadata_path = os.path.join(output_dir, "stage1_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump({
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "cfg_pm": cfg_pm,
            }, f)

    @classmethod
    @torch.no_grad()
    def execute(
        cls,
        generator,
        image,  # torch.Tensor [B, H, W, C]
        mask,   # torch.Tensor [B, H, W] or [H, W]
        pointmap,
        seed: int,
        stage1_steps: int = 25,
        stage1_cfg: float = 7.5,
        stage1_cfg_pm: float = 0.0,
        stage2_steps: int = 25,
        stage2_cfg: float = 5.0,
        use_distillation: bool = False,
    ):
        """
        Generate SLAT from image, mask, and pointmap.

        This method runs in an isolated subprocess with its own Python environment.

        Internally runs Stage 1 (sparse) and Stage 2 (SLAT) with lazy loading.
        Caches Stage 1 output - if params match, skips Stage 1.
        """
        # These imports happen in the isolated subprocess
        import os
        import torch
        import comfy.utils
        import numpy as np
        from pathlib import Path
        from PIL import Image

        import folder_paths
        import comfy.model_management as mm
        from .utils.stages import run_stage1, run_stage2
        from .utils.vram_log import vram

        log.info("GenerateSLAT: Starting SLAT generation...")
        vram("GenerateSLAT: start")

        # Create per-run output directory
        output_dir = cls._get_next_run_dir(folder_paths.get_output_directory())

        # Convert ComfyUI IMAGE to PIL
        if image.dim() == 4:
            image_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        else:
            image_np = (image.cpu().numpy() * 255).astype(np.uint8)
        image_pil = Image.fromarray(image_np)

        # Convert ComfyUI MASK to PIL (ensure 2D)
        mask_np = mask.squeeze().cpu().numpy()
        if mask_np.ndim == 3:
            mask_np = mask_np[..., 0]  # Take first channel if RGB
        mask_np = (mask_np * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_np)

        # Move pointmap to device
        device = mm.get_torch_device()
        pointmap = pointmap.to(device)
        log.info("Pointmap shape: %s", pointmap.shape)

        # Get config path from generator model
        config_path = generator["config_path"]

        # Check Stage 1 cache
        use_cached_stage1 = cls._check_stage1_cache(output_dir, seed, stage1_steps, stage1_cfg, stage1_cfg_pm)
        sparse_path = os.path.join(output_dir, "sparse_structure.pt")
        debug_image_path = None

        # Stage 1: Sparse structure generation
        if use_cached_stage1 and os.path.exists(sparse_path):
            log.info("Using cached Stage 1 output")
            stage1_output = comfy.utils.load_torch_file(sparse_path, safe_load=False)
            # Check for cached debug image
            cached_debug = os.path.join(output_dir, "debug_preprocessed_stage1.png")
            if os.path.exists(cached_debug):
                debug_image_path = cached_debug
        else:
            log.info("Running Stage 1 (sparse structure)...")
            result = run_stage1(
                config_path,
                image_pil,
                mask_pil,
                pointmap,
                seed=seed,
                inference_steps=stage1_steps,
                cfg_strength=stage1_cfg,
                cfg_strength_pm=stage1_cfg_pm,
                output_dir=output_dir,
                precision=generator.get("precision", "bf16"),
            )
            # Load sparse structure from saved file
            stage1_file = result["output"]["files"]["sparse_structure"]
            stage1_output = comfy.utils.load_torch_file(stage1_file, safe_load=False)
            debug_image_path = result.get("debug_image")

            # Copy to expected location if different
            if stage1_file != sparse_path:
                torch.save(stage1_output, sparse_path)
            cls._save_stage1_metadata(output_dir, seed, stage1_steps, stage1_cfg, stage1_cfg_pm)
            log.info("Stage 1 complete, saved to: %s", sparse_path)

        # Stage 2: SLAT generation
        log.info("Running Stage 2 (SLAT generation)...")
        stage2_result = run_stage2(
            config_path,
            image_pil,
            mask_pil,
            stage1_output,
            seed=seed,
            inference_steps=stage2_steps,
            cfg_strength=stage2_cfg,
            output_dir=output_dir,
            precision=generator.get("precision", "bf16"),
        )
        # Get SLAT path from stage2 result (already saved by run_stage2_lazy)
        # Note: The saved file contains full dict with stage1_data for pose info
        slat_path = stage2_result["output"]["files"]["slat"]
        log.info("Stage 2 complete: %s", slat_path)

        # Load debug image if available
        debug_image = None
        if debug_image_path and os.path.exists(debug_image_path):
            try:
                pil_img = Image.open(debug_image_path).convert("RGB")
                img_np = np.array(pil_img).astype(np.float32) / 255.0
                debug_image = torch.from_numpy(img_np).unsqueeze(0)  # [1, H, W, C]
                log.debug("Debug image loaded: %s", debug_image.shape)
            except Exception as e:
                log.warning("Failed to load debug image: %s", e)

        # Create placeholder if no debug image
        if debug_image is None:
            debug_image = torch.zeros(1, 64, 64, 3)  # Placeholder

        vram("GenerateSLAT: done")
        log.info("GenerateSLAT completed: %s", slat_path)
        return io.NodeOutput(slat_path, debug_image)
