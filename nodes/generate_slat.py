"""SAM3DGenerateSLAT node - merged Stage 1 + Stage 2 with internal caching."""

import os
import json
import hashlib
from pathlib import Path
from typing import Any

import torch
import numpy as np
from PIL import Image

from .utils import comfy_image_to_pil, comfy_mask_to_numpy
from .subprocess_bridge import get_bridge, run_generate_slat


class SAM3DGenerateSLAT:
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
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generator": ("SAM3D_MODEL", {"tooltip": "Generator from LoadSAM3DModel"}),
                "image": ("IMAGE", {"tooltip": "Input RGB image"}),
                "mask": ("MASK", {"tooltip": "Binary mask for object"}),
                "pointmap_path": ("STRING", {"forceInput": True, "tooltip": "Path to pointmap.pt from SAM3DDepthEstimate"}),
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
                    "tooltip": "Pointmap guidance strength for Stage 1. Higher = more depth influence on structure"
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
                "use_distillation": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use distilled models for faster generation (less quality)"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("slat_path", "debug_preprocessed")
    OUTPUT_TOOLTIPS = (
        "Path to SLAT file. Pass to GaussianDecode and MeshDecode",
        "Debug: The exact 518x518 cropped image fed to DINO embedder (crop around mask bbox)",
    )
    FUNCTION = "generate_slat"
    CATEGORY = "SAM3DObjects"
    DESCRIPTION = "Generate SLAT from image+mask+depth. For batch processing, use SAM3DSceneGenerate."

    def _check_stage1_cache(self, output_dir: str, seed: int, steps: int, cfg: float, cfg_pm: float) -> bool:
        """Check if Stage 1 output exists with matching params."""
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
        except:
            pass

        return False

    def _save_stage1_metadata(self, output_dir: str, seed: int, steps: int, cfg: float, cfg_pm: float):
        """Save Stage 1 params for cache validation."""
        metadata_path = os.path.join(output_dir, "stage1_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump({
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "cfg_pm": cfg_pm,
            }, f)

    def generate_slat(
        self,
        generator,
        image,
        mask,
        pointmap_path: str,
        seed: int,
        stage1_steps: int = 12,
        stage1_cfg: float = 7.5,
        stage1_cfg_pm: float = 0.0,
        stage2_steps: int = 12,
        stage2_cfg: float = 5.0,
        use_distillation: bool = False,
    ):
        """
        Generate SLAT from image, mask, and pointmap.

        Internally runs Stage 1 (sparse) and Stage 2 (SLAT) with lazy loading.
        Caches Stage 1 output - if params match, skips Stage 1.
        """
        # Derive output_dir from pointmap_path (same directory created by DepthEstimate)
        output_dir = os.path.dirname(pointmap_path)

        # Convert inputs
        image_pil = comfy_image_to_pil(image)
        mask_np = comfy_mask_to_numpy(mask)

        # Check Stage 1 cache
        use_cached_stage1 = self._check_stage1_cache(output_dir, seed, stage1_steps, stage1_cfg, stage1_cfg_pm)

        if use_cached_stage1:
            print(f"[SAM3DObjects] GenerateSLAT: Using cached Stage 1 output")
        else:
            print(f"[SAM3DObjects] GenerateSLAT: Running Stage 1 (sparse structure)...")

        print(f"[SAM3DObjects] GenerateSLAT: Running Stage 2 (SLAT generation)...")

        # Get bridge and run combined generation
        bridge = get_bridge()

        # Get config path from generator model
        config_path = generator.config_path

        try:
            result = run_generate_slat(
                bridge=bridge,
                config_path=config_path,
                image=image_pil,
                mask=mask_np,
                pointmap_path=pointmap_path,
                output_dir=output_dir,
                seed=seed,
                stage1_steps=stage1_steps,
                stage1_cfg=stage1_cfg,
                stage1_cfg_pm=stage1_cfg_pm,
                stage2_steps=stage2_steps,
                stage2_cfg=stage2_cfg,
                skip_stage1=use_cached_stage1,
                use_distillation=use_distillation,
            )
        except Exception as e:
            raise RuntimeError(f"SAM3D SLAT generation failed: {e}") from e

        # Save Stage 1 metadata for future cache hits (if we ran Stage 1)
        if not use_cached_stage1:
            self._save_stage1_metadata(output_dir, seed, stage1_steps, stage1_cfg, stage1_cfg_pm)

        # Extract SLAT path
        slat_path = result.get("slat_path")
        if not slat_path:
            # Check files dict
            if "files" in result and "slat" in result["files"]:
                slat_path = result["files"]["slat"]

        if not slat_path:
            raise RuntimeError("SLAT file was not generated")

        # Load debug image if available
        debug_image = None
        debug_image_path = result.get("debug_image")
        if not debug_image_path and "files" in result:
            debug_image_path = result["files"].get("debug_image")

        if debug_image_path and os.path.exists(debug_image_path):
            try:
                pil_img = Image.open(debug_image_path).convert("RGB")
                img_np = np.array(pil_img).astype(np.float32) / 255.0
                debug_image = torch.from_numpy(img_np).unsqueeze(0)  # [1, H, W, C]
                print(f"[SAM3DObjects] Debug image loaded: {debug_image.shape}")
            except Exception as e:
                print(f"[SAM3DObjects] Failed to load debug image: {e}")

        # Create placeholder if no debug image
        if debug_image is None:
            debug_image = torch.zeros(1, 64, 64, 3)  # Placeholder

        print(f"[SAM3DObjects] GenerateSLAT completed: {slat_path}")
        return (slat_path, debug_image)
