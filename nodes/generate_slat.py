"""SAM3DGenerateSLAT node - merged Stage 1 + Stage 2 with internal caching."""

import os
import json
from typing import Any

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
        except:
            pass

        return False

    def _save_stage1_metadata(self, output_dir: str, seed: int, steps: int, cfg: float, cfg_pm: float):
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

    def generate_slat(
        self,
        generator,
        image,  # torch.Tensor [B, H, W, C]
        mask,   # torch.Tensor [B, H, W] or [H, W]
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

        This method runs in an isolated subprocess with its own Python environment.

        Internally runs Stage 1 (sparse) and Stage 2 (SLAT) with lazy loading.
        Caches Stage 1 output - if params match, skips Stage 1.
        """
        # These imports happen in the isolated subprocess
        import os
        import torch
        import numpy as np
        from pathlib import Path
        from PIL import Image

        from .utils.stages import run_stage1, run_stage2
        from .utils.helpers import load_pointmap_from_file

        print(f"[SAM3DObjects] GenerateSLAT: Starting SLAT generation...")

        # Derive output_dir from pointmap_path (same directory created by DepthEstimate)
        output_dir = os.path.dirname(pointmap_path)

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

        # Load pointmap
        print(f"[SAM3DObjects] Loading pointmap from: {pointmap_path}")
        pointmap = load_pointmap_from_file(pointmap_path)
        print(f"[SAM3DObjects] Pointmap shape: {pointmap.shape}")

        # Get config path from generator model
        config_path = generator["config_path"]

        # Check Stage 1 cache
        use_cached_stage1 = self._check_stage1_cache(output_dir, seed, stage1_steps, stage1_cfg, stage1_cfg_pm)
        sparse_path = os.path.join(output_dir, "sparse_structure.pt")
        debug_image_path = None

        # Stage 1: Sparse structure generation
        if use_cached_stage1 and os.path.exists(sparse_path):
            print(f"[SAM3DObjects] Using cached Stage 1 output")
            stage1_output = torch.load(sparse_path, weights_only=False)
            # Check for cached debug image
            cached_debug = os.path.join(output_dir, "debug_preprocessed_stage1.png")
            if os.path.exists(cached_debug):
                debug_image_path = cached_debug
        else:
            print(f"[SAM3DObjects] Running Stage 1 (sparse structure)...")
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
            )
            # Load sparse structure from saved file
            stage1_file = result["output"]["files"]["sparse_structure"]
            stage1_output = torch.load(stage1_file, weights_only=False)
            debug_image_path = result.get("debug_image")

            # Copy to expected location if different
            if stage1_file != sparse_path:
                torch.save(stage1_output, sparse_path)
            self._save_stage1_metadata(output_dir, seed, stage1_steps, stage1_cfg, stage1_cfg_pm)
            print(f"[SAM3DObjects] Stage 1 complete, saved to: {sparse_path}")

        # Stage 2: SLAT generation
        print(f"[SAM3DObjects] Running Stage 2 (SLAT generation)...")
        stage2_result = run_stage2(
            config_path,
            image_pil,
            mask_pil,
            stage1_output,
            seed=seed,
            inference_steps=stage2_steps,
            cfg_strength=stage2_cfg,
            output_dir=output_dir,
        )
        # Get SLAT path from stage2 result (already saved by run_stage2_lazy)
        # Note: The saved file contains full dict with stage1_data for pose info
        slat_path = stage2_result["output"]["files"]["slat"]
        print(f"[SAM3DObjects] Stage 2 complete: {slat_path}")

        # Load debug image if available
        debug_image = None
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
