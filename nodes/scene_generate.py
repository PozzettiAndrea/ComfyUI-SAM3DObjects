"""SAM3DSceneGenerate node - batch process multiple masks to 3D objects."""

import os
import json
import hashlib
import shutil
from pathlib import Path
from typing import Any

import torch
import numpy as np
from PIL import Image

from .utils import comfy_image_to_pil
from .subprocess_bridge import InferenceWorkerBridge, run_generate_slat, run_decode, run_texture_bake_direct


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
    def get_bridge(cls):
        node_root = Path(__file__).parent.parent
        return InferenceWorkerBridge.get_instance(node_root)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generator": ("SAM3D_MODEL", {"tooltip": "Generator from LoadSAM3DModel"}),
                "slat_decoder_gs": ("SAM3D_MODEL", {"tooltip": "Gaussian decoder from LoadSAM3DModel (needed if add_textures=True)"}),
                "slat_decoder_mesh": ("SAM3D_MODEL", {"tooltip": "Mesh decoder from LoadSAM3DModel"}),
                "image": ("IMAGE", {"tooltip": "Input RGB image"}),
                "masks": ("MASK", {"tooltip": "Batch of masks [N, H, W] - each becomes a 3D object"}),
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

    def _extract_path(self, result: dict, key: str, files_key: str) -> str:
        """Extract path from nested result structure."""
        path = result.get(f"{key}_path", None)

        if not path and "output" in result:
            output = result["output"]
            if isinstance(output, dict) and "files" in output:
                path = output["files"].get(files_key)

        if not path and "file_output" in result:
            file_output = result["file_output"]
            if isinstance(file_output, dict) and "files" in file_output:
                path = file_output["files"].get(files_key)

        if not path and "files" in result and files_key in result["files"]:
            path = result["files"][files_key]

        return path

    def generate_scene(
        self,
        generator,
        slat_decoder_gs,
        slat_decoder_mesh,
        image,
        masks,
        pointmap_path: str,
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

        Returns path to output folder containing object_0/, object_1/, etc.
        """
        # Get batch size from mask tensor [N, H, W]
        if len(masks.shape) == 3:
            batch_size = masks.shape[0]
        else:
            batch_size = 1
            masks = masks.unsqueeze(0)

        print(f"[SAM3DObjects] SceneGenerate: Processing {batch_size} object(s)")
        if add_textures:
            print(f"[SAM3DObjects] SceneGenerate: Texture baking enabled (mode={texture_mode}, size={texture_size})")

        # Derive base output dir from pointmap path
        base_output_dir = os.path.dirname(pointmap_path)

        # Convert image once (shared across all masks)
        image_pil = comfy_image_to_pil(image)

        # Get config paths from models
        generator_config = generator.config_path
        mesh_config = slat_decoder_mesh.config_path
        gs_config = slat_decoder_gs.config_path

        # Start bridge worker once for all operations
        bridge = self.get_bridge()
        bridge.start_worker()

        for idx in range(batch_size):
            print(f"\n[SAM3DObjects] SceneGenerate: === Object {idx + 1}/{batch_size} ===")

            # Extract single mask
            single_mask = masks[idx]  # [H, W]
            mask_np = single_mask.cpu().numpy()

            # Create object directory
            object_dir = os.path.join(base_output_dir, f"object_{idx}")
            os.makedirs(object_dir, exist_ok=True)

            # Copy pointmap to object directory if not exists
            object_pointmap_path = os.path.join(object_dir, "pointmap.pt")
            if not os.path.exists(object_pointmap_path):
                shutil.copy(pointmap_path, object_pointmap_path)

            # === Step 1: Generate SLAT ===
            use_cached_stage1 = self._check_stage1_cache(
                object_dir, seed, stage1_steps, stage1_cfg, stage1_cfg_pm
            )

            if use_cached_stage1:
                print(f"[SAM3DObjects] SceneGenerate [{idx}]: Using cached Stage 1")
            else:
                print(f"[SAM3DObjects] SceneGenerate [{idx}]: Running Stage 1 + 2...")

            try:
                slat_result = run_generate_slat(
                    bridge=bridge,
                    config_path=generator_config,
                    image=image_pil,
                    mask=mask_np,
                    pointmap_path=object_pointmap_path,
                    output_dir=object_dir,
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
                raise RuntimeError(f"SLAT generation failed for object {idx}: {e}") from e

            if not use_cached_stage1:
                self._save_stage1_metadata(object_dir, seed, stage1_steps, stage1_cfg, stage1_cfg_pm)

            slat_path = slat_result.get("slat_path")
            if not slat_path and "files" in slat_result:
                slat_path = slat_result["files"].get("slat")
            if not slat_path:
                raise RuntimeError(f"SLAT not generated for object {idx}")

            print(f"[SAM3DObjects] SceneGenerate [{idx}]: SLAT -> {slat_path}")

            # === Step 2: Decode to Mesh ===
            print(f"[SAM3DObjects] SceneGenerate [{idx}]: Decoding mesh...")

            try:
                mesh_result = run_decode(
                    bridge=bridge,
                    config_path=mesh_config,
                    slat_path=slat_path,
                    output_dir=object_dir,
                    decode_format="mesh",
                    with_postprocess=with_postprocess,
                    simplify=simplify,
                )
            except Exception as e:
                raise RuntimeError(f"Mesh decode failed for object {idx}: {e}") from e

            glb_path = self._extract_path(mesh_result, "glb", "glb")
            if not glb_path:
                raise RuntimeError(f"GLB not generated for object {idx}")

            print(f"[SAM3DObjects] SceneGenerate [{idx}]: Mesh -> {glb_path}")

            # === Step 3 (optional): Texture Baking ===
            if add_textures:
                # First decode Gaussian
                print(f"[SAM3DObjects] SceneGenerate [{idx}]: Decoding Gaussian...")

                try:
                    gs_result = run_decode(
                        bridge=bridge,
                        config_path=gs_config,
                        slat_path=slat_path,
                        output_dir=object_dir,
                        decode_format="gaussian",
                    )
                except Exception as e:
                    raise RuntimeError(f"Gaussian decode failed for object {idx}: {e}") from e

                ply_path = self._extract_path(gs_result, "ply", "ply")
                if not ply_path:
                    raise RuntimeError(f"PLY not generated for object {idx}")

                print(f"[SAM3DObjects] SceneGenerate [{idx}]: Gaussian -> {ply_path}")

                # Then bake texture
                print(f"[SAM3DObjects] SceneGenerate [{idx}]: Baking texture...")

                try:
                    texture_result = run_texture_bake_direct(
                        ply_path=ply_path,
                        glb_path=glb_path,
                        output_dir=object_dir,
                        texture_mode=texture_mode,
                        texture_size=texture_size,
                        rendering_engine="nvdiffrast",
                    )
                except Exception as e:
                    raise RuntimeError(f"Texture baking failed for object {idx}: {e}") from e

                textured_glb = texture_result.get("glb_path")
                if textured_glb:
                    print(f"[SAM3DObjects] SceneGenerate [{idx}]: Textured -> {textured_glb}")

            print(f"[SAM3DObjects] SceneGenerate [{idx}]: Done!")

        print(f"\n[SAM3DObjects] SceneGenerate: Completed {batch_size} object(s)")
        print(f"[SAM3DObjects] SceneGenerate: Output folder: {base_output_dir}")

        return (base_output_dir,)
