"""SAM3DSparseGen node for generating sparse 3D structure."""

import os
import torch
from typing import Any

from .utils import (
    comfy_image_to_pil,
    comfy_mask_to_numpy,
)


class SAM3DSparseGen:
    """
    Sparse Structure Generation.

    Generates sparse voxel coordinates using the sparse structure diffusion model.
    Fast stage (~3 seconds) that produces the structural skeleton for 3D generation.

    Output can be passed to SAM3DSLATGen to continue the pipeline.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ss_generator": ("SAM3D_MODEL", {"tooltip": "Sparse structure generator from LoadSAM3DModel"}),
                "image": ("IMAGE", {"tooltip": "Input RGB image"}),
                "mask": ("MASK", {"tooltip": "Binary mask indicating the object region"}),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 2**31 - 1,
                    "control_after_generate": "fixed",
                    "tooltip": "Random seed for reproducible generation"
                }),
            },
            "optional": {
                "pointmap_path": ("STRING", {
                    "tooltip": "Path to pointmap tensor file (.pt) from SAM3D_DepthEstimate. If provided, skips internal depth estimation."
                }),
                "intrinsics": ("SAM3D_INTRINSICS", {
                    "tooltip": "Pre-computed camera intrinsics from SAM3D_DepthEstimate"
                }),
                "stage1_inference_steps": ("INT", {
                    "default": 25,
                    "min": 1,
                    "max": 100,
                    "tooltip": "Number of denoising steps for Stage 1 (sparse structure). Higher = better quality but slower"
                }),
                "stage1_cfg_strength": ("FLOAT", {
                    "default": 7.0,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.1,
                    "tooltip": "Classifier-free guidance strength for Stage 1. Higher = stronger adherence to input image"
                }),
                "use_stage1_distillation": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable distillation mode for faster inference. Disables CFG guidance but uses learned shortcuts."
                }),
            }
        }

    RETURN_TYPES = ("STRING", "SAM3D_POSE")
    RETURN_NAMES = ("sparse_structure_path", "pose")
    OUTPUT_TOOLTIPS = (
        "Path to saved sparse voxel structure - pass to SAM3DSLATGen",
        "Object pose (rotation, translation, scale)"
    )
    FUNCTION = "generate_sparse"
    CATEGORY = "SAM3DObjects"
    DESCRIPTION = "Generate sparse voxel structure from image and mask (~3 seconds)."

    def generate_sparse(
        self,
        ss_generator: Any,
        image: torch.Tensor,
        mask: torch.Tensor,
        seed: int,
        pointmap_path: str = None,
        intrinsics: Any = None,
        stage1_inference_steps: int = 25,
        stage1_cfg_strength: float = 7.0,
        use_stage1_distillation: bool = False,
    ):
        """
        Generate sparse voxel structure.

        Args:
            ss_generator: SAM3D sparse structure generator
            image: Input image tensor [B, H, W, C]
            mask: Input mask tensor [N, H, W]
            seed: Random seed
            pointmap_path: Path to pointmap tensor file (.pt) (optional - skips depth estimation if provided)
            intrinsics: Pre-computed camera intrinsics (optional)
            stage1_inference_steps: Denoising steps for Stage 1
            stage1_cfg_strength: CFG strength for Stage 1
            use_stage1_distillation: Enable distillation mode for faster inference

        Returns:
            Tuple of (sparse_structure_path, pose)
        """
        print(f"[SAM3DObjects] SparseGen: Generating sparse structure (seed: {seed})")
        if pointmap_path is not None:
            print(f"[SAM3DObjects] Using pre-computed pointmap from: {pointmap_path}")

        # Convert ComfyUI tensors to formats expected by SAM3D
        image_pil = comfy_image_to_pil(image)
        mask_np = comfy_mask_to_numpy(mask)

        print(f"[SAM3DObjects] Image size: {image_pil.size}")
        print(f"[SAM3DObjects] Mask shape: {mask_np.shape}")
        print(f"[SAM3DObjects] Sparse parameters: steps={stage1_inference_steps}, cfg={stage1_cfg_strength}, distillation={use_stage1_distillation}")

        # Derive output_dir from pointmap_path (same directory)
        output_dir = os.path.dirname(pointmap_path) if pointmap_path else None

        # Run sparse structure generation only
        try:
            print("[SAM3DObjects] Running sparse structure generation...")
            sparse_output = ss_generator(
                image_pil, mask_np,
                seed=seed,
                stage1_only=True,  # CRITICAL: Only run sparse generation
                stage1_inference_steps=stage1_inference_steps,
                stage1_cfg_strength=stage1_cfg_strength,
                use_stage1_distillation=use_stage1_distillation,
                pointmap_path=pointmap_path,  # Pass pointmap tensor path if available
                intrinsics=intrinsics,  # Pass pre-computed intrinsics if available
                output_dir=output_dir,  # Derived from pointmap_path directory
            )

        except Exception as e:
            raise RuntimeError(f"SAM3D sparse generation failed: {e}") from e

        print("[SAM3DObjects] Sparse structure generation completed!")

        # Extract file path from output
        if isinstance(sparse_output, dict) and "files" in sparse_output and "sparse_structure" in sparse_output["files"]:
            sparse_path = sparse_output["files"]["sparse_structure"]
            print(f"[SAM3DObjects] - Saved to: {sparse_path}")

            # Extract pose information
            pose = {
                "rotation": sparse_output.get("rotation"),
                "translation": sparse_output.get("translation"),
                "scale": sparse_output.get("scale"),
            }
            print(f"[SAM3DObjects] - Pose extracted: rotation={pose['rotation'] is not None}, translation={pose['translation'] is not None}, scale={pose['scale'] is not None}")

            return (sparse_path, pose)

        # Fallback/Error
        print(f"[SAM3DObjects] Warning: Could not find file path in output: {sparse_output.keys() if isinstance(sparse_output, dict) else sparse_output}")
        raise RuntimeError("Failed to get sparse structure file path from worker")
