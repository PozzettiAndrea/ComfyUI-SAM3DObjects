"""SAM3D_ScenePoseOptimize node - optimize poses for all objects in a scene folder."""

import logging
import os
import re
import json
from typing import Any, Dict, List, Tuple

log = logging.getLogger("sam3dobjects")

class SAM3D_ScenePoseOptimize:
    """
    Scene Pose Optimization - Apply/refine poses for all objects in a scene folder.

    Takes the output_folder from SAM3DSceneGenerate and applies pose transforms
    to each object using the saved data (intrinsics, masks, poses, pointmaps).

    Supports three optimization modes:
    - pose_only: Just apply the predicted pose (fastest, no optimization)
    - icp_only: Pose + ICP point cloud alignment refinement
    - render_only: Pose + render-and-compare optimization (best quality, slowest)

    Outputs list of paths to aligned GLB files and quality scores.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_folder": ("STRING", {
                    "default": "",
                    "tooltip": "Output folder containing object_N/ subdirectories (from SAM3DSceneGenerate or manual path)"
                }),
                "optimization_mode": (["pose_only", "icp_only", "render_only"], {
                    "default": "pose_only",
                    "tooltip": "Optimization mode: pose_only (just apply predicted pose), icp_only (+ ICP refinement), render_only (+ render-and-compare optimization)"
                }),
            },
        }

    RETURN_TYPES = ("STRING", "FLOAT")
    RETURN_NAMES = ("output_folder", "iou_scores")
    OUTPUT_IS_LIST = (False, True)
    OUTPUT_TOOLTIPS = (
        "Path to folder containing aligned GLB files (object_0.glb, object_1.glb, ...)",
        "List of IOU scores (quality metric, -1 if optimization failed or skipped)"
    )
    FUNCTION = "optimize_poses"
    CATEGORY = "SAM3DObjects"
    DESCRIPTION = "Optimize poses for all objects in a scene folder using alignment algorithms."

    def _discover_objects(self, output_folder: str) -> List[str]:
        """Discover all object_N/ folders in sorted order."""
        import os
        import re

        object_dirs = []
        pattern = re.compile(r'^object_(\d+)$')

        if not os.path.exists(output_folder):
            return object_dirs

        for name in os.listdir(output_folder):
            match = pattern.match(name)
            if match:
                object_path = os.path.join(output_folder, name)
                if os.path.isdir(object_path):
                    idx = int(match.group(1))
                    object_dirs.append((idx, object_path))

        # Sort by index
        object_dirs.sort(key=lambda x: x[0])
        return [path for _, path in object_dirs]

    def _check_cache(self, pose_opt_folder: str, optimization_mode: str, num_objects: int) -> Tuple[bool, List[float]]:
        """Check if cached results exist with matching params."""
        import os
        import json

        metadata_path = os.path.join(pose_opt_folder, "pose_opt_metadata.json")

        if not os.path.exists(pose_opt_folder):
            return False, []

        if not os.path.exists(metadata_path):
            return False, []

        try:
            with open(metadata_path, 'r') as f:
                cached = json.load(f)

            if (cached.get("optimization_mode") == optimization_mode and
                cached.get("num_objects") == num_objects):
                # Return cached IoU scores
                return True, cached.get("iou_scores", [])
        except Exception as e:
            log.debug("Failed to read pose optimization cache metadata: %s", e)

        return False, []

    def _save_cache_metadata(self, pose_opt_folder: str, optimization_mode: str, num_objects: int, iou_scores: List[float]):
        """Save metadata for cache validation."""
        import json

        metadata_path = os.path.join(pose_opt_folder, "pose_opt_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump({
                "optimization_mode": optimization_mode,
                "num_objects": num_objects,
                "iou_scores": iou_scores,
            }, f)

    def optimize_poses(
        self,
        output_folder: str,
        optimization_mode: str = "manual_icp_render",
    ) -> Tuple[str, List[float]]:
        """
        Optimize poses for all objects in the scene folder.

        This method runs in an isolated subprocess with its own Python environment.

        Args:
            output_folder: Path to folder containing object_N/ subdirectories
            optimization_mode: One of "pose_only", "manual_only", "manual_icp", "manual_icp_render"

        Returns:
            Tuple of (path to output folder, list of IOU scores)
        """
        # These imports happen in the isolated subprocess
        import os
        import time
        import pickle
        import base64
        import torch
        import comfy.utils
        import numpy as np
        import trimesh
        from pathlib import Path

        from .utils.pose_optimization import run_pose_optimization_batch

        start_time = time.time()
        log.info("ScenePoseOptimize: Starting pose optimization")
        log.info("ScenePoseOptimize: Mode = %s", optimization_mode)
        log.info("ScenePoseOptimize: Folder = %s", output_folder)

        # Validate folder exists
        if not os.path.exists(output_folder):
            raise ValueError(f"Output folder does not exist: {output_folder}")

        # Check for required files
        intrinsics_path = os.path.join(output_folder, "intrinsics.pt")
        image_path = os.path.join(output_folder, "image.png")

        if not os.path.exists(intrinsics_path):
            raise ValueError(f"Intrinsics file not found: {intrinsics_path}. Run SAM3DSceneGenerate first.")

        # Load intrinsics (our own intermediate file, may contain numpy arrays)
        intrinsics = torch.load(intrinsics_path, weights_only=False)
        log.info("ScenePoseOptimize: Loaded intrinsics from %s", intrinsics_path)

        # Discover object folders
        object_dirs = self._discover_objects(output_folder)
        if not object_dirs:
            log.warning("ScenePoseOptimize: No object folders found")
            return ("", [])

        log.info("ScenePoseOptimize: Found %d object(s)", len(object_dirs))

        # Define output folder for pose-optimized meshes
        pose_opt_folder = f"{output_folder}_pose_optimized"

        # Check cache - skip if already processed with same settings
        cache_hit, cached_scores = self._check_cache(pose_opt_folder, optimization_mode, len(object_dirs))
        if cache_hit:
            log.info("ScenePoseOptimize: Using cached results from %s", pose_opt_folder)
            return (pose_opt_folder, cached_scores)

        # Determine optimization flags (no manual alignment in any mode)
        enable_icp = optimization_mode == "icp_only"
        enable_render = optimization_mode == "render_only"

        # Create output folder
        os.makedirs(pose_opt_folder, exist_ok=True)
        log.info("ScenePoseOptimize: Output folder = %s", pose_opt_folder)

        # Serialize helper
        def serialize_tensor(tensor):
            if tensor is None:
                return None
            if isinstance(tensor, torch.Tensor):
                arr = tensor.cpu().numpy()
            elif isinstance(tensor, np.ndarray):
                arr = tensor
            elif isinstance(tensor, list):
                arr = np.array(tensor)
            else:
                arr = np.array(tensor)
            return base64.b64encode(pickle.dumps(arr)).decode('utf-8')

        def serialize_pose(rotation, translation, scale):
            return {
                "rotation": serialize_tensor(rotation),
                "translation": serialize_tensor(translation),
                "scale": serialize_tensor(scale),
            }

        # Serialize intrinsics
        if isinstance(intrinsics, torch.Tensor):
            intrinsics_np = intrinsics.cpu().numpy()
        else:
            intrinsics_np = np.array(intrinsics)

        iou_scores = []

        for idx, object_dir in enumerate(object_dirs):
            log.info("ScenePoseOptimize: === Object %d/%d ===", idx + 1, len(object_dirs))

            # Check required files for this object (fall back to base dir)
            base_dir = os.path.dirname(object_dir)

            pointmap_path = os.path.join(object_dir, "pointmap.pt")
            if not os.path.exists(pointmap_path):
                pointmap_path = os.path.join(base_dir, "pointmap.pt")

            mask_path = os.path.join(object_dir, "mask.npy")

            mesh_path = os.path.join(object_dir, "mesh.glb")
            if not os.path.exists(mesh_path):
                mesh_path = os.path.join(base_dir, f"object_{idx}.glb")

            # Check for textured mesh first
            textured_mesh_path = os.path.join(object_dir, "textured_mesh.glb")
            if not os.path.exists(textured_mesh_path):
                textured_candidate = os.path.join(base_dir, f"object_{idx}_textured.glb")
                if os.path.exists(textured_candidate):
                    textured_mesh_path = textured_candidate
            if os.path.exists(textured_mesh_path):
                mesh_path = textured_mesh_path

            # Validate required files
            missing = []
            if not os.path.exists(pointmap_path):
                missing.append("pointmap.pt")
            if not os.path.exists(mask_path):
                missing.append("mask.npy")
            if not os.path.exists(mesh_path):
                missing.append("mesh.glb")

            if missing:
                log.warning("ScenePoseOptimize [%d]: Missing files: %s, skipping", idx, missing)
                iou_scores.append(-1.0)
                continue

            # Load initial pose from sparse_structure.pt (computed in Stage 1)
            sparse_path = os.path.join(object_dir, "sparse_structure.pt")
            if os.path.exists(sparse_path):
                sparse_data = torch.load(sparse_path, weights_only=False)
                rotation = sparse_data.get("rotation")
                translation = sparse_data.get("translation")
                scale = sparse_data.get("scale")

                # Convert tensors to lists for serialization
                if rotation is not None and hasattr(rotation, 'tolist'):
                    rotation = rotation.cpu().tolist() if hasattr(rotation, 'cpu') else rotation.tolist()
                if translation is not None and hasattr(translation, 'tolist'):
                    translation = translation.cpu().tolist() if hasattr(translation, 'cpu') else translation.tolist()
                if scale is not None and hasattr(scale, 'tolist'):
                    scale = scale.cpu().tolist() if hasattr(scale, 'cpu') else scale.tolist()

                # Ensure list-of-list format for serialize_pose (expects [[...]])
                if rotation is not None:
                    if not isinstance(rotation, list):
                        rotation = [rotation]
                    if not isinstance(rotation[0], list):
                        rotation = [rotation]
                if translation is not None:
                    if not isinstance(translation, list):
                        translation = [translation]
                    if not isinstance(translation[0], list):
                        translation = [translation]
                if scale is not None:
                    if not isinstance(scale, list):
                        scale = [scale]
                    if not isinstance(scale[0], list):
                        scale = [scale]

                log.info("ScenePoseOptimize [%d]: Loaded pose from sparse_structure.pt", idx)
            else:
                # Fallback to identity pose if no sparse_structure.pt
                rotation = [[1, 0, 0, 0]]  # Identity quaternion (wxyz)
                translation = [[0, 0, 0]]
                scale = [[1, 1, 1]]
                log.info("ScenePoseOptimize [%d]: No sparse_structure.pt, using identity pose", idx)

            # Output path in the pose-optimized folder
            output_glb_path = os.path.join(pose_opt_folder, f"object_{idx}.glb")

            # Handle pose_only mode - just apply the pose without optimization
            if optimization_mode == "pose_only":
                try:
                    log.info("ScenePoseOptimize [%d]: Applying pose directly (no optimization)", idx)
                    from .utils.pose_optimization import _apply_and_save_pose

                    # Load mesh
                    mesh = trimesh.load(mesh_path)
                    if isinstance(mesh, trimesh.Scene):
                        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
                        if not meshes:
                            raise ValueError("No mesh found in GLB file")
                        mesh = meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)

                    rot_np = np.array(rotation).squeeze()
                    trans_np = np.array(translation).squeeze()
                    scale_np = np.array(scale).squeeze()

                    response = _apply_and_save_pose(
                        mesh, rot_np, trans_np, scale_np,
                        output_glb_path, object_dir, mesh_path,
                        iou=-1.0, used_icp=False, used_render_opt=False
                    )

                    log.info("ScenePoseOptimize [%d]: Saved: %s", idx, output_glb_path)
                    iou_scores.append(response.get("iou", -1.0))

                except Exception as e:
                    log.error("ScenePoseOptimize [%d]: Pose-only transform failed", idx, exc_info=True)
                    iou_scores.append(-1.0)
                continue

            # Load mask for optimization modes
            mask = np.load(mask_path)
            log.info("ScenePoseOptimize [%d]: Loaded mask %s", idx, mask.shape)

            request = {
                "object_dir": object_dir,
                "glb_path": mesh_path,
                "pointmap_path": pointmap_path,
                "intrinsics_b64": serialize_tensor(intrinsics_np),
                "pose_b64": serialize_pose(rotation, translation, scale),
                "mask_b64": base64.b64encode(pickle.dumps(mask.astype(np.float32))).decode('utf-8'),
                "enable_manual_alignment": False,  # Never use height-based manual alignment
                "enable_icp": enable_icp,
                "enable_render_opt": enable_render,
                "image_path": image_path if os.path.exists(image_path) else None,
                "output_glb_path": output_glb_path,
            }

            try:
                log.info("ScenePoseOptimize [%d]: Running optimization...", idx)
                response = run_pose_optimization_batch(request)

                if response.get("status") == "error":
                    log.error("ScenePoseOptimize [%d]: Worker error: %s", idx, response.get('error'))
                    iou_scores.append(-1.0)
                    continue

                # Extract results
                result_glb_path = response.get("output_glb_path", output_glb_path)
                iou = response.get("iou", -1.0)

                log.info("ScenePoseOptimize [%d]: Done (IOU: %.3f)", idx, iou)
                log.info("ScenePoseOptimize [%d]: Output: %s", idx, result_glb_path)

                iou_scores.append(float(iou))

            except Exception as e:
                log.error("ScenePoseOptimize [%d]: Pose optimization failed", idx, exc_info=True)
                iou_scores.append(-1.0)

        elapsed = time.time() - start_time
        log.info("Pose optimization done: %.0fs (%d objects)", elapsed, len(object_dirs))

        # Save cache metadata for future runs
        self._save_cache_metadata(pose_opt_folder, optimization_mode, len(object_dirs), iou_scores)

        return (pose_opt_folder, iou_scores)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "SAM3D_ScenePoseOptimize": SAM3D_ScenePoseOptimize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3D_ScenePoseOptimize": "SAM3D Scene Pose Optimize",
}
