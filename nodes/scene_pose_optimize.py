"""SAM3D_ScenePoseOptimize node - optimize poses for all objects in a scene folder."""

import os
import re
import json
from typing import Any, Dict, List, Tuple

from comfy_env import isolated


@isolated(env="sam3dobjects", import_paths=[".", "../vendor"])
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
        except:
            pass

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
        import numpy as np
        import trimesh
        from pathlib import Path

        from utils.pose_optimization import run_pose_optimization_batch

        start_time = time.time()
        print(f"[SAM3DObjects] ScenePoseOptimize: Starting pose optimization")
        print(f"[SAM3DObjects] ScenePoseOptimize: Mode = {optimization_mode}")
        print(f"[SAM3DObjects] ScenePoseOptimize: Folder = {output_folder}")

        # Validate folder exists
        if not os.path.exists(output_folder):
            raise ValueError(f"Output folder does not exist: {output_folder}")

        # Check for required files
        intrinsics_path = os.path.join(output_folder, "intrinsics.pt")
        image_path = os.path.join(output_folder, "image.png")

        if not os.path.exists(intrinsics_path):
            raise ValueError(f"Intrinsics file not found: {intrinsics_path}. Run SAM3DSceneGenerate first.")

        # Load intrinsics
        intrinsics = torch.load(intrinsics_path, weights_only=False)
        print(f"[SAM3DObjects] ScenePoseOptimize: Loaded intrinsics from {intrinsics_path}")

        # Discover object folders
        object_dirs = self._discover_objects(output_folder)
        if not object_dirs:
            print(f"[SAM3DObjects] ScenePoseOptimize: No object folders found")
            return ("", [])

        print(f"[SAM3DObjects] ScenePoseOptimize: Found {len(object_dirs)} object(s)")

        # Define output folder for pose-optimized meshes
        pose_opt_folder = f"{output_folder}_pose_optimized"

        # Check cache - skip if already processed with same settings
        cache_hit, cached_scores = self._check_cache(pose_opt_folder, optimization_mode, len(object_dirs))
        if cache_hit:
            print(f"[SAM3DObjects] ScenePoseOptimize: Using cached results from {pose_opt_folder}")
            return (pose_opt_folder, cached_scores)

        # Determine optimization flags (no manual alignment in any mode)
        enable_icp = optimization_mode == "icp_only"
        enable_render = optimization_mode == "render_only"

        # Create output folder
        os.makedirs(pose_opt_folder, exist_ok=True)
        print(f"[SAM3DObjects] ScenePoseOptimize: Output folder = {pose_opt_folder}")

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
            print(f"\n[SAM3DObjects] ScenePoseOptimize: === Object {idx + 1}/{len(object_dirs)} ===")

            # Check required files for this object
            pointmap_path = os.path.join(object_dir, "pointmap.pt")
            mask_path = os.path.join(object_dir, "mask.npy")
            mesh_path = os.path.join(object_dir, "mesh.glb")

            # Check for textured mesh first
            textured_mesh_path = os.path.join(object_dir, "textured_mesh.glb")
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
                print(f"[SAM3DObjects] ScenePoseOptimize [{idx}]: Missing files: {missing}, skipping")
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

                print(f"[SAM3DObjects] ScenePoseOptimize [{idx}]: Loaded pose from sparse_structure.pt")
            else:
                # Fallback to identity pose if no sparse_structure.pt
                rotation = [[1, 0, 0, 0]]  # Identity quaternion (wxyz)
                translation = [[0, 0, 0]]
                scale = [[1, 1, 1]]
                print(f"[SAM3DObjects] ScenePoseOptimize [{idx}]: No sparse_structure.pt, using identity pose")

            # Output path in the pose-optimized folder
            output_glb_path = os.path.join(pose_opt_folder, f"object_{idx}.glb")

            # Handle pose_only mode - just apply the pose without optimization
            if optimization_mode == "pose_only":
                try:
                    print(f"[SAM3DObjects] ScenePoseOptimize [{idx}]: Applying pose directly (no optimization)")

                    # Load mesh
                    mesh = trimesh.load(mesh_path)
                    if isinstance(mesh, trimesh.Scene):
                        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
                        if not meshes:
                            raise ValueError("No mesh found in GLB file")
                        mesh = meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)

                    # Convert pose to tensors
                    from pytorch3d.transforms import quaternion_to_matrix

                    rot_np = np.array(rotation).squeeze()
                    trans_np = np.array(translation).squeeze()
                    scale_np = np.array(scale).squeeze()

                    # Ensure scale is 3D
                    if scale_np.ndim == 0:
                        scale_np = np.array([float(scale_np)] * 3)
                    elif scale_np.size == 1:
                        scale_np = np.array([float(scale_np.flat[0])] * 3)
                    else:
                        scale_np = scale_np.flatten()[:3]

                    # Convert quaternion to rotation matrix
                    quat_tensor = torch.from_numpy(rot_np).float().unsqueeze(0)
                    rot_matrix = quaternion_to_matrix(quat_tensor).squeeze(0).numpy()

                    # Z-up to Y-up conversion matrix (SAM3D decoder outputs Z-up)
                    z_up_to_y_up = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).T

                    # Apply transformation: Z-up → Y-up, then scale, rotate, translate
                    # Note: PyTorch3D quaternion_to_matrix returns row-vector convention (v @ R)
                    # Do NOT transpose - that would apply inverse rotation
                    vertices = mesh.vertices.copy()
                    vertices_y_up = vertices @ z_up_to_y_up
                    vertices_scaled = vertices_y_up * scale_np
                    vertices_rotated = vertices_scaled @ rot_matrix
                    vertices_transformed = vertices_rotated + trans_np

                    # Flip X and Y axes to match image coordinate convention
                    # PyTorch3D: X-left, Y-up, Z-inward
                    # Image coords: X-right, Y-down
                    vertices_transformed[:, 0] = -vertices_transformed[:, 0]
                    vertices_transformed[:, 1] = -vertices_transformed[:, 1]

                    mesh.vertices = vertices_transformed
                    mesh.export(output_glb_path)

                    print(f"[SAM3DObjects] ScenePoseOptimize [{idx}]: Saved: {output_glb_path}")
                    iou_scores.append(1.0)  # No IoU computed in pose_only mode

                except Exception as e:
                    print(f"[SAM3DObjects] ScenePoseOptimize [{idx}]: Error: {e}")
                    import traceback
                    traceback.print_exc()
                    iou_scores.append(-1.0)
                continue

            # Load mask for optimization modes
            mask = np.load(mask_path)
            print(f"[SAM3DObjects] ScenePoseOptimize [{idx}]: Loaded mask {mask.shape}")

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
                print(f"[SAM3DObjects] ScenePoseOptimize [{idx}]: Running optimization...")
                response = run_pose_optimization_batch(request)

                if response.get("status") == "error":
                    print(f"[SAM3DObjects] ScenePoseOptimize [{idx}]: Worker error: {response.get('error')}")
                    iou_scores.append(-1.0)
                    continue

                # Extract results
                result_glb_path = response.get("output_glb_path", output_glb_path)
                iou = response.get("iou", -1.0)

                print(f"[SAM3DObjects] ScenePoseOptimize [{idx}]: Done (IOU: {iou:.3f})")
                print(f"[SAM3DObjects] ScenePoseOptimize [{idx}]: Output: {result_glb_path}")

                iou_scores.append(float(iou))

            except Exception as e:
                print(f"[SAM3DObjects] ScenePoseOptimize [{idx}]: Error: {e}")
                import traceback
                traceback.print_exc()
                iou_scores.append(-1.0)

        elapsed = time.time() - start_time
        print(f"\n[SAM3DObjects] ✓ Pose optimization done: {elapsed:.0f}s ({len(object_dirs)} objects)")

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
