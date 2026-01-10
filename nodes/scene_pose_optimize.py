"""SAM3D_ScenePoseOptimize node - optimize poses for all objects in a scene folder."""

import os
import re
import torch
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .subprocess_bridge import get_bridge


class SAM3D_ScenePoseOptimize:
    """
    Scene Pose Optimization - Refine poses for all objects in a scene folder.

    Takes the output_folder from SAM3DSceneGenerate and runs pose alignment
    on each object using the saved data (intrinsics, masks, poses, pointmaps).

    Supports three optimization modes:
    - manual_only: Height-based alignment only (fastest)
    - manual_icp: Manual alignment + ICP refinement
    - manual_icp_render: Full pipeline with render-and-compare optimization (best quality)

    Outputs list of paths to aligned GLB files and quality scores.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_folder": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Output folder from SAM3DSceneGenerate containing object_N/ subdirectories"
                }),
                "optimization_mode": (["manual_only", "manual_icp", "manual_icp_render"], {
                    "default": "manual_icp_render",
                    "tooltip": "Optimization mode: manual_only (fast), manual_icp (balanced), manual_icp_render (best quality)"
                }),
            },
        }

    RETURN_TYPES = ("STRING", "FLOAT")
    RETURN_NAMES = ("glb_paths", "iou_scores")
    OUTPUT_IS_LIST = (True, True)
    OUTPUT_TOOLTIPS = (
        "List of paths to aligned GLB files",
        "List of IOU scores (quality metric, -1 if optimization failed or skipped)"
    )
    FUNCTION = "optimize_poses"
    CATEGORY = "SAM3DObjects"
    DESCRIPTION = "Optimize poses for all objects in a scene folder using alignment algorithms."

    def _discover_objects(self, output_folder: str) -> List[str]:
        """Discover all object_N/ folders in sorted order."""
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

    def optimize_poses(
        self,
        output_folder: str,
        optimization_mode: str = "manual_icp_render",
    ) -> Tuple[List[str], List[float]]:
        """
        Optimize poses for all objects in the scene folder.

        Args:
            output_folder: Path to folder containing object_N/ subdirectories
            optimization_mode: One of "manual_only", "manual_icp", "manual_icp_render"

        Returns:
            Tuple of (list of aligned GLB paths, list of IOU scores)
        """
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
            return ([], [])

        print(f"[SAM3DObjects] ScenePoseOptimize: Found {len(object_dirs)} object(s)")

        # Determine optimization flags
        enable_icp = optimization_mode in ["manual_icp", "manual_icp_render"]
        enable_render = optimization_mode == "manual_icp_render"

        # Get bridge for worker communication
        bridge = get_bridge()

        glb_paths = []
        iou_scores = []

        for idx, object_dir in enumerate(object_dirs):
            print(f"\n[SAM3DObjects] ScenePoseOptimize: === Object {idx + 1}/{len(object_dirs)} ===")

            # Check required files for this object
            pointmap_path = os.path.join(object_dir, "pointmap.pt")
            mask_path = os.path.join(object_dir, "mask.npy")
            pose_path = os.path.join(object_dir, "pose.pt")
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
                glb_paths.append(mesh_path if os.path.exists(mesh_path) else "")
                iou_scores.append(-1.0)
                continue

            # Load pose (or use identity)
            if os.path.exists(pose_path):
                pose_data = torch.load(pose_path, weights_only=False)
                rotation = pose_data.get("rotation")
                translation = pose_data.get("translation")
                scale = pose_data.get("scale")
                print(f"[SAM3DObjects] ScenePoseOptimize [{idx}]: Loaded pose from {pose_path}")
            else:
                # Use identity pose
                rotation = [[1, 0, 0, 0]]  # Identity quaternion (wxyz)
                translation = [[0, 0, 0]]
                scale = [[1, 1, 1]]
                print(f"[SAM3DObjects] ScenePoseOptimize [{idx}]: Using identity pose")

            # Load mask
            mask = np.load(mask_path)
            print(f"[SAM3DObjects] ScenePoseOptimize [{idx}]: Loaded mask {mask.shape}")

            # Build request for worker
            import pickle
            import base64

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

            request = {
                "object_dir": object_dir,
                "glb_path": mesh_path,
                "pointmap_path": pointmap_path,
                "intrinsics_b64": serialize_tensor(intrinsics_np),
                "pose_b64": serialize_pose(rotation, translation, scale),
                "mask_b64": base64.b64encode(pickle.dumps(mask.astype(np.float32))).decode('utf-8'),
                "enable_icp": enable_icp,
                "enable_render_opt": enable_render,
                "image_path": image_path if os.path.exists(image_path) else None,
            }

            try:
                print(f"[SAM3DObjects] ScenePoseOptimize [{idx}]: Running optimization...")
                response = bridge.call("pose_optimization_batch", timeout=180.0, **request)

                if response.get("status") == "error":
                    print(f"[SAM3DObjects] ScenePoseOptimize [{idx}]: Worker error: {response.get('error')}")
                    glb_paths.append(mesh_path)
                    iou_scores.append(-1.0)
                    continue

                # Extract results
                output_glb_path = response.get("output_glb_path", mesh_path)
                iou = response.get("iou", -1.0)

                print(f"[SAM3DObjects] ScenePoseOptimize [{idx}]: Done (IOU: {iou:.3f})")
                print(f"[SAM3DObjects] ScenePoseOptimize [{idx}]: Output: {output_glb_path}")

                glb_paths.append(output_glb_path)
                iou_scores.append(float(iou))

            except Exception as e:
                print(f"[SAM3DObjects] ScenePoseOptimize [{idx}]: Error: {e}")
                import traceback
                traceback.print_exc()
                glb_paths.append(mesh_path)
                iou_scores.append(-1.0)

        print(f"\n[SAM3DObjects] ScenePoseOptimize: Completed {len(object_dirs)} object(s)")
        return (glb_paths, iou_scores)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "SAM3D_ScenePoseOptimize": SAM3D_ScenePoseOptimize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3D_ScenePoseOptimize": "SAM3D Scene Pose Optimize",
}
