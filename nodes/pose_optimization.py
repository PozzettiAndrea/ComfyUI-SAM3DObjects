"""SAM3D_PoseOptimization node for refining object pose using ICP and render optimization."""

import os
import torch
import numpy as np
import pickle
import base64
from typing import Any, Dict
from pathlib import Path


class SAM3D_PoseOptimization:
    """
    Refine object pose using ICP and render-based optimization.

    Takes initial pose from sparse generation and refines it by:
    1. Height alignment - Match mesh to point cloud height
    2. ICP registration - Iterative Closest Point for fine alignment
    3. Render optimization - Gradient descent to maximize mask IoU

    Outputs the refined pose, IoU score, and a reposed GLB file saved to disk.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "glb_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Path to GLB mesh file from mesh decode"
                }),
                "pointmap_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Path to pointmap .pt file from SAM3D_DepthEstimate"
                }),
                "intrinsics": ("SAM3D_INTRINSICS", {
                    "tooltip": "Camera intrinsics from SAM3D_DepthEstimate"
                }),
                "pose": ("SAM3D_POSE", {
                    "tooltip": "Initial pose from SAM3DSparseGen (rotation, translation, scale)"
                }),
                "mask": ("MASK", {
                    "tooltip": "Original binary mask of the object"
                }),
            },
            "optional": {
                "enable_icp": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable ICP (Iterative Closest Point) refinement step"
                }),
                "enable_render_opt": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable render-and-compare optimization step"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "SAM3D_POSE", "FLOAT")
    RETURN_NAMES = ("glb_path", "pose", "iou")
    OUTPUT_TOOLTIPS = (
        "Path to reposed GLB file (mesh transformed with refined pose)",
        "Refined pose (rotation, translation, scale)",
        "Final IoU score (quality metric, -1 if optimization skipped)"
    )
    FUNCTION = "optimize_pose"
    CATEGORY = "SAM3DObjects"
    DESCRIPTION = "Refine object pose using ICP and render-based optimization. Outputs reposed GLB."

    def _serialize_tensor(self, tensor) -> str:
        """Serialize tensor to base64 numpy array."""
        if tensor is None:
            return None
        # Convert to numpy array regardless of input type
        if isinstance(tensor, torch.Tensor):
            arr = tensor.cpu().numpy()
        elif isinstance(tensor, np.ndarray):
            arr = tensor
        elif isinstance(tensor, (list, tuple)):
            arr = np.array(tensor)
        else:
            arr = np.array(tensor)
        return base64.b64encode(pickle.dumps(arr)).decode('utf-8')

    def _serialize_pose(self, pose: Dict[str, Any]) -> Dict[str, str]:
        """Serialize pose dict with tensors to base64."""
        return {
            "rotation": self._serialize_tensor(pose.get("rotation")),
            "translation": self._serialize_tensor(pose.get("translation")),
            "scale": self._serialize_tensor(pose.get("scale")),
        }

    def optimize_pose(
        self,
        glb_path: str,
        pointmap_path: str,
        intrinsics: torch.Tensor,
        pose: Dict[str, Any],
        mask: torch.Tensor,
        enable_icp: bool = True,
        enable_render_opt: bool = True,
    ):
        """
        Optimize object pose using ICP and render optimization.
        Delegates to worker subprocess where sam3d_objects is available.
        """
        print(f"[SAM3DObjects] PoseOptimization: Starting pose refinement")

        # Validate inputs
        if not glb_path:
            raise ValueError("GLB path is required")
        if not pointmap_path:
            raise ValueError("Pointmap path is required")

        # Check pose data
        rotation = pose.get("rotation")
        translation = pose.get("translation")
        scale = pose.get("scale")

        if rotation is None or translation is None or scale is None:
            print("[SAM3DObjects] Warning: Incomplete pose data, returning original")
            return (glb_path, pose, -1.0)

        # Get the worker bridge
        from .subprocess_bridge import InferenceWorkerBridge
        node_root = Path(__file__).parent.parent
        bridge = InferenceWorkerBridge.get_instance(node_root)

        # Prepare mask for serialization
        if mask.dim() == 3:
            mask_2d = mask[0].cpu().numpy()
        else:
            mask_2d = mask.cpu().numpy()

        # Serialize data for worker
        request = {
            "command": "pose_optimization",
            "glb_path": glb_path,
            "pointmap_path": pointmap_path,
            "intrinsics_b64": self._serialize_tensor(intrinsics),
            "pose_b64": self._serialize_pose(pose),
            "mask_b64": base64.b64encode(pickle.dumps(mask_2d.astype(np.float32))).decode('utf-8'),
            "enable_icp": enable_icp,
            "enable_render_opt": enable_render_opt,
        }

        # Send to worker
        try:
            response = bridge._send_request(request, timeout=120.0)

            if response.get("status") == "error":
                print(f"[SAM3DObjects] Worker error: {response.get('error')}")
                return (glb_path, pose, -1.0)

            # Extract results
            output_glb_path = response.get("output_glb_path", glb_path)
            iou = response.get("iou", -1.0)

            # Deserialize refined pose
            refined_pose_b64 = response.get("refined_pose_b64", {})
            refined_pose = {}
            for key in ["rotation", "translation", "scale"]:
                if refined_pose_b64.get(key):
                    data = pickle.loads(base64.b64decode(refined_pose_b64[key]))
                    refined_pose[key] = torch.tensor(data)
                else:
                    refined_pose[key] = pose.get(key)

            print(f"[SAM3DObjects] Pose optimization completed (IoU: {iou:.3f})")
            return (output_glb_path, refined_pose, float(iou))

        except Exception as e:
            print(f"[SAM3DObjects] Error during pose optimization: {e}")
            import traceback
            traceback.print_exc()
            return (glb_path, pose, -1.0)
