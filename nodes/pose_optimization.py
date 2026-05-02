"""SAM3D_PoseOptimization node for refining object pose using ICP and render optimization."""

import logging
import os
from typing import Any, Dict

import torch
from comfy_api.latest import io

log = logging.getLogger("sam3dobjects")

class SAM3D_PoseOptimization(io.ComfyNode):
    """
    Refine object pose using ICP and render-based optimization.

    Takes initial pose from sparse generation and refines it by:
    1. Height alignment - Match mesh to point cloud height
    2. ICP registration - Iterative Closest Point for fine alignment
    3. Render optimization - Gradient descent to maximize mask IoU

    Outputs the refined pose, IoU score, and a reposed GLB file saved to disk.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SAM3D_PoseOptimization",
            category="SAM3DObjects",
            description="Refine object pose using ICP and render-based optimization. Outputs reposed GLB.",
            inputs=[
                io.String.Input("glb_path",
                    default="",
                    multiline=False,
                    tooltip="Path to GLB mesh file from mesh decode"
                ),
                io.String.Input("pointmap_path",
                    default="",
                    multiline=False,
                    tooltip="Path to pointmap .pt file from SAM3D_DepthEstimate"
                ),
                io.Custom("SAM3D_INTRINSICS").Input("intrinsics",
                    tooltip="Camera intrinsics from SAM3D_DepthEstimate"
                ),
                io.Custom("SAM3D_POSE").Input("pose",
                    tooltip="Initial pose from SAM3DSparseGen (rotation, translation, scale)"
                ),
                io.Mask.Input("mask",
                    tooltip="Original binary mask of the object"
                ),
                io.Boolean.Input("enable_icp",
                    default=True,
                    tooltip="Enable ICP (Iterative Closest Point) refinement step",
                    optional=True,
                ),
                io.Boolean.Input("enable_render_opt",
                    default=True,
                    tooltip="Enable render-and-compare optimization step",
                    optional=True,
                ),
            ],
            outputs=[
                io.String.Output(display_name="glb_path", tooltip="Path to reposed GLB file (mesh transformed with refined pose)"),
                io.Custom("SAM3D_POSE").Output(display_name="pose", tooltip="Refined pose (rotation, translation, scale)"),
                io.Float.Output(display_name="iou", tooltip="Final IoU score (quality metric, -1 if optimization skipped)"),
            ],
        )

    @classmethod
    @torch.no_grad()
    def execute(
        cls,
        glb_path: str,
        pointmap_path: str,
        intrinsics,  # numpy array or tensor
        pose: Dict[str, Any],
        mask,  # torch.Tensor
        enable_icp: bool = True,
        enable_render_opt: bool = True,
    ):
        """
        Optimize object pose using ICP and render optimization.

        This method runs in an isolated subprocess with its own Python environment.
        """
        # These imports happen in the isolated subprocess
        import os
        import pickle
        import base64
        import torch
        import numpy as np

        from .utils.pose_optimization import run_pose_optimization

        log.info("PoseOptimization: Starting pose refinement")

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
            log.warning("Incomplete pose data, returning original")
            return io.NodeOutput(glb_path, pose, -1.0)

        # Serialize helper
        def serialize_tensor(tensor):
            if tensor is None:
                return None
            if isinstance(tensor, torch.Tensor):
                arr = tensor.cpu().numpy()
            elif isinstance(tensor, np.ndarray):
                arr = tensor
            elif isinstance(tensor, (list, tuple)):
                arr = np.array(tensor)
            else:
                arr = np.array(tensor)
            return base64.b64encode(pickle.dumps(arr)).decode('utf-8')

        def serialize_pose(pose_dict):
            return {
                "rotation": serialize_tensor(pose_dict.get("rotation")),
                "translation": serialize_tensor(pose_dict.get("translation")),
                "scale": serialize_tensor(pose_dict.get("scale")),
            }

        # Prepare mask for serialization
        if mask.dim() == 3:
            mask_2d = mask[0].cpu().numpy()
        else:
            mask_2d = mask.cpu().numpy()

        # Build request for worker
        request = {
            "glb_path": glb_path,
            "pointmap_path": pointmap_path,
            "intrinsics_b64": serialize_tensor(intrinsics),
            "pose_b64": serialize_pose(pose),
            "mask_b64": base64.b64encode(pickle.dumps(mask_2d.astype(np.float32))).decode('utf-8'),
            "enable_icp": enable_icp,
            "enable_render_opt": enable_render_opt,
        }

        # Run pose optimization
        response = run_pose_optimization(request)

        if response.get("status") == "error":
            log.error("Worker error: %s", response.get('error'))
            return io.NodeOutput(glb_path, pose, -1.0)

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

        log.info("Pose optimization completed (IoU: %.3f)", iou)
        return io.NodeOutput(output_glb_path, refined_pose, float(iou))
