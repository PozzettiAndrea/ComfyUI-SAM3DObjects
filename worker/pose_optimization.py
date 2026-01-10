"""
Pose optimization for SAM3D scene generation.

This module contains pose optimization using layout_post_optimization.
"""

import sys
import os
import base64
import pickle
import traceback
from typing import Any, Dict
import numpy as np
import torch


def run_pose_optimization(request: Dict[str, Any]) -> Dict[str, Any]:
    """Run pose optimization using layout_post_optimization."""
    try:
        from pathlib import Path

        # Setup environment for sam3d_objects imports
        vendor_path = Path(__file__).parent.parent / "vendor"
        if str(vendor_path) not in sys.path:
            sys.path.insert(0, str(vendor_path))

        import trimesh
        from pytorch3d.transforms import quaternion_to_matrix
        from sam3d_objects.pipeline.inference_utils import layout_post_optimization

        print("[Worker] Running pose optimization", file=sys.stderr)

        # Extract parameters
        glb_path = request["glb_path"]
        pointmap_path = request["pointmap_path"]
        enable_icp = request.get("enable_icp", True)
        enable_render_opt = request.get("enable_render_opt", True)

        # Deserialize intrinsics
        intrinsics_np = pickle.loads(base64.b64decode(request["intrinsics_b64"]))
        intrinsics = torch.from_numpy(intrinsics_np).float()

        # Deserialize pose
        pose_b64 = request["pose_b64"]
        rotation_np = pickle.loads(base64.b64decode(pose_b64["rotation"]))
        translation_np = pickle.loads(base64.b64decode(pose_b64["translation"]))
        scale_np = pickle.loads(base64.b64decode(pose_b64["scale"]))

        # Deserialize mask
        mask_np = pickle.loads(base64.b64decode(request["mask_b64"]))

        # Use GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Convert to tensors
        rotation = torch.from_numpy(rotation_np).float().to(device)
        translation = torch.from_numpy(translation_np).float().to(device)
        scale = torch.from_numpy(scale_np).float().to(device)
        mask_tensor = torch.from_numpy(mask_np).float().to(device)
        intrinsics = intrinsics.to(device)

        # Ensure correct shapes
        if rotation.dim() == 1:
            rotation = rotation.unsqueeze(0)
        if translation.dim() == 1:
            translation = translation.unsqueeze(0)
        if scale.dim() == 0:
            scale = scale.unsqueeze(0).unsqueeze(0).expand(1, 3)
        elif scale.dim() == 1:
            scale = scale.unsqueeze(0)

        # Load mesh
        mesh = trimesh.load(glb_path)
        if isinstance(mesh, trimesh.Scene):
            meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if not meshes:
                raise ValueError("No mesh found in GLB file")
            mesh = meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)

        # Load pointmap
        pointmap_data = torch.load(pointmap_path, weights_only=False)
        if isinstance(pointmap_data, dict):
            pointmap_tensor = pointmap_data.get("pointmap") or pointmap_data.get("data")
        else:
            pointmap_tensor = pointmap_data
        pointmap_tensor = pointmap_tensor.float().to(device)

        # Run optimization
        (
            refined_quat,
            refined_trans,
            refined_scale,
            final_iou,
            used_icp,
            used_render_opt,
        ) = layout_post_optimization(
            Mesh=mesh,
            Quaternion=rotation,
            Translation=translation,
            Scale=scale,
            Mask=mask_tensor,
            Point_Map=pointmap_tensor,
            Intrinsics=intrinsics,
            Enable_shape_ICP=enable_icp,
            Enable_rendering_optimization=enable_render_opt,
            device=device,
        )

        print(f"[Worker] Optimization complete: IoU={final_iou:.3f}", file=sys.stderr)

        # Convert results to numpy
        quat_np = refined_quat.cpu().numpy() if hasattr(refined_quat, 'cpu') else np.array(refined_quat)
        trans_np = refined_trans.cpu().numpy() if hasattr(refined_trans, 'cpu') else np.array(refined_trans)
        scale_np_out = refined_scale.cpu().numpy() if hasattr(refined_scale, 'cpu') else np.array(refined_scale)

        # Ensure correct shapes
        if quat_np.ndim > 1:
            quat_np = quat_np.squeeze()
        if trans_np.ndim > 1:
            trans_np = trans_np.squeeze()
        if scale_np_out.ndim > 1:
            scale_np_out = scale_np_out.squeeze()

        # Convert quaternion to rotation matrix using pytorch3d (already in wxyz format)
        quat_tensor = torch.from_numpy(quat_np).float().unsqueeze(0)
        rot_matrix = quaternion_to_matrix(quat_tensor).squeeze(0).numpy()

        # Apply transformation to mesh
        scale_val = scale_np_out.mean() if scale_np_out.ndim > 0 else float(scale_np_out)
        vertices = mesh.vertices.copy()
        vertices_transformed = (vertices @ (rot_matrix.T * scale_val)) + trans_np
        mesh.vertices = vertices_transformed

        # Save reposed GLB
        input_dir = os.path.dirname(glb_path)
        input_name = os.path.splitext(os.path.basename(glb_path))[0]
        output_glb_path = os.path.join(input_dir, f"{input_name}_reposed.glb")
        mesh.export(output_glb_path)

        print(f"[Worker] Saved reposed GLB: {output_glb_path}", file=sys.stderr)

        # Serialize refined pose for response
        refined_pose_b64 = {
            "rotation": base64.b64encode(pickle.dumps(quat_np)).decode('utf-8'),
            "translation": base64.b64encode(pickle.dumps(trans_np)).decode('utf-8'),
            "scale": base64.b64encode(pickle.dumps(scale_np_out)).decode('utf-8'),
        }

        return {
            "status": "success",
            "output_glb_path": output_glb_path,
            "refined_pose_b64": refined_pose_b64,
            "iou": float(final_iou),
            "used_icp": bool(used_icp),
            "used_render_opt": bool(used_render_opt),
        }

    except Exception as e:
        print(f"[Worker] Pose optimization error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def run_pose_optimization_batch(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run pose optimization for batch scene generation.

    This function handles the pose_optimization_batch command from SAM3D_ScenePoseOptimize.
    It's similar to run_pose_optimization but saves output as aligned_mesh.glb in the object dir.
    """
    try:
        from pathlib import Path

        # Setup environment for sam3d_objects imports
        vendor_path = Path(__file__).parent.parent / "vendor"
        if str(vendor_path) not in sys.path:
            sys.path.insert(0, str(vendor_path))

        import trimesh
        from pytorch3d.transforms import quaternion_to_matrix
        from sam3d_objects.pipeline.inference_utils import layout_post_optimization

        print("[Worker] Running batch pose optimization", file=sys.stderr)

        # Extract parameters
        object_dir = request.get("object_dir", "")
        glb_path = request["glb_path"]
        pointmap_path = request["pointmap_path"]
        enable_icp = request.get("enable_icp", True)
        enable_render_opt = request.get("enable_render_opt", True)

        # Deserialize intrinsics
        intrinsics_np = pickle.loads(base64.b64decode(request["intrinsics_b64"]))
        intrinsics = torch.from_numpy(intrinsics_np).float()

        # Deserialize pose
        pose_b64 = request["pose_b64"]
        rotation_np = pickle.loads(base64.b64decode(pose_b64["rotation"]))
        translation_np = pickle.loads(base64.b64decode(pose_b64["translation"]))
        scale_np = pickle.loads(base64.b64decode(pose_b64["scale"]))

        # Deserialize mask
        mask_np = pickle.loads(base64.b64decode(request["mask_b64"]))

        # Use GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Convert to tensors
        rotation = torch.from_numpy(rotation_np).float().to(device)
        translation = torch.from_numpy(translation_np).float().to(device)
        scale = torch.from_numpy(scale_np).float().to(device)
        mask_tensor = torch.from_numpy(mask_np).float().to(device)
        intrinsics = intrinsics.to(device)

        # Ensure correct shapes
        if rotation.dim() == 1:
            rotation = rotation.unsqueeze(0)
        if translation.dim() == 1:
            translation = translation.unsqueeze(0)
        if scale.dim() == 0:
            scale = scale.unsqueeze(0).unsqueeze(0).expand(1, 3)
        elif scale.dim() == 1:
            scale = scale.unsqueeze(0)

        # Load mesh
        print(f"[Worker] Loading mesh: {glb_path}", file=sys.stderr)
        mesh = trimesh.load(glb_path)
        if isinstance(mesh, trimesh.Scene):
            meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if not meshes:
                raise ValueError("No mesh found in GLB file")
            mesh = meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)

        # Load pointmap
        print(f"[Worker] Loading pointmap: {pointmap_path}", file=sys.stderr)
        pointmap_data = torch.load(pointmap_path, weights_only=False)
        if isinstance(pointmap_data, dict):
            pointmap_tensor = pointmap_data.get("pointmap") or pointmap_data.get("data")
        else:
            pointmap_tensor = pointmap_data
        pointmap_tensor = pointmap_tensor.float().to(device)

        print(f"[Worker] Running layout_post_optimization...", file=sys.stderr)
        print(f"[Worker] - enable_icp: {enable_icp}", file=sys.stderr)
        print(f"[Worker] - enable_render_opt: {enable_render_opt}", file=sys.stderr)

        # Run optimization
        (
            refined_quat,
            refined_trans,
            refined_scale,
            final_iou,
            used_icp,
            used_render_opt,
        ) = layout_post_optimization(
            Mesh=mesh,
            Quaternion=rotation,
            Translation=translation,
            Scale=scale,
            Mask=mask_tensor,
            Point_Map=pointmap_tensor,
            Intrinsics=intrinsics,
            Enable_shape_ICP=enable_icp,
            Enable_rendering_optimization=enable_render_opt,
            device=device,
        )

        print(f"[Worker] Optimization complete: IoU={final_iou:.3f}", file=sys.stderr)
        print(f"[Worker] - used_icp: {used_icp}, used_render_opt: {used_render_opt}", file=sys.stderr)

        # Convert results to numpy
        quat_np = refined_quat.cpu().numpy() if hasattr(refined_quat, 'cpu') else np.array(refined_quat)
        trans_np = refined_trans.cpu().numpy() if hasattr(refined_trans, 'cpu') else np.array(refined_trans)
        scale_np_out = refined_scale.cpu().numpy() if hasattr(refined_scale, 'cpu') else np.array(refined_scale)

        # Ensure correct shapes
        if quat_np.ndim > 1:
            quat_np = quat_np.squeeze()
        if trans_np.ndim > 1:
            trans_np = trans_np.squeeze()
        if scale_np_out.ndim > 1:
            scale_np_out = scale_np_out.squeeze()

        # Convert quaternion to rotation matrix using pytorch3d (already in wxyz format)
        quat_tensor = torch.from_numpy(quat_np).float().unsqueeze(0)
        rot_matrix = quaternion_to_matrix(quat_tensor).squeeze(0).numpy()

        # Apply transformation to mesh
        scale_val = scale_np_out.mean() if scale_np_out.ndim > 0 else float(scale_np_out)
        vertices = mesh.vertices.copy()
        vertices_transformed = (vertices @ (rot_matrix.T * scale_val)) + trans_np
        mesh.vertices = vertices_transformed

        # Save aligned GLB - use object_dir if provided, else use input path
        if object_dir:
            output_glb_path = os.path.join(object_dir, "aligned_mesh.glb")
        else:
            input_dir = os.path.dirname(glb_path)
            output_glb_path = os.path.join(input_dir, "aligned_mesh.glb")

        mesh.export(output_glb_path)
        print(f"[Worker] Saved aligned GLB: {output_glb_path}", file=sys.stderr)

        return {
            "status": "success",
            "output_glb_path": output_glb_path,
            "iou": float(final_iou),
            "used_icp": bool(used_icp),
            "used_render_opt": bool(used_render_opt),
        }

    except Exception as e:
        print(f"[Worker] Batch pose optimization error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
