"""
Pose optimization for SAM3D scene generation.

This module contains pose optimization that can skip manual (height-based) alignment.
Supports: icp_only, render_only modes (no manual alignment).
"""

import logging
import sys
import os
import base64
import pickle
import traceback
from typing import Any, Dict
import numpy as np
import torch
import comfy.utils
import comfy.model_management

log = logging.getLogger("sam3dobjects")


def depth_edge(depth: np.ndarray, rtol: float = 0.03, mask: np.ndarray = None) -> np.ndarray:
    """
    Detect edges in depth map based on relative depth discontinuities.

    Args:
        depth: 2D depth map
        rtol: Relative tolerance for edge detection
        mask: Optional mask to apply

    Returns:
        Boolean mask where True indicates an edge pixel
    """
    # Compute gradients in x and y directions
    grad_x = np.abs(np.diff(depth, axis=1, prepend=depth[:, :1]))
    grad_y = np.abs(np.diff(depth, axis=0, prepend=depth[:1, :]))

    # Compute relative gradients
    depth_safe = np.maximum(np.abs(depth), 1e-6)
    rel_grad_x = grad_x / depth_safe
    rel_grad_y = grad_y / depth_safe

    # Edge where relative gradient exceeds threshold
    edge_mask = (rel_grad_x > rtol) | (rel_grad_y > rtol)

    if mask is not None:
        edge_mask = edge_mask & mask

    return edge_mask


def run_pose_optimization(request: Dict[str, Any]) -> Dict[str, Any]:
    """Run pose optimization using layout_post_optimization."""
    try:
        import trimesh
        from pytorch3d.transforms import quaternion_to_matrix
        from ..sam3d.pipeline import layout_post_optimization

        log.info("Running pose optimization")

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
        device = comfy.model_management.get_torch_device()

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
        pointmap_data = comfy.utils.load_torch_file(pointmap_path, safe_load=False)
        if isinstance(pointmap_data, dict):
            pointmap_tensor = pointmap_data.get("pointmap")
            if pointmap_tensor is None:
                pointmap_tensor = pointmap_data.get("data")
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

        log.info("Optimization complete: IoU=%.3f", final_iou)

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
        # SAM3D decoder outputs Z-up meshes. The GLB files contain Z-up coordinates.
        # layout_post_optimization works in Y-up (pytorch3d convention).
        # We must convert Z-up -> Y-up before applying the refined transform.
        z_up_to_y_up_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).T  # Z-up to Y-up

        # Ensure scale is per-axis (3 values)
        if scale_np_out.ndim == 0:
            scale_vec = np.array([float(scale_np_out)] * 3)
        elif scale_np_out.size == 1:
            scale_vec = np.array([float(scale_np_out.flat[0])] * 3)
        else:
            scale_vec = scale_np_out.flatten()[:3]

        vertices = mesh.vertices.copy()

        # 1. Convert Z-up (SAM3D decoder) to Y-up (pytorch3d convention)
        vertices_y_up = vertices @ z_up_to_y_up_matrix

        # 2. Apply refined transform (correct order: scale -> rotate -> translate)
        # This matches the original apply_transform() in layout_post_optimization_utils.py
        vertices_scaled = vertices_y_up * scale_vec  # Per-axis scale
        vertices_rotated = vertices_scaled @ rot_matrix  # Rotate
        vertices_transformed = vertices_rotated + trans_np  # Translate

        mesh.vertices = vertices_transformed

        # Save reposed GLB
        input_dir = os.path.dirname(glb_path)
        input_name = os.path.splitext(os.path.basename(glb_path))[0]
        output_glb_path = os.path.join(input_dir, f"{input_name}_reposed.glb")
        mesh.export(output_glb_path)

        log.info("Saved reposed GLB: %s", output_glb_path)

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
        log.error("Pose optimization error: %s", e)
        traceback.print_exc(file=sys.stderr)
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def run_pose_optimization_batch(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run pose optimization for batch scene generation.

    This function handles optimization without manual (height-based) alignment.
    Supports: icp_only, render_only modes.
    """
    try:
        import trimesh
        from pytorch3d.structures import Meshes
        from pytorch3d.transforms import quaternion_to_matrix, Transform3d, matrix_to_quaternion
        from pytorch3d.renderer import TexturesVertex
        from ..sam3d.transforms import compose_transform
        from ..sam3d.pipeline import (
            get_mesh,
            get_mask_renderer,
            compute_iou,
            run_ICP,
            run_render_compare,
            apply_transform,
            check_occlusion,
            set_seed,
        )

        log.info("Running batch pose optimization (no manual alignment)")

        # Extract parameters
        object_dir = request.get("object_dir", "")
        glb_path = request["glb_path"]
        pointmap_path = request["pointmap_path"]
        enable_manual_alignment = request.get("enable_manual_alignment", False)
        enable_icp = request.get("enable_icp", False)
        enable_render_opt = request.get("enable_render_opt", False)

        log.debug("enable_manual_alignment: %s", enable_manual_alignment)
        log.debug("enable_icp: %s", enable_icp)
        log.debug("enable_render_opt: %s", enable_render_opt)

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
        device = comfy.model_management.get_torch_device()

        # Convert to tensors
        rotation = torch.from_numpy(rotation_np).float().to(device)
        translation = torch.from_numpy(translation_np).float().to(device)
        scale = torch.from_numpy(scale_np).float().to(device)
        mask_tensor = torch.from_numpy(mask_np).float().to(device)
        intrinsics = intrinsics.to(device)

        # Ensure correct shapes for pose tensors
        if rotation.dim() == 1:
            rotation = rotation.unsqueeze(0)
        if translation.dim() == 1:
            translation = translation.unsqueeze(0)
        if scale.dim() == 0:
            scale = scale.unsqueeze(0).unsqueeze(0).expand(1, 3)
        elif scale.dim() == 1:
            scale = scale.unsqueeze(0)

        # Load mesh
        log.info("Loading mesh: %s", glb_path)
        mesh_trimesh = trimesh.load(glb_path)
        if isinstance(mesh_trimesh, trimesh.Scene):
            meshes = [g for g in mesh_trimesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if not meshes:
                raise ValueError("No mesh found in GLB file")
            mesh_trimesh = meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)

        # Load pointmap
        log.info("Loading pointmap: %s", pointmap_path)
        pointmap_data = comfy.utils.load_torch_file(pointmap_path, safe_load=False)
        if isinstance(pointmap_data, dict):
            pointmap_tensor = pointmap_data.get("pointmap")
            if pointmap_tensor is None:
                pointmap_tensor = pointmap_data.get("data")
        else:
            pointmap_tensor = pointmap_data
        pointmap_tensor = pointmap_tensor.float().to(device)

        set_seed(100)

        # Initialize transformation and process mesh
        Rotation = quaternion_to_matrix(rotation.squeeze(1) if rotation.dim() > 2 else rotation)
        center = translation[0].clone()
        tfm_ori = compose_transform(scale=scale, rotation=Rotation, translation=translation)
        mesh, faces_idx, textures = get_mesh(mesh_trimesh, tfm_ori, device)

        # Setup renderer (resizes mask to 512 internally)
        mask_processed, renderer = get_mask_renderer(mask_tensor, 512, intrinsics, device)

        # Check occlusion using original full-resolution mask and pointmap
        # (mask_tensor is already at same size as pointmap: 4480x6720)
        if check_occlusion(mask_tensor.cpu().numpy(), pointmap_tensor.cpu().numpy()):
            log.info("Occluded, skipping optimization")
            # Return original pose
            return _apply_and_save_pose(
                mesh_trimesh, rotation_np, translation_np, scale_np,
                request.get("output_glb_path"), object_dir, glb_path,
                iou=-1.0, used_icp=False, used_render_opt=False
            )

        # Compute initial IoU
        rendered = renderer(mesh)
        initial_iou = compute_iou(rendered[..., 3][0][None, None], mask_processed, threshold=0.5)
        final_iou = initial_iou.cpu().item()
        log.info("Initial IoU: %.3f", final_iou)

        # Track transforms
        tfm = tfm_ori
        used_icp = False
        used_render_opt = False

        # Get target points from pointmap (for ICP)
        if enable_icp:
            # Get target points from full-resolution pointmap using original mask
            mask_2d = mask_tensor.bool().cpu().numpy()
            depth = pointmap_tensor[..., 2].cpu().numpy()
            depth_edge_mask = depth_edge(depth, rtol=0.03, mask=mask_2d)
            cleaned_mask = mask_2d & ~depth_edge_mask
            cleaned_mask_tensor = torch.from_numpy(cleaned_mask).to(device)
            target_points = pointmap_tensor[cleaned_mask_tensor]
            finite_mask = torch.isfinite(target_points).all(dim=1)
            target_points = target_points[finite_mask]

            if target_points.shape[0] > 0:
                source_points = mesh.verts_packed()

                # Run ICP
                log.info("Running ICP...")
                points_aligned_icp, transformation = run_ICP(
                    mesh, source_points, target_points, threshold=0.05
                )
                mesh_ICP = Meshes(
                    verts=[points_aligned_icp], faces=[faces_idx], textures=textures
                )
                rendered_icp = renderer(mesh_ICP)
                icp_iou = compute_iou(
                    rendered_icp[..., 3][0][None, None], mask_processed, threshold=0.5
                )

                # Accept ICP if it improves IoU
                if icp_iou > initial_iou:
                    mesh = mesh_ICP
                    final_iou = icp_iou.cpu().item()
                    used_icp = True
                    log.info("ICP improved IoU: %.3f", final_iou)

                    # Update center and transform
                    T_o3d = torch.tensor(transformation, dtype=torch.float32, device=device)
                    T_o3d = T_o3d.T
                    A = T_o3d[:3, :3]
                    t = T_o3d[3, :3]
                    icp_scale = A.norm(dim=1)
                    R = A / icp_scale[:, None]
                    center = ((center[None] * icp_scale) @ R + t)[0]
                    tfm2 = (
                        Transform3d(device=device)
                        .scale(icp_scale[None])
                        .rotate(R[None])
                        .translate(t[None])
                    )
                    tfm = tfm.compose(tfm2)
                else:
                    log.info("ICP rejected (IoU: %.3f)", icp_iou.cpu().item())

        # Run render-and-compare optimization
        if enable_render_opt:
            log.info("Running render optimization...")
            ori_iou = final_iou

            quat, trans_opt, scale_opt, R = run_render_compare(
                mesh, center, renderer, mask_processed, device
            )
            with torch.no_grad():
                transformed = apply_transform(mesh, center, quat, trans_opt, scale_opt)
                rendered_opt = renderer(transformed)
            optimized_iou = compute_iou(
                rendered_opt[..., 3][0][None, None], mask_processed, threshold=0.5
            )

            # Accept if IoU > 0.5 and improves
            if optimized_iou > 0.5 and optimized_iou > ori_iou:
                used_render_opt = True
                final_iou = optimized_iou.detach().cpu().item()
                log.info("Render optimization improved IoU: %.3f", final_iou)

                tfm3 = (
                    Transform3d(device=device)
                    .translate(-center[None])
                    .scale(scale_opt.expand(3)[None])
                    .rotate(R.T[None])
                    .translate(center[None])
                    .translate(trans_opt[None])
                )
                tfm = tfm.compose(tfm3)
            else:
                log.info("Render optimization rejected (IoU: %.3f)", optimized_iou.cpu().item())

        # Extract final pose from transform
        M = tfm.get_matrix()[0]
        T_final = M[3, :3]
        A = M[:3, :3]
        scale_final = A.norm(dim=1)
        R_final = A / scale_final[:, None]
        quat_final = matrix_to_quaternion(R_final)

        # Convert to numpy
        quat_np = quat_final.cpu().numpy()
        trans_np = T_final.cpu().numpy()
        scale_np_out = scale_final.cpu().numpy()

        log.info("Final IoU: %.3f", final_iou)

        return _apply_and_save_pose(
            mesh_trimesh, quat_np, trans_np, scale_np_out,
            request.get("output_glb_path"), object_dir, glb_path,
            iou=final_iou, used_icp=used_icp, used_render_opt=used_render_opt
        )

    except Exception as e:
        log.error("Batch pose optimization error: %s", e)
        traceback.print_exc(file=sys.stderr)
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def _apply_and_save_pose(
    mesh_trimesh, quat_np, trans_np, scale_np_out,
    output_glb_path, object_dir, glb_path,
    iou, used_icp, used_render_opt
) -> Dict[str, Any]:
    """Apply final pose to trimesh and save GLB."""
    from pytorch3d.transforms import quaternion_to_matrix

    # Ensure correct shapes
    if quat_np.ndim > 1:
        quat_np = quat_np.squeeze()
    if trans_np.ndim > 1:
        trans_np = trans_np.squeeze()
    if scale_np_out.ndim > 1:
        scale_np_out = scale_np_out.squeeze()

    # Convert quaternion to rotation matrix
    quat_tensor = torch.from_numpy(quat_np).float().unsqueeze(0)
    rot_matrix = quaternion_to_matrix(quat_tensor).squeeze(0).numpy()

    # Z-up to Y-up conversion
    z_up_to_y_up_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).T

    # Ensure scale is per-axis
    if scale_np_out.ndim == 0:
        scale_vec = np.array([float(scale_np_out)] * 3)
    elif scale_np_out.size == 1:
        scale_vec = np.array([float(scale_np_out.flat[0])] * 3)
    else:
        scale_vec = scale_np_out.flatten()[:3]

    vertices = mesh_trimesh.vertices.copy()

    # 1. Convert Z-up to Y-up
    vertices_y_up = vertices @ z_up_to_y_up_matrix

    # 2. Apply transform: scale -> rotate -> translate
    vertices_scaled = vertices_y_up * scale_vec
    vertices_rotated = vertices_scaled @ rot_matrix
    vertices_transformed = vertices_rotated + trans_np

    mesh_trimesh.vertices = vertices_transformed

    # Determine output path
    if not output_glb_path:
        if object_dir:
            output_glb_path = os.path.join(object_dir, "aligned_mesh.glb")
        else:
            input_dir = os.path.dirname(glb_path)
            output_glb_path = os.path.join(input_dir, "aligned_mesh.glb")

    mesh_trimesh.export(output_glb_path)
    log.info("Saved aligned GLB: %s", output_glb_path)

    return {
        "status": "success",
        "output_glb_path": output_glb_path,
        "iou": float(iou),
        "used_icp": bool(used_icp),
        "used_render_opt": bool(used_render_opt),
    }
