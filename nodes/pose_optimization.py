"""SAM3D_PoseOptimization node for refining object pose using ICP and render optimization."""

import os
import torch
import numpy as np
from typing import Any, Dict


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

        Args:
            glb_path: Path to GLB mesh file
            pointmap_path: Path to pointmap .pt file
            intrinsics: Camera intrinsics matrix (3, 3)
            pose: Pose dict with 'rotation', 'translation', 'scale'
            mask: Binary mask tensor
            enable_icp: Enable ICP step
            enable_render_opt: Enable render optimization step

        Returns:
            Tuple of (reposed_glb_path, refined_pose, iou_score)
        """
        import trimesh
        from pytorch3d.transforms import quaternion_to_matrix

        print(f"[SAM3DObjects] PoseOptimization: Starting pose refinement")
        print(f"[SAM3DObjects]   - GLB path: {glb_path}")
        print(f"[SAM3DObjects]   - Pointmap path: {pointmap_path}")
        print(f"[SAM3DObjects]   - Enable ICP: {enable_icp}")
        print(f"[SAM3DObjects]   - Enable render optimization: {enable_render_opt}")

        # Load mesh
        if not glb_path:
            raise ValueError("GLB path is required")

        mesh = trimesh.load(glb_path)
        original_scene = None
        if isinstance(mesh, trimesh.Scene):
            original_scene = mesh
            # Extract mesh from scene
            meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if not meshes:
                raise ValueError("No mesh found in GLB file")
            mesh = meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)

        print(f"[SAM3DObjects]   - Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        # Extract pose components
        rotation = pose.get("rotation")
        translation = pose.get("translation")
        scale = pose.get("scale")

        if rotation is None or translation is None or scale is None:
            print("[SAM3DObjects] Warning: Incomplete pose data, returning original")
            return (glb_path, pose, -1.0)

        # Convert to tensors if needed
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not isinstance(rotation, torch.Tensor):
            rotation = torch.tensor(rotation, dtype=torch.float32, device=device)
        else:
            rotation = rotation.to(device)

        if not isinstance(translation, torch.Tensor):
            translation = torch.tensor(translation, dtype=torch.float32, device=device)
        else:
            translation = translation.to(device)

        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale, dtype=torch.float32, device=device)
        else:
            scale = scale.to(device)

        # Ensure correct shapes
        if rotation.dim() == 1:
            rotation = rotation.unsqueeze(0)  # (1, 4)
        if translation.dim() == 1:
            translation = translation.unsqueeze(0)  # (1, 3)
        if scale.dim() == 0:
            scale = scale.unsqueeze(0).unsqueeze(0).expand(1, 3)  # (1, 3)
        elif scale.dim() == 1:
            scale = scale.unsqueeze(0)  # (1, 3)

        print(f"[SAM3DObjects]   - Rotation shape: {rotation.shape}")
        print(f"[SAM3DObjects]   - Translation shape: {translation.shape}")
        print(f"[SAM3DObjects]   - Scale shape: {scale.shape}")

        # Load pointmap from file
        if not pointmap_path:
            raise ValueError("Pointmap path is required")

        print(f"[SAM3DObjects] Loading pointmap from: {pointmap_path}")
        pointmap_data = torch.load(pointmap_path, weights_only=False)

        # Extract pointmap tensor from loaded data
        if isinstance(pointmap_data, dict):
            pointmap_tensor = pointmap_data.get("pointmap")
            if pointmap_tensor is None:
                pointmap_tensor = pointmap_data.get("data")
        else:
            pointmap_tensor = pointmap_data

        if pointmap_tensor is None:
            raise ValueError("Pointmap tensor not found in file")

        if not isinstance(pointmap_tensor, torch.Tensor):
            pointmap_tensor = torch.tensor(pointmap_tensor, dtype=torch.float32, device=device)
        else:
            pointmap_tensor = pointmap_tensor.to(device)

        print(f"[SAM3DObjects]   - Pointmap shape: {pointmap_tensor.shape}")

        # Get intrinsics
        if not isinstance(intrinsics, torch.Tensor):
            intrinsics = torch.tensor(intrinsics, dtype=torch.float32, device=device)
        else:
            intrinsics = intrinsics.to(device)

        print(f"[SAM3DObjects]   - Intrinsics shape: {intrinsics.shape}")

        # Get mask
        if mask.dim() == 3:
            mask_2d = mask[0]  # Take first mask if batched
        else:
            mask_2d = mask

        mask_2d = mask_2d.to(device)
        print(f"[SAM3DObjects]   - Mask shape: {mask_2d.shape}")

        # Import the optimization function
        try:
            from sam3d_objects.pipeline.inference_utils import layout_post_optimization
        except ImportError as e:
            print(f"[SAM3DObjects] Error: Could not import layout_post_optimization: {e}")
            print("[SAM3DObjects] Returning original without optimization")
            return (glb_path, pose, -1.0)

        # Run optimization
        print("[SAM3DObjects] Running layout post-optimization...")
        try:
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
                Mask=mask_2d,
                Point_Map=pointmap_tensor,
                Intrinsics=intrinsics,
                Enable_shape_ICP=enable_icp,
                Enable_rendering_optimization=enable_render_opt,
                device=device,
            )

            print(f"[SAM3DObjects] Optimization complete:")
            print(f"[SAM3DObjects]   - Final IoU: {final_iou}")
            print(f"[SAM3DObjects]   - Used ICP: {used_icp}")
            print(f"[SAM3DObjects]   - Used render optimization: {used_render_opt}")

            # Create refined pose dict
            refined_pose = {
                "rotation": refined_quat.cpu() if hasattr(refined_quat, 'cpu') else refined_quat,
                "translation": refined_trans.cpu() if hasattr(refined_trans, 'cpu') else refined_trans,
                "scale": refined_scale.cpu() if hasattr(refined_scale, 'cpu') else refined_scale,
            }

            # Apply refined pose to mesh and save reposed GLB
            print("[SAM3DObjects] Applying refined pose to mesh...")

            # Get pose as numpy for trimesh
            if hasattr(refined_quat, 'cpu'):
                quat_np = refined_quat.cpu().numpy()
            else:
                quat_np = np.array(refined_quat)

            if hasattr(refined_trans, 'cpu'):
                trans_np = refined_trans.cpu().numpy()
            else:
                trans_np = np.array(refined_trans)

            if hasattr(refined_scale, 'cpu'):
                scale_np = refined_scale.cpu().numpy()
            else:
                scale_np = np.array(refined_scale)

            # Ensure correct shapes
            if quat_np.ndim > 1:
                quat_np = quat_np.squeeze()
            if trans_np.ndim > 1:
                trans_np = trans_np.squeeze()
            if scale_np.ndim > 1:
                scale_np = scale_np.squeeze()

            # Convert quaternion to rotation matrix using pytorch3d
            quat_tensor = torch.tensor(quat_np, dtype=torch.float32).unsqueeze(0)
            rot_matrix = quaternion_to_matrix(quat_tensor).squeeze().cpu().numpy()

            # Build 4x4 transformation matrix
            transform = np.eye(4)
            # Apply scale to rotation matrix
            scale_val = scale_np.mean() if scale_np.ndim > 0 else float(scale_np)
            transform[:3, :3] = rot_matrix * scale_val
            transform[:3, 3] = trans_np

            print(f"[SAM3DObjects]   - Scale: {scale_val:.4f}")
            print(f"[SAM3DObjects]   - Translation: [{trans_np[0]:.4f}, {trans_np[1]:.4f}, {trans_np[2]:.4f}]")

            # Apply transformation to mesh vertices
            vertices = mesh.vertices.copy()
            # Transform: v' = R * s * v + t
            vertices_transformed = (vertices @ (rot_matrix.T * scale_val)) + trans_np
            mesh.vertices = vertices_transformed

            # Generate output path
            input_dir = os.path.dirname(glb_path)
            input_name = os.path.splitext(os.path.basename(glb_path))[0]
            output_glb_path = os.path.join(input_dir, f"{input_name}_reposed.glb")

            # Save reposed GLB
            print(f"[SAM3DObjects] Saving reposed GLB to: {output_glb_path}")
            mesh.export(output_glb_path)

            file_size = os.path.getsize(output_glb_path)
            print(f"[SAM3DObjects] Reposed GLB saved: {file_size:,} bytes")

            return (output_glb_path, refined_pose, float(final_iou))

        except Exception as e:
            print(f"[SAM3DObjects] Error during optimization: {e}")
            import traceback
            traceback.print_exc()
            print("[SAM3DObjects] Returning original")
            return (glb_path, pose, -1.0)
