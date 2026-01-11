"""
Utility functions for SAM3D inference worker.

This module contains:
- Serialization/deserialization helpers
- Coordinate transformation functions
- File I/O helpers
"""

import sys
import json
import pickle
import base64
import traceback
from pathlib import Path
from typing import Any, Dict
import numpy as np
from PIL import Image
import io
import torch


def deserialize_image(image_b64: str) -> Image.Image:
    """Deserialize base64-encoded image."""
    image_bytes = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(image_bytes))


def deserialize_mask(mask_b64: str) -> np.ndarray:
    """Deserialize base64-encoded mask."""
    mask_bytes = base64.b64decode(mask_b64)
    return pickle.loads(mask_bytes)


def transform_to_global_coordinates(output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform Gaussian splat and mesh to global (scene) coordinates using pose data.

    This applies rotation, translation, and scale from depth estimation so that
    multiple objects from the same image can be correctly positioned when combined.
    """
    rotation = output.get("rotation")
    translation = output.get("translation")
    scale = output.get("scale")

    # If no pose data, nothing to transform
    if rotation is None or translation is None or scale is None:
        print("[Worker] No pose data available, skipping global coordinate transform", file=sys.stderr)
        return output

    print(f"[Worker] Transforming to global coordinates", file=sys.stderr)
    print(f"[Worker] Pose: rotation shape={rotation.shape if hasattr(rotation, 'shape') else 'N/A'}, translation shape={translation.shape if hasattr(translation, 'shape') else 'N/A'}, scale={scale}", file=sys.stderr)

    # Transform Gaussian if present
    if "gs" in output and output["gs"] is not None:
        try:
            gs = output["gs"]

            # Import required utilities
            from sam3d_objects.utils.visualization.scene_visualizer import SceneVisualizer
            from pytorch3d.transforms import quaternion_multiply, quaternion_invert

            # Get Gaussian positions in local coordinates
            xyz_local = gs.get_xyz  # (N, 3)

            # Transform to global coordinates using pose
            # SceneVisualizer.object_pointcloud expects batched input
            PC = SceneVisualizer.object_pointcloud(
                points_local=xyz_local.unsqueeze(0),  # (1, N, 3)
                quat_l2c=rotation,
                trans_l2c=translation,
                scale_l2c=scale,
            )
            # Set transformed positions
            gs.from_xyz(PC.points_list()[0])

            # Rotate the Gaussian rotation parameters
            gs.from_rotation(
                quaternion_multiply(
                    quaternion_invert(rotation),
                    gs.get_rotation,
                )
            )

            # Scale the Gaussian scaling parameters
            adjusted_scale = gs.get_scaling * scale
            # Ensure minimum kernel size is maintained
            if hasattr(gs, 'mininum_kernel_size'):
                gs.mininum_kernel_size *= scale[0, 0].item()
                adjusted_scale = torch.maximum(
                    adjusted_scale,
                    torch.tensor(
                        gs.mininum_kernel_size * 1.1,
                        device=adjusted_scale.device,
                    ),
                )
            gs.from_scaling(adjusted_scale)

            print(f"[Worker] Transformed Gaussian to global coordinates", file=sys.stderr)

        except Exception as e:
            print(f"[Worker] Warning: Failed to transform Gaussian: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    # Transform mesh vertices if present
    if "mesh" in output and output["mesh"] is not None:
        try:
            mesh_list = output["mesh"]
            if isinstance(mesh_list, list) and len(mesh_list) > 0:
                mesh = mesh_list[0]

                # Import required utilities
                from sam3d_objects.utils.visualization.scene_visualizer import SceneVisualizer

                # Get mesh vertices
                if hasattr(mesh, 'vertices'):
                    vertices = mesh.vertices  # (N, 3)
                    if hasattr(vertices, 'unsqueeze'):
                        # Transform using pose
                        PC = SceneVisualizer.object_pointcloud(
                            points_local=vertices.unsqueeze(0),
                            quat_l2c=rotation,
                            trans_l2c=translation,
                            scale_l2c=scale,
                        )
                        mesh.vertices = PC.points_list()[0]
                        print(f"[Worker] Transformed mesh vertices to global coordinates", file=sys.stderr)

        except Exception as e:
            print(f"[Worker] Warning: Failed to transform mesh: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    # Transform trimesh GLB if it's already been converted
    if "glb" in output and output["glb"] is not None:
        try:
            import trimesh
            if isinstance(output["glb"], trimesh.Trimesh):
                glb_mesh = output["glb"]

                # Import required utilities
                from sam3d_objects.utils.visualization.scene_visualizer import SceneVisualizer

                # Transform vertices
                vertices = torch.from_numpy(glb_mesh.vertices).float()
                if torch.cuda.is_available():
                    vertices = vertices.cuda()

                PC = SceneVisualizer.object_pointcloud(
                    points_local=vertices.unsqueeze(0),
                    quat_l2c=rotation,
                    trans_l2c=translation,
                    scale_l2c=scale,
                )
                glb_mesh.vertices = PC.points_list()[0].cpu().numpy()
                print(f"[Worker] Transformed GLB mesh to global coordinates", file=sys.stderr)

        except Exception as e:
            print(f"[Worker] Warning: Failed to transform GLB mesh: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    return output


def save_output_to_disk(output: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    """
    Save output to disk and return file paths.

    This is much more robust than trying to serialize complex objects through IPC.
    Following ComfyUI's standard pattern of saving outputs to disk.

    The output_dir should be a sam3d_inference_N directory created by depth_estimate.
    Files are saved directly to this directory (no subdirectory creation).
    """
    # Use provided directory directly (created by depth_estimate node)
    save_dir = Path(output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Worker] Saving outputs to: {save_dir}", file=sys.stderr)

    result = {
        "output_dir": str(save_dir),
        "files": {},
        "metadata": {}
    }

    # Save sparse structure (Stage 1 output)
    # Identify sparse structure by keys present in stage 1 but not others
    if "coords" in output and "pointmap" in output and "slat" not in output:
        sparse_path = save_dir / "sparse_structure.pt"
        torch.save(output, sparse_path)
        result["files"]["sparse_structure"] = str(sparse_path)
        print(f"[Worker] Saved sparse structure: {sparse_path}", file=sys.stderr)

    # Save SLAT (Stage 2 intermediate output)
    if "slat" in output:
        slat_path = save_dir / "slat.pt"
        torch.save(output, slat_path)
        result["files"]["slat"] = str(slat_path)
        print(f"[Worker] Saved SLAT: {slat_path}", file=sys.stderr)

    # Save GLB file (textured mesh)
    if "glb" in output and output["glb"] is not None:
        glb_path = save_dir / "mesh.glb"

        # Check if it's a Trimesh object that needs to be exported
        import trimesh
        if isinstance(output["glb"], trimesh.Trimesh):
            # Export Trimesh to GLB format
            glb_bytes = output["glb"].export(file_type="glb")
            with open(glb_path, 'wb') as f:
                f.write(glb_bytes)
            print(f"[Worker] Saved GLB: {glb_path} ({len(glb_bytes)} bytes)", file=sys.stderr)
        else:
            # Already bytes
            with open(glb_path, 'wb') as f:
                f.write(output["glb"])
            print(f"[Worker] Saved GLB: {glb_path} ({len(output['glb'])} bytes)", file=sys.stderr)

        result["files"]["glb"] = str(glb_path)

    # Save Gaussian Splat PLY file (colored point cloud)
    if "gs" in output and output["gs"] is not None:
        ply_path = save_dir / "gaussian.ply"
        try:
            output["gs"].save_ply(str(ply_path))
            result["files"]["ply"] = str(ply_path)
            print(f"[Worker] Saved Gaussian PLY: {ply_path}", file=sys.stderr)
        except Exception as e:
            print(f"[Worker] Warning: Failed to save Gaussian PLY: {e}", file=sys.stderr)

    # Save metadata (simple types only)
    metadata = {}
    for key, value in output.items():
        if isinstance(value, (int, float, str, bool)):
            metadata[key] = value
        elif isinstance(value, torch.Tensor):
            # Convert torch tensors to lists for JSON serialization
            metadata[key] = value.cpu().tolist()
        elif isinstance(value, np.ndarray):
            # Convert numpy arrays to lists for JSON serialization
            metadata[key] = value.tolist()
        elif isinstance(value, dict) and key not in ["glb", "gaussian_splat", "mesh"]:
            # Save simple dict metadata
            try:
                json.dumps(value)  # Test if it's JSON-serializable
                metadata[key] = value
            except:
                pass

    if metadata:
        metadata_path = save_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        result["files"]["metadata"] = str(metadata_path)
        result["metadata"] = metadata

    return result


