# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Geometry operations with Open3D/Trimesh fallback support.

This module centralizes all 3D geometry operations that depend on Open3D,
providing trimesh-based fallbacks when Open3D is not available (e.g., on Windows
where Open3D pip wheels have DLL issues).
"""

import numpy as np
import torch
import trimesh
from loguru import logger

# =============================================================================
# Open3D conditional import
# =============================================================================

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    o3d = None
    HAS_OPEN3D = False
    logger.warning(
        "Open3D not available - using trimesh fallbacks. "
        "Some features (ICP registration) will be skipped."
    )


# =============================================================================
# Mesh operations
# =============================================================================

def trimesh_to_o3d_mesh(trimesh_mesh):
    """
    Convert a trimesh mesh to Open3D TriangleMesh.

    Returns None if Open3D is not available.
    """
    if not HAS_OPEN3D:
        return None

    verts = np.asarray(trimesh_mesh.vertices)
    faces = np.asarray(trimesh_mesh.faces)
    return o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts),
        o3d.utility.Vector3iVector(faces)
    )


def load_and_simplify_mesh(mesh, device, target_triangles=5000):
    """
    Clean and simplify a mesh to target triangle count.

    Args:
        mesh: trimesh.Trimesh object
        device: torch device
        target_triangles: target number of triangles after simplification

    Returns:
        verts: torch.Tensor of vertices
        faces: torch.Tensor of faces
    """
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)

    if HAS_OPEN3D:
        # Use Open3D for cleaning and simplification
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)

        mesh_o3d.remove_duplicated_vertices()
        mesh_o3d.remove_degenerate_triangles()
        mesh_o3d.remove_duplicated_triangles()
        mesh_o3d.remove_non_manifold_edges()

        if len(mesh_o3d.triangles) > target_triangles:
            mesh_simplified = mesh_o3d.simplify_quadric_decimation(target_triangles)
        else:
            mesh_simplified = mesh_o3d

        verts = torch.tensor(
            np.asarray(mesh_simplified.vertices), dtype=torch.float32, device=device
        )
        faces = torch.tensor(
            np.asarray(mesh_simplified.triangles), dtype=torch.int64, device=device
        )
    else:
        # Trimesh fallback
        # Create a copy to avoid modifying the original
        mesh_copy = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        # Clean the mesh
        mesh_copy.remove_duplicate_faces()
        mesh_copy.remove_degenerate_faces()
        mesh_copy.merge_vertices()

        # Simplify if needed
        if len(mesh_copy.faces) > target_triangles:
            mesh_copy = mesh_copy.simplify_quadric_decimation(target_triangles)

        verts = torch.tensor(
            np.asarray(mesh_copy.vertices), dtype=torch.float32, device=device
        )
        faces = torch.tensor(
            np.asarray(mesh_copy.faces), dtype=torch.int64, device=device
        )

    return verts, faces


def voxelize_mesh(mesh, resolution=64):
    """
    Convert a mesh to a voxel grid.

    Args:
        mesh: trimesh.Trimesh object (or Open3D mesh if HAS_OPEN3D)
        resolution: voxel grid resolution

    Returns:
        ss: torch.Tensor voxel grid (1, resolution, resolution, resolution)
        scale: float or None
        center: numpy array or None
    """
    verts = np.asarray(mesh.vertices)
    # rotate mesh (from z-up to y-up)
    verts = verts @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).T

    # normalize vertices
    if np.abs(verts.min() + 0.5) < 1e-3 and np.abs(verts.max() - 0.5) < 1e-3:
        vertices, scale, center = verts, None, None
    else:
        vertices, scale, center = _normalize_mesh_verts(verts)

    vertices = np.clip(vertices, -0.5 + 1e-6, 0.5 - 1e-6)

    if HAS_OPEN3D:
        # Use Open3D's VoxelGrid
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
            mesh,
            voxel_size=1 / 64,
            min_bound=(-0.5, -0.5, -0.5),
            max_bound=(0.5, 0.5, 0.5),
        )
        vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
        vertices = (vertices + 0.5) / 64 - 0.5
    else:
        # Trimesh fallback - use trimesh's voxelization
        mesh_copy = trimesh.Trimesh(
            vertices=vertices,
            faces=np.asarray(mesh.faces),
            process=False
        )
        # Voxelize using trimesh
        pitch = 1.0 / 64
        voxel_grid = mesh_copy.voxelized(pitch=pitch)
        # Get voxel centers
        vertices = voxel_grid.points
        # Clip to bounds
        mask = np.all((vertices >= -0.5) & (vertices <= 0.5), axis=1)
        vertices = vertices[mask]

    coords = ((torch.tensor(vertices) + 0.5) * resolution).int().contiguous()
    ss = torch.zeros(1, resolution, resolution, resolution, dtype=torch.long)
    # Clip coords to valid range
    coords = torch.clamp(coords, 0, resolution - 1)
    ss[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
    return ss, scale, center


def _normalize_mesh_verts(verts):
    """Normalize mesh vertices to [-0.5, 0.5] range."""
    center = (verts.max(axis=0) + verts.min(axis=0)) / 2
    max_extent = (verts.max(axis=0) - verts.min(axis=0)).max()
    if max_extent < 1e-6:
        scale = 1.0
        vertices = verts - center
    else:
        scale = 1.0 / max_extent
        vertices = (verts - center) * scale
    return vertices, scale, center


# =============================================================================
# Point cloud operations
# =============================================================================

def tensor_to_point_cloud(tensor):
    """
    Convert a torch tensor to a point cloud representation.

    Returns Open3D PointCloud if available, otherwise numpy array.
    """
    points = tensor.cpu().numpy() if torch.is_tensor(tensor) else tensor

    if HAS_OPEN3D:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd
    else:
        return points


def point_cloud_to_tensor(pcd, device=None):
    """
    Convert a point cloud to torch tensor.

    Args:
        pcd: Open3D PointCloud or numpy array
        device: optional torch device
    """
    if HAS_OPEN3D and hasattr(pcd, 'points'):
        points = np.asarray(pcd.points)
    else:
        points = pcd if isinstance(pcd, np.ndarray) else np.asarray(pcd)

    tensor = torch.tensor(points, dtype=torch.float32)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def plane_estimation(points):
    """
    Estimate a plane from 3D points using RANSAC.

    Args:
        points: numpy array of shape (N, 3)

    Returns:
        plane_model: [a, b, c, d] where ax + by + cz + d = 0
        inliers: indices of inlier points
        clean_points: points after removing flying points
        normal: unit normal vector
        v1, v2: basis vectors in the plane
        centroid: center point of the plane
        u_extent, v_extent: extent of points along v1, v2 directions
    """
    if HAS_OPEN3D:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        plane_model, inliers = pcd.segment_plane(0.02, 3, 1000)
        inlier_points = np.asarray(pcd.points)[inliers]
    else:
        # Fallback: use scipy RANSAC-like approach
        plane_model, inliers = _ransac_plane_fit(points)
        inlier_points = points[inliers]

    [a, b, c, d] = plane_model
    logger.info(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    # Adaptive flying point removal based on Z-range
    z_range = np.max(inlier_points[:, 2]) - np.min(inlier_points[:, 2])
    if z_range > 6.0:
        thresh = 0.90
    elif z_range > 2.0:
        thresh = 0.93
    else:
        thresh = 0.95

    depth_quantile = np.quantile(inlier_points[:, 2], thresh)
    clean_points = inlier_points[inlier_points[:, 2] <= depth_quantile]

    logger.info(f"Flying point removal: {len(inlier_points)} -> {len(clean_points)} points")

    # Get the normal vector of the plane
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)

    # Create two orthogonal vectors in the plane
    if abs(normal[2]) < 0.9:
        tangent = np.array([0, 0, 1])
    else:
        tangent = np.array([1, 0, 0])

    v1 = np.cross(normal, tangent)
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(normal, v1)
    v2 = v2 / np.linalg.norm(v2)

    if np.dot(np.cross(v1, v2), normal) < 0:
        v2 = -v2

    # Calculate centroid using bounding box center
    min_vals = np.min(clean_points, axis=0)
    max_vals = np.max(clean_points, axis=0)
    centroid = (min_vals + max_vals) / 2

    # Project clean points onto the plane's coordinate system
    relative_points = clean_points - centroid
    u_coords = np.dot(relative_points, v1)
    v_coords = np.dot(relative_points, v2)

    u_min, u_max = np.percentile(u_coords, [0, 100])
    v_min, v_max = np.percentile(v_coords, [0, 100])

    u_extent = max(u_max - u_min, 0.1)
    v_extent = max(v_max - v_min, 0.1)

    return {
        'plane_model': plane_model,
        'inliers': inliers,
        'clean_points': clean_points,
        'normal': normal,
        'v1': v1,
        'v2': v2,
        'centroid': centroid,
        'u_extent': u_extent,
        'v_extent': v_extent,
    }


def segment_plane(points, distance_threshold=0.02, ransac_n=3, num_iterations=1000):
    """
    Segment a plane from point cloud using RANSAC.

    Uses Open3D if available, otherwise falls back to numpy RANSAC.

    Args:
        points: numpy array of shape (N, 3)
        distance_threshold: max distance for inliers
        ransac_n: number of points to sample
        num_iterations: RANSAC iterations

    Returns:
        plane_model: [a, b, c, d] where ax + by + cz + d = 0
        inliers: indices of inlier points
    """
    if HAS_OPEN3D:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        plane_model, inliers = pcd.segment_plane(
            distance_threshold, ransac_n, num_iterations
        )
        return list(plane_model), list(inliers)
    else:
        return _ransac_plane_fit(points, num_iterations, distance_threshold)


def _ransac_plane_fit(points, n_iterations=1000, threshold=0.02):
    """
    Simple RANSAC plane fitting fallback when Open3D is not available.
    """
    best_inliers = []
    best_plane = [0, 0, 1, 0]
    n_points = len(points)

    for _ in range(n_iterations):
        # Random sample 3 points
        idx = np.random.choice(n_points, 3, replace=False)
        p1, p2, p3 = points[idx]

        # Compute plane normal
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        normal = normal / norm

        # Plane equation: ax + by + cz + d = 0
        d = -np.dot(normal, p1)

        # Compute distances
        distances = np.abs(np.dot(points, normal) + d)
        inliers = np.where(distances < threshold)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_plane = [normal[0], normal[1], normal[2], d]

    return best_plane, best_inliers


# =============================================================================
# ICP Registration
# =============================================================================

def run_ICP(source_points_mesh, source_points, target_points, threshold):
    """
    Run ICP (Iterative Closest Point) registration.

    Args:
        source_points_mesh: pytorch3d Meshes object
        source_points: torch.Tensor source points
        target_points: torch.Tensor target points
        threshold: ICP threshold

    Returns:
        points_aligned: aligned source points
        transformation: 4x4 transformation matrix
    """
    if not HAS_OPEN3D:
        logger.warning("ICP registration skipped (requires Open3D)")
        # Return unchanged points with identity transform
        mesh_points = source_points_mesh.verts_padded().squeeze(0)
        return mesh_points, np.eye(4)

    # Convert to Open3D point clouds
    mesh_src_pcd = tensor_to_point_cloud(source_points_mesh.verts_padded().squeeze(0))
    src_pcd = tensor_to_point_cloud(source_points)
    tgt_pcd = tensor_to_point_cloud(target_points)

    # Run ICP
    trans_init = np.eye(4)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        src_pcd,
        tgt_pcd,
        threshold,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )

    # Apply transformation
    mesh_src_pcd.transform(reg_p2p.transformation)
    points_aligned_icp = point_cloud_to_tensor(mesh_src_pcd, device=source_points.device)

    return points_aligned_icp, reg_p2p.transformation
