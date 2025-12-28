"""
Texture baking for SAM3D inference.

This module contains direct texture baking without model loading.
"""

import sys
from pathlib import Path
from typing import Any, Dict
import numpy as np
import torch


def run_texture_bake_direct(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run texture baking directly without loading any models.

    Loads Gaussian from PLY and Mesh from GLB, then calls to_glb() for texture baking.
    No embedders, generators, or other models are needed.

    NOTE: Mesh simplification should be done in MeshDecode, not here.
    This function only handles texture baking.

    Args:
        request: Dict with ply_path, glb_path, output_dir, texture_mode, etc.

    Returns:
        Dict with status and output (glb_path)
    """
    import trimesh

    print("[Worker] Running direct texture baking (no models)", file=sys.stderr)

    # Extract parameters
    ply_path = request["ply_path"]
    glb_path = request["glb_path"]
    output_dir = request["output_dir"]
    texture_mode = request.get("texture_mode", "opt")
    texture_size = request.get("texture_size", 1024)
    rendering_engine = request.get("rendering_engine", "nvdiffrast")

    print(f"[Worker] PLY: {ply_path}", file=sys.stderr)
    print(f"[Worker] GLB: {glb_path}", file=sys.stderr)
    print(f"[Worker] Mode: {texture_mode}, Size: {texture_size}", file=sys.stderr)

    # Setup environment for sam3d_objects imports
    vendor_path = Path(__file__).parent.parent / "vendor"
    if str(vendor_path) not in sys.path:
        sys.path.insert(0, str(vendor_path))

    # Import required modules
    from sam3d_objects.model.backbone.tdfy_dit.representations.gaussian import Gaussian
    from sam3d_objects.model.backbone.tdfy_dit.representations.mesh.cube2mesh import MeshExtractResult
    from sam3d_objects.model.backbone.tdfy_dit.utils.postprocessing_utils import to_glb

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load Gaussian from PLY
    print(f"[Worker] Loading Gaussian from PLY...", file=sys.stderr)
    gaussian = Gaussian(
        aabb=[-1, -1, -1, 2, 2, 2],  # Default AABB
        sh_degree=0,
        device=device
    )
    gaussian.load_ply(ply_path)
    print(f"[Worker] Loaded Gaussian with {gaussian._xyz.shape[0]} points", file=sys.stderr)

    # Load Mesh from GLB
    print(f"[Worker] Loading Mesh from GLB...", file=sys.stderr)
    loaded = trimesh.load(glb_path)

    # Handle Scene vs Mesh
    if isinstance(loaded, trimesh.Scene):
        meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise RuntimeError("No mesh geometries found in GLB")
        trimesh_mesh = meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)
    else:
        trimesh_mesh = loaded

    print(f"[Worker] Loaded mesh with {len(trimesh_mesh.vertices)} vertices", file=sys.stderr)

    # Convert to MeshExtractResult format expected by to_glb
    # NOTE: The GLB from _run_decode_lazy is saved in Z-up (raw decoder output)
    # The Gaussian PLY is also in Z-up. Both are aligned - no transformation needed.
    # to_glb() will apply the final Z-up→Y-up transform when saving.
    vertices_np = np.array(trimesh_mesh.vertices)
    vertices_tensor = torch.tensor(vertices_np, dtype=torch.float32, device=device)
    faces_tensor = torch.tensor(np.array(trimesh_mesh.faces), dtype=torch.long, device=device)

    # Get vertex colors if available
    if trimesh_mesh.visual is not None and hasattr(trimesh_mesh.visual, 'vertex_colors') and trimesh_mesh.visual.vertex_colors is not None:
        vertex_colors = np.array(trimesh_mesh.visual.vertex_colors)[:, :3] / 255.0
    else:
        vertex_colors = np.ones((len(trimesh_mesh.vertices), 3), dtype=np.float32)

    vertex_attrs_tensor = torch.tensor(vertex_colors, dtype=torch.float32, device=device)

    mesh = MeshExtractResult(
        vertices=vertices_tensor,
        faces=faces_tensor,
        vertex_attrs=vertex_attrs_tensor,
        res=64
    )

    # Run texture baking using to_glb
    # NOTE: Mesh simplification is done in MeshDecode, not here
    print(f"[Worker] Running texture baking...", file=sys.stderr)
    result_mesh = to_glb(
        gaussian,
        mesh,
        simplify=0,  # No simplification - should be done in MeshDecode
        fill_holes=False,
        texture_size=texture_size,
        verbose=False,
        with_mesh_postprocess=False,  # No postprocessing - should be done in MeshDecode
        with_texture_baking=True,
        rendering_engine=rendering_engine,
        texture_mode=texture_mode,
    )

    # Save textured GLB
    output_path = Path(output_dir) / "mesh_textured.glb"
    glb_bytes = result_mesh.export(file_type="glb")
    with open(output_path, 'wb') as f:
        f.write(glb_bytes)

    print(f"[Worker] Saved textured GLB: {output_path}", file=sys.stderr)

    # Cleanup GPU memory
    del gaussian, mesh
    torch.cuda.empty_cache()

    return {
        "status": "success",
        "output": {
            "glb_path": str(output_path),
        }
    }
