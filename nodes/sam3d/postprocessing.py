# Copyright (c) Meta Platforms, Inc. and affiliates.
# Consolidated from:
#   - vendor/.../utils/random_utils.py
#   - vendor/.../utils/render_utils.py
#   - vendor/.../utils/postprocessing_utils.py

from typing import *
import sys
import numpy as np
import torch
import comfy.model_management
import utils3d
from PIL import Image
from tqdm import tqdm
import trimesh
import trimesh.visual
import xatlas
import pyvista as pv
from pymeshfix import _meshfix
import igraph
from loguru import logger

from .image_ops import inpaint
from .renderers import GaussianRenderer, OctreeRenderer
from .representations import Octree, Gaussian, MeshExtractResult, Strivec


# ============================================================================
# random_utils.py
# ============================================================================

PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]


def radical_inverse(base, n):
    val = 0
    inv_base = 1.0 / base
    inv_base_n = inv_base
    while n > 0:
        digit = n % base
        val += digit * inv_base_n
        n //= base
        inv_base_n *= inv_base
    return val


def halton_sequence(dim, n):
    return [radical_inverse(PRIMES[dim], n) for dim in range(dim)]


def hammersley_sequence(dim, n, num_samples):
    return [n / num_samples] + halton_sequence(dim - 1, n)


def sphere_hammersley_sequence(n, num_samples, offset=(0, 0), remap=False):
    u, v = hammersley_sequence(2, n, num_samples)
    u += offset[0] / num_samples
    v += offset[1]
    if remap:
        u = 2 * u if u < 0.25 else 2 / 3 * u + 1 / 3
    theta = np.arccos(1 - 2 * u) - np.pi / 2
    phi = v * 2 * np.pi
    return [phi, theta]


# ============================================================================
# render_utils.py
# ============================================================================

def yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, rs, fovs):
    is_list = isinstance(yaws, list)
    if not is_list:
        yaws = [yaws]
        pitchs = [pitchs]
    if not isinstance(rs, list):
        rs = [rs] * len(yaws)
    if not isinstance(fovs, list):
        fovs = [fovs] * len(yaws)
    device = comfy.model_management.get_torch_device()
    extrinsics = []
    intrinsics = []
    for yaw, pitch, r, fov in zip(yaws, pitchs, rs, fovs):
        fov = torch.deg2rad(torch.tensor(float(fov))).to(device)
        yaw = torch.tensor(float(yaw)).to(device)
        pitch = torch.tensor(float(pitch)).to(device)
        orig = (
            torch.tensor(
                [
                    torch.sin(yaw) * torch.cos(pitch),
                    torch.cos(yaw) * torch.cos(pitch),
                    torch.sin(pitch),
                ]
            ).to(device)
            * r
        )
        extr = utils3d.torch.extrinsics_look_at(
            orig,
            torch.tensor([0, 0, 0]).float().to(device),
            torch.tensor([0, 0, 1]).float().to(device),
        )
        intr = utils3d.torch.intrinsics_from_fov(fov_x=fov, fov_y=fov)
        extrinsics.append(extr)
        intrinsics.append(intr)
    if not is_list:
        extrinsics = extrinsics[0]
        intrinsics = intrinsics[0]
    return extrinsics, intrinsics


def render_frames(
    sample,
    extrinsics,
    intrinsics,
    options={},
    colors_overwrite=None,
    verbose=True,
    **kwargs,
):
    if isinstance(sample, Octree):
        renderer = OctreeRenderer()
        renderer.rendering_options.resolution = options.get("resolution", 512)
        renderer.rendering_options.near = options.get("near", 0.8)
        renderer.rendering_options.far = options.get("far", 1.6)
        renderer.rendering_options.bg_color = options.get("bg_color", (0, 0, 0))
        renderer.rendering_options.ssaa = options.get("ssaa", 4)
        renderer.pipe.primitive = sample.primitive
    elif isinstance(sample, Gaussian):
        renderer = GaussianRenderer()
        renderer.rendering_options.resolution = options.get("resolution", 512)
        renderer.rendering_options.near = options.get("near", 0.8)
        renderer.rendering_options.far = options.get("far", 1.6)
        renderer.rendering_options.bg_color = options.get("bg_color", (0, 0, 0))
        renderer.rendering_options.ssaa = options.get("ssaa", 1)
        renderer.rendering_options.backend = options.get("backend", "gsplat")  # Changed from "inria"
        renderer.pipe.kernel_size = kwargs.get("kernel_size", 0.1)
        renderer.pipe.use_mip_gaussian = True
    elif isinstance(sample, MeshExtractResult):
        renderer = MeshRenderer()
        renderer.rendering_options.resolution = options.get("resolution", 512)
        renderer.rendering_options.near = options.get("near", 1)
        renderer.rendering_options.far = options.get("far", 100)
        renderer.rendering_options.ssaa = options.get("ssaa", 4)
    else:
        raise ValueError(f"Unsupported sample type: {type(sample)}")

    rets = {}
    for j, (extr, intr) in tqdm(
        enumerate(zip(extrinsics, intrinsics)),
        total=len(extrinsics),
        desc="Rendering views",
        disable=not verbose,
        file=sys.stderr,
        ncols=80,
        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}'
    ):
        if not isinstance(sample, MeshExtractResult):
            res = renderer.render(sample, extr, intr, colors_overwrite=colors_overwrite)
            if "color" not in rets:
                rets["color"] = []
            if "depth" not in rets:
                rets["depth"] = []
            rets["color"].append(
                np.clip(
                    res["color"].detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255
                ).astype(np.uint8)
            )
            if "percent_depth" in res:
                rets["depth"].append(res["percent_depth"].detach().cpu().numpy())
            elif "depth" in res:
                rets["depth"].append(res["depth"].detach().cpu().numpy())
            else:
                rets["depth"].append(None)
        else:
            res = renderer.render(sample, extr, intr)
            if "normal" not in rets:
                rets["normal"] = []
            rets["normal"].append(
                np.clip(
                    res["normal"].detach().cpu().numpy().transpose(1, 2, 0) * 255,
                    0,
                    255,
                ).astype(np.uint8)
            )
    return rets


def render_gaussian_color_stay_in_device(
    sample,
    extrinsics,
    intrinsics,
    options={},
    colors_overwrite=None,
    verbose=True,
    **kwargs,
):
    assert isinstance(sample, Gaussian)
    renderer = GaussianRenderer()
    renderer.rendering_options.resolution = options.get("resolution", 512)
    renderer.rendering_options.near = options.get("near", 0.8)
    renderer.rendering_options.far = options.get("far", 1.6)
    renderer.rendering_options.bg_color = options.get("bg_color", (0, 0, 0))
    renderer.rendering_options.ssaa = options.get("ssaa", 1)
    renderer.rendering_options.backend = options.get("backend", "gsplat")  # Changed from "inria"
    renderer.pipe.kernel_size = kwargs.get("kernel_size", 0.1)
    renderer.pipe.use_mip_gaussian = True

    rets = {}
    for _, (extr, intr) in tqdm(
        enumerate(zip(extrinsics, intrinsics)), desc="Rendering", disable=not verbose
    ):
        res = renderer.render(sample, extr, intr, colors_overwrite=colors_overwrite)
        color = (res["color"].permute(1, 2, 0) * 255).to(torch.uint8)
        if "color" not in rets:
            rets["color"] = []
        rets["color"].append(color)
    return rets

def render_video(
    sample,
    resolution=512,
    bg_color=(0, 0, 0),
    num_frames=300,
    r=2,
    fov=40,
    backend="gsplat",  # Changed from "inria"
    **kwargs,
):
    yaws = torch.linspace(0, 2 * 3.1415, num_frames)
    pitch = 0.25 + 0.5 * torch.sin(torch.linspace(0, 2 * 3.1415, num_frames))
    yaws = yaws.tolist()
    pitch = pitch.tolist()
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(
        yaws, pitch, r, fov
    )
    return render_frames(
        sample,
        extrinsics,
        intrinsics,
        {"resolution": resolution, "bg_color": bg_color, "backend": backend},
        **kwargs,
    )


def render_multiview(sample, resolution=512, nviews=30, verbose=True):
    r = 2
    fov = 40
    cams = [sphere_hammersley_sequence(i, nviews) for i in range(nviews)]
    yaws = [cam[0] for cam in cams]
    pitchs = [cam[1] for cam in cams]
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(
        yaws, pitchs, r, fov
    )
    res = render_frames(
        sample,
        extrinsics,
        intrinsics,
        {"resolution": resolution, "bg_color": (0, 0, 0), "backend": "gsplat"},
        verbose=verbose,
    )
    return res["color"], extrinsics, intrinsics


def render_snapshot(
    samples,
    resolution=512,
    bg_color=(0, 0, 0),
    offset=(-16 / 180 * np.pi, 20 / 180 * np.pi),
    r=10,
    fov=8,
    **kwargs,
):
    yaw = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
    yaw_offset = offset[0]
    yaw = [y + yaw_offset for y in yaw]
    pitch = [offset[1] for _ in range(4)]
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(
        yaw, pitch, r, fov
    )
    return render_frames(
        samples,
        extrinsics,
        intrinsics,
        {"resolution": resolution, "bg_color": bg_color},
        **kwargs,
    )


# ============================================================================
# postprocessing_utils.py
# ============================================================================

@torch.no_grad()
def _fill_holes(
    verts,
    faces,
    max_hole_size=0.04,
    max_hole_nbe=32,
    resolution=128,
    num_views=500,
    debug=False,
    verbose=False,
):
    """
    Rasterize a mesh from multiple views and remove invisible faces.
    Also includes postprocessing to:
        1. Remove connected components that are have low visibility.
        2. Mincut to remove faces at the inner side of the mesh connected to the outer side with a small hole.

    Args:
        verts (torch.Tensor): Vertices of the mesh. Shape (V, 3).
        faces (torch.Tensor): Faces of the mesh. Shape (F, 3).
        max_hole_size (float): Maximum area of a hole to fill.
        resolution (int): Resolution of the rasterization.
        num_views (int): Number of views to rasterize the mesh.
        verbose (bool): Whether to print progress.
    """
    # Construct cameras
    yaws = []
    pitchs = []
    for i in range(num_views):
        y, p = sphere_hammersley_sequence(i, num_views)
        yaws.append(y)
        pitchs.append(p)
    device = comfy.model_management.get_torch_device()
    yaws = torch.tensor(yaws).to(device)
    pitchs = torch.tensor(pitchs).to(device)
    radius = 2.0
    fov = torch.deg2rad(torch.tensor(40)).to(device)
    projection = utils3d.torch.perspective_from_fov_xy(fov, fov, 1, 3)
    views = []
    for yaw, pitch in zip(yaws, pitchs):
        orig = (
            torch.tensor(
                [
                    torch.sin(yaw) * torch.cos(pitch),
                    torch.cos(yaw) * torch.cos(pitch),
                    torch.sin(pitch),
                ]
            )
            .to(device)
            .float()
            * radius
        )
        view = utils3d.torch.view_look_at(
            orig,
            torch.tensor([0, 0, 0]).float().to(device),
            torch.tensor([0, 0, 1]).float().to(device),
        )
        views.append(view)
    views = torch.stack(views, dim=0)

    # Rasterize
    visblity = torch.zeros(faces.shape[0], dtype=torch.int32, device=verts.device)
    rastctx = utils3d.torch.RastContext(backend="cuda")
    for i in tqdm(
        range(views.shape[0]),
        total=views.shape[0],
        disable=not verbose,
        desc="Rasterizing",
    ):
        view = views[i]
        buffers = utils3d.torch.rasterize_triangle_faces(
            rastctx,
            verts[None],
            faces,
            resolution,
            resolution,
            view=view,
            projection=projection,
        )
        face_id = buffers["face_id"][0][buffers["mask"][0] > 0.95] - 1
        face_id = torch.unique(face_id).long()
        visblity[face_id] += 1
    visblity = visblity.float() / num_views

    # Mincut
    ## construct outer faces
    edges, face2edge, edge_degrees = utils3d.torch.compute_edges(faces)
    boundary_edge_indices = torch.nonzero(edge_degrees == 1).reshape(-1)
    connected_components = utils3d.torch.compute_connected_components(
        faces, edges, face2edge
    )
    outer_face_indices = torch.zeros(
        faces.shape[0], dtype=torch.bool, device=faces.device
    )
    for i in range(len(connected_components)):
        outer_face_indices[connected_components[i]] = visblity[
            connected_components[i]
        ] > min(max(visblity[connected_components[i]].quantile(0.75).item(), 0.25), 0.5)
    outer_face_indices = outer_face_indices.nonzero().reshape(-1)

    ## construct inner faces
    inner_face_indices = torch.nonzero(visblity == 0).reshape(-1)
    if verbose:
        tqdm.write(f"Found {inner_face_indices.shape[0]} invisible faces")
    if inner_face_indices.shape[0] == 0:
        return verts, faces

    ## Construct dual graph (faces as nodes, edges as edges)
    dual_edges, dual_edge2edge = utils3d.torch.compute_dual_graph(face2edge)
    dual_edge2edge = edges[dual_edge2edge]
    dual_edges_weights = torch.norm(
        verts[dual_edge2edge[:, 0]] - verts[dual_edge2edge[:, 1]], dim=1
    )
    if verbose:
        tqdm.write(f"Dual graph: {dual_edges.shape[0]} edges")

    ## solve mincut problem
    ### construct main graph
    g = igraph.Graph()
    g.add_vertices(faces.shape[0])
    g.add_edges(dual_edges.cpu().numpy())
    g.es["weight"] = dual_edges_weights.cpu().numpy()

    ### source and target
    g.add_vertex("s")
    g.add_vertex("t")

    ### connect invisible faces to source
    g.add_edges(
        [(f, "s") for f in inner_face_indices],
        attributes={
            "weight": torch.ones(inner_face_indices.shape[0], dtype=torch.float32)
            .cpu()
            .numpy()
        },
    )

    ### connect outer faces to target
    g.add_edges(
        [(f, "t") for f in outer_face_indices],
        attributes={
            "weight": torch.ones(outer_face_indices.shape[0], dtype=torch.float32)
            .cpu()
            .numpy()
        },
    )

    ### solve mincut
    cut = g.mincut("s", "t", (np.array(g.es["weight"]) * 1000).tolist())
    remove_face_indices = torch.tensor(
        [v for v in cut.partition[0] if v < faces.shape[0]],
        dtype=torch.long,
        device=faces.device,
    )
    if verbose:
        tqdm.write(f"Mincut solved, start checking the cut")

    ### check if the cut is valid with each connected component
    to_remove_cc = utils3d.torch.compute_connected_components(
        faces[remove_face_indices]
    )
    if debug:
        tqdm.write(f"Number of connected components of the cut: {len(to_remove_cc)}")
    valid_remove_cc = []
    cutting_edges = []
    for cc in to_remove_cc:
        #### check if the connected component has low visibility
        visblity_median = visblity[remove_face_indices[cc]].median()
        if debug:
            tqdm.write(f"visblity_median: {visblity_median}")
        if visblity_median > 0.25:
            continue

        #### check if the cuting loop is small enough
        cc_edge_indices, cc_edges_degree = torch.unique(
            face2edge[remove_face_indices[cc]], return_counts=True
        )
        cc_boundary_edge_indices = cc_edge_indices[cc_edges_degree == 1]
        cc_new_boundary_edge_indices = cc_boundary_edge_indices[
            ~torch.isin(cc_boundary_edge_indices, boundary_edge_indices)
        ]
        if len(cc_new_boundary_edge_indices) > 0:
            cc_new_boundary_edge_cc = utils3d.torch.compute_edge_connected_components(
                edges[cc_new_boundary_edge_indices]
            )
            cc_new_boundary_edges_cc_center = [
                verts[edges[cc_new_boundary_edge_indices[edge_cc]]]
                .mean(dim=1)
                .mean(dim=0)
                for edge_cc in cc_new_boundary_edge_cc
            ]
            cc_new_boundary_edges_cc_area = []
            for i, edge_cc in enumerate(cc_new_boundary_edge_cc):
                _e1 = (
                    verts[edges[cc_new_boundary_edge_indices[edge_cc]][:, 0]]
                    - cc_new_boundary_edges_cc_center[i]
                )
                _e2 = (
                    verts[edges[cc_new_boundary_edge_indices[edge_cc]][:, 1]]
                    - cc_new_boundary_edges_cc_center[i]
                )
                cc_new_boundary_edges_cc_area.append(
                    torch.norm(torch.cross(_e1, _e2, dim=-1), dim=1).sum() * 0.5
                )
            if debug:
                cutting_edges.append(cc_new_boundary_edge_indices)
                tqdm.write(f"Area of the cutting loop: {cc_new_boundary_edges_cc_area}")
            if any([l > max_hole_size for l in cc_new_boundary_edges_cc_area]):
                continue

        valid_remove_cc.append(cc)

    if debug:
        face_v = verts[faces].mean(dim=1).cpu().numpy()
        vis_dual_edges = dual_edges.cpu().numpy()
        vis_colors = np.zeros((faces.shape[0], 3), dtype=np.uint8)
        vis_colors[inner_face_indices.cpu().numpy()] = [0, 0, 255]
        vis_colors[outer_face_indices.cpu().numpy()] = [0, 255, 0]
        vis_colors[remove_face_indices.cpu().numpy()] = [255, 0, 255]
        if len(valid_remove_cc) > 0:
            vis_colors[
                remove_face_indices[torch.cat(valid_remove_cc)].cpu().numpy()
            ] = [255, 0, 0]
        utils3d.io.write_ply(
            "dbg_dual.ply", face_v, edges=vis_dual_edges, vertex_colors=vis_colors
        )

        vis_verts = verts.cpu().numpy()
        vis_edges = edges[torch.cat(cutting_edges)].cpu().numpy()
        utils3d.io.write_ply("dbg_cut.ply", vis_verts, edges=vis_edges)

    if len(valid_remove_cc) > 0:
        remove_face_indices = remove_face_indices[torch.cat(valid_remove_cc)]
        mask = torch.ones(faces.shape[0], dtype=torch.bool, device=faces.device)
        mask[remove_face_indices] = 0
        faces = faces[mask]
        faces, verts = utils3d.torch.remove_unreferenced_vertices(faces, verts)
        if verbose:
            tqdm.write(f"Removed {(~mask).sum()} faces by mincut")
    else:
        if verbose:
            tqdm.write(f"Removed 0 faces by mincut")

    mesh = _meshfix.PyTMesh()
    mesh.load_array(verts.cpu().numpy(), faces.cpu().numpy())
    mesh.fill_small_boundaries(nbe=max_hole_nbe, refine=True)
    verts, faces = mesh.return_arrays()
    verts, faces = torch.tensor(
        verts, device="cuda", dtype=torch.float32
    ), torch.tensor(faces, device="cuda", dtype=torch.int32)

    return verts, faces


def postprocess_mesh(
    vertices: np.array,
    faces: np.array,
    simplify: bool = True,
    simplify_ratio: float = 0.9,
    fill_holes: bool = True,
    fill_holes_max_hole_size: float = 0.04,
    fill_holes_max_hole_nbe: int = 32,
    fill_holes_resolution: int = 1024,
    fill_holes_num_views: int = 1000,
    debug: bool = False,
    verbose: bool = False,
):
    """
    Postprocess a mesh by simplifying, removing invisible faces, and removing isolated pieces.

    Args:
        vertices (np.array): Vertices of the mesh. Shape (V, 3).
        faces (np.array): Faces of the mesh. Shape (F, 3).
        simplify (bool): Whether to simplify the mesh, using quadric edge collapse.
        simplify_ratio (float): Ratio of faces to keep after simplification.
        fill_holes (bool): Whether to fill holes in the mesh.
        fill_holes_max_hole_size (float): Maximum area of a hole to fill.
        fill_holes_max_hole_nbe (int): Maximum number of boundary edges of a hole to fill.
        fill_holes_resolution (int): Resolution of the rasterization.
        fill_holes_num_views (int): Number of views to rasterize the mesh.
        verbose (bool): Whether to print progress.
    """

    if verbose:
        tqdm.write(
            f"Before postprocess: {vertices.shape[0]} vertices, {faces.shape[0]} faces"
        )

    # Simplify
    if simplify and simplify_ratio > 0:
        mesh = pv.PolyData(
            vertices, np.concatenate([np.full((faces.shape[0], 1), 3), faces], axis=1)
        )
        mesh = mesh.decimate(simplify_ratio, progress_bar=verbose)
        vertices, faces = mesh.points, mesh.faces.reshape(-1, 4)[:, 1:]
        if verbose:
            tqdm.write(
                f"After decimate: {vertices.shape[0]} vertices, {faces.shape[0]} faces"
            )

    # Remove invisible faces
    if fill_holes:
        device = comfy.model_management.get_torch_device()
        vertices, faces = (
            torch.tensor(vertices).to(device),
            torch.tensor(faces.astype(np.int32)).to(device),
        )
        vertices, faces = _fill_holes(
            vertices,
            faces,
            max_hole_size=fill_holes_max_hole_size,
            max_hole_nbe=fill_holes_max_hole_nbe,
            resolution=fill_holes_resolution,
            num_views=fill_holes_num_views,
            debug=debug,
            verbose=verbose,
        )
        vertices, faces = vertices.cpu().numpy(), faces.cpu().numpy()
        if verbose:
            tqdm.write(
                f"After remove invisible faces: {vertices.shape[0]} vertices, {faces.shape[0]} faces"
            )

    return vertices, faces


def parametrize_mesh(vertices: np.array, faces: np.array):
    """
    Parametrize a mesh to a texture space, using xatlas.

    Args:
        vertices (np.array): Vertices of the mesh. Shape (V, 3).
        faces (np.array): Faces of the mesh. Shape (F, 3).
    """

    vmapping, indices, uvs = xatlas.parametrize(vertices, faces)

    vertices = vertices[vmapping]
    faces = indices

    return vertices, faces, uvs

@torch.inference_mode(False)
@torch.enable_grad()
def bake_texture(
    vertices: np.array,
    faces: np.array,
    uvs: np.array,
    observations: List[np.array],
    masks: List[np.array],
    extrinsics: List[np.array],
    intrinsics: List[np.array],
    texture_size: int = 2048,
    near: float = 0.1,
    far: float = 10.0,
    mode: Literal["fast", "opt"] = "opt",
    lambda_tv: float = 1e-2,
    verbose: bool = False,
    rendering_engine: str = "nvdiffrast",  # nvdiffrast OR "pytorch3d"
    device: str = "cuda",

):
    """
    Bake texture to a mesh from multiple observations.

    Args:
        vertices (np.array): Vertices of the mesh. Shape (V, 3).
        faces (np.array): Faces of the mesh. Shape (F, 3).
        uvs (np.array): UV coordinates of the mesh. Shape (V, 2).
        observations (List[np.array]): List of observations. Each observation is a 2D image. Shape (H, W, 3).
        masks (List[np.array]): List of masks. Each mask is a 2D image. Shape (H, W).
        extrinsics (List[np.array]): List of extrinsics. Shape (4, 4).
        intrinsics (List[np.array]): List of intrinsics. Shape (3, 3).
        texture_size (int): Size of the texture.
        near (float): Near plane of the camera.
        far (float): Far plane of the camera.
        mode (Literal['fast', 'opt']): Mode of texture baking.
        lambda_tv (float): Weight of total variation loss in optimization.
        verbose (bool): Whether to print progress.
    """


    vertices = torch.tensor(vertices).to(device)
    faces = torch.tensor(faces.astype(np.int32)).to(device)
    uvs = torch.tensor(uvs).to(device)
    # Keep observations and masks as numpy arrays - load to GPU on-demand to save ~1.2GB VRAM
    # observations = [torch.tensor(obs / 255.0).float().to(device) for obs in observations]
    # masks = [torch.tensor(m > 0).bool().to(device) for m in masks]
    # Note: utils3d functions create internal CPU tensors, so we compute on CPU then move to GPU
    views = [
        utils3d.torch.extrinsics_to_view(torch.tensor(extr)).to(device)
        for extr in extrinsics
    ]
    projections = [
        utils3d.torch.intrinsics_to_perspective(torch.tensor(intr), near, far).to(device)
        for intr in intrinsics
    ]

    if mode == "fast":
        texture = torch.zeros(
            (texture_size * texture_size, 3), dtype=torch.float32
        ).to(device)
        texture_weights = torch.zeros(
            (texture_size * texture_size), dtype=torch.float32
        ).to(device)
        rastctx = utils3d.torch.RastContext(backend=device if device.startswith("cuda") else "cuda")
        total_views = len(views)

        # Use tqdm with file=sys.stderr for subprocess compatibility
        pbar = tqdm(
            enumerate(zip(views, projections)),
            total=total_views,
            disable=not verbose,
            desc="Texture baking (fast)",
            file=sys.stderr,
            ncols=80,
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}'
        )
        for i, (view, projection) in pbar:
            # Load observation and mask on-demand (saves ~1.2GB VRAM)
            observation = torch.tensor(observations[i] / 255.0).float().to(device)
            obs_mask = torch.tensor(masks[i] > 0).bool().to(device)

            with torch.no_grad():
                rast = utils3d.torch.rasterize_triangle_faces(
                    rastctx,
                    vertices[None],
                    faces,
                    observation.shape[1],
                    observation.shape[0],
                    uv=uvs[None],
                    view=view,
                    projection=projection,
                )
                uv_map = rast["uv"][0].detach().flip(0)
                mask = rast["mask"][0].detach().bool() & obs_mask  # Fixed: use correct mask for each view

            # nearest neighbor interpolation
            uv_map = (uv_map * texture_size).floor().long()
            obs = observation[mask]
            uv_map = uv_map[mask]
            idx = uv_map[:, 0] + (texture_size - uv_map[:, 1] - 1) * texture_size
            texture = texture.scatter_add(0, idx.view(-1, 1).expand(-1, 3), obs)
            texture_weights = texture_weights.scatter_add(
                0,
                idx,
                torch.ones((obs.shape[0]), dtype=torch.float32, device=texture.device),
            )

            # Free memory periodically
            del observation, obs_mask, rast, uv_map, obs, mask
            if i % 20 == 0:
                torch.cuda.empty_cache()

        mask = texture_weights > 0
        texture[mask] /= texture_weights[mask][:, None]
        texture = np.clip(
            texture.reshape(texture_size, texture_size, 3).cpu().numpy() * 255, 0, 255
        ).astype(np.uint8)

        # inpaint
        mask = (
            (texture_weights == 0)
            .cpu()
            .numpy()
            .astype(np.uint8)
            .reshape(texture_size, texture_size)
        )
        texture = inpaint(texture, mask, inpaint_radius=3)

    elif mode == "opt":
        rastctx = utils3d.torch.RastContext(backend=device if device.startswith("cuda") else "cuda")
        # For "opt" mode, we need all observations on GPU for 2500 optimization steps
        # Convert to tensors and flip (observations are numpy arrays at this point)
        observations_gpu = [torch.tensor(obs / 255.0).float().to(device).flip(0) for obs in observations]
        masks_gpu = [torch.tensor(m > 0).bool().to(device).flip(0) for m in masks]
        _uv = []
        _uv_dr = []
        total_views = len(views)

        # Use tqdm with file=sys.stderr for subprocess compatibility
        uv_pbar = tqdm(
            enumerate(zip(observations_gpu, views, projections)),
            total=total_views,
            disable=not verbose,
            desc="Preparing UV maps",
            file=sys.stderr,
            ncols=80,
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}'
        )
        for i, (observation, view, projection) in uv_pbar:
            with torch.no_grad():
                rast = utils3d.torch.rasterize_triangle_faces(
                    rastctx,
                    vertices[None],
                    faces,
                    observation.shape[1],
                    observation.shape[0],
                    uv=uvs[None],
                    view=view,
                    projection=projection,
                )
                _uv.append(rast["uv"].detach())
                _uv_dr.append(rast["uv_dr"].detach())

        texture = torch.nn.Parameter(
            torch.zeros((1, texture_size, texture_size, 3), dtype=torch.float32).to(device)
        )
        optimizer = torch.optim.Adam([texture], betas=(0.5, 0.9), lr=1e-2)

        def exp_anealing(optimizer, step, total_steps, start_lr, end_lr):
            return start_lr * (end_lr / start_lr) ** (step / total_steps)

        def cosine_anealing(optimizer, step, total_steps, start_lr, end_lr):
            return end_lr + 0.5 * (start_lr - end_lr) * (
                1 + np.cos(np.pi * step / total_steps)
            )

        def tv_loss(texture):
            return torch.nn.functional.l1_loss(
                texture[:, :-1, :, :], texture[:, 1:, :, :]
            ) + torch.nn.functional.l1_loss(texture[:, :, :-1, :], texture[:, :, 1:, :])



        def render_pt3d_texture(texture, uv, uv_dr=None):
            import torch.nn.functional as F
            texture_perm = texture.permute(0, 3, 1, 2)
            grid = uv * 2 - 1
            if grid.dim() == 3:
                grid = grid.unsqueeze(0)  # (1, H, W, 2)
            elif grid.dim() == 4 and grid.shape[0] == 1:
                pass
            elif grid.dim() == 4 and grid.shape[1] == 1:
                grid = grid.squeeze(1)  # remove extra batch dimension if necessary
            else:
                raise ValueError(f"Unexpected grid shape: {grid.shape}")
            render = F.grid_sample(
                texture_perm, grid, mode='bilinear', padding_mode='border', align_corners=True
            )
            render = render.permute(0, 2, 3, 1)[0]  # (H_out, W_out, 3)
            return render


        total_steps = 2500

        # Use tqdm with file=sys.stderr for subprocess compatibility
        pbar = tqdm(
            range(total_steps),
            disable=not verbose,
            desc="Texture baking (opt)",
            file=sys.stderr,
            ncols=80,
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{postfix}]'
        )

        for step in pbar:
            optimizer.zero_grad()
            selected = np.random.randint(0, len(views))
            uv, uv_dr, observation, mask = (
                _uv[selected],
                _uv_dr[selected],
                observations_gpu[selected],
                masks_gpu[selected],
            )

            if rendering_engine == "nvdiffrast":
                import nvdiffrast.torch as dr
                render = dr.texture(texture, uv, uv_dr)[0]

            if rendering_engine == "pytorch3d":
                render = render_pt3d_texture(texture, uv)

            loss = torch.nn.functional.l1_loss(render[mask], observation[mask])
            if lambda_tv > 0:
                loss += lambda_tv * tv_loss(texture)
            loss.backward()
            optimizer.step()
            # annealing
            optimizer.param_groups[0]["lr"] = cosine_anealing(
                optimizer, step, total_steps, 1e-2, 1e-5
                )

            # Update progress bar with loss
            if step % 100 == 0:
                pbar.set_postfix_str(f"loss={loss.item():.4f}")

        # Free GPU memory before final texture processing
        del observations_gpu, masks_gpu, _uv, _uv_dr
        torch.cuda.empty_cache()

        texture = np.clip(
            texture[0].flip(0).detach().cpu().numpy() * 255, 0, 255
        ).astype(np.uint8)
        mask = 1 - utils3d.torch.rasterize_triangle_faces(
            rastctx, (uvs * 2 - 1)[None], faces, texture_size, texture_size
        )["mask"][0].detach().cpu().numpy().astype(np.uint8)
        texture = inpaint(texture, mask, inpaint_radius=3)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return texture


def to_glb(
    app_rep: Union[Strivec, Gaussian],
    mesh: MeshExtractResult,
    simplify: float = 0.95,
    fill_holes: bool = True,
    fill_holes_max_size: float = 0.04,
    texture_size: int = 1024,
    debug: bool = False,
    verbose: bool = True,
    with_mesh_postprocess=True,
    with_texture_baking=True,
    use_vertex_color=False,
    rendering_engine: str = "nvdiffrast",  # nvdiffrast OR "pytorch3d"
    texture_mode: str = "opt",  # "fast" (5s, nearest neighbor) or "opt" (30-60s, gradient descent)
) -> trimesh.Trimesh:
    """
    Convert a generated asset to a glb file.

    Args:
        app_rep (Union[Strivec, Gaussian]): Appearance representation.
        mesh (MeshExtractResult): Extracted mesh.
        simplify (float): Ratio of faces to remove in simplification.
        fill_holes (bool): Whether to fill holes in the mesh.
        fill_holes_max_size (float): Maximum area of a hole to fill.
        texture_size (int): Size of the texture.
        debug (bool): Whether to print debug information.
        verbose (bool): Whether to print progress.
    """
    vertices = mesh.vertices.float().cpu().numpy()
    faces = mesh.faces.cpu().numpy()
    vert_colors = mesh.vertex_attrs[:, :3].cpu().numpy()

    if with_mesh_postprocess:
        # mesh postprocess
        vertices, faces = postprocess_mesh(
            vertices,
            faces,
            simplify=simplify > 0,
            simplify_ratio=simplify,
            fill_holes=fill_holes,
            fill_holes_max_hole_size=fill_holes_max_size,
            fill_holes_max_hole_nbe=int(250 * np.sqrt(1 - simplify)),
            fill_holes_resolution=1024,
            fill_holes_num_views=1000,
            debug=debug,
            verbose=verbose,
        )

    if with_texture_baking:
        # parametrize mesh
        vertices, faces, uvs = parametrize_mesh(vertices, faces)
        logger.info("Baking texture ...")

        # bake texture - render 100 views from Gaussian
        observations, extrinsics, intrinsics = render_multiview(
            app_rep, resolution=1024, nviews=100, verbose=verbose
        )
        masks = [np.any(observation > 0, axis=-1) for observation in observations]
        extrinsics = [extrinsics[i].cpu().numpy() for i in range(len(extrinsics))]
        intrinsics = [intrinsics[i].cpu().numpy() for i in range(len(intrinsics))]

        # Free GPU memory before texture baking (Gaussian no longer needed)
        # Note: We can't delete app_rep since it's owned by caller, but we can clear cache
        torch.cuda.empty_cache()

        texture = bake_texture(
            vertices,
            faces,
            uvs,
            observations,
            masks,
            extrinsics,
            intrinsics,
            texture_size=texture_size,
            mode=texture_mode,
            lambda_tv=0.01,
            verbose=verbose,
            rendering_engine=rendering_engine
        )

        # Free memory after texture baking
        del observations, masks
        torch.cuda.empty_cache()

        texture = Image.fromarray(texture)
        material = trimesh.visual.material.PBRMaterial(
            roughnessFactor=1.0,
            baseColorTexture=texture,
            baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        )

    # rotate mesh (from z-up to y-up)
    vertices = vertices @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

    if not with_mesh_postprocess and not with_texture_baking and use_vertex_color:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh.visual.vertex_colors = vert_colors
    else:
        mesh = trimesh.Trimesh(
            vertices,
            faces,
            visual=(
                trimesh.visual.TextureVisuals(uv=uvs, material=material)
                if with_texture_baking
                else None
            ),
        )

    return mesh


def simplify_gs(
    gs: Gaussian,
    simplify: float = 0.95,
    verbose: bool = True,
):
    """
    Simplify 3D Gaussians
    NOTE: this function is not used in the current implementation for the unsatisfactory performance.

    Args:
        gs (Gaussian): 3D Gaussian.
        simplify (float): Ratio of Gaussians to remove in simplification.
    """
    if simplify <= 0:
        return gs

    # simplify
    observations, extrinsics, intrinsics = render_multiview(
        gs, resolution=1024, nviews=100, verbose=verbose
    )
    device = comfy.model_management.get_torch_device()
    observations = [
        torch.tensor(obs / 255.0).float().to(device).permute(2, 0, 1)
        for obs in observations
    ]

    # Following https://arxiv.org/pdf/2411.06019
    renderer = GaussianRenderer(
        {
            "resolution": 1024,
            "near": 0.8,
            "far": 1.6,
            "ssaa": 1,
            "bg_color": (0, 0, 0),
        }
    )
    new_gs = Gaussian(**gs.init_params)
    new_gs._features_dc = gs._features_dc.clone()
    new_gs._features_rest = (
        gs._features_rest.clone() if gs._features_rest is not None else None
    )
    new_gs._opacity = torch.nn.Parameter(gs._opacity.clone())
    new_gs._rotation = torch.nn.Parameter(gs._rotation.clone())
    new_gs._scaling = torch.nn.Parameter(gs._scaling.clone())
    new_gs._xyz = torch.nn.Parameter(gs._xyz.clone())

    start_lr = [1e-4, 1e-3, 5e-3, 0.025]
    end_lr = [1e-6, 1e-5, 5e-5, 0.00025]
    optimizer = torch.optim.Adam(
        [
            {"params": new_gs._xyz, "lr": start_lr[0]},
            {"params": new_gs._rotation, "lr": start_lr[1]},
            {"params": new_gs._scaling, "lr": start_lr[2]},
            {"params": new_gs._opacity, "lr": start_lr[3]},
        ],
        lr=start_lr[0],
    )

    def exp_anealing(optimizer, step, total_steps, start_lr, end_lr):
        return start_lr * (end_lr / start_lr) ** (step / total_steps)

    def cosine_anealing(optimizer, step, total_steps, start_lr, end_lr):
        return end_lr + 0.5 * (start_lr - end_lr) * (
            1 + np.cos(np.pi * step / total_steps)
        )

    _zeta = new_gs.get_opacity.clone().detach().squeeze()
    _lambda = torch.zeros_like(_zeta)
    _delta = 1e-7
    _interval = 10
    num_target = int((1 - simplify) * _zeta.shape[0])

    with tqdm(total=2500, disable=not verbose, desc="Simplifying Gaussian") as pbar:
        for i in range(2500):
            # prune
            if i % 100 == 0:
                mask = new_gs.get_opacity.squeeze() > 0.05
                mask = torch.nonzero(mask).squeeze()
                new_gs._xyz = torch.nn.Parameter(new_gs._xyz[mask])
                new_gs._rotation = torch.nn.Parameter(new_gs._rotation[mask])
                new_gs._scaling = torch.nn.Parameter(new_gs._scaling[mask])
                new_gs._opacity = torch.nn.Parameter(new_gs._opacity[mask])
                new_gs._features_dc = new_gs._features_dc[mask]
                new_gs._features_rest = (
                    new_gs._features_rest[mask]
                    if new_gs._features_rest is not None
                    else None
                )
                _zeta = _zeta[mask]
                _lambda = _lambda[mask]
                # update optimizer state
                for param_group, new_param in zip(
                    optimizer.param_groups,
                    [new_gs._xyz, new_gs._rotation, new_gs._scaling, new_gs._opacity],
                ):
                    stored_state = optimizer.state[param_group["params"][0]]
                    if "exp_avg" in stored_state:
                        stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                        stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                    del optimizer.state[param_group["params"][0]]
                    param_group["params"][0] = new_param
                    optimizer.state[param_group["params"][0]] = stored_state

            opacity = new_gs.get_opacity.squeeze()

            # sparisfy
            if i % _interval == 0:
                _zeta = _lambda + opacity.detach()
                if opacity.shape[0] > num_target:
                    index = _zeta.topk(num_target)[1]
                    _m = torch.ones_like(_zeta, dtype=torch.bool)
                    _m[index] = 0
                    _zeta[_m] = 0
                _lambda = _lambda + opacity.detach() - _zeta

            # sample a random view
            view_idx = np.random.randint(len(observations))
            observation = observations[view_idx]
            extrinsic = extrinsics[view_idx]
            intrinsic = intrinsics[view_idx]

            color = renderer.render(new_gs, extrinsic, intrinsic)["color"]
            rgb_loss = torch.nn.functional.l1_loss(color, observation)
            loss = rgb_loss + _delta * torch.sum(
                torch.pow(_lambda + opacity - _zeta, 2)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update lr
            for j in range(len(optimizer.param_groups)):
                optimizer.param_groups[j]["lr"] = cosine_anealing(
                    optimizer, i, 2500, start_lr[j], end_lr[j]
                )

            pbar.set_postfix(
                {
                    "loss": rgb_loss.item(),
                    "num": opacity.shape[0],
                    "lambda": _lambda.mean().item(),
                }
            )
            pbar.update()

    new_gs._xyz = new_gs._xyz.data
    new_gs._rotation = new_gs._rotation.data
    new_gs._scaling = new_gs._scaling.data
    new_gs._opacity = new_gs._opacity.data

    return new_gs
