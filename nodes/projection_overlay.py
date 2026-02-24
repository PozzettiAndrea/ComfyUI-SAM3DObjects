"""SAM3D_ProjectionOverlay node - visualize projected meshes over original image."""

import logging
import os
import re
import time
from typing import Tuple

log = logging.getLogger("sam3dobjects")

class SAM3D_ProjectionOverlay:
    """
    Projection Overlay - Visualize pose-optimized meshes projected onto the original image.

    Takes the output folder from SAM3D_ScenePoseOptimize and creates an overlay
    visualization showing how the 3D meshes project onto the 2D image.

    Useful for debugging and verifying pose optimization quality.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_opt_folder": ("STRING", {
                    "default": "",
                    "tooltip": "Path to pose-optimized folder (output from SAM3D_ScenePoseOptimize)"
                }),
                "render_mode": (["3d_mesh_textured", "3d_mesh", "mask_colors"], {
                    "default": "3d_mesh_textured",
                    "tooltip": "Rendering mode: mask_colors (colored points), 3d_mesh (solid mesh), 3d_mesh_textured (mesh with textures)"
                }),
                "point_size": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 10,
                    "tooltip": "Size of projected points in pixels (only for mask_colors mode)"
                }),
                "alpha": ("FLOAT", {
                    "default": 0.6,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Opacity of projected mesh"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("overlay_image",)
    OUTPUT_TOOLTIPS = ("Overlay image showing projected meshes on original image",)
    FUNCTION = "create_overlay"
    CATEGORY = "SAM3DObjects"
    DESCRIPTION = "Visualize pose-optimized meshes projected onto the original image."

    def create_overlay(
        self,
        pose_opt_folder: str,
        render_mode: str = "mask_colors",
        point_size: int = 2,
        alpha: float = 0.6,
    ) -> Tuple:
        """
        Create overlay visualization of projected meshes.

        Args:
            pose_opt_folder: Path to pose-optimized folder
            render_mode: Rendering mode (mask_colors, 3d_mesh, 3d_mesh_textured)
            point_size: Size of projected points (only for mask_colors)
            alpha: Opacity of mesh overlay

        Returns:
            Tuple containing ComfyUI IMAGE tensor
        """
        import os
        import re
        import time
        import torch
        import comfy.utils
        import numpy as np
        import trimesh
        from PIL import Image

        start_time = time.time()
        log.info("ProjectionOverlay: Creating overlay (mode=%s)", render_mode)

        # Validate folder exists
        if not os.path.exists(pose_opt_folder):
            raise ValueError(f"Pose optimization folder does not exist: {pose_opt_folder}")

        # Derive source folder (remove _pose_optimized suffix)
        if pose_opt_folder.endswith("_pose_optimized"):
            source_folder = pose_opt_folder[:-len("_pose_optimized")]
        else:
            # Try to find source folder
            source_folder = pose_opt_folder
            log.warning("ProjectionOverlay: folder doesn't end with _pose_optimized")

        # Load original image
        image_path = os.path.join(source_folder, "image.png")
        if not os.path.exists(image_path):
            raise ValueError(f"Original image not found: {image_path}")

        orig_img = Image.open(image_path).convert("RGB")
        img_array = np.array(orig_img).astype(np.float32) / 255.0
        H, W = img_array.shape[:2]
        log.info("ProjectionOverlay: Loaded image %dx%d", W, H)

        # Load intrinsics
        intrinsics_path = os.path.join(source_folder, "intrinsics.pt")
        if not os.path.exists(intrinsics_path):
            raise ValueError(f"Intrinsics not found: {intrinsics_path}")

        intrinsics = torch.load(intrinsics_path, weights_only=False)
        intrinsics = np.array(intrinsics)

        # Denormalize intrinsics
        K = intrinsics.copy()
        K[0, 0] *= W  # fx
        K[1, 1] *= H  # fy
        K[0, 2] *= W  # cx
        K[1, 2] *= H  # cy
        log.info("ProjectionOverlay: Intrinsics loaded")

        # Find all mesh files
        mesh_files = []
        pattern = re.compile(r'^object_(\d+)\.glb$')
        for name in os.listdir(pose_opt_folder):
            match = pattern.match(name)
            if match:
                idx = int(match.group(1))
                mesh_files.append((idx, os.path.join(pose_opt_folder, name)))

        mesh_files.sort(key=lambda x: x[0])
        log.info("ProjectionOverlay: Found %d meshes", len(mesh_files))

        if not mesh_files:
            log.warning("ProjectionOverlay: No meshes found, returning original image")
            img_tensor = torch.from_numpy(img_array).unsqueeze(0)
            return (img_tensor,)

        # Generate distinct colors for each object
        colors = [
            [1.0, 0.5, 0.0],  # Orange
            [0.0, 0.5, 1.0],  # Blue
            [0.0, 0.8, 0.2],  # Green
            [0.8, 0.0, 0.2],  # Red
            [0.6, 0.0, 0.8],  # Purple
            [0.6, 0.4, 0.2],  # Brown
            [1.0, 0.0, 0.5],  # Pink
            [0.0, 0.8, 0.8],  # Cyan
            [0.8, 0.8, 0.0],  # Yellow
            [0.5, 0.5, 0.5],  # Gray
        ]

        # Load all meshes first
        loaded_meshes = []
        for idx, mesh_path in mesh_files:
            try:
                mesh = trimesh.load(mesh_path)
                if isinstance(mesh, trimesh.Scene):
                    mesh_list = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
                    if not mesh_list:
                        continue
                    mesh = mesh_list[0] if len(mesh_list) == 1 else trimesh.util.concatenate(mesh_list)
                loaded_meshes.append((idx, mesh))
            except Exception as e:
                log.error("ProjectionOverlay: Error loading object %d: %s", idx, e)
                continue

        log.info("ProjectionOverlay: Loaded %d meshes", len(loaded_meshes))

        # Render based on mode
        if render_mode == "mask_colors":
            overlay = self._render_mask_colors(loaded_meshes, img_array, K, colors, point_size, alpha)
        elif render_mode == "3d_mesh":
            overlay = self._render_3d_mesh(loaded_meshes, img_array, K, colors, alpha, H, W)
        elif render_mode == "3d_mesh_textured":
            overlay = self._render_3d_mesh_textured(loaded_meshes, img_array, K, alpha, H, W)
        else:
            raise ValueError(f"Unknown render_mode: {render_mode}")

        # Convert to ComfyUI IMAGE format (B, H, W, C)
        overlay = np.clip(overlay, 0, 1)
        img_tensor = torch.from_numpy(overlay).float().unsqueeze(0)

        elapsed = time.time() - start_time
        log.info("ProjectionOverlay: Done in %.1fs, output shape %s", elapsed, img_tensor.shape)
        return (img_tensor,)

    def _render_mask_colors(self, loaded_meshes, img_array, K, colors, point_size, alpha):
        """Render colored points based on mask index (original behavior)."""
        import numpy as np

        H, W = img_array.shape[:2]
        overlay = img_array.copy()

        for idx, mesh in loaded_meshes:
            verts = mesh.vertices

            # Filter vertices in front of camera (Z > 0)
            mask = verts[:, 2] > 0.1
            v = verts[mask]

            if len(v) == 0:
                continue

            # Project vertices
            # Mesh is in PyTorch3D convention (X left, Y up, Z into scene)
            # OpenCV projection expects (X right, Y down), so negate X and Y
            X, Y, Z = v[:, 0], v[:, 1], v[:, 2]
            x = (K[0, 0] * (-X) / Z + K[0, 2]).astype(int)
            y = (K[1, 1] * (-Y) / Z + K[1, 2]).astype(int)

            # Filter points within image bounds
            valid = (x >= 0) & (x < W) & (y >= 0) & (y < H)
            x, y = x[valid], y[valid]

            # Get color for this object
            color = np.array(colors[idx % len(colors)])

            # Vectorized point drawing with size
            offsets = np.arange(-point_size, point_size + 1)
            dx, dy = np.meshgrid(offsets, offsets)
            dx, dy = dx.flatten(), dy.flatten()

            all_x = (x[:, None] + dx[None, :]).flatten()
            all_y = (y[:, None] + dy[None, :]).flatten()

            valid = (all_x >= 0) & (all_x < W) & (all_y >= 0) & (all_y < H)
            all_x, all_y = all_x[valid], all_y[valid]

            overlay[all_y, all_x] = overlay[all_y, all_x] * (1 - alpha) + color * alpha
            log.info("ProjectionOverlay: Object %d: %d points projected", idx, len(x))

        return overlay

    def _render_3d_mesh(self, loaded_meshes, img_array, K, colors, alpha, H, W):
        """Render solid colored meshes with proper depth buffering using pyrender."""
        import numpy as np
        import pyrender
        import os

        # Set offscreen rendering backend (EGL on Linux, Pyglet/WGL on Windows)
        import sys
        if sys.platform != 'win32':
            os.environ['PYOPENGL_PLATFORM'] = 'egl'

        overlay = img_array.copy()

        # Create scene
        scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.5, 0.5, 0.5])

        # Add camera with intrinsics
        camera = pyrender.IntrinsicsCamera(
            fx=K[0, 0], fy=K[1, 1],
            cx=K[0, 2], cy=K[1, 2],
            znear=0.01, zfar=100.0
        )
        # Camera at origin looking down +Z (pyrender handles CV intrinsics internally)
        camera_pose = np.eye(4, dtype=np.float32)
        scene.add(camera, pose=camera_pose)

        # Add a light
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
        scene.add(light, pose=camera_pose)

        # Add meshes with solid colors
        for idx, mesh in loaded_meshes:
            color = colors[idx % len(colors)]
            material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[*color, 1.0],
                metallicFactor=0.0,
                roughnessFactor=0.8
            )

            # Transform mesh coordinates for pyrender (OpenGL conventions)
            # Our pose-optimized mesh is in PyTorch3D convention:
            #   X+ = left, Y+ = up, Z+ = into scene
            # pyrender/OpenGL expects:
            #   X+ = right, Y+ = up, Z+ = out of scene (camera looks along -Z)
            # Transform: negate X, keep Y, negate Z
            import trimesh as tr
            mesh_gl = tr.Trimesh(
                vertices=mesh.vertices.copy(),
                faces=mesh.faces.copy(),
                vertex_colors=mesh.visual.vertex_colors if hasattr(mesh.visual, 'vertex_colors') else None,
            )
            mesh_gl.vertices[:, 0] = -mesh_gl.vertices[:, 0]  # X: PT3D left -> OpenGL right
            mesh_gl.vertices[:, 2] = -mesh_gl.vertices[:, 2]  # Z: PT3D into scene -> OpenGL out

            pyrender_mesh = pyrender.Mesh.from_trimesh(mesh_gl, material=material)
            scene.add(pyrender_mesh)

        # Render
        renderer = pyrender.OffscreenRenderer(W, H)
        color_img, depth = renderer.render(scene)
        renderer.delete()

        # Replace pixels where meshes are rendered (solid, no transparency)
        mask = depth > 0
        color_normalized = color_img.astype(np.float32) / 255.0
        overlay[mask] = color_normalized[mask]

        log.info("ProjectionOverlay: Rendered %d meshes (3d_mesh)", len(loaded_meshes))
        return overlay

    def _render_3d_mesh_textured(self, loaded_meshes, img_array, K, alpha, H, W):
        """Render textured meshes using pyrender."""
        import numpy as np
        import pyrender
        import os

        # Set offscreen rendering backend (EGL on Linux, Pyglet/WGL on Windows)
        import sys
        if sys.platform != 'win32':
            os.environ['PYOPENGL_PLATFORM'] = 'egl'

        overlay = img_array.copy()

        # Create scene
        scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.5, 0.5, 0.5])

        # Add camera with intrinsics
        camera = pyrender.IntrinsicsCamera(
            fx=K[0, 0], fy=K[1, 1],
            cx=K[0, 2], cy=K[1, 2],
            znear=0.01, zfar=100.0
        )
        camera_pose = np.eye(4)
        scene.add(camera, pose=camera_pose)

        # Add a light
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
        scene.add(light, pose=camera_pose)

        # Add meshes (pyrender will use vertex colors or textures if available)
        for idx, mesh in loaded_meshes:
            # Transform mesh coordinates for pyrender (OpenGL conventions)
            # PyTorch3D: X left, Y up, Z into scene
            # OpenGL: X right, Y up, Z out of scene
            # Transform: negate X, keep Y, negate Z
            import trimesh as tr
            mesh_gl = tr.Trimesh(
                vertices=mesh.vertices.copy(),
                faces=mesh.faces.copy(),
                visual=mesh.visual,  # Preserve textures/vertex colors
            )
            mesh_gl.vertices[:, 0] = -mesh_gl.vertices[:, 0]  # X: PT3D left -> OpenGL right
            mesh_gl.vertices[:, 2] = -mesh_gl.vertices[:, 2]  # Z: PT3D into scene -> OpenGL out

            pyrender_mesh = pyrender.Mesh.from_trimesh(mesh_gl, smooth=True)
            scene.add(pyrender_mesh)

        # Render
        renderer = pyrender.OffscreenRenderer(W, H)
        color_img, depth = renderer.render(scene)
        renderer.delete()

        # Blend with original image where depth > 0
        mask = depth > 0
        color_normalized = color_img.astype(np.float32) / 255.0
        overlay[mask] = color_normalized[mask]

        log.info("ProjectionOverlay: Rendered %d textured meshes", len(loaded_meshes))
        return overlay


# Node mappings
NODE_CLASS_MAPPINGS = {
    "SAM3D_ProjectionOverlay": SAM3D_ProjectionOverlay,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3D_ProjectionOverlay": "SAM3D Projection Overlay",
}
