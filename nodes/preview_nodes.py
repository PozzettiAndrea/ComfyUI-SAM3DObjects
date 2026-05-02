"""
Preview nodes for Point Clouds and Gaussian Splats
"""

import logging

import torch
from comfy_api.latest import io

log = logging.getLogger("sam3dobjects")

class SAM3D_PreviewPointCloud(io.ComfyNode):
    """
    Preview point cloud PLY files in the browser using VTK.js
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SAM3D_PreviewPointCloud",
            category="SAM3DObjects",
            description="""
Preview point cloud PLY files in 3D using VTK.js (scientific visualization).

Inputs:
- file_path: Path to PLY file

Features:
- VTK.js rendering engine
- Trackball camera controls
- Axis orientation widget
- Adjustable point size
- Max 2M points

Controls:
- Left Mouse: Rotate view
- Right Mouse: Pan camera
- Mouse Wheel: Zoom in/out
- Slider: Adjust point size
""",
            is_output_node=True,
            inputs=[
                io.String.Input("file_path", default=""),
            ],
            outputs=[],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        """Force re-execution when file_path changes."""
        # Return deterministic value based on inputs for proper cache invalidation
        file_path = kwargs.get('file_path', '')
        return f"{file_path}"

    @classmethod
    @torch.no_grad()
    def execute(cls, file_path=""):
        """
        Preview the point cloud using VTK.js.

        Args:
            file_path: Path to existing PLY file
        """
        log.debug("preview() called with file_path='%s'", file_path)

        if not file_path or file_path.strip() == "":
            # No input provided
            return {"ui": {"file_path": [""]}}

        # Return the file path directly
        return {
            "ui": {
                "file_path": [file_path]
            }
        }
