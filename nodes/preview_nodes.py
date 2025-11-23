"""
Preview nodes for Point Clouds and Gaussian Splats
"""

class SAM3D_PreviewPointCloud:
    """
    Preview point cloud PLY files in the browser using VTK.js
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "file_path": ("STRING", {"default": ""}),
            },
            "optional": {
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "preview"
    OUTPUT_NODE = True
    CATEGORY = "SAM3DObjects"
    DESCRIPTION = """
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
"""

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Force re-execution when file_path changes."""
        # Return deterministic value based on inputs for proper cache invalidation
        file_path = kwargs.get('file_path', '')
        return f"{file_path}"

    def preview(self, file_path=""):
        """
        Preview the point cloud using VTK.js.

        Args:
            file_path: Path to existing PLY file
        """
        print(f"[SAM3D Preview] preview() called with file_path='{file_path}'")

        if not file_path or file_path.strip() == "":
            # No input provided
            return {"ui": {"file_path": [""]}}

        # Return the file path directly
        return {
            "ui": {
                "file_path": [file_path]
            }
        }
