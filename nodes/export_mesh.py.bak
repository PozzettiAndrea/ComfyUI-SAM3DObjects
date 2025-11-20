"""SAM3DExportMesh node for exporting meshes to various 3D formats."""

import os
from pathlib import Path
from typing import Any, Dict
from comfy_api.latest import io
import folder_paths


class SAM3DExportMesh(io.ComfyNode):
    """
    Export 3D mesh to various file formats (OBJ, GLB, etc.).

    Converts the generated 3D representation to standard mesh formats
    that can be imported into 3D modeling software.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SAM3DExportMesh",
            display_name="SAM3D Export Mesh",
            category="SAM3DObjects",
            inputs=[
                io.Any.Input(
                    "output_data",
                    tooltip="Full output data from SAM3DGenerate node (mesh output)."
                ),
                io.Combo.Input(
                    "format",
                    options=["obj", "glb", "ply_mesh"],
                    default="obj",
                    tooltip="Output mesh format."
                ),
                io.String.Input(
                    "filename",
                    default="mesh",
                    multiline=False,
                    tooltip="Output filename (without extension)."
                ),
                io.String.Input(
                    "output_dir",
                    default="",
                    multiline=False,
                    tooltip="Output directory path. Leave empty to use ComfyUI/output/sam3d/"
                ),
                io.Int.Input(
                    "resolution",
                    default=512,
                    min=128,
                    max=2048,
                    step=64,
                    tooltip="Mesh resolution (higher = more detailed but larger file)."
                ),
            ],
            outputs=[
                io.String.Output(
                    "filepath",
                    tooltip="Full path to the saved mesh file."
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        output_data: Dict[str, Any],
        format: str,
        filename: str,
        output_dir: str,
        resolution: int,
    ) -> io.NodeOutput:
        """
        Export mesh to file.

        Args:
            output_data: Full output dict from SAM3DGenerate
            format: Output format (obj, glb, ply_mesh)
            filename: Output filename
            output_dir: Output directory
            resolution: Mesh resolution

        Returns:
            Path to saved file
        """
        print(f"[SAM3DObjects] Exporting mesh to {format.upper()} format...")

        # Determine output directory
        if output_dir and output_dir.strip():
            save_dir = Path(output_dir)
        else:
            output_base = Path(folder_paths.get_output_directory())
            save_dir = output_base / "sam3d"

        save_dir.mkdir(parents=True, exist_ok=True)

        # Clean filename and add extension
        clean_filename = cls._sanitize_filename(filename)
        if not clean_filename.endswith(f'.{format}'):
            clean_filename = f"{clean_filename}.{format}"

        output_path = save_dir / clean_filename

        # Extract mesh from output data
        # The output_data should contain gaussian model and other info
        try:
            if "gaussian" in output_data:
                gaussian = output_data["gaussian"]
            else:
                raise ValueError("Output data does not contain 'gaussian' key")

            print(f"[SAM3DObjects] Converting to mesh (resolution: {resolution})...")

            # Convert Gaussian to mesh using marching cubes or similar
            mesh = cls._gaussian_to_mesh(gaussian, resolution)

            # Save mesh in requested format
            print(f"[SAM3DObjects] Saving to: {output_path}")

            if format == "obj":
                cls._save_obj(mesh, output_path)
            elif format == "glb":
                cls._save_glb(mesh, output_path)
            elif format == "ply_mesh":
                cls._save_ply_mesh(mesh, output_path)
            else:
                raise ValueError(f"Unsupported format: {format}")

        except Exception as e:
            raise RuntimeError(f"Failed to export mesh: {e}") from e

        # Verify file was created
        if not output_path.exists():
            raise RuntimeError(f"Mesh file was not created at {output_path}")

        file_size_mb = output_path.stat().st_size / (1024 * 1024)

        print(f"[SAM3DObjects] Mesh exported successfully!")
        print(f"[SAM3DObjects] - Path: {output_path}")
        print(f"[SAM3DObjects] - Size: {file_size_mb:.2f} MB")

        return io.NodeOutput(str(output_path))

    @staticmethod
    def _gaussian_to_mesh(gaussian, resolution: int):
        """
        Convert Gaussian representation to mesh.

        This is a placeholder - actual implementation depends on
        the SAM3D output format and available conversion tools.

        Args:
            gaussian: Gaussian model from SAM3D
            resolution: Target mesh resolution

        Returns:
            Mesh object (trimesh or similar)
        """
        import trimesh
        import numpy as np

        # TODO: Implement actual conversion from Gaussian to mesh
        # This may involve:
        # 1. Extracting point cloud from Gaussian
        # 2. Running Poisson surface reconstruction
        # 3. Or using built-in SAM3D mesh export if available

        # For now, this is a placeholder that shows the structure
        # The actual SAM3D output may already contain mesh data

        print("[SAM3DObjects] Converting Gaussian representation to mesh...")
        print("[SAM3DObjects] Note: This may take some time depending on resolution.")

        # Placeholder - in real implementation, this would use
        # SAM3D's built-in mesh extraction or convert the Gaussian
        # representation to a proper mesh

        raise NotImplementedError(
            "Mesh conversion not yet implemented. "
            "Please check SAM3D documentation for mesh export methods."
        )

    @staticmethod
    def _save_obj(mesh, path: Path):
        """Save mesh as OBJ format."""
        import trimesh
        mesh.export(str(path))

    @staticmethod
    def _save_glb(mesh, path: Path):
        """Save mesh as GLB format."""
        import trimesh
        mesh.export(str(path))

    @staticmethod
    def _save_ply_mesh(mesh, path: Path):
        """Save mesh as PLY format."""
        import trimesh
        mesh.export(str(path))

    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        """Sanitize filename by removing invalid characters."""
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        filename = filename.strip('. ')
        if not filename:
            filename = "mesh"
        return filename


class SAM3DExtractMesh(io.ComfyNode):
    """
    Extract mesh data from SAM3D output for further processing.

    This node extracts the mesh representation without saving to file,
    allowing it to be passed to other mesh processing nodes.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SAM3DExtractMesh",
            display_name="SAM3D Extract Mesh Data",
            category="SAM3DObjects",
            inputs=[
                io.Any.Input(
                    "output_data",
                    tooltip="Full output data from SAM3DGenerate node."
                ),
                io.Int.Input(
                    "resolution",
                    default=512,
                    min=128,
                    max=2048,
                    step=64,
                    tooltip="Mesh resolution."
                ),
            ],
            outputs=[
                io.Any.Output(
                    "mesh",
                    tooltip="Extracted mesh data."
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        output_data: Dict[str, Any],
        resolution: int,
    ) -> io.NodeOutput:
        """
        Extract mesh from output data.

        Args:
            output_data: Full output dict from SAM3DGenerate
            resolution: Mesh resolution

        Returns:
            Mesh data
        """
        print(f"[SAM3DObjects] Extracting mesh data (resolution: {resolution})...")

        try:
            if "gaussian" in output_data:
                gaussian = output_data["gaussian"]
            else:
                raise ValueError("Output data does not contain 'gaussian' key")

            # Convert to mesh
            mesh = SAM3DExportMesh._gaussian_to_mesh(gaussian, resolution)

            print("[SAM3DObjects] Mesh extraction completed!")

            return io.NodeOutput(mesh)

        except Exception as e:
            raise RuntimeError(f"Failed to extract mesh: {e}") from e
