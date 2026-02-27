"""SAM3DExportPLY node for exporting Gaussian Splats to PLY format."""

import logging
import os
from pathlib import Path
from typing import Any
import folder_paths

import torch
from comfy_api.latest import io

log = logging.getLogger("sam3dobjects")


class SAM3DExportPLY(io.ComfyNode):
    """
    Export 3D Gaussian Splat to PLY file format.

    Saves the Gaussian Splat representation to a .ply file that can be
    viewed in 3D viewers or imported into other software.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SAM3DExportPLY",
            category="SAM3DObjects",
            inputs=[
                io.Custom("SAM3D_GAUSSIAN").Input("gaussian_splat"),
                io.String.Input("filename", default="output", multiline=False),
                io.String.Input("output_dir", default="", multiline=False),
            ],
            outputs=[
                io.String.Output(display_name="filepath"),
            ],
        )

    @classmethod
    @torch.no_grad()
    def execute(
        cls,
        gaussian_splat: Any,
        filename: str,
        output_dir: str,
    ):
        """
        Export Gaussian Splat to PLY file.

        Args:
            gaussian_splat: Gaussian Splat object
            filename: Output filename (without extension)
            output_dir: Output directory path

        Returns:
            Path to saved file
        """
        log.info("Exporting Gaussian Splat to PLY...")

        # Determine output directory
        if output_dir and output_dir.strip():
            save_dir = Path(output_dir)
        else:
            # Use ComfyUI output directory
            output_base = Path(folder_paths.get_output_directory())
            save_dir = output_base / "sam3d"

        # Create directory if it doesn't exist
        save_dir.mkdir(parents=True, exist_ok=True)

        # Clean filename and add extension
        clean_filename = cls._sanitize_filename(filename)
        if not clean_filename.endswith('.ply'):
            clean_filename += '.ply'

        # Full output path
        output_path = save_dir / clean_filename

        # Save PLY file
        try:
            log.info("Saving to: %s", output_path)
            gaussian_splat.save_ply(str(output_path))

        except Exception as e:
            raise RuntimeError(f"Failed to save PLY file: {e}") from e

        # Verify file was created
        if not output_path.exists():
            raise RuntimeError(f"PLY file was not created at {output_path}")

        # Get file size
        file_size = output_path.stat().st_size
        file_size_mb = file_size / (1024 * 1024)

        log.info("PLY file saved successfully!")
        log.info("- Path: %s", output_path)
        log.info("- Size: %.2f MB", file_size_mb)

        return io.NodeOutput(str(output_path),)

    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        """
        Sanitize filename by removing invalid characters.

        Args:
            filename: Original filename

        Returns:
            Sanitized filename
        """
        # Remove or replace invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')

        # Remove leading/trailing spaces and dots
        filename = filename.strip('. ')

        # Ensure filename is not empty
        if not filename:
            filename = "output"

        return filename


class SAM3DExportPLYBatch(io.ComfyNode):
    """
    Export multiple Gaussian Splats to PLY files with automatic numbering.

    Useful for batch processing multiple objects.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SAM3DExportPLYBatch",
            category="SAM3DObjects",
            inputs=[
                io.Custom("SAM3D_GAUSSIAN").Input("gaussian_splat"),
                io.String.Input("prefix", default="object", multiline=False),
                io.Int.Input("index", default=0, min=0, max=9999),
                io.String.Input("output_dir", default="", multiline=False),
            ],
            outputs=[
                io.String.Output(display_name="filepath"),
            ],
        )

    @classmethod
    @torch.no_grad()
    def execute(
        cls,
        gaussian_splat: Any,
        prefix: str,
        index: int,
        output_dir: str,
    ):
        """
        Export Gaussian Splat to numbered PLY file.

        Args:
            gaussian_splat: Gaussian Splat object
            prefix: Filename prefix
            index: File index number
            output_dir: Output directory path

        Returns:
            Path to saved file
        """
        # Generate filename with zero-padded index
        filename = f"{prefix}_{index:04d}.ply"

        # Use the same export logic
        log.info("Exporting batch item %d...", index)

        # Determine output directory
        if output_dir and output_dir.strip():
            save_dir = Path(output_dir)
        else:
            output_base = Path(folder_paths.get_output_directory())
            save_dir = output_base / "sam3d"

        save_dir.mkdir(parents=True, exist_ok=True)

        # Full output path
        output_path = save_dir / filename

        # Save PLY file
        try:
            gaussian_splat.save_ply(str(output_path))
        except Exception as e:
            raise RuntimeError(f"Failed to save PLY file: {e}") from e

        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        log.info("Saved: %s (%.2f MB)", filename, file_size_mb)

        return io.NodeOutput(str(output_path),)
