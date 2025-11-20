#!/usr/bin/env python3
"""
Script to convert nodes from comfy_api.latest to standard NODE_CLASS_MAPPINGS format.
"""

import re
from pathlib import Path

def convert_file(filepath):
    """Convert a single node file."""
    print(f"Converting {filepath.name}...")

    content = filepath.read_text()

    # Remove comfy_api import
    content = re.sub(r'from comfy_api\.latest import io\n?', '', content)

    # Replace class inheritance
    content = re.sub(r'class (\w+)\(io\.ComfyNode\):', r'class \1:', content)

    # Replace @classmethod execute with instance method
    content = re.sub(r'@classmethod\s+def execute\(cls,', 'def execute(self,', content, flags=re.MULTILINE)

    # Replace io.NodeOutput returns with tuples
    content = re.sub(r'return io\.NodeOutput\((.*?)\)', r'return (\1)', content, flags=re.DOTALL)

    # Fix cls -> self in execute method bodies (careful with INPUT_TYPES)
    # This is tricky, so we'll skip it for now and do manually if needed

    filepath.write_text(content)
    print(f"  ✓ {filepath.name} converted")

# Convert remaining files
nodes_dir = Path(__file__).parent / "nodes"
for filename in ["export_ply.py", "export_mesh.py", "visualizer.py"]:
    convert_file(nodes_dir / filename)

print("\nConversion complete! Now update schemas manually.")
