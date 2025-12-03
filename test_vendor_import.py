#!/usr/bin/env python3
"""
Test that the worker can import sam3d_objects from vendor directory.
"""

import json
import subprocess
import sys
from pathlib import Path


def test_vendor_import():
    """Test sam3d_objects import in worker."""
    # Get paths
    node_root = Path(__file__).parent
    sys.path.insert(0, str(node_root))

    from nodes.env_manager import SAM3DEnvironmentManager

    env_mgr = SAM3DEnvironmentManager(node_root)
    python_exe = env_mgr.get_python_executable()

    print("Testing vendor imports in isolated environment...")
    print(f"Python: {python_exe}")
    print()

    # Test import directly
    print("Testing: import sam3d_objects")
    result = subprocess.run(
        [
            str(python_exe),
            "-c",
            "import os; os.environ['LIDRA_SKIP_INIT'] = '1'; import sys; sys.path.insert(0, 'vendor'); import sam3d_objects; print('[OK] sam3d_objects imported successfully')"
        ],
        capture_output=True,
        text=True,
        cwd=node_root
    )

    print(result.stdout)
    if result.returncode != 0:
        print("ERROR:")
        print(result.stderr)
        return 1

    print()
    print("Testing: import sam3d_objects.pipeline.inference_pipeline_pointmap")
    result = subprocess.run(
        [
            str(python_exe),
            "-c",
            "import os; os.environ['LIDRA_SKIP_INIT'] = '1'; import sys; sys.path.insert(0, 'vendor'); from sam3d_objects.pipeline.inference_pipeline_pointmap import InferencePipelinePointMap; print('[OK] InferencePipelinePointMap imported successfully')"
        ],
        capture_output=True,
        text=True,
        cwd=node_root
    )

    print(result.stdout)
    if result.returncode != 0:
        print("ERROR:")
        print(result.stderr)
        return 1

    print()
    print("=" * 60)
    print("[OK] All vendor imports working correctly!")
    print("[OK] Worker can access sam3d_objects module!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(test_vendor_import())
