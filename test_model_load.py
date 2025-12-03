#!/usr/bin/env python3
"""
Test that the worker can actually load the SAM3D model from vendor.
"""

import json
import subprocess
import sys
import base64
import io
from pathlib import Path
from PIL import Image
import numpy as np
import pickle


def serialize_image(image):
    """Serialize PIL Image to base64."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def serialize_mask(mask):
    """Serialize numpy mask to base64."""
    return base64.b64encode(pickle.dumps(mask)).decode('utf-8')


def test_model_loading():
    """Test that the worker can load sam3d_objects and the model."""
    # Get paths
    node_root = Path(__file__).parent
    sys.path.insert(0, str(node_root))

    from nodes.env_manager import SAM3DEnvironmentManager

    env_mgr = SAM3DEnvironmentManager(node_root)
    python_exe = env_mgr.get_python_executable()
    worker_script = node_root / "inference_worker.py"

    # Find config file
    config_path = Path("/home/shadeform/sam3do_node/ComfyUI/models/sam3d/hf/checkpoints/pipeline.yaml")
    if not config_path.exists():
        print(f"Config not found at {config_path}")
        print("Skipping model load test - config file needed")
        return 0

    print(f"Testing model loading...")
    print(f"Config: {config_path}")
    print()

    # Start worker
    print("Starting worker process...")
    proc = subprocess.Popen(
        [str(python_exe), str(worker_script)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    try:
        # Wait for worker to be ready
        import time
        time.sleep(1)

        # Create dummy image and mask
        print("Creating test data...")
        dummy_image = Image.new('RGB', (256, 256), color='red')
        dummy_mask = np.ones((256, 256), dtype=np.uint8)

        # Send inference request
        print("Sending inference request...")
        request = {
            "config_path": str(config_path),
            "compile": False,
            "image": serialize_image(dummy_image),
            "mask": serialize_mask(dummy_mask),
            "seed": 42
        }

        request_json = json.dumps(request) + "\n"
        proc.stdin.write(request_json)
        proc.stdin.flush()

        print("Waiting for response (this may take a while as model loads)...")

        # Read response (with timeout)
        response_line = proc.stdout.readline()
        if not response_line:
            print("ERROR: No response from worker")
            stderr_output = proc.stderr.read()
            print("Worker stderr:")
            print(stderr_output)
            return 1

        response = json.loads(response_line)

        if response.get("status") == "success":
            print()
            print("[OK] Model loaded successfully!")
            print("[OK] sam3d_objects imports working!")
            print("[OK] Inference completed!")
            return 0
        elif response.get("status") == "error":
            print()
            print(f"ERROR: {response.get('error')}")
            print()
            print("Traceback:")
            print(response.get('traceback', 'No traceback'))
            return 1
        else:
            print(f"Unexpected response: {response}")
            return 1

    finally:
        # Shutdown
        print()
        print("Shutting down worker...")
        try:
            shutdown_request = json.dumps({"command": "shutdown"}) + "\n"
            proc.stdin.write(shutdown_request)
            proc.stdin.flush()
            proc.wait(timeout=5)
        except Exception:
            proc.terminate()
            proc.wait(timeout=2)

        # Print stderr
        stderr_output = proc.stderr.read() if proc.stderr else ""
        if stderr_output:
            print()
            print("Worker output:")
            print(stderr_output)


if __name__ == "__main__":
    sys.exit(test_model_loading())
