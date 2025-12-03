#!/usr/bin/env python3
"""
Quick test to verify the isolated environment worker can start and load modules.
"""

import json
import subprocess
import sys
from pathlib import Path


def test_worker():
    """Test that the worker can start and import sam3d_objects."""
    # Get paths
    node_root = Path(__file__).parent
    sys.path.insert(0, str(node_root))

    from nodes.env_manager import SAM3DEnvironmentManager

    env_mgr = SAM3DEnvironmentManager(node_root)
    python_exe = env_mgr.get_python_executable()
    worker_script = node_root / "inference_worker.py"

    if not python_exe.exists():
        print(f"ERROR: Python executable not found at {python_exe}")
        print("Please run install.py first")
        return 1

    print(f"Testing isolated environment...")
    print(f"Python: {python_exe}")
    print(f"Worker: {worker_script}")
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
        # Send ping
        print("Sending ping command...")
        ping_request = json.dumps({"command": "ping"}) + "\n"
        proc.stdin.write(ping_request)
        proc.stdin.flush()

        # Read response
        response_line = proc.stdout.readline()
        if not response_line:
            print("ERROR: No response from worker")
            stderr_output = proc.stderr.read()
            print("Worker stderr:")
            print(stderr_output)
            return 1

        response = json.loads(response_line)
        print(f"Response: {response}")

        if response.get("status") == "pong":
            print()
            print("[OK] Worker started successfully!")
            print("[OK] Environment is functional!")

            # Read stderr to see initialization messages
            import time
            time.sleep(0.5)  # Give it a moment to output stderr

            return 0
        else:
            print(f"ERROR: Unexpected response: {response}")
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

        # Print any stderr output
        stderr_output = proc.stderr.read() if proc.stderr else ""
        if stderr_output:
            print()
            print("Worker output:")
            print(stderr_output)


if __name__ == "__main__":
    sys.exit(test_worker())
