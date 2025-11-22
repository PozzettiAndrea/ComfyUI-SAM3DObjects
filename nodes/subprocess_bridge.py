"""
Subprocess bridge for communicating with the isolated SAM3D inference worker.

This module manages the worker process lifecycle and handles IPC communication.
"""

import json
import pickle
import base64
import subprocess
import io
from pathlib import Path
from typing import Any, Dict, Optional
import threading
import queue
from PIL import Image
import numpy as np


class InferenceWorkerBridge:
    """Manages communication with the isolated inference worker process."""

    _instance: Optional['InferenceWorkerBridge'] = None
    _lock = threading.Lock()

    def __init__(self, python_exe: Path, worker_script: Path):
        """
        Initialize the bridge.

        Args:
            python_exe: Path to Python executable in isolated environment
            worker_script: Path to inference_worker.py
        """
        self.python_exe = python_exe
        self.worker_script = worker_script
        self.process: Optional[subprocess.Popen] = None
        self._process_lock = threading.Lock()

    @classmethod
    def get_instance(cls, node_root: Path) -> 'InferenceWorkerBridge':
        """
        Get or create singleton instance.

        Args:
            node_root: Root directory of the ComfyUI-SAM3DObjects node

        Returns:
            Bridge instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    from .env_manager import SAM3DEnvironmentManager

                    env_mgr = SAM3DEnvironmentManager(node_root)
                    python_exe = env_mgr.get_python_executable()
                    worker_script = node_root / "inference_worker.py"

                    if not python_exe.exists():
                        raise RuntimeError(
                            f"Isolated environment not found. Please run install.py first.\n"
                            f"Expected Python at: {python_exe}"
                        )

                    cls._instance = cls(python_exe, worker_script)

        return cls._instance

    def start_worker(self) -> None:
        """Start the worker process if not already running."""
        with self._process_lock:
            if self.process is not None and self.process.poll() is None:
                # Worker already running
                return

            print(f"[SAM3DObjects] Starting inference worker...")
            print(f"[SAM3DObjects] Python: {self.python_exe}")
            print(f"[SAM3DObjects] Worker: {self.worker_script}")

            self.process = subprocess.Popen(
                [str(self.python_exe), str(self.worker_script)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
            )

            # Start stderr reader thread
            self._stderr_thread = threading.Thread(
                target=self._read_stderr,
                daemon=True
            )
            self._stderr_thread.start()

            # Test connection
            response = self._send_request({"command": "ping"})
            if response.get("status") != "pong":
                raise RuntimeError(f"Worker failed to respond to ping: {response}")

            print("[SAM3DObjects] Inference worker started successfully")

    def _read_stderr(self) -> None:
        """Read stderr from worker process and print to console."""
        if self.process and self.process.stderr:
            for line in self.process.stderr:
                print(line.rstrip())

    def stop_worker(self) -> None:
        """Stop the worker process gracefully."""
        with self._process_lock:
            if self.process is None or self.process.poll() is not None:
                return

            print("[SAM3DObjects] Stopping inference worker...")

            try:
                # Send shutdown command
                self._send_request({"command": "shutdown"}, timeout=5.0)
            except Exception:
                pass

            # Wait for graceful shutdown
            try:
                self.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                print("[SAM3DObjects] Worker did not shut down gracefully, terminating...")
                self.process.terminate()
                try:
                    self.process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self.process.kill()

            self.process = None
            print("[SAM3DObjects] Inference worker stopped")

    def _send_request(self, request: Dict[str, Any], timeout: float = 300.0) -> Dict[str, Any]:
        """
        Send a request to the worker and wait for response.

        Args:
            request: Request dict
            timeout: Timeout in seconds

        Returns:
            Response dict
        """
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("Worker process is not running")

        # Send request
        request_json = json.dumps(request) + "\n"
        self.process.stdin.write(request_json)
        self.process.stdin.flush()

        # Read response
        response_line = self.process.stdout.readline()
        if not response_line:
            raise RuntimeError("Worker process closed unexpectedly")

        return json.loads(response_line)

    def serialize_image(self, image: Image.Image) -> str:
        """Serialize PIL Image to base64."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def serialize_mask(self, mask: np.ndarray) -> str:
        """Serialize numpy mask to base64."""
        return base64.b64encode(pickle.dumps(mask)).decode('utf-8')

    def deserialize_output(self, serialized: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize output from worker."""
        deserialized = {}

        for key, value in serialized.items():
            if value is None:
                deserialized[key] = None
            elif isinstance(value, dict) and "_type" in value:
                # Handle special serialized types
                if value["_type"] == "numpy":
                    deserialized[key] = pickle.loads(base64.b64decode(value["_data"]))
                elif value["_type"] == "pickle":
                    deserialized[key] = pickle.loads(base64.b64decode(value["_data"]))
                elif value["_type"] in ("list", "tuple"):
                    items = [self.deserialize_output({"v": v})["v"] for v in value["_data"]]
                    deserialized[key] = items if value["_type"] == "list" else tuple(items)
                else:
                    deserialized[key] = value
            elif isinstance(value, dict):
                # Recursively deserialize dicts
                deserialized[key] = self.deserialize_output(value)
            else:
                deserialized[key] = value

        return deserialized

    def run_inference(
        self,
        config_path: str,
        image: Image.Image,
        mask: np.ndarray,
        seed: int = 42,
        compile: bool = False,
        with_mesh_postprocess: bool = True
    ) -> Dict[str, Any]:
        """
        Run inference on the isolated worker.

        Args:
            config_path: Path to pipeline config
            image: Input PIL image
            mask: Input numpy mask
            seed: Random seed
            compile: Whether to compile model
            with_mesh_postprocess: Whether to perform mesh postprocessing

        Returns:
            Inference output dict
        """
        # Ensure worker is running
        self.start_worker()

        print(f"[SAM3DObjects] Sending inference request to isolated worker...")

        # Prepare request
        request = {
            "config_path": config_path,
            "compile": compile,
            "image": self.serialize_image(image),
            "mask": self.serialize_mask(mask),
            "seed": seed,
            "with_mesh_postprocess": with_mesh_postprocess,
        }

        # Send request
        response = self._send_request(request, timeout=600.0)  # 10 minute timeout

        # Check status
        if response["status"] == "error":
            error_msg = response.get("error", "Unknown error")
            traceback_msg = response.get("traceback", "")
            raise RuntimeError(
                f"Inference worker error: {error_msg}\n"
                f"Traceback:\n{traceback_msg}"
            )

        # Deserialize output
        output = self.deserialize_output(response["output"])

        print(f"[SAM3DObjects] Inference completed successfully")
        return output

    def __del__(self):
        """Cleanup when bridge is destroyed."""
        self.stop_worker()
