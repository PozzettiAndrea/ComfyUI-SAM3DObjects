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
from typing import Any, Dict, Optional, Union
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
        """Read stderr from worker process and print filtered output to console."""
        if not self.process or not self.process.stderr:
            return

        # Patterns to show (important events)
        SHOW_PATTERNS = [
            "[Worker] SAM3D inference worker started",
            "[Worker] Ready for requests",
            "[Worker] Inference completed",
            "[Worker] Optimization complete",
            "[Worker] Saved ",
            "[Worker] Error",
            "[Worker] Warning",
            "[Worker] Shutdown",
            "[Worker] Worker shutting down",
            "[Worker] Running ",
            "Traceback",
            "Error:",
        ]

        # Patterns to hide (verbose details)
        HIDE_PATTERNS = [
            "[Worker] Python:",
            "[Worker] Working directory:",
            "[Worker] PyTorch version:",
            "[Worker] PyTorch3D version:",
            "[Worker] CUDA available:",
            "[Worker] Loading model",
            "[Worker] Added ",
            "[Worker] Set ",
            "[Worker] Found ",
            "[Worker] Model cache",
            "[Worker] Loaded ",
            "[Worker] Image ",
            "[Worker] Mask ",
            "[Worker] Pointmap ",
            "[Worker] Intrinsics ",
            "[Worker] Stage ",
            "[Worker] Output ",
            "[Worker] Converting ",
            "[Worker] Assembled ",
            "[Worker] Transformed ",
            "[Worker] GLB ",
            "[Worker] PLY ",
            "[Worker] No pose data",
            "| INFO",
            "| DEBUG",
            "Warp ",
            "CUDA Toolkit",
            "Devices:",
            '"cpu"',
            '"cuda:0"',
            "Kernel cache:",
            "Rendering:",
            "it/s]",
            "% done",
        ]

        for line in self.process.stderr:
            line = line.rstrip()
            if not line:
                continue

            # Always show if matches important patterns
            if any(p in line for p in SHOW_PATTERNS):
                print(line)
                continue

            # Hide if matches verbose patterns
            if any(p in line for p in HIDE_PATTERNS):
                continue

            # Show anything else (unknown messages, errors, etc.)
            print(line)

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

        # Read response with robust JSON parsing
        # Skip any non-JSON lines that may be polluting stdout from libraries
        import time
        start_time = time.time()
        max_attempts = 100  # Limit attempts to prevent infinite loops
        attempts = 0

        while attempts < max_attempts:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Timeout waiting for JSON response after {timeout}s")

            response_line = self.process.stdout.readline()
            if not response_line:
                raise RuntimeError("Worker process closed unexpectedly")

            response_line = response_line.strip()
            if not response_line:
                # Empty line, skip
                attempts += 1
                continue

            # Try to parse as JSON
            # Filter out obvious log messages that start with [ but aren't JSON arrays
            # Log messages look like: [Gaussian], [SPARSE], [Worker], [PLY Export], etc.
            if response_line.startswith('[') and not response_line.startswith('[{') and not response_line.startswith('["'):
                # This is a log message like "[Gaussian] ...", not a JSON array
                # Skip silently - these are worker debug messages
                attempts += 1
                continue

            if response_line.startswith('{') or response_line.startswith('['):
                try:
                    return json.loads(response_line)
                except json.JSONDecodeError:
                    # Malformed JSON - skip silently (likely partial output)
                    attempts += 1
                    continue
            else:
                # Not JSON - likely a log message that slipped through (skip silently)
                attempts += 1
                continue

        raise RuntimeError(f"Failed to get valid JSON response after {max_attempts} attempts")

    def serialize_image(self, image: Image.Image) -> str:
        """Serialize PIL Image to base64."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def serialize_mask(self, mask: np.ndarray) -> str:
        """Serialize numpy mask to base64."""
        return base64.b64encode(pickle.dumps(mask)).decode('utf-8')

    def serialize_stage1_output(self, stage1_output: Optional[dict]) -> Optional[str]:
        """Serialize stage1_output dict to base64."""
        if stage1_output is None:
            return None
        return base64.b64encode(pickle.dumps(stage1_output)).decode('utf-8')

    def _serialize_tensor(self, tensor) -> Optional[str]:
        """Serialize tensor/numpy array to base64."""
        if tensor is None:
            return None
        # Convert to numpy if it's a tensor
        if hasattr(tensor, 'cpu'):
            tensor = tensor.cpu().numpy()
        elif hasattr(tensor, 'numpy'):
            tensor = tensor.numpy()
        return base64.b64encode(pickle.dumps(tensor)).decode('utf-8')

    def serialize_stage2_output(self, stage2_output: Optional[dict]) -> Optional[str]:
        """Serialize stage2_output dict to base64."""
        if stage2_output is None:
            return None
        # Check if this needs to combine separate Gaussian and Mesh serialized data
        # DON'T deserialize here! Just mark it for worker to handle
        if isinstance(stage2_output, dict) and stage2_output.get("_needs_combination"):
            print(f"[SAM3DObjects] Passing separate Gaussian and Mesh data to worker for combination")
            # Return a special marker that tells worker to combine them
            # The worker will receive this via the stage2_output_b64 parameter
            return base64.b64encode(pickle.dumps(stage2_output)).decode('utf-8')
        # Check if this is already serialized Stage 2 data from previous node
        if isinstance(stage2_output, dict) and "_serialized_stage2_output" in stage2_output:
            return stage2_output["_serialized_stage2_output"]
        # Otherwise serialize it
        return base64.b64encode(pickle.dumps(stage2_output)).decode('utf-8')

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

    def load_output_from_disk(self, saved_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load output from disk using file paths.

        Returns file paths (not file contents) - following ComfyUI pattern.
        This allows nodes to decide whether to load/preview the files.
        """
        from pathlib import Path

        result = {
            "output_dir": saved_output["output_dir"],
            "metadata": saved_output.get("metadata", {})
        }

        # Return GLB file path (not contents!)
        if "glb" in saved_output.get("files", {}):
            glb_path = Path(saved_output["files"]["glb"])
            if glb_path.exists():
                result["glb_path"] = str(glb_path)
                print(f"[SAM3DObjects] GLB saved to: {glb_path}")
            else:
                print(f"[SAM3DObjects] Warning: GLB file not found: {glb_path}")
                result["glb_path"] = None

        # Return Gaussian PLY file path
        if "ply" in saved_output.get("files", {}):
            ply_path = Path(saved_output["files"]["ply"])
            if ply_path.exists():
                result["ply_path"] = str(ply_path)
                print(f"[SAM3DObjects] Gaussian PLY saved to: {ply_path}")
            else:
                print(f"[SAM3DObjects] Warning: PLY file not found: {ply_path}")
                result["ply_path"] = None

        return result

    def run_inference(
        self,
        config_path: str,
        image: Image.Image,
        mask: np.ndarray,
        seed: int = 42,
        compile: bool = False,
        use_gpu_cache: bool = True,
        stage1_inference_steps: int = 25,
        stage2_inference_steps: int = 25,
        stage1_cfg_strength: float = 7.0,
        stage2_cfg_strength: float = 5.0,
        texture_size: int = 1024,
        simplify: float = 0.95,
        output_dir: str = None,
        stage1_only: bool = False,
        stage1_output: Optional[dict] = None,
        stage2_only: bool = False,
        stage2_output: Optional[dict] = None,
        slat_only: bool = False,
        slat_output: Optional[dict] = None,
        gaussian_only: bool = False,
        mesh_only: bool = False,
        save_files: bool = False,
        with_mesh_postprocess: bool = False,
        with_texture_baking: bool = True,
        use_vertex_color: bool = False,
        use_stage1_distillation: bool = False,
        use_stage2_distillation: bool = False,
        # NEW: Depth estimation and memory management
        depth_only: bool = False,
        unload_model: str = None,
        pointmap_path: str = None,
        intrinsics: Any = None,
        # Texture baking mode
        texture_mode: str = "opt",
        # Rendering engine
        rendering_engine: str = "pytorch3d",
        # Mask merge controls
        merge_mask: bool = True,
        auto_resize_mask: bool = True,
    ) -> Dict[str, Any]:
        """
        Run inference on the isolated worker.

        Args:
            config_path: Path to pipeline config
            image: Input PIL image
            mask: Input numpy mask
            seed: Random seed
            compile: Whether to compile model
            use_gpu_cache: Keep models on GPU between stages (higher VRAM, faster)
            stage1_inference_steps: Denoising steps for Stage 1
            stage2_inference_steps: Denoising steps for Stage 2
            stage1_cfg_strength: CFG strength for Stage 1
            stage2_cfg_strength: CFG strength for Stage 2
            texture_size: Texture resolution
            simplify: Mesh simplification ratio
            output_dir: Directory to save outputs (defaults to ComfyUI output directory)
            stage1_only: If True, only run Stage 1 (sparse structure generation)
            stage1_output: Stage 1 output to resume from (for Stage 2 only mode)

        Returns:
            Inference output dict with file paths
        """
        # Ensure worker is running
        self.start_worker()

        # Determine output directory
        if output_dir is None:
            # Try to find ComfyUI's output directory
            import os
            comfy_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            output_dir = os.path.join(comfy_dir, "output")

        print(f"[SAM3DObjects] Sending inference request to isolated worker...")
        print(f"[SAM3DObjects] Output directory: {output_dir}")

        # Prepare request
        request = {
            "config_path": config_path,
            "compile": compile,
            "use_cache": not use_gpu_cache,  # Invert: gpu_cache=True means internal use_cache=False
            "image": self.serialize_image(image) if image is not None else None,
            "mask": self.serialize_mask(mask) if mask is not None else None,
            "seed": seed,
            "stage1_inference_steps": stage1_inference_steps,
            "stage2_inference_steps": stage2_inference_steps,
            "stage1_cfg_strength": stage1_cfg_strength,
            "stage2_cfg_strength": stage2_cfg_strength,
            "texture_size": texture_size,
            "simplify": simplify,
            "output_dir": output_dir,  # Pass output directory to worker
            "stage1_only": stage1_only,
            "stage1_output": self.serialize_stage1_output(stage1_output) if isinstance(stage1_output, dict) else None,
            "stage1_output_path": stage1_output if isinstance(stage1_output, str) else None,
            "stage2_only": stage2_only,
            "stage2_output": self.serialize_stage2_output(stage2_output),
            "slat_only": slat_only,
            "slat_output": self.serialize_stage2_output(slat_output) if isinstance(slat_output, dict) else None,
            "slat_output_path": slat_output if isinstance(slat_output, str) else None,
            "gaussian_only": gaussian_only,
            "mesh_only": mesh_only,
            "save_files": save_files,
            "with_mesh_postprocess": with_mesh_postprocess,
            "with_texture_baking": with_texture_baking,
            "use_vertex_color": use_vertex_color,
            "use_stage1_distillation": use_stage1_distillation,
            "use_stage2_distillation": use_stage2_distillation,
            # NEW: Depth estimation and memory management
            "depth_only": depth_only,
            "unload_model": unload_model,
            "pointmap_path": pointmap_path,  # Pass pointmap tensor path directly (no serialization needed)
            "intrinsics": self._serialize_tensor(intrinsics) if intrinsics is not None else None,
            # Texture baking mode
            "texture_mode": texture_mode,
            # Rendering engine
            "rendering_engine": rendering_engine,
            # Mask merge controls
            "merge_mask": merge_mask,
            "auto_resize_mask": auto_resize_mask,
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

        # Handle depth_only response
        if response.get("depth_only", False):
            print(f"[SAM3DObjects] Depth estimation completed")
            result = {"status": "success", "depth_only": True}
            # Deserialize pointmap and intrinsics
            if response.get("pointmap"):
                result["pointmap"] = pickle.loads(base64.b64decode(response["pointmap"]))
            if response.get("intrinsics"):
                result["intrinsics"] = pickle.loads(base64.b64decode(response["intrinsics"]))
            return result

        # Handle unload_model response
        if unload_model is not None:
            print(f"[SAM3DObjects] Model unload completed: {response.get('unloaded', unload_model)}")
            return response

        # Check if this is Stage 1 output (serialized intermediate data or file path)
        if response.get("stage1_mode", False):
            output_data = response["output"]
            if isinstance(output_data, dict) and "files" in output_data:
                print(f"[SAM3DObjects] Stage 1 output saved to disk")
                # Include pose data directly in the response
                result = output_data.copy()
                if response.get("rotation") is not None:
                    result["rotation"] = response["rotation"]
                if response.get("translation") is not None:
                    result["translation"] = response["translation"]
                if response.get("scale") is not None:
                    result["scale"] = response["scale"]
                return result

            print(f"[SAM3DObjects] Deserializing Stage 1 intermediate output")
            output = pickle.loads(base64.b64decode(output_data))
            print(f"[SAM3DObjects] Stage 1 output keys: {list(output.keys())}")
            return output

        # Check if this is Stage 2 output (serialized Gaussian + Mesh data)
        # IMPORTANT: Don't deserialize here! Keep as base64 string.
        # Stage 2 objects require sam3d_objects module which only exists in isolated env.
        # We'll deserialize when sending back to worker for Stage 3.
        if response.get("stage2_mode", False):
            output_data = response["output"]
            # Check if it's a file-based output (slat_only)
            if isinstance(output_data, dict) and "files" in output_data:
                print(f"[SAM3DObjects] Stage 2/SLAT output saved to disk")
                return output_data

            print(f"[SAM3DObjects] Received Stage 2 output (kept as serialized data)")
            # Return the base64 string wrapped in a dict so Stage 3 node can identify it
            result = {
                "_serialized_stage2_output": output_data,
                "_stage2_mode": True
            }
            # Include file paths if they were returned (for gaussian_only/mesh_only modes)
            if "file_output" in response:
                result.update(response["file_output"])
            return result

        # Normal mode: Load files from disk
        output = self.load_output_from_disk(response["output"])

        print(f"[SAM3DObjects] Inference completed successfully")
        return output

    def __del__(self):
        """Cleanup when bridge is destroyed."""
        self.stop_worker()
