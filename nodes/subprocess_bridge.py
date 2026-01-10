"""
Subprocess bridge for communicating with the isolated SAM3D inference worker.

This module manages the worker process lifecycle and handles IPC communication.
Uses comfyui-isolation package for process isolation.

Environment configuration is loaded from comfyui_isolation_reqs.toml in the node root.
"""

import io
import pickle
import base64
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image
import numpy as np

from comfyui_isolation import WorkerBridge


# Node root directory
NODE_ROOT = Path(__file__).parent.parent

# Singleton bridge instance
_bridge: Optional[WorkerBridge] = None


def get_bridge() -> WorkerBridge:
    """Get or create the worker bridge singleton."""
    global _bridge
    if _bridge is None:
        def log(msg):
            print(f"[SAM3DObjects] {msg}")

        # Load environment config from comfyui_isolation_reqs.toml
        _bridge = WorkerBridge.from_config_file(
            node_dir=NODE_ROOT,
            worker_script=NODE_ROOT / "inference_worker.py",
            log_callback=log,
        )
    return _bridge


# Legacy class name for backward compatibility
class InferenceWorkerBridge:
    """
    Legacy wrapper for backward compatibility.

    New code should use get_bridge() directly.
    """

    _instance: Optional['InferenceWorkerBridge'] = None

    def __init__(self):
        self._bridge = get_bridge()

    @classmethod
    def get_instance(cls, node_root: Path = None) -> 'InferenceWorkerBridge':
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def start_worker(self):
        """Start the worker process."""
        self._bridge.start()

    def stop_worker(self):
        """Stop the worker process."""
        self._bridge.stop()

    @property
    def python_exe(self):
        return self._bridge.python_exe

    def run_inference(self, **kwargs) -> Dict[str, Any]:
        """Run inference on the worker."""
        # Serialize complex objects before sending
        kwargs = _serialize_args(kwargs)
        response = self._bridge.call("inference", timeout=600.0, **kwargs)
        return _process_inference_response(response, kwargs)


def _serialize_args(kwargs: dict) -> dict:
    """Serialize PIL Images and numpy arrays to base64."""
    serialized = {}
    for key, value in kwargs.items():
        if value is None:
            serialized[key] = None
        elif isinstance(value, Image.Image):
            buffer = io.BytesIO()
            value.save(buffer, format="PNG")
            serialized[key] = base64.b64encode(buffer.getvalue()).decode('utf-8')
        elif isinstance(value, np.ndarray):
            serialized[key] = base64.b64encode(pickle.dumps(value)).decode('utf-8')
        elif hasattr(value, 'cpu') and hasattr(value, 'numpy'):
            # PyTorch tensor
            arr = value.cpu().numpy()
            serialized[key] = base64.b64encode(pickle.dumps(arr)).decode('utf-8')
        elif isinstance(value, dict) and "_serialized_stage2_output" in value:
            # Already serialized stage2 output
            serialized[key] = value["_serialized_stage2_output"]
        elif isinstance(value, dict):
            # Recursively serialize dicts
            serialized[key] = _serialize_args(value)
        else:
            serialized[key] = value
    return serialized


def _process_inference_response(response: Any, request: dict) -> Dict[str, Any]:
    """Process and deserialize inference response."""
    if isinstance(response, dict):
        # Check for error
        if response.get("status") == "error":
            error_msg = response.get("error", "Unknown error")
            traceback_msg = response.get("traceback", "")
            raise RuntimeError(f"Worker error: {error_msg}\n{traceback_msg}")

        # Handle depth_only response
        if response.get("depth_only", False):
            print(f"[SAM3DObjects] Depth estimation completed")
            result = {"status": "success", "depth_only": True}
            if response.get("pointmap"):
                result["pointmap"] = pickle.loads(base64.b64decode(response["pointmap"]))
            if response.get("intrinsics"):
                result["intrinsics"] = pickle.loads(base64.b64decode(response["intrinsics"]))
            return result

        # Handle unload_model response
        if request.get("unload_model") is not None:
            print(f"[SAM3DObjects] Model unload completed")
            return response

        # Handle stage1_mode
        if response.get("stage1_mode", False):
            output_data = response.get("output")
            if isinstance(output_data, dict) and "files" in output_data:
                print(f"[SAM3DObjects] Stage 1 output saved to disk")
                result = output_data.copy()
                for key in ["rotation", "translation", "scale"]:
                    if response.get(key) is not None:
                        result[key] = response[key]
                return result
            print(f"[SAM3DObjects] Deserializing Stage 1 intermediate output")
            return pickle.loads(base64.b64decode(output_data))

        # Handle stage2_mode
        if response.get("stage2_mode", False):
            output_data = response.get("output")
            if isinstance(output_data, dict) and "files" in output_data:
                print(f"[SAM3DObjects] Stage 2/SLAT output saved to disk")
                return output_data
            result = {"_serialized_stage2_output": output_data, "_stage2_mode": True}
            if "file_output" in response:
                result.update(response["file_output"])
            return result

        # Handle file output
        if "output" in response and isinstance(response["output"], dict):
            return _load_output_from_disk(response["output"])

    return response


def _load_output_from_disk(saved_output: Dict[str, Any]) -> Dict[str, Any]:
    """Load output from disk using file paths."""
    result = {
        "output_dir": saved_output.get("output_dir"),
        "metadata": saved_output.get("metadata", {})
    }

    files = saved_output.get("files", {})
    if "glb" in files:
        glb_path = Path(files["glb"])
        if glb_path.exists():
            result["glb_path"] = str(glb_path)
            print(f"[SAM3DObjects] GLB saved to: {glb_path}")
        else:
            result["glb_path"] = None

    if "ply" in files:
        ply_path = Path(files["ply"])
        if ply_path.exists():
            result["ply_path"] = str(ply_path)
            print(f"[SAM3DObjects] Gaussian PLY saved to: {ply_path}")
        else:
            result["ply_path"] = None

    return result


# =============================================================================
# Helper functions for backward compatibility
# =============================================================================

def run_texture_bake_direct(
    ply_path: str,
    glb_path: str,
    output_dir: str,
    texture_mode: str = "opt",
    texture_size: int = 1024,
    rendering_engine: str = "nvdiffrast",
) -> Dict[str, Any]:
    """Run texture baking directly without loading any models."""
    print(f"[SAM3DObjects] Running direct texture baking")
    print(f"[SAM3DObjects] PLY: {ply_path}, GLB: {glb_path}")

    bridge = get_bridge()
    response = bridge.call(
        "texture_bake_direct",
        timeout=300.0,
        ply_path=ply_path,
        glb_path=glb_path,
        output_dir=output_dir,
        texture_mode=texture_mode,
        texture_size=texture_size,
        rendering_engine=rendering_engine,
    )

    if isinstance(response, dict) and response.get("status") == "error":
        raise RuntimeError(f"Texture baking failed: {response.get('error')}")

    return response.get("output", response) if isinstance(response, dict) else response


def run_generate_slat(
    bridge,  # Ignored, kept for API compatibility
    config_path: str,
    image: Image.Image,
    mask: np.ndarray,
    pointmap_path: str,
    output_dir: str,
    seed: int = 42,
    stage1_steps: int = 12,
    stage1_cfg: float = 7.5,
    stage1_cfg_pm: float = 0.0,
    stage2_steps: int = 12,
    stage2_cfg: float = 5.0,
    skip_stage1: bool = False,
    use_distillation: bool = False,
) -> Dict[str, Any]:
    """Run SLAT generation (Stage 1 + Stage 2) with lazy loading."""
    # Serialize image
    img_buffer = io.BytesIO()
    image.save(img_buffer, format='PNG')
    image_b64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

    # Serialize mask
    mask_b64 = base64.b64encode(pickle.dumps(mask)).decode('utf-8')

    response = get_bridge().call(
        "generate_slat",
        timeout=600.0,
        config_path=config_path,
        image=image_b64,
        mask=mask_b64,
        pointmap_path=pointmap_path,
        output_dir=output_dir,
        seed=seed,
        stage1_steps=stage1_steps,
        stage1_cfg=stage1_cfg,
        stage1_cfg_pm=stage1_cfg_pm,
        stage2_steps=stage2_steps,
        stage2_cfg=stage2_cfg,
        skip_stage1=skip_stage1,
        use_distillation=use_distillation,
    )

    if isinstance(response, dict) and response.get("status") == "error":
        raise RuntimeError(f"SLAT generation failed: {response.get('error')}")

    return response


def run_scene_generate_batch(
    bridge,  # Ignored, kept for API compatibility
    image: Image.Image,
    masks: list,
    pointmap_path: str,
    base_output_dir: str,
    config_path: str,
    mesh_config_path: str,
    gs_config_path: Optional[str] = None,
    seed: int = 42,
    stage1_steps: int = 12,
    stage1_cfg: float = 7.5,
    stage1_cfg_pm: float = 0.0,
    stage2_steps: int = 12,
    stage2_cfg: float = 5.0,
    with_postprocess: bool = False,
    simplify: float = 0.95,
    add_textures: bool = False,
    texture_mode: str = "opt",
    texture_size: int = 1024,
) -> Dict[str, Any]:
    """Run batch scene generation with phase-based model loading."""
    # Serialize image
    img_buffer = io.BytesIO()
    image.save(img_buffer, format='PNG')
    image_b64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

    # Serialize all masks
    masks_b64 = [base64.b64encode(pickle.dumps(mask)).decode('utf-8') for mask in masks]

    timeout = max(600.0, len(masks) * 120.0)
    response = get_bridge().call(
        "scene_generate_batch",
        timeout=timeout,
        image=image_b64,
        masks=masks_b64,
        pointmap_path=pointmap_path,
        base_output_dir=base_output_dir,
        config_path=config_path,
        mesh_config_path=mesh_config_path,
        gs_config_path=gs_config_path,
        seed=seed,
        stage1_steps=stage1_steps,
        stage1_cfg=stage1_cfg,
        stage1_cfg_pm=stage1_cfg_pm,
        stage2_steps=stage2_steps,
        stage2_cfg=stage2_cfg,
        with_postprocess=with_postprocess,
        simplify=simplify,
        add_textures=add_textures,
        texture_mode=texture_mode,
        texture_size=texture_size,
    )

    if isinstance(response, dict) and response.get("status") == "error":
        raise RuntimeError(f"Batch generation failed: {response.get('error')}")

    return response


def run_decode(
    bridge,  # Ignored, kept for API compatibility
    config_path: str,
    slat_path: str,
    output_dir: str,
    decode_format: str = "gaussian",
    with_postprocess: bool = False,
    simplify: float = 0.95,
    up_axis: str = "Y-up (standard)",
) -> Dict[str, Any]:
    """Run decode command - converts SLAT to Gaussian or Mesh."""
    response = get_bridge().call(
        "decode",
        timeout=300.0,
        config_path=config_path,
        slat_path=slat_path,
        output_dir=output_dir,
        decode_format=decode_format,
        with_postprocess=with_postprocess,
        simplify=simplify,
        up_axis=up_axis,
    )

    if isinstance(response, dict) and response.get("status") == "error":
        raise RuntimeError(f"Decode failed: {response.get('error')}")

    return response
