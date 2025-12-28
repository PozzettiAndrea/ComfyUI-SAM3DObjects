"""
Inference worker for SAM3D that runs in the isolated environment.

This worker loads the SAM3D model and handles inference requests
via IPC (stdin/stdout communication).

The actual implementation is split into modules in inference_scripts/:
- lazy_manager: Lazy loading of models for low-VRAM GPUs
- utils: Serialization, coordinate transforms, file I/O
- preprocessing: Image/mask preprocessing
- stages: Pipeline stages (sparse gen, SLAT gen, decode)
- depth: Depth estimation
- texture_baking: Texture baking from Gaussian to mesh
- pose_optimization: Pose optimization for scene generation
- inference: Main inference orchestration
"""

import sys
import json
import traceback
from pathlib import Path

# Import from modular inference scripts
from inference_scripts import (
    run_inference,
    run_texture_bake_direct,
    run_pose_optimization,
    run_generate_slat,
)


def main():
    """Main worker loop - reads requests from stdin, writes responses to stdout."""

    # CRITICAL: Suppress all library output to prevent stdout pollution
    # Libraries like OmegaConf, Hydra, PyTorch, CUDA can print to stdout,
    # which interferes with our JSON-based IPC protocol
    import warnings
    import logging
    import os

    # Suppress Python warnings from all libraries
    warnings.filterwarnings("ignore")

    # Suppress TensorFlow logs (if used by any dependency)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Suppress Hydra full error traces
    os.environ['HYDRA_FULL_ERROR'] = '0'

    # Disable all Python logging from libraries
    logging.disable(logging.CRITICAL)

    # Configure loguru to only show errors (suppress INFO/WARNING spam from vendor code)
    try:
        from loguru import logger
        logger.remove()  # Remove default handler
        logger.add(sys.stderr, level="ERROR", format="{message}")
    except ImportError:
        pass  # loguru not available yet

    print("[Worker] SAM3D inference worker started", file=sys.stderr)
    print(f"[Worker] Python: {sys.executable}", file=sys.stderr)
    print(f"[Worker] Working directory: {Path.cwd()}", file=sys.stderr)

    # Verify critical imports
    try:
        import torch
        import pytorch3d
        print(f"[Worker] PyTorch version: {torch.__version__}", file=sys.stderr)
        print(f"[Worker] PyTorch3D version: {pytorch3d.__version__}", file=sys.stderr)
        print(f"[Worker] CUDA available: {torch.cuda.is_available()}", file=sys.stderr)
    except Exception as e:
        print(f"[Worker] Warning: Could not verify dependencies: {e}", file=sys.stderr)

    print("[Worker] Ready for requests", file=sys.stderr)

    # Read requests from stdin
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)

            # Handle special commands
            if request.get("command") == "ping":
                response = {"status": "pong"}
            elif request.get("command") == "shutdown":
                print("[Worker] Shutdown requested", file=sys.stderr)
                response = {"status": "shutdown"}
                print(json.dumps(response), flush=True)
                break
            elif request.get("command") == "pose_optimization":
                response = run_pose_optimization(request)
            elif request.get("command") == "texture_bake_direct":
                response = run_texture_bake_direct(request)
            elif request.get("command") == "generate_slat":
                response = run_generate_slat(request)
            else:
                # Run inference
                response = run_inference(request)

            # Send response
            print(json.dumps(response), flush=True)

        except Exception as e:
            print(f"[Worker] Error processing request: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            error_response = {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            print(json.dumps(error_response), flush=True)

    print("[Worker] Worker shutting down", file=sys.stderr)


if __name__ == "__main__":
    main()
