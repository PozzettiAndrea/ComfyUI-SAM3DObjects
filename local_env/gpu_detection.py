"""
GPU detection for automatic CUDA version selection.

Detects Blackwell GPUs (RTX 50xx, B100, B200) which require CUDA 12.8,
vs older GPUs which use CUDA 12.4.

This runs BEFORE PyTorch is installed, so we use nvidia-smi directly.
"""

import subprocess
from typing import List, Dict, Optional


def detect_gpu_info() -> List[Dict[str, str]]:
    """
    Detect GPU name and compute capability using nvidia-smi.

    Returns:
        List of dicts with 'name' and 'compute_cap' keys.
        Empty list if detection fails.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            gpus = []
            for line in result.stdout.strip().split('\n'):
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(',')]
                name = parts[0] if parts else "Unknown"
                cc = parts[1] if len(parts) > 1 else "0.0"
                gpus.append({"name": name, "compute_cap": cc})
            return gpus
    except FileNotFoundError:
        # nvidia-smi not found - no NVIDIA GPU or driver not installed
        pass
    except subprocess.TimeoutExpired:
        pass
    except Exception:
        pass
    return []


def is_blackwell_gpu(name: str, compute_cap: str) -> bool:
    """
    Check if a GPU is Blackwell architecture.

    Args:
        name: GPU name from nvidia-smi
        compute_cap: Compute capability string (e.g., "8.9", "12.0")

    Returns:
        True if Blackwell (requires CUDA 12.8)
    """
    name_upper = name.upper()

    # Check by name patterns
    blackwell_patterns = [
        "RTX 50",      # RTX 5090, 5080, 5070, etc.
        "RTX50",       # Without space
        "B100",        # Datacenter Blackwell
        "B200",        # Datacenter Blackwell
        "GB202",       # Blackwell die
        "GB203",
        "GB205",
        "GB206",
        "GB207",
    ]

    if any(pattern in name_upper for pattern in blackwell_patterns):
        return True

    # Check by compute capability (10.0+ = Blackwell)
    try:
        cc = float(compute_cap)
        if cc >= 10.0:
            return True
    except (ValueError, TypeError):
        pass

    return False


def needs_cuda_128() -> bool:
    """
    Check if any detected GPU requires CUDA 12.8.

    Returns:
        True if Blackwell GPU detected, False otherwise.
    """
    gpus = detect_gpu_info()

    for gpu in gpus:
        if is_blackwell_gpu(gpu["name"], gpu["compute_cap"]):
            return True

    return False


def get_recommended_cuda_version() -> str:
    """
    Get recommended CUDA version based on detected GPU.

    Returns:
        "12.8" for Blackwell GPUs, "12.4" for all others.
    """
    return "12.8" if needs_cuda_128() else "12.4"


def get_gpu_summary() -> str:
    """
    Get a human-readable summary of detected GPUs.

    Returns:
        Summary string for logging.
    """
    gpus = detect_gpu_info()

    if not gpus:
        return "No NVIDIA GPU detected"

    lines = []
    for i, gpu in enumerate(gpus):
        is_blackwell = is_blackwell_gpu(gpu["name"], gpu["compute_cap"])
        tag = " [Blackwell - CUDA 12.8]" if is_blackwell else ""
        lines.append(f"  GPU {i}: {gpu['name']} (sm_{gpu['compute_cap'].replace('.', '')}){tag}")

    return "\n".join(lines)
