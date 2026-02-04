"""SAM3D model configuration - just a dict for JSON serialization."""


def IsolatedSAM3DModel(config_path: str, compile: bool = False, use_gpu_cache: bool = True, depth_backend: str = "moge2") -> dict:
    """Create model config dict."""
    return {
        "config_path": str(config_path),
        "compile": compile,
        "use_gpu_cache": use_gpu_cache,
        "depth_backend": depth_backend,
    }


# Backward compat
SAM3DModelConfig = IsolatedSAM3DModel
