# ComfyUI-SAM3DObjects

ComfyUI custom nodes for SAM3D (Single-view 3D Objects) - generates 3D meshes and Gaussian splats from single images with masks.

## Architecture

```
ComfyUI Process                    Isolated Subprocess (@isolated decorator)
┌────────────────────┐            ┌─────────────────────────┐
│  nodes/*.py        │  auto-     │  Auto-generated worker  │
│  @isolated class   │──spawn────►│  in _env_sam3dobjects/  │
│  - INPUT_TYPES     │            │                         │
│  - FUNCTION method │  JSON/IPC  │  Imports from:          │
│    (runs in        │◄──────────►│  - nodes/worker/*.py    │
│     subprocess)    │            │  - vendor/sam3d_objects │
└────────────────────┘            └─────────────────────────┘
```

**Process Isolation**: Each `@isolated` decorated node method runs in a separate Python subprocess with its own venv (`_env_sam3dobjects/`). The decorator auto-generates worker code, handles serialization (ComfyUI IMAGE/MASK → numpy/torch), and manages subprocess lifecycle.

## Directory Structure

```
ComfyUI-SAM3DObjects/
├── nodes/                    # ComfyUI node definitions
│   ├── load_model.py         # LoadSAM3DModel (no isolation - just config)
│   ├── depth_estimate.py     # @isolated SAM3D_DepthEstimate (MoGe)
│   ├── generate_slat.py      # @isolated SAM3DGenerateSLAT (Stage1+2)
│   ├── gaussian_decode.py    # @isolated SAM3DGaussianDecode
│   ├── mesh_decode.py        # @isolated SAM3DMeshDecode
│   ├── postprocess.py        # @isolated SAM3DTextureBake
│   ├── scene_generate.py     # @isolated SAM3DSceneGenerate (batch)
│   ├── scene_pose_optimize.py # @isolated SAM3D_ScenePoseOptimize
│   ├── pose_optimization.py  # @isolated SAM3D_PoseOptimization
│   ├── isolated_model.py     # SAM3DModelConfig (simple data class)
│   ├── unload_model.py       # No-op pass-through (for compatibility)
│   └── worker/               # Importable worker modules
│       ├── __init__.py       # Exports key functions
│       ├── lazy_manager.py   # Model loading/caching
│       ├── stages.py         # run_stage1_lazy, run_stage2_lazy, run_decode_lazy
│       ├── inference.py      # High-level inference API
│       ├── scene_batch.py    # run_scene_generate_batch
│       ├── pose_optimization.py  # run_pose_optimization, run_pose_optimization_batch
│       ├── depth.py          # run_depth_only
│       └── texture_baking.py # run_texture_bake_direct
├── vendor/                   # Vendored dependencies
│   ├── sam3d_objects/        # Core SAM3D library
│   ├── moge/                 # MoGe depth estimation
│   └── cv2/                  # OpenCV shim (DLL fix)
├── comfyui_isolation_reqs.toml  # Isolated env config
└── _env_sam3dobjects/        # Isolated Python venv (auto-created)
```

## @isolated Decorator Pattern

Each GPU node uses the `@isolated` decorator from `comfyui-isolation`:

```python
from comfyui_isolation import isolated

@isolated(
    env="sam3dobjects",
    config="comfyui_isolation_reqs.toml",
    import_paths=[".", "../vendor"],  # "." = nodes/, "../vendor" = vendor/
    timeout=600.0,
)
class SAM3D_DepthEstimate:
    FUNCTION = "estimate_depth"

    def estimate_depth(self, depth_model, image):
        # This entire method runs in isolated subprocess
        # Imports happen here, inside the subprocess
        from worker.depth import run_depth_only
        from worker.lazy_manager import get_model_manager

        # depth_model is SAM3DModelConfig with config_path, etc.
        manager = get_model_manager(depth_model.config_path)
        result = run_depth_only(manager, image, ...)
        return (result,)
```

**Key points:**
- Method body runs in subprocess with isolated Python environment
- ComfyUI IMAGE/MASK tensors auto-serialized via JSON IPC
- `import_paths=[".", "../vendor"]` adds `nodes/` and `vendor/` to subprocess's sys.path
- ComfyUI base auto-detected, so `folder_paths` works in subprocess
- Each node invocation spawns fresh subprocess → auto VRAM cleanup

## Pipeline

### Single Object
```
LoadSAM3DModel → SAM3D_DepthEstimate → SAM3DGenerateSLAT → Decode → (TextureBake)
                     ↓                        ↓
                 pointmap.pt              slat.pt (cached)
```

### Batch (Scene)
```
SAM3DSceneGenerate (image + masks batch)
    → Phase-based: Load Stage1 once → all masks → unload
    → Phase-based: Load Stage2 once → all SLATs → unload
    → Phase-based: Load Decoder once → all meshes → unload
    → Output: object_0/, object_1/, ... (world-coordinate GLBs)
```

## Coordinate Systems

| Context | System | Notes |
|---------|--------|-------|
| Internal generation | Z-up | SAM3D native |
| Pose computation | Z-up | Camera frame |
| GLB export | Y-up | Standard (toggle available) |
| Output meshes/Gaussians | World coords | Pose baked in (not centered) |

## Key Files for Modifications

### Adding/Modifying Nodes
- `nodes/*.py` - Node class definitions (add `@isolated` decorator for GPU work)
- `__init__.py` - Node registration (NODE_CLASS_MAPPINGS)

### Modifying Inference
- `nodes/worker/stages.py` - Core decode logic, pose application
- `nodes/worker/lazy_manager.py` - Model loading/caching
- `nodes/worker/inference.py` - High-level inference orchestration

### Serialization
- Automatic: ComfyUI IMAGE/MASK handled by comfyui-isolation
- Manual for complex types: Use `base64.b64encode(pickle.dumps(data))`

## Considerations

### Process Isolation
- Each @isolated node method runs in separate subprocess
- Subprocess has different Python/CUDA than ComfyUI
- Auto-generated worker code lives in `_generated_{env}/`
- Subprocess exits when method returns → automatic VRAM cleanup

### VRAM Management
- Fresh subprocess per call = automatic cleanup
- No need for explicit unload_model (kept for compatibility)
- ~16GB VRAM minimum (24GB+ recommended)

### Caching
- SLAT generation caches by seed/steps/cfg hash
- Stage 1 output cached in output folder
- Scene pose optimization caches by mode/object count

### Model Config
- `LoadSAM3DModel` returns `SAM3DModelConfig` (simple data class)
- Contains: config_path, compile, use_gpu_cache, depth_backend
- Serialized automatically across IPC boundary

### Error Handling
- Worker exceptions propagate back via IPC
- Check ComfyUI console for `[SAM3DObjects]` prefixed messages

## Common Tasks

### Add a new @isolated node
1. Create `nodes/your_node.py` with @isolated decorator
2. Import and register in `__init__.py`
3. Method body imports from `worker/` and runs GPU code

### Modify decode output
- Edit `nodes/worker/stages.py` → `run_decode_lazy()`
- Mesh path: ~line 723
- Gaussian path: ~line 685

### Change model loading
- Edit `nodes/worker/lazy_manager.py`
- Models registered in `_parse_config()`

### Debug subprocess
- Worker logs to stderr with `print(f"...", file=sys.stderr)`
- Check ComfyUI console for output
- Generated worker code in `_generated_sam3dobjects/__main__.py`
