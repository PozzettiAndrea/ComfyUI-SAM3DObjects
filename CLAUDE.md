# ComfyUI-SAM3DObjects

ComfyUI custom nodes for SAM3D (Single-view 3D Objects) - generates 3D meshes and Gaussian splats from single images with masks.

## Architecture

```
ComfyUI Process                    Isolated Worker Process
┌────────────────────┐            ┌─────────────────────────┐
│  nodes/*.py        │◄──IPC────►│  worker/               │
│  (UI, scheduling)  │  pickle   │  (GPU inference)       │
│                    │  base64   │                         │
│  subprocess_bridge │           │  lazy_manager.py       │
│  ────────────────► │           │  stages.py             │
│  get_bridge()      │           │  inference.py          │
└────────────────────┘            └─────────────────────────┘
```

**Process Isolation**: The worker runs in a separate Python environment (`_env_sam3dobjects/`) with specific CUDA/PyTorch versions. This is managed by `comfyui-isolation` package and configured in `comfyui_isolation_reqs.toml`.

## Directory Structure

```
ComfyUI-SAM3DObjects/
├── nodes/                    # ComfyUI node definitions
│   ├── load_model.py         # LoadSAM3DModel
│   ├── depth_estimate.py     # SAM3D_DepthEstimate (MoGe)
│   ├── generate_slat.py      # SAM3DGenerateSLAT (Stage1+2)
│   ├── gaussian_decode.py    # SAM3DGaussianDecode
│   ├── mesh_decode.py        # SAM3DMeshDecode
│   ├── postprocess.py        # SAM3DTextureBake
│   ├── scene_generate.py     # SAM3DSceneGenerate (batch)
│   ├── scene_pose_optimize.py # SAM3D_ScenePoseOptimize
│   ├── pose_optimization.py  # SAM3D_PoseOptimization
│   └── subprocess_bridge.py  # IPC bridge to worker
├── worker/                   # Isolated inference code
│   ├── __main__.py           # Worker entry point
│   ├── lazy_manager.py       # On-demand model loading
│   ├── stages.py             # Stage1, Stage2, Decode implementations
│   ├── inference.py          # High-level inference API
│   ├── scene_batch.py        # Batch scene processing
│   ├── pose_optimization.py  # ICP + render optimization
│   └── depth.py              # MoGe depth estimation
├── vendor/                   # Vendored dependencies
│   ├── sam3d_objects/        # Core SAM3D library
│   ├── moge/                 # MoGe depth estimation
│   └── cv2/                  # OpenCV shim (DLL fix)
├── comfyui_isolation_reqs.toml  # Isolated env config
└── _env_sam3dobjects/        # Isolated Python venv (auto-created)
```

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

**Key**: All decode outputs are in **world coordinates** by default. The pose (rotation, translation, scale) from Stage1 is applied during decode, not stored separately.

## Key Files for Modifications

### Adding/Modifying Nodes
- `nodes/*.py` - Node class definitions (INPUT_TYPES, FUNCTION, etc.)
- `__init__.py` - Node registration (NODE_CLASS_MAPPINGS)

### Modifying Inference
- `worker/stages.py` - Core decode logic, pose application
- `worker/lazy_manager.py` - Model loading/unloading
- `worker/inference.py` - High-level inference orchestration

### IPC Communication
- `nodes/subprocess_bridge.py` - Serialization, bridge management
- `worker/__main__.py` - Command dispatch

## Considerations

### Process Isolation
- Worker has different Python/CUDA than ComfyUI
- All data crosses process boundary via pickle/base64
- Large tensors (images, pointmaps) serialized as numpy arrays
- `get_bridge()` returns singleton WorkerBridge

### VRAM Management
- `LazyModelManager` loads models on-demand
- Models can be unloaded after use (`unload_after=True`)
- Batch processing loads each model type once per phase
- ~16GB VRAM minimum (24GB+ recommended)

### Caching
- SLAT generation caches by seed/steps/cfg hash
- Scene pose optimization caches by mode/object count
- Check `_check_*_cache()` methods before re-running

### Pose Application
- Stage1 computes pose (quaternion wxyz, translation, scale)
- Decode applies pose to vertices/Gaussians before export
- `_apply_pose_to_vertices()` - mesh transform
- `_apply_pose_to_gaussian()` - Gaussian transform (positions + rotation composition + log-scale)

### Error Handling
- Worker errors return `{"status": "error", "error": str, "traceback": str}`
- Node exceptions propagate to ComfyUI
- Check worker stderr for detailed logs

## Common Tasks

### Add a new node
1. Create `nodes/your_node.py` with class definition
2. Import and register in `__init__.py`
3. If needs GPU: add worker command in `worker/__main__.py`

### Modify decode output
- Edit `worker/stages.py` → `run_decode_lazy()`
- Mesh path: ~line 723
- Gaussian path: ~line 685

### Change model loading
- Edit `worker/lazy_manager.py`
- Models registered in `_parse_config()`

### Debug worker
- Worker logs to stderr
- Add `print(f"[Worker] ...", file=sys.stderr)`
- Check ComfyUI console for `[SAM3DObjects]` prefixed messages
