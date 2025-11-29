
# ComfyUI

https://github.com/user-attachments/assets/f8580108-8b59-4938-9339-b3ef8b72039f

-SAM3DObjects

ComfyUI custom nodes for generating 3D objects from single images using [SAM 3D Objects](https://github.com/bennyguo/sam-3d-objects).

## Features

- Generate 3D Gaussian Splats and meshes from single images
- Support for both image+mask and RGBA input formats
- Export to PLY, OBJ, and GLB formats
- Integrated 3D visualization within ComfyUI
- Automatic model checkpoint management
- Batch processing support

## Requirements

### Hardware
- **Recommended**: NVIDIA RTX 30xx/40xx or A100/H100 with **32GB+ VRAM** (supports bfloat16 precision)
- **Minimum**: NVIDIA RTX 30xx with **24GB VRAM**
- **Older GPUs**: RTX 20xx/GTX 10xx supported with **automatic precision fallback** to float16
- CUDA 12.1 or compatible
- The node automatically detects your GPU capabilities and selects optimal precision

### Software
- ComfyUI (with modern `comfy_api.latest` support)
- Python 3.10+
- PyTorch with CUDA support
- All dependencies installed automatically!

## Installation

### Quick Install (One Command!)

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/yourusername/ComfyUI-SAM3DObjects.git
cd ComfyUI-SAM3DObjects
python install.py
```

**That's it!** The node is now **completely standalone** - no manual repository cloning or directory setup required!

### What the installer does:

1. Auto-detects and uses conda/mamba/micromamba if available (fastest for pytorch3d)
2. Installs sam3d_objects package automatically from GitHub
3. Installs pytorch3d via conda (much faster than pip)
4. Falls back to pip installation if needed
5. Installs all other dependencies

### Restart ComfyUI

Restart ComfyUI to load the new nodes. You should see them under the "SAM3DObjects" category.

## Nodes

### Core Nodes

#### LoadSAM3DModel
Load the SAM 3D Objects inference pipeline.

**Inputs:**
- `model_tag`: Model variant ("hf" for HuggingFace release)
- `compile`: Enable torch.compile for faster inference (slower first run)
- `force_reload`: Force reload even if cached
- `dtype`: Model precision - "bfloat16" (default, RTX 30xx+), "float16" (older GPUs), "float32" (most compatible), or "auto" (automatic GPU detection)
- `keep_model_loaded`: Keep model in GPU memory between inferences (default: True). Disable to free VRAM after each inference

**Outputs:**
- `model`: SAM3D inference pipeline

**Notes:**
- Checkpoints are automatically downloaded to `ComfyUI/models/sam3d/`
- Models are cached globally to avoid reloading
- Automatic GPU capability detection ensures compatibility with all NVIDIA GPUs

#### SAM3DGenerate
Generate 3D object from image and mask.

**Inputs:**
- `model`: SAM3D model from LoadSAM3DModel
- `image`: Input image (RGB)
- `mask`: Binary mask indicating object region
- `seed`: Random seed for reproducibility
- `stage1_inference_steps`: Denoising steps for Stage 1 (sparse structure, default: 25)
- `stage2_inference_steps`: Denoising steps for Stage 2 (SLAT generation, default: 25)
- `stage1_cfg_strength`: CFG strength for Stage 1 (default: 7.0, higher = stronger adherence to input)
- `stage2_cfg_strength`: CFG strength for Stage 2 (default: 5.0)
- `texture_size`: Texture resolution - 512/1024/2048/4096 (default: 1024)
- `simplify`: Mesh simplification ratio 0.0-1.0 (default: 0.95, keep 95% of faces)

**Outputs:**
- `glb_filepath`: Path to the generated GLB mesh file (textured 3D mesh)
- `ply_filepath`: Path to the generated Gaussian PLY file (colored point cloud)
- `pose_data`: Object pose information (rotation, translation, scale)

#### SAM3DGenerateRGBA
Generate 3D object from RGBA image (alpha as mask).

**Inputs:**
- `model`: SAM3D model from LoadSAM3DModel
- `rgba_image`: RGBA input image
- `seed`: Random seed
- `alpha_threshold`: Threshold for converting alpha to binary mask (default: 0.5)

**Outputs:**
- Same as SAM3DGenerate

### Export Nodes

#### SAM3DExportPLY
Export Gaussian Splat to PLY file.

**Inputs:**
- `gaussian_splat`: From SAM3DGenerate
- `filename`: Output filename (without extension)
- `output_dir`: Output directory (default: `ComfyUI/output/sam3d/`)

**Outputs:**
- `filepath`: Path to saved PLY file

#### SAM3DExportPLYBatch
Batch export with automatic numbering.

**Inputs:**
- `gaussian_splat`: From SAM3DGenerate
- `prefix`: Filename prefix
- `index`: File index number (zero-padded)
- `output_dir`: Output directory

**Outputs:**
- `filepath`: Path to saved PLY file

#### SAM3DExportMesh
Export mesh to various formats.

**Inputs:**
- `output_data`: Mesh output from SAM3DGenerate
- `format`: Output format ("obj", "glb", "ply_mesh")
- `filename`: Output filename
- `output_dir`: Output directory
- `resolution`: Mesh resolution (128-2048)

**Outputs:**
- `filepath`: Path to saved mesh file

**Note:** Mesh conversion is currently a placeholder - implementation depends on SAM3D's mesh export capabilities.

#### SAM3DExtractMesh
Extract mesh data without saving to file.

**Inputs:**
- `output_data`: From SAM3DGenerate
- `resolution`: Mesh resolution

**Outputs:**
- `mesh`: Extracted mesh data

### Visualization Nodes

#### SAM3DVisualizer
Render multiple views of the 3D object.

**Inputs:**
- `gaussian_splat`: From SAM3DGenerate
- `num_views`: Number of views (1-16)
- `resolution`: Output resolution (256-2048)
- `elevation`: Camera elevation angle (-90 to 90)
- `distance`: Camera distance (0.5-10.0)

**Outputs:**
- Batch of rendered images (ComfyUI IMAGE format)

#### SAM3DRenderSingle
Render single view with full camera control.

**Inputs:**
- `gaussian_splat`: From SAM3DGenerate
- `resolution`: Output resolution
- `azimuth`: Horizontal rotation angle
- `elevation`: Vertical angle
- `distance`: Camera distance

**Outputs:**
- Single rendered image

**Note:** Rendering is currently a placeholder - implementation depends on SAM3D's rendering API.

## Example Workflows

### Basic Workflow

1. **Load Image** → Image with object
2. **Create Mask** → Binary mask of object
3. **LoadSAM3DModel** → Load the model
4. **SAM3DGenerate** → Generate 3D object
5. **SAM3DExportPLY** → Save as PLY file
6. **SAM3DVisualizer** → Preview in ComfyUI

### RGBA Workflow

1. **Load RGBA Image** → Image with alpha channel
2. **LoadSAM3DModel** → Load the model
3. **SAM3DGenerateRGBA** → Generate 3D object (alpha as mask)
4. **SAM3DExportPLY** → Save as PLY file

### Batch Processing

1. **Load Images** → Multiple images
2. **Create Masks** → Corresponding masks
3. **LoadSAM3DModel** → Load model once
4. **Loop:**
   - **SAM3DGenerate** → Generate 3D for each image
   - **SAM3DExportPLYBatch** → Export with auto-numbering

## Output Formats

### PLY (Point Cloud)
Gaussian Splat representation, best for viewing in 3D viewers that support point clouds.

**Recommended Viewers:**
- CloudCompare
- MeshLab
- Blender (with PLY import)

### OBJ (Mesh)
Standard mesh format, widely supported.

**Use Cases:**
- Import into 3D modeling software
- Further editing and processing

### GLB (Binary glTF)
Modern 3D format with good compression.

**Use Cases:**
- Web-based 3D viewers
- AR/VR applications
- Game engines

## Directory Structure

```
ComfyUI-SAM3DObjects/
├── __init__.py              # Extension entry point
├── install.py               # Installation script
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── nodes/
│   ├── load_model.py       # LoadSAM3DModel node
│   ├── generate.py         # Generation nodes
│   ├── export_ply.py       # PLY export nodes
│   ├── export_mesh.py      # Mesh export nodes
│   ├── visualizer.py       # Visualization nodes
│   └── utils.py            # Utility functions
├── workflows/               # Example workflows (empty for now)
└── assets/                  # Assets and examples (empty for now)
```

## Model Checkpoints

Checkpoints are stored in: `ComfyUI/models/sam3d/{model_tag}/`

On first use, the LoadSAM3DModel node will automatically download checkpoints from HuggingFace.

### Manual Download

If automatic download fails, you can manually download:

```bash
cd ComfyUI/models/sam3d
# Download checkpoints here
# Ensure pipeline.yaml is present in the model directory
```

## Troubleshooting

### "CUDA out of memory"
- SAM3D requires 32GB+ VRAM
- Try reducing batch size
- Close other GPU applications
- Consider using a machine with more VRAM

### "Failed to load model"
- Check that `pipeline.yaml` exists in the checkpoint directory
- Verify all dependencies are installed
- Check CUDA compatibility

### "Rendering shows blank images"
- Rendering implementation is currently a placeholder
- Check SAM3D documentation for proper rendering API
- Implementation may need to be updated based on SAM3D version

## Known Limitations

1. **Mesh Export**: Mesh conversion from Gaussian Splats is not yet fully implemented
2. **Rendering**: Visualization nodes use placeholder implementation
3. **VRAM**: High memory requirements may limit usability

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project follows the same license as SAM 3D Objects.

## Acknowledgments

- [SAM 3D Objects](https://github.com/bennyguo/sam-3d-objects) by Benny Guo et al.
- ComfyUI community
- All contributors

## Support

For issues and questions:
- Open an issue on GitHub
- Check SAM 3D Objects documentation
- Visit ComfyUI community forums

## Attribution

This package includes a vendored copy of **SAM 3D Objects** from Meta AI Research:

- **Original Repository**: https://github.com/facebookresearch/sam-3d-objects
- **License**: SAM License (see `vendor/SAM3D_LICENSE`)
- **Authors**: Meta AI Research
- **Paper**: [SAM 3D Objects: Generating 3D Objects from Single Images](https://github.com/facebookresearch/sam-3d-objects)

The vendored code is located in `vendor/sam3d_objects/` and is redistributed under the terms of the SAM License, which permits:
- Use, reproduction, distribution, and modification
- Creating derivative works
- Redistribution under the same license terms

We gratefully acknowledge Meta AI Research for making SAM 3D Objects available to the research community.

## Version History

### 1.0.0 (Initial Release)
- LoadSAM3DModel node with auto-download
- SAM3DGenerate and SAM3DGenerateRGBA nodes
- Export nodes for PLY and mesh formats
- Visualization nodes (placeholder implementation)
- Batch processing support
- **Vendored sam3d_objects code** - completely standalone!
