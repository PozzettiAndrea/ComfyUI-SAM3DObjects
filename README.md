# ComfyUI-SAM3DObjects

## Installation

Three options, in order of speed → reliability:

1. **ComfyUI Manager (recommended)** — search for `SAM3DObjects` in the Manager and click Install from the highest version displayed. If that doesn't work, try nightly.
2. **Manager via Git URL** — in ComfyUI Manager: "Install via Git URL" with `https://github.com/PozzettiAndrea/ComfyUI-SAM3DObjects.git`.
3. **Manual (most reliable)**:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/PozzettiAndrea/ComfyUI-SAM3DObjects.git
   cd ComfyUI-SAM3DObjects
   pip install -r requirements.txt --upgrade
   python install.py
   ```

> **Please report any problems** you hit during installation or use of my nodes — open a [Discussion](https://github.com/PozzettiAndrea/ComfyUI-SAM3DObjects/discussions) or [Issue](https://github.com/PozzettiAndrea/ComfyUI-SAM3DObjects/issues). Very grateful for your help! 🙏

---


<div align="center">
<a href="https://pozzettiandrea.github.io/ComfyUI-SAM3DObjects/">
<img src="https://pozzettiandrea.github.io/ComfyUI-SAM3DObjects/gallery-preview.png" alt="Workflow Test Gallery" width="800">
</a>
<br>
<b><a href="https://pozzettiandrea.github.io/ComfyUI-SAM3DObjects/">View Live Test Gallery →</a></b>
</div>

https://github.com/user-attachments/assets/f8580108-8b59-4938-9339-b3ef8b72039f

ComfyUI custom nodes for generating 3D objects from single images using [SAM 3D Objects](https://github.com/bennyguo/sam-3d-objects).



## Community

Questions or feature requests? Open a [Discussion](https://github.com/PozzettiAndrea/ComfyUI-SAM3DObjects/discussions) on GitHub.

Join the [Comfy3D Discord](https://discord.gg/bcdQCUjnHE) for help, updates, and chat about 3D workflows in ComfyUI.

## License

This project follows the same license as SAM 3D Objects.

## Acknowledgments

- [SAM 3D Objects](https://github.com/bennyguo/sam-3d-objects) by Benny Guo et al.
- ComfyUI community
- All contributors

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
