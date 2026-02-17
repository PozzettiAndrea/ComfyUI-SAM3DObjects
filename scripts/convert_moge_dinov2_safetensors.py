#!/usr/bin/env python3
"""Convert MoGe and DINOv2 checkpoints to safetensors and upload to HuggingFace.

Usage:
    python convert_moge_dinov2_safetensors.py                     # Convert only (local)
    python convert_moge_dinov2_safetensors.py --upload             # Convert + upload to HF
    python convert_moge_dinov2_safetensors.py --upload --token FILE # Upload with token from file

Converts:
  - moge-vitl/model.pt                  → moge_vitl.safetensors  + moge_vitl_config.json
  - dinov2_vitl14_reg4_pretrain.pth      → dinov2_vitl14_reg.safetensors

Both are uploaded to the existing apozz/sam-3d-objects-safetensors repo.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file


TARGET_REPO = "apozz/sam-3d-objects-safetensors"

# Local paths in the ComfyUI models folder
MODELS_DIR = Path("/home/shadeform/sam3dobjects/ComfyUI/models/sam3dobjects")

CONVERSIONS = [
    (MODELS_DIR / "moge-vitl" / "model.pt", "moge_vitl.safetensors"),
    (MODELS_DIR / "dinov2" / "dinov2_vitl14_reg4_pretrain.pth", "dinov2_vitl14_reg.safetensors"),
]


def convert_one(src_path: Path, dst_path: Path):
    """Load a checkpoint and save as safetensors."""
    print(f"Loading {src_path}...")
    data = torch.load(str(src_path), map_location="cpu", weights_only=True)

    # Extract state_dict from various checkpoint formats
    if isinstance(data, dict):
        # Save model_config as sibling JSON if present (e.g. MoGe)
        if "model_config" in data:
            config_path = dst_path.with_name(dst_path.stem + "_config.json")
            with open(config_path, "w") as f:
                json.dump(data["model_config"], f, indent=2)
            print(f"  Saved config: {config_path}")

        if "state_dict" in data:
            state_dict = data["state_dict"]
        elif "model" in data:
            state_dict = data["model"]
        elif "model_state_dict" in data:
            state_dict = data["model_state_dict"]
        else:
            # Assume it's already a flat state dict
            state_dict = data
    else:
        sys.exit(f"ERROR: unexpected format in {src_path}: {type(data)}")

    # Clean: only tensors, must be contiguous, clone to break shared memory
    clean = {}
    skipped = []
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            clean[k] = v.contiguous().clone()
        else:
            skipped.append(k)
    if skipped:
        print(f"  Skipped {len(skipped)} non-tensor keys: {skipped[:5]}...")

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving {dst_path} ({len(clean)} tensors)...")
    save_file(clean, str(dst_path))
    print(f"  Size: {dst_path.stat().st_size / 1e6:.1f} MB")


def convert(output_dir: Path):
    """Convert local checkpoints to safetensors."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for src_path, dst_name in CONVERSIONS:
        if not src_path.exists():
            sys.exit(f"ERROR: source not found: {src_path}")
        convert_one(src_path, output_dir / dst_name)

    print("Conversion complete!")
    return output_dir


def upload(output_dir: Path, token: str = None):
    """Upload converted files to HuggingFace."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)

    print(f"Uploading to {TARGET_REPO}...")
    for f in output_dir.iterdir():
        if f.suffix in (".safetensors", ".json"):
            print(f"  Uploading {f.name}...")
            api.upload_file(
                path_or_fileobj=str(f),
                path_in_repo=f.name,
                repo_id=TARGET_REPO,
                repo_type="model",
            )
    print(f"Upload complete: https://huggingface.co/{TARGET_REPO}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/sam3dobjects_moge_dinov2"))
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--token", type=str, default=None, help="HF token or path to token file")
    args = parser.parse_args()

    convert(args.output_dir)

    if args.upload:
        token = args.token
        if token and Path(token).is_file():
            token = Path(token).read_text().strip()
        upload(args.output_dir, token=token)


if __name__ == "__main__":
    main()
