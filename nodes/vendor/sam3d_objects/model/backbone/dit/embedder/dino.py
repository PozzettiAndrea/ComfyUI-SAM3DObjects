# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch
from typing import Optional, Dict, Any
import warnings
import time
import os
from pathlib import Path
from torchvision.transforms import Normalize
import torch.nn.functional as F
from loguru import logger


def _get_comfyui_dinov2_path() -> Optional[Path]:
    """Check if DINOv2 is downloaded to ComfyUI models folder."""
    try:
        import folder_paths
        dinov2_path = Path(folder_paths.models_dir) / "sam3dobjects" / "dinov2"
        if (dinov2_path / "hubconf.py").exists():
            return dinov2_path
    except ImportError:
        pass
    return None


class Dino(torch.nn.Module):
    def __init__(
        self,
        input_size: int = 224,
        repo_or_dir: str = "facebookresearch/dinov2",
        dino_model: str = "dinov2_vitb14",
        source: str = "github",
        backbone_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        # for backward compatible
        prenorm_features: bool = False,
        freeze_backbone: bool = True,
        prune_network: bool = False,  # False for backward compatible
    ):
        super().__init__()
        if backbone_kwargs is None:
            backbone_kwargs = {}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Check for local ComfyUI copy first (avoids network issues in subprocesses)
            local_path = _get_comfyui_dinov2_path()
            if local_path is not None:
                repo_or_dir = str(local_path)
                source = "local"
                logger.info(f"Loading DINO model: {dino_model} from local path {local_path}")

                # Pass local weights path to avoid network download (SSL issues in isolated envs)
                # dino_model is like "dinov2_vitl14_reg", weights file is "dinov2_vitl14_reg4_pretrain.pth"
                weights_filename = dino_model.replace("_reg", "_reg4") + "_pretrain.pth"
                local_weights = local_path / weights_filename
                if local_weights.exists():
                    backbone_kwargs['weights'] = str(local_weights)
                    logger.info(f"Using local weights: {local_weights}")
            else:
                logger.info(f"Loading DINO model: {dino_model} from {repo_or_dir} (source: {source})")

            if backbone_kwargs:
                logger.info(f"DINO backbone kwargs: {backbone_kwargs}")

            # Retry logic for torch.hub.load() to handle transient network errors
            max_retries = 3
            retry_delay = 2  # seconds
            last_error = None

            for attempt in range(max_retries):
                try:
                    self.backbone = torch.hub.load(
                        repo_or_dir=repo_or_dir,
                        model=dino_model,
                        source=source,
                        verbose=False,
                        **backbone_kwargs,
                    )
                    break  # Success, exit retry loop
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"Failed to load DINO model (attempt {attempt + 1}/{max_retries}): {e}")
                        logger.info(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Failed to load DINO model after {max_retries} attempts")
                        raise last_error

            # Log model properties after loading
            logger.info(f"Loaded DINO model - type: {type(self.backbone)}, "
                        f"embed_dim: {self.backbone.embed_dim}, "
                        f"patch_size: {getattr(self.backbone.patch_embed, 'patch_size', 'N/A')}")


        self.resize_input_size = (input_size, input_size)
        self.embed_dim = self.backbone.embed_dim
        self.input_size = input_size
        self.input_channels = 3
        self.normalize_images = normalize_images
        self.prenorm_features = prenorm_features
        self.register_buffer('mean', torch.as_tensor([[0.485, 0.456, 0.406]]).view(-1, 1, 1), persistent=False)
        self.register_buffer('std', torch.as_tensor([[0.229, 0.224, 0.225]]).view(-1, 1, 1), persistent=False)

        # freeze
        if freeze_backbone:
            self.requires_grad_(False)
            self.eval()
        elif not prune_network:
            logger.warning(
                "Unfreeze encoder w/o prune parameter may lead to error in ddp/fp16 training"
            )

        if prune_network:
            self._prune_network()

    def _preprocess_input(self, x):
        _resized_images = torch.nn.functional.interpolate(
            x,
            size=self.resize_input_size,
            mode="bilinear",
            align_corners=False,
        )

        if x.shape[1] == 1:
            _resized_images = _resized_images.repeat(1, 3, 1, 1)

        if self.normalize_images:
            _resized_images = _resized_images.sub_(self.mean.to(_resized_images.device)).div_(self.std.to(_resized_images.device))

        return _resized_images

    def _forward_intermediate_layers(
        self, input_img, intermediate_layers, cls_token=True
    ):
        return self.backbone.get_intermediate_layers(
            input_img,
            intermediate_layers,
            return_class_token=cls_token,
        )

    def _forward_last_layer(self, input_img):
        # Move backbone's orphan params (cls_token, pos_embed, etc.) to input
        # device — needed for ComfyUI lowvram where forward_features() is called
        # directly (bypassing __call__ hooks).
        device = input_img.device
        for p in self.backbone.parameters(recurse=False):
            if p.data.device != device:
                p.data = p.data.to(device)
        for b in self.backbone.buffers(recurse=False):
            if b.device != device:
                b.data = b.data.to(device)
        output = self.backbone.forward_features(input_img)
        if self.prenorm_features:
            features = output["x_prenorm"]
            tokens = F.layer_norm(features, features.shape[-1:])
        else:
            tokens = torch.cat(
                [
                    output["x_norm_clstoken"].unsqueeze(1),
                    output["x_norm_patchtokens"],
                ],
                dim=1,
            )
        return tokens

    def forward(self, x, **kwargs):
        _resized_images = self._preprocess_input(x)
        tokens = self._forward_last_layer(_resized_images)
        return tokens.to(x.dtype)

    def _prune_network(self):
        """
        Ran this script:
        out = model(input)
        loss = out.sum()
        loss.backward()

        for name, p in dino_model.named_parameters():
            if p.grad is None:
                print(name)
        model.zero_grad()
        """
        self.backbone.mask_token = None
        if self.prenorm_features:
            self.backbone.norm = torch.nn.Identity()


class DinoForMasks(torch.nn.Module):
    def __init__(
        self,
        backbone: Dino,
    ):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = self.backbone.embed_dim

    def forward(self, image, mask):
        return self.backbone.forward(mask)
