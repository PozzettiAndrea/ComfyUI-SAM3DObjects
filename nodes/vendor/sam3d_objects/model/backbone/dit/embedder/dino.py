# Copyright (c) Meta Platforms, Inc. and affiliates.
import importlib
import torch
from typing import Optional, Dict, Any
from pathlib import Path
import torch.nn.functional as F
from loguru import logger


def _wrap_attn_comfy(module):
    """Class-swap a DINOv2 Attention module to use comfy-attn dispatch (sage/flash/sdpa)."""
    import torch
    from comfy_attn import dispatch_attention

    class _W(module.__class__):
        def forward(self, x, **kwargs):
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
            q, k, v = torch.unbind(qkv, 2)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            out = dispatch_attention(q, k, v)
            out = out.transpose(1, 2).contiguous().view(B, N, C)
            return self.proj_drop(self.proj(out))

    module.__class__ = _W



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

        import comfy.utils
        import folder_paths

        # Build model architecture from vendored MoGe dinov2 code (no network needed)
        factory = getattr(
            importlib.import_module("moge.model.dinov2.hub.backbones"),
            dino_model,
        )
        self.backbone = factory(pretrained=False)

        # Load weights from safetensors in ComfyUI models folder
        weights_path = (
            Path(folder_paths.models_dir)
            / "sam3dobjects"
            / f"{dino_model}.safetensors"
        )
        state_dict = comfy.utils.load_torch_file(str(weights_path))
        self.backbone.load_state_dict(state_dict, strict=True)
        logger.info(f"Loaded DINOv2 from {weights_path.name} "
                     f"(embed_dim={self.backbone.embed_dim})")

        # Route attention through comfy-attn (sage/flash/sdpa)
        for block in self.backbone.blocks:
            _wrap_attn_comfy(block.attn)
        from comfy_attn import set_backend, get_backend_label
        set_backend("auto")
        logger.info(f"DINOv2 attention: {get_backend_label()} ({len(self.backbone.blocks)} blocks)")

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
