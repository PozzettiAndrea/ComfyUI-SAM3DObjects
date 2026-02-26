"""
Pipeline stages for SAM3D inference.

This module contains all pipeline stages:
- Stage 1: Sparse structure generation
- Stage 2: SLAT generation
- Stage 3: Gaussian/Mesh decoding
- Texture baking

Each function loads its own models directly - no shared state.
"""

import gc
import importlib
import logging
import sys
import base64
import pickle
import yaml
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger("sam3dobjects")

import numpy as np
import torch
import comfy.model_management
import comfy.model_patcher
import comfy.utils

from .helpers import preprocess_image_lazy, save_output_to_disk
from .vram_log import vram


# Inline decoder configs — used as fallback when YAML files are missing.
# These are small, static configs that never change between releases.
_DEFAULT_DECODER_CONFIGS = {
    "slat_decoder_gs": {
        "_target_": "sam3d_objects.model.backbone.tdfy_dit.models.structured_latent_vae.decoder_gs.SLatGaussianDecoderTdfyWrapper",
        "resolution": 64,
        "model_channels": 768,
        "latent_channels": 8,
        "num_blocks": 12,
        "num_heads": 12,
        "mlp_ratio": 4,
        "attn_mode": "swin",
        "window_size": 8,
        "representation_config": {
            "lr": {"_xyz": 1.0, "_features_dc": 1.0, "_opacity": 1.0, "_scaling": 1.0, "_rotation": 0.1},
            "perturb_offset": True,
            "voxel_size": 1.5,
            "num_gaussians": 32,
            "2d_filter_kernel_size": 0.1,
            "3d_filter_kernel_size": 0.0009,
            "scaling_bias": 0.004,
            "opacity_bias": 0.1,
            "scaling_activation": "softplus",
        },
        "use_fp16": True,
    },
    "slat_decoder_gs_4": {
        "_target_": "sam3d_objects.model.backbone.tdfy_dit.models.structured_latent_vae.decoder_gs.SLatGaussianDecoderTdfyWrapper",
        "resolution": 64,
        "model_channels": 768,
        "latent_channels": 8,
        "num_blocks": 12,
        "num_heads": 12,
        "mlp_ratio": 4,
        "attn_mode": "swin",
        "window_size": 8,
        "representation_config": {
            "lr": {"_xyz": 1.0, "_features_dc": 1.0, "_opacity": 1.0, "_scaling": 1.0, "_rotation": 0.1},
            "perturb_offset": True,
            "voxel_size": 1.5,
            "num_gaussians": 4,
            "2d_filter_kernel_size": 0.1,
            "3d_filter_kernel_size": 0.0009,
            "scaling_bias": 0.004,
            "opacity_bias": 0.1,
            "scaling_activation": "softplus",
        },
        "use_fp16": True,
    },
    "slat_decoder_mesh": {
        "_target_": "sam3d_objects.model.backbone.tdfy_dit.models.structured_latent_vae.decoder_mesh.SLatMeshDecoderTdfyWrapper",
        "resolution": 64,
        "model_channels": 768,
        "latent_channels": 8,
        "num_blocks": 12,
        "num_heads": 12,
        "mlp_ratio": 4,
        "attn_mode": "swin",
        "window_size": 8,
        "representation_config": {"use_color": True},
        "use_fp16": True,
    },
}


# =============================================================================
# Hydra Replacement: Config Loading + Recursive Instantiation
# =============================================================================

class _DotDict(dict):
    """Dict with attribute access for OmegaConf compatibility."""
    def __getattr__(self, name):
        try:
            val = self[name]
        except KeyError:
            raise AttributeError(name)
        if isinstance(val, dict) and not isinstance(val, _DotDict):
            val = _DotDict(val)
            self[name] = val
        return val


# Registry: Hydra _target_ string -> (relative_module, class_name)
# All modules use relative imports from this package (nodes.utils).
_NATIVE_CLASS_MAP = {
    # Flow models
    'sam3d_objects.model.backbone.tdfy_dit.models.sparse_structure_flow.SparseStructureFlowModel':
        ('..sam3d.model', 'SparseStructureFlowModel'),
    'sam3d_objects.model.backbone.tdfy_dit.models.mot_sparse_structure_flow.MOTSparseStructureFlowModel':
        ('..sam3d.model', 'MOTSparseStructureFlowModel'),
    'sam3d_objects.model.backbone.tdfy_dit.models.mot_sparse_structure_flow.SparseStructureFlowTdfyWrapper':
        ('..sam3d.model', 'MOTSparseStructureFlowTdfyWrapper'),
    'sam3d_objects.model.backbone.tdfy_dit.models.structured_latent_flow.SLatFlowModel':
        ('..sam3d.model', 'SLatFlowModel'),
    'sam3d_objects.model.backbone.tdfy_dit.models.structured_latent_flow.SLatFlowModelTdfyWrapper':
        ('..sam3d.model', 'SLatFlowModelTdfyWrapper'),
    # VAE
    'sam3d_objects.model.backbone.tdfy_dit.models.sparse_structure_vae.SparseStructureDecoderTdfyWrapper':
        ('..sam3d.vae', 'SparseStructureDecoderTdfyWrapper'),
    'sam3d_objects.model.backbone.tdfy_dit.models.sparse_structure_vae.SparseStructureEncoderTdfyWrapper':
        ('..sam3d.vae', 'SparseStructureEncoderTdfyWrapper'),
    'sam3d_objects.model.backbone.tdfy_dit.models.structured_latent_vae.decoder_gs.SLatGaussianDecoderTdfyWrapper':
        ('..sam3d.vae', 'SLatGaussianDecoderTdfyWrapper'),
    'sam3d_objects.model.backbone.tdfy_dit.models.structured_latent_vae.decoder_mesh.SLatMeshDecoderTdfyWrapper':
        ('..sam3d.vae', 'SLatMeshDecoderTdfyWrapper'),
    'sam3d_objects.model.backbone.tdfy_dit.models.structured_latent_vae.encoder.SLatEncoderTdfyWrapper':
        ('..sam3d.vae', 'SLatEncoderTdfyWrapper'),
    # Embedders
    'sam3d_objects.model.backbone.dit.embedder.embedder_fuser.EmbedderFuser':
        ('..sam3d.model', 'EmbedderFuser'),
    'sam3d_objects.model.backbone.dit.embedder.dino.Dino':
        ('..sam3d.model', 'Dino'),
    'sam3d_objects.model.backbone.dit.embedder.dino.DinoForMasks':
        ('..sam3d.model', 'DinoForMasks'),
    'sam3d_objects.model.backbone.dit.embedder.pointmap.PointPatchEmbed':
        ('..sam3d.model', 'PointPatchEmbed'),
    # MM Latent
    'sam3d_objects.model.backbone.tdfy_dit.models.mm_latent.Latent':
        ('..sam3d.model', 'Latent'),
    'sam3d_objects.model.backbone.tdfy_dit.models.mm_latent.LearntPositionEmbedder':
        ('..sam3d.model', 'LearntPositionEmbedder'),
    'sam3d_objects.model.backbone.tdfy_dit.models.mm_latent.ShapePositionEmbedder':
        ('..sam3d.model', 'ShapePositionEmbedder'),
    # Generators
    'sam3d_objects.model.backbone.generator.shortcut.model.ShortCut':
        ('..sam3d.generators', 'ShortCut'),
    'sam3d_objects.model.backbone.generator.flow_matching.model.FlowMatching':
        ('..sam3d.generators', 'FlowMatching'),
    'sam3d_objects.model.backbone.generator.flow_matching.model.lognorm_sampler':
        ('..sam3d.generators', 'lognorm_sampler'),
    'sam3d_objects.model.backbone.generator.classifier_free_guidance.ClassifierFreeGuidance':
        ('..sam3d.generators', 'ClassifierFreeGuidance'),
    'sam3d_objects.model.backbone.generator.classifier_free_guidance.ClassifierFreeGuidanceWithExternalUnconditionalProbability':
        ('..sam3d.generators', 'ClassifierFreeGuidanceWithExternalUnconditionalProbability'),
    'sam3d_objects.model.backbone.generator.classifier_free_guidance.PointmapCFG':
        ('..sam3d.generators', 'PointmapCFG'),
    # Transforms / Preprocessor
    'sam3d_objects.data.dataset.tdfy.preprocessor.PreProcessor':
        ('..sam3d.transforms', 'PreProcessor'),
    'sam3d_objects.data.dataset.tdfy.img_and_mask_transforms.resize_all_to_same_size':
        ('..sam3d.transforms', 'resize_all_to_same_size'),
    'sam3d_objects.data.dataset.tdfy.img_and_mask_transforms.crop_around_mask_with_padding':
        ('..sam3d.transforms', 'crop_around_mask_with_padding'),
    'sam3d_objects.data.dataset.tdfy.img_and_mask_transforms.ObjectCentricSSI':
        ('..sam3d.transforms', 'ObjectCentricSSI'),
    'sam3d_objects.data.dataset.tdfy.img_processing.pad_to_square_centered':
        ('..sam3d.transforms', 'pad_to_square_centered'),
    # Pipeline
    'sam3d_objects.pipeline.depth_models.moge.MoGe':
        ('..sam3d.pipeline', 'MoGe'),
    # MoGe
    'moge.model.v1.MoGeModel.from_pretrained':
        ('..sam3d.moge', 'MoGeModel.from_pretrained'),
}

# Prefix-based fallback for any sam3d_objects targets not in the explicit map
_MODULE_REMAP = [
    ('sam3d_objects.model.backbone.tdfy_dit.models', '..sam3d.model'),
    ('sam3d_objects.model.backbone.dit.embedder', '..sam3d.model'),
    ('sam3d_objects.model.backbone.generator', '..sam3d.generators'),
    ('sam3d_objects.data.dataset.tdfy', '..sam3d.transforms'),
    ('sam3d_objects.data.utils', '..sam3d.data_utils'),
    ('sam3d_objects.pipeline', '..sam3d.pipeline'),
    ('moge.model', '..sam3d.moge'),
]


def _make_dict(**kwargs):
    """Replacement for sam3d_objects.config.utils.make_dict."""
    return dict(**kwargs)


def _resolve_class(target: str):
    """Resolve a _target_ string to a class. Uses relative imports from nodes.utils."""
    # Special case: make_dict utility
    if target == 'sam3d_objects.config.utils.make_dict':
        return _make_dict

    # Explicit map (fast path)
    if target in _NATIVE_CLASS_MAP:
        module_path, attr_chain = _NATIVE_CLASS_MAP[target]
        module = importlib.import_module(module_path, package=__package__)
        # Handle chained attrs like 'MoGeModel.from_pretrained'
        obj = module
        for attr in attr_chain.split('.'):
            obj = getattr(obj, attr)
        return obj

    # Prefix-based remapping for vendor targets not in the explicit map
    class_name = target.rsplit('.', 1)[1]
    for old_prefix, new_module in _MODULE_REMAP:
        if target.startswith(old_prefix + '.'):
            module = importlib.import_module(new_module, package=__package__)
            return getattr(module, class_name)

    # External classes (torchvision, etc.) — absolute import
    parts = target.rsplit('.', 1)
    module = importlib.import_module(parts[0])
    return getattr(module, parts[1])


def _instantiate(config):
    """Recursively instantiate a Hydra-style config dict (replaces hydra.utils.instantiate).

    Handles:
    - _target_: fully-qualified class/function path
    - _partial_: true -> return functools.partial instead of calling
    - Nested dicts with _target_ are recursively instantiated
    - Lists are recursively processed
    """
    if isinstance(config, dict):
        if '_target_' in config:
            target = config['_target_']
            cls = _resolve_class(target)
            is_partial = config.get('_partial_', False)
            kwargs = {}
            for k, v in config.items():
                if k.startswith('_'):  # Skip _target_, _recursive_, _partial_, etc.
                    continue
                kwargs[k] = _instantiate(v)
            if is_partial:
                return partial(cls, **kwargs)
            return cls(**kwargs)
        else:
            return {k: _instantiate(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [_instantiate(v) for v in config]
    else:
        return config


def _filter_and_remove_prefix(sd: dict, prefix: str) -> dict:
    """Filter state dict to keys starting with prefix, then remove the prefix."""
    n = len(prefix)
    return {k[n:]: v for k, v in sd.items() if k.startswith(prefix)}


def _remove_prefix(sd: dict, prefix: str) -> dict:
    """Remove prefix from state dict keys that have it."""
    n = len(prefix)
    return {(k[n:] if k.startswith(prefix) else k): v for k, v in sd.items()}


def _fix_meta_buffers(model: torch.nn.Module, device):
    """Reinitialize any buffers left on meta device after assign=True loading."""
    for name, buf in model.named_buffers():
        if buf.device.type == "meta":
            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            parent._buffers[parts[-1]] = torch.zeros_like(buf, device=device)


def _enable_lowvram_cast(model: torch.nn.Module) -> None:
    """Swap leaf modules to comfy.ops.disable_weight_init versions for lowvram support.

    ComfyUI's ModelPatcher can only partially-load modules that have the
    `comfy_cast_weights` attribute (from CastWeightBiasOp).  Native ComfyUI models
    use `comfy.ops.disable_weight_init.*` layers, but third-party models use plain
    `torch.nn.*`.  This function retroactively swaps __class__ on every leaf module
    so that ModelPatcher's lowvram layer-streaming works transparently.

    Also installs forward pre-hooks on non-leaf modules that own direct parameters
    or buffers (e.g. cls_token, pos_embed on ViT) so they get moved to the input
    device on-the-fly during lowvram inference.
    """
    try:
        from comfy.ops import disable_weight_init
    except ImportError:
        return

    _CLASS_MAP = {
        torch.nn.Linear: disable_weight_init.Linear,
        torch.nn.Conv1d: disable_weight_init.Conv1d,
        torch.nn.Conv2d: disable_weight_init.Conv2d,
        torch.nn.Conv3d: disable_weight_init.Conv3d,
        torch.nn.GroupNorm: disable_weight_init.GroupNorm,
        torch.nn.LayerNorm: disable_weight_init.LayerNorm,
        torch.nn.ConvTranspose2d: disable_weight_init.ConvTranspose2d,
        torch.nn.ConvTranspose1d: disable_weight_init.ConvTranspose1d,
        torch.nn.Embedding: disable_weight_init.Embedding,
    }

    cast_count = 0
    for _name, module in model.named_modules():
        comfy_cls = _CLASS_MAP.get(type(module))
        if comfy_cls is not None:
            module.__class__ = comfy_cls
            cast_count += 1

    # Install forward pre-hooks on non-leaf modules that own orphan params/buffers.
    # These are things like cls_token, pos_embed on ViT — tiny tensors that aren't
    # inside any leaf layer, so comfy_cast_weights doesn't cover them.
    hook_count = 0
    for _name, module in model.named_modules():
        if hasattr(module, 'comfy_cast_weights'):
            continue  # Leaf module — handled by cast_weights mechanism
        direct_params = list(module.named_parameters(recurse=False))
        direct_bufs = list(module.named_buffers(recurse=False))
        if not direct_params and not direct_bufs:
            continue  # No orphan tensors

        def _move_orphans_hook(mod, args, kwargs=None):
            # Infer target device from first tensor in args or kwargs
            device = None
            for a in args:
                if isinstance(a, torch.Tensor):
                    device = a.device
                    break
                # SparseTensor and similar wrapper types have a .feats tensor
                if hasattr(a, 'feats') and isinstance(a.feats, torch.Tensor):
                    device = a.feats.device
                    break
            if device is None and kwargs:
                for v in kwargs.values():
                    if isinstance(v, torch.Tensor):
                        device = v.device
                        break
                    if hasattr(v, 'feats') and isinstance(v.feats, torch.Tensor):
                        device = v.feats.device
                        break
            if device is None:
                # Last resort: check if any child already on CUDA
                for p in mod.parameters():
                    if p.device.type == 'cuda':
                        device = p.device
                        break
            if device is None:
                return
            for _, p in mod.named_parameters(recurse=False):
                if p.data.device != device:
                    p.data = p.data.to(device)
            for _, b in mod.named_buffers(recurse=False):
                if b.device != device:
                    b.data = b.data.to(device)

        module.register_forward_pre_hook(_move_orphans_hook, with_kwargs=True)
        hook_count += 1

    if cast_count or hook_count:
        log.debug("Enabled lowvram: %d cast modules, %d orphan-param hooks", cast_count, hook_count)


# =============================================================================
# Config / Model Loading Helpers
# =============================================================================

def _load_config(config_path: str):
    """Load pipeline.yaml and return config + checkpoint_dir."""
    with open(config_path, 'r') as f:
        config = _DotDict(yaml.safe_load(f))
    checkpoint_dir = Path(config_path).parent
    return config, checkpoint_dir


def _prefer_safetensors(ckpt_path: Path) -> Path:
    """Prefer .safetensors over .ckpt if available (mmap-friendly, zero-copy loading)."""
    safetensors_path = ckpt_path.with_suffix('.safetensors')
    if safetensors_path.exists():
        return safetensors_path
    return ckpt_path


def _resolve_dtype(precision: str) -> torch.dtype:
    """Convert precision string to torch dtype.

    Args:
        precision: One of "bf16", "fp16", "fp32".
    """
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return torch.float32


def _load_generator(config_path: str, generator_type: str, precision: str = "bf16"):
    """
    Load a generator model (ss_generator or slat_generator) wrapped in ModelPatcher.

    Args:
        config_path: Path to pipeline.yaml
        generator_type: 'ss' or 'slat'
        precision: "bf16", "fp16", or "fp32"

    Returns:
        comfy.model_patcher.ModelPatcher wrapping the generator model
    """

    config, checkpoint_dir = _load_config(config_path)

    gen_config_path = checkpoint_dir / config[f"{generator_type}_generator_config_path"]
    gen_ckpt_path = _prefer_safetensors(checkpoint_dir / config[f"{generator_type}_generator_ckpt_path"])

    with open(gen_config_path, 'r') as f:
        gen_config = yaml.safe_load(f)
    model_config = gen_config["module"]["generator"]["backbone"]

    offload_device = comfy.model_management.unet_offload_device()

    # Build on meta device (zero memory), then load weights directly
    vram(f"_load_generator({generator_type}): before meta build")
    with torch.device("meta"):
        model = _instantiate(model_config)

    sd = comfy.utils.load_torch_file(str(gen_ckpt_path))
    # Training checkpoints nest weights under "state_dict"; safetensors are flat
    if 'state_dict' in sd:
        sd = sd['state_dict']
    sd = _filter_and_remove_prefix(sd, "_base_models.generator.")
    model.load_state_dict(sd, strict=False, assign=True)
    _fix_meta_buffers(model, offload_device)
    vram(f"_load_generator({generator_type}): after load_state_dict")

    model.eval()
    model.requires_grad_(False)

    # Native models use manual_cast — dtype is handled by the operations tier.
    # No convert_to_fp16() needed.

    _enable_lowvram_cast(model)
    patcher = comfy.model_patcher.ModelPatcher(
        model,
        load_device=comfy.model_management.get_torch_device(),
        offload_device=offload_device,
    )

    return patcher


def _load_decoder(config_path: str, decoder_type: str, precision: str = "bf16"):
    """
    Load a decoder model (ss_decoder, slat_decoder_gs, slat_decoder_mesh) wrapped in ModelPatcher.

    Args:
        config_path: Path to pipeline.yaml
        decoder_type: 'ss', 'slat_decoder_gs', 'slat_decoder_mesh'
        precision: "bf16", "fp16", or "fp32"

    Returns:
        comfy.model_patcher.ModelPatcher wrapping the decoder model
    """

    config, checkpoint_dir = _load_config(config_path)

    if decoder_type == 'ss':
        dec_config_path = checkpoint_dir / config.ss_decoder_config_path
        dec_ckpt_path = _prefer_safetensors(checkpoint_dir / config.ss_decoder_ckpt_path)
    else:
        dec_config_path = checkpoint_dir / config[f"{decoder_type}_config_path"]
        dec_ckpt_path = _prefer_safetensors(checkpoint_dir / config[f"{decoder_type}_ckpt_path"])

    if dec_config_path.exists():
        with open(dec_config_path, 'r') as f:
            dec_config = yaml.safe_load(f)
    else:
        dec_config = _DEFAULT_DECODER_CONFIGS.get(decoder_type)
        if dec_config is None:
            raise FileNotFoundError(
                f"No config file and no inline default for {decoder_type}: {dec_config_path}"
            )
        dec_config = dict(dec_config)  # shallow copy to avoid mutating the default
        log.warning("Config file missing, using inline default: %s", dec_config_path)
    # Remove pretrained_ckpt_path if present (training artifact)
    dec_config.pop('pretrained_ckpt_path', None)

    offload_device = comfy.model_management.unet_offload_device()

    # Decoders are small (~170-364 MB) and contain utility objects (FlexiCubes,
    # SparseFeatures2Mesh) that create real data tensors in __init__, so skip
    # the meta device pattern here — not worth the complexity.
    vram(f"_load_decoder({decoder_type}): before instantiate")
    model = _instantiate(dec_config)

    sd = comfy.utils.load_torch_file(str(dec_ckpt_path))
    # Decoder checkpoints have weights at root level with "module." prefix
    sd = _remove_prefix(sd, "module.")
    model.load_state_dict(sd, strict=False, assign=True)
    vram(f"_load_decoder({decoder_type}): after load_state_dict")

    model.eval()
    model.requires_grad_(False)

    # Cast to requested precision (sageattention requires fp16/bf16)
    dtype = _resolve_dtype(precision)
    if dtype != torch.float32:
        model.to(dtype=dtype)

    # Re-init FlexiCubes / SparseFeatures2Mesh lookup tables on the target device.
    if hasattr(model, 'mesh_extractor'):
        from ..sam3d.representations import SparseFeatures2Mesh
        me = model.mesh_extractor
        if isinstance(me, SparseFeatures2Mesh):
            device = str(comfy.model_management.get_torch_device())
            model.mesh_extractor = SparseFeatures2Mesh(
                device=device, res=me.res, use_color=me.use_color,
            )
            log.info("Re-initialized mesh_extractor on %s", device)

    _enable_lowvram_cast(model)
    patcher = comfy.model_patcher.ModelPatcher(
        model,
        load_device=comfy.model_management.get_torch_device(),
        offload_device=offload_device,
    )

    return patcher


def _load_condition_embedder(config_path: str, embedder_type: str, precision: str = "bf16"):
    """
    Load a condition embedder (ss or slat) wrapped in ModelPatcher.

    Args:
        config_path: Path to pipeline.yaml
        embedder_type: 'ss' or 'slat'
        precision: "bf16", "fp16", or "fp32"

    Returns:
        comfy.model_patcher.ModelPatcher wrapping the embedder model
    """

    config, checkpoint_dir = _load_config(config_path)

    gen_config_path = checkpoint_dir / config[f"{embedder_type}_generator_config_path"]
    gen_ckpt_path = _prefer_safetensors(checkpoint_dir / config[f"{embedder_type}_generator_ckpt_path"])

    with open(gen_config_path, 'r') as f:
        gen_config = yaml.safe_load(f)
    embedder_config = gen_config["module"]["condition_embedder"]["backbone"]

    offload_device = comfy.model_management.unet_offload_device()

    # Build on meta device (zero memory), then load weights directly
    vram(f"_load_condition_embedder({embedder_type}): before meta build")
    with torch.device("meta"):
        embedder = _instantiate(embedder_config)

    sd = comfy.utils.load_torch_file(str(gen_ckpt_path))
    # Training checkpoints nest weights under "state_dict"; safetensors are flat
    if 'state_dict' in sd:
        sd = sd['state_dict']
    sd = _filter_and_remove_prefix(sd, "_base_models.condition_embedder.")
    embedder.load_state_dict(sd, strict=False, assign=True)
    _fix_meta_buffers(embedder, offload_device)
    vram(f"_load_condition_embedder({embedder_type}): after load_state_dict")

    embedder.eval()
    embedder.requires_grad_(False)

    _reinit_dino_buffers(embedder)
    _share_dino_backbones(embedder)
    _wrap_dino_attention(embedder)

    _enable_lowvram_cast(embedder)
    patcher = comfy.model_patcher.ModelPatcher(
        embedder,
        load_device=comfy.model_management.get_torch_device(),
        offload_device=offload_device,
    )

    return patcher


def _get_preprocessor(config_path: str, preprocessor_type: str):
    """Get preprocessor from config."""
    config, _ = _load_config(config_path)
    preprocessor_config = config.get(f'{preprocessor_type}_preprocessor')

    if preprocessor_config and isinstance(preprocessor_config, dict) and '_target_' in preprocessor_config:
        return _instantiate(preprocessor_config)
    else:
        from ..sam3d.pipeline import get_default_preprocessor
        return get_default_preprocessor()


def _reinit_dino_buffers(model):
    """Reinitialize Dino mean/std buffers after meta-device loading.

    Meta-device init creates these as empty tensors. They need correct
    ImageNet normalization values for proper preprocessing.
    """
    for module in model.modules():
        if type(module).__name__ == 'Dino' and hasattr(module, 'mean'):
            device = module.mean.device
            dtype = module.mean.dtype
            module.mean = torch.as_tensor([[0.485, 0.456, 0.406]]).view(-1, 1, 1).to(device=device, dtype=dtype)
            module.std = torch.as_tensor([[0.229, 0.224, 0.225]]).view(-1, 1, 1).to(device=device, dtype=dtype)


def _share_dino_backbones(model):
    """Share frozen DINOv2 backbone between Dino instances. Saves ~1.1GB VRAM.

    The EmbedderFuser has two Dino modules (one for image, one for mask)
    with identical frozen backbones loaded from the same pretrained weights.
    """
    if not hasattr(model, 'module_list'):
        return

    # Match both native and vendor Dino classes by checking for backbone attribute
    dino_modules = [
        m for m in model.module_list
        if type(m).__name__ in ('Dino', 'DinoForMasks') and hasattr(m, 'backbone')
    ]

    if len(dino_modules) > 1:
        shared = dino_modules[0].backbone
        for dino in dino_modules[1:]:
            dino.backbone = shared
        log.info("Shared DINOv2 backbone across %d Dino instances (saved ~1.1GB)", len(dino_modules))
        gc.collect()
        comfy.model_management.soft_empty_cache()


def _wrap_dino_attention(model):
    """Wrap DINOv2 attention blocks with ComfyUI's attention dispatch after meta-device loading."""
    from ..sam3d.model import _wrap_attn_comfy
    for module in model.modules():
        if type(module).__name__ == 'Dino' and hasattr(module, 'backbone'):
            for block in module.backbone.blocks:
                _wrap_attn_comfy(block.attn)
            log.info("Wrapped DINOv2 attention blocks (%d blocks)", len(module.backbone.blocks))



# =============================================================================
# Pose Transformation Helpers
# =============================================================================

def _apply_pose_to_gaussian(gaussian, pose_data: Dict, device=None):
    """
    Apply pose transformation (rotation, translation, scale) to a Gaussian object.
    """
    from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion, quaternion_multiply, Transform3d

    if device is None:
        device = comfy.model_management.get_torch_device()

    rotation = pose_data.get("rotation")
    translation = pose_data.get("translation")
    scale = pose_data.get("scale")

    if rotation is None or translation is None or scale is None:
        log.warning("Missing pose data, skipping Gaussian pose application")
        return gaussian

    # Convert to tensors
    if hasattr(rotation, 'cpu'):
        rotation = rotation.cpu()
    rotation = torch.tensor(rotation, dtype=torch.float32, device=device).squeeze()

    if hasattr(translation, 'cpu'):
        translation = translation.cpu()
    translation = torch.tensor(translation, dtype=torch.float32, device=device).squeeze()

    if hasattr(scale, 'cpu'):
        scale = scale.cpu()
    scale = torch.tensor(scale, dtype=torch.float32, device=device).squeeze()

    # Ensure correct shapes
    if rotation.dim() == 1:
        rotation = rotation.unsqueeze(0)
    if translation.dim() == 1:
        translation = translation.unsqueeze(0)
    if scale.dim() == 0:
        scale = scale.unsqueeze(0).expand(3)
    elif scale.dim() == 1 and scale.shape[0] == 1:
        scale = scale.expand(3)
    scale_val = scale.mean()

    # Build Transform3d: scale -> rotate -> translate
    rot_matrix = quaternion_to_matrix(rotation)
    tfm = (
        Transform3d(device=device)
        .scale(scale_val.expand(3)[None])
        .rotate(rot_matrix)
        .translate(translation)
    )

    # 1. Transform positions
    positions = gaussian.get_xyz
    positions_world = tfm.transform_points(positions.unsqueeze(0)).squeeze(0)
    gaussian.from_xyz(positions_world)

    # 2. Apply scale to Gaussian scaling (in log-space)
    log_scale = torch.log(scale_val).expand_as(gaussian._scaling)
    gaussian._scaling = gaussian._scaling + log_scale

    # 3. Compose rotations
    tfm_matrix = tfm.get_matrix()[0]
    rotation_matrix = tfm_matrix[:3, :3]
    scale_factors = rotation_matrix.norm(dim=0)
    pure_rotation_matrix = rotation_matrix / scale_factors[None, :]
    pose_rotation_quat = matrix_to_quaternion(pure_rotation_matrix[None])

    current_rotations = gaussian.get_rotation
    new_rotations = quaternion_multiply(pose_rotation_quat, current_rotations)
    gaussian.from_rotation(new_rotations)

    log.info(f"Applied pose to Gaussian: scale={scale_val:.4f}, trans={translation.squeeze().tolist()}")
    return gaussian


def _apply_pose_to_vertices(vertices: np.ndarray, pose_data: Dict) -> np.ndarray:
    """
    Apply pose transformation (rotation, translation, scale) to vertices.
    """
    from pytorch3d.transforms import quaternion_to_matrix

    rotation = pose_data.get("rotation")
    translation = pose_data.get("translation")
    scale = pose_data.get("scale")

    if rotation is None or translation is None or scale is None:
        log.warning("Missing pose data, skipping pose application")
        return vertices

    # Convert to numpy arrays
    if hasattr(rotation, 'cpu'):
        rotation = rotation.cpu().numpy()
    rotation = np.array(rotation).squeeze()

    if hasattr(translation, 'cpu'):
        translation = translation.cpu().numpy()
    translation = np.array(translation).squeeze()

    if hasattr(scale, 'cpu'):
        scale = scale.cpu().numpy()
    scale = np.array(scale).squeeze()

    # Convert quaternion (wxyz) to rotation matrix
    quat_tensor = torch.from_numpy(rotation.astype(np.float32)).unsqueeze(0)
    rot_matrix = quaternion_to_matrix(quat_tensor).squeeze(0).numpy()

    # Apply scale (use mean for uniform scaling)
    scale_val = scale.mean() if scale.ndim > 0 else float(scale)

    # Apply transformation: (R * S * v) + t
    vertices_transformed = (vertices @ (rot_matrix.T * scale_val)) + translation
    return vertices_transformed


# =============================================================================
# Stage 1: Sparse Structure Generation
# =============================================================================

def run_stage1(
    config_path: str,
    image,
    mask,
    pointmap,
    seed: int = 42,
    inference_steps: int = 25,
    cfg_strength: float = 7.0,
    cfg_strength_pm: float = 0.0,
    output_dir: str = None,
    precision: str = "bf16",
) -> Dict[str, Any]:
    """
    Run Stage 1 (sparse structure generation).

    Models loaded: ss_generator (~6.7 GB), ss_condition_embedder (~1.2 GB), ss_decoder (~150 MB)
    Peak VRAM: ~8-9 GB
    """
    from ..sam3d.pipeline import (
        downsample_sparse_structure,
        prune_sparse_structure,
        get_pose_decoder,
    )

    log.info("Running Stage 1 (sparse gen)")

    config, checkpoint_dir = _load_config(config_path)
    dtype = _resolve_dtype(precision)

    # Set seed
    torch.manual_seed(seed)

    # Convert image/mask to numpy
    image_np = np.array(image)
    mask_np = np.array(mask) if mask is not None else None

    # Ensure RGBA format - USE MASK AS ALPHA CHANNEL for proper cropping
    if image_np.ndim == 2:
        image_np = np.stack([image_np] * 3 + [np.full_like(image_np, 255)], axis=-1)
    elif image_np.shape[-1] == 3:
        if mask_np is not None:
            if mask_np.dtype == np.float32 or mask_np.dtype == np.float64:
                alpha = (mask_np * 255).astype(np.uint8)
            else:
                alpha = mask_np.astype(np.uint8)
            if alpha.ndim == 2:
                alpha = alpha[:, :, np.newaxis]
        else:
            alpha = np.full((image_np.shape[0], image_np.shape[1], 1), 255, dtype=np.uint8)
        image_np = np.concatenate([image_np, alpha], axis=-1)

    # Get preprocessor and preprocess image
    log.info("Preprocessing image...")
    ss_preprocessor = _get_preprocessor(config_path, 'ss')

    # Convert pointmap to tensor for preprocessing
    if isinstance(pointmap, np.ndarray):
        pointmap_tensor = torch.from_numpy(pointmap).float()
    else:
        pointmap_tensor = pointmap

    # Preprocess
    ss_input_dict = preprocess_image_lazy(image_np, mask_np, ss_preprocessor, pointmap=pointmap_tensor)

    # Save debug image
    debug_image_path = None
    if output_dir:
        try:
            from PIL import Image as PILImage
            debug_img = ss_input_dict.get("image")
            if debug_img is not None:
                if debug_img.dim() == 4:
                    debug_img = debug_img[0]
                debug_img_np = debug_img.cpu().permute(1, 2, 0).numpy()
                debug_img_np = (debug_img_np * 255).clip(0, 255).astype(np.uint8)
                debug_pil = PILImage.fromarray(debug_img_np)
                debug_image_path = str(Path(output_dir) / "debug_preprocessed_stage1.png")
                debug_pil.save(debug_image_path)
                log.debug("Saved debug image: %s", debug_image_path)
        except Exception as e:
            log.warning("Failed to save debug image: %s", e)

    # Store pointmap scale/shift for pose decoding
    pointmap_scale = ss_input_dict.get("pointmap_scale", None)
    pointmap_shift = ss_input_dict.get("pointmap_shift", None)

    # Load models via ModelPatcher (ComfyUI manages VRAM, per-layer offloading)
    log.info("Loading Stage 1 models...")
    vram("Stage1: before loading models")
    ss_gen_patcher = _load_generator(config_path, 'ss', precision)
    ss_dec_patcher = _load_decoder(config_path, 'ss', precision)
    ss_emb_patcher = _load_condition_embedder(config_path, 'ss', precision)

    vram("Stage1: before load_models_gpu")
    comfy.model_management.load_models_gpu([ss_gen_patcher, ss_dec_patcher, ss_emb_patcher])
    vram("Stage1: after load_models_gpu")

    ss_generator = ss_gen_patcher.model
    ss_decoder = ss_dec_patcher.model
    ss_embedder = ss_emb_patcher.model

    # Configure generator (match original override_ss_generator_cfg_config)
    ss_generator.no_shortcut = True
    ss_generator.reverse_fn.strength = cfg_strength
    ss_generator.reverse_fn.strength_pm = cfg_strength_pm
    ss_generator.inference_steps = inference_steps
    # Critical settings from original that were missing:
    ss_generator.reverse_fn.interval = getattr(config, 'ss_cfg_interval', [0, 500])
    ss_generator.rescale_t = getattr(config, 'ss_rescale_t', 3)
    ss_generator.reverse_fn.backbone.condition_embedder.normalize_images = True
    ss_generator.reverse_fn.unconditional_handling = "add_flag"

    log.info("Running sparse structure generation...")
    vram("Stage1: before inference")

    downsample_ss_dist = getattr(config, 'downsample_ss_dist', 0)
    device = comfy.model_management.get_torch_device()

    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=dtype):
            image_tensor = ss_input_dict["image"]
            bs = image_tensor.shape[0]

            # Check if MM-DiT architecture
            has_latent_mapping = hasattr(ss_generator.reverse_fn.backbone, "latent_mapping")

            if has_latent_mapping:
                latent_shape_dict = {
                    k: (bs,) + (v.pos_emb.shape[0], v.input_layer.in_features)
                    for k, v in ss_generator.reverse_fn.backbone.latent_mapping.items()
                }
            else:
                latent_shape_dict = (bs,) + (4096, 8)

            # Get condition input mapping from config
            ss_condition_input_mapping = getattr(config, 'ss_condition_input_mapping', ['image'])

            # Get condition embeddings
            condition_args = [ss_input_dict[k] for k in ss_condition_input_mapping if k in ss_input_dict]
            condition_kwargs = {k: v for k, v in ss_input_dict.items() if k not in ss_condition_input_mapping}

            # Embed conditions
            if ss_embedder is not None and len(condition_args) > 0:
                tokens = ss_embedder(*condition_args, **condition_kwargs)
                condition_args = (tokens,)
                condition_kwargs = {}
            elif ss_embedder is not None:
                tokens = ss_embedder(**condition_kwargs)
                condition_args = (tokens,)
                condition_kwargs = {}

            # Run generator with progress reporting
            steps = ss_generator.inference_steps
            pbar = comfy.utils.ProgressBar(steps)
            gen_iter = ss_generator.generate_iter(
                latent_shape_dict,
                image_tensor.device,
                *condition_args,
                **condition_kwargs,
            )
            try:
                from tqdm import tqdm
                gen_iter = tqdm(gen_iter, total=steps, desc="Stage 1 (sparse structure)")
            except ImportError:
                pass
            for _, xt, _ in gen_iter:
                pbar.update(1)
            return_dict = xt

            if not has_latent_mapping:
                return_dict = {"shape": return_dict}

            shape_latent = return_dict["shape"]

            # Decode sparse structure
            ss = ss_decoder(
                shape_latent.permute(0, 2, 1)
                .contiguous()
                .view(shape_latent.shape[0], 8, 16, 16, 16)
            )
            coords = torch.argwhere(ss > 0)[:, [0, 2, 3, 4]].int()

            # Downsample output
            return_dict["coords_original"] = coords
            original_shape = coords.shape

            if downsample_ss_dist > 0:
                coords = prune_sparse_structure(
                    coords,
                    max_neighbor_axes_dist=downsample_ss_dist,
                )

            coords, downsample_factor = downsample_sparse_structure(coords)
            log.debug("Downsampled coords from %d to %d", original_shape[0], coords.shape[0])

            return_dict["coords"] = coords
            return_dict["downsample_factor"] = downsample_factor

    vram("Stage1: after inference")

    # Free preprocessing tensors — no longer needed after inference
    del ss_input_dict

    # Apply pose decoding
    log.info("Decoding pose...")
    pose_decoder_name = getattr(config, 'pose_decoder_name', 'default')
    pose_decoder = get_pose_decoder(pose_decoder_name)
    pose_result = pose_decoder(
        return_dict,
        scene_scale=pointmap_scale,
        scene_shift=pointmap_shift,
    )
    return_dict.update(pose_result)

    # Rescale after downsampling
    return_dict["scale"] = return_dict["scale"] * return_dict["downsample_factor"]

    # Add voxel coordinates
    return_dict["voxel"] = return_dict["coords"][:, 1:] / 64 - 0.5

    log.info("Stage 1 complete")

    # Save sparse structure
    if output_dir:
        save_dir = Path(output_dir)
    else:
        import tempfile
        save_dir = Path(tempfile.mkdtemp())
    save_dir.mkdir(parents=True, exist_ok=True)

    sparse_path = save_dir / "sparse_structure.pt"
    torch.save(return_dict, sparse_path)
    log.info("Saved sparse structure: %s", sparse_path)

    saved_output = {
        "output_dir": str(save_dir),
        "files": {"sparse_structure": str(sparse_path)},
        "metadata": {}
    }
    if debug_image_path:
        saved_output["files"]["debug_image"] = debug_image_path

    # Extract pose data for direct access
    rotation = return_dict.get("rotation")
    translation = return_dict.get("translation")
    scale = return_dict.get("scale")

    # Convert tensors to lists for JSON serialization
    if rotation is not None and hasattr(rotation, 'tolist'):
        rotation = rotation.cpu().tolist() if hasattr(rotation, 'cpu') else rotation.tolist()
    if translation is not None and hasattr(translation, 'tolist'):
        translation = translation.cpu().tolist() if hasattr(translation, 'cpu') else translation.tolist()
    if scale is not None and hasattr(scale, 'tolist'):
        scale = scale.cpu().tolist() if hasattr(scale, 'cpu') else scale.tolist()

    return {
        "status": "success",
        "stage1_mode": True,
        "output": saved_output,
        "data": return_dict,
        "rotation": rotation,
        "translation": translation,
        "scale": scale,
        "debug_image": debug_image_path,
    }


# =============================================================================
# Stage 2: SLAT Generation
# =============================================================================

def run_stage2(
    config_path: str,
    image,
    mask,
    stage1_output: Dict,
    seed: int = 42,
    inference_steps: int = 25,
    cfg_strength: float = 5.0,
    output_dir: str = None,
    precision: str = "bf16",
) -> Dict[str, Any]:
    """
    Run Stage 2 (SLAT generation).

    Models loaded: slat_generator (~4.9 GB), slat_condition_embedder (~1.2 GB)
    Peak VRAM: ~6-7 GB
    """
    from ..sam3d.sparse import SparseTensor

    log.info("Running Stage 2 (SLAT gen)")

    config, checkpoint_dir = _load_config(config_path)
    dtype = _resolve_dtype(precision)

    # Set seed
    torch.manual_seed(seed)

    # Convert image/mask to numpy
    image_np = np.array(image)
    mask_np = np.array(mask) if mask is not None else None

    # Ensure RGBA format
    if image_np.ndim == 2:
        image_np = np.stack([image_np] * 3 + [np.full_like(image_np, 255)], axis=-1)
    elif image_np.shape[-1] == 3:
        if mask_np is not None:
            if mask_np.dtype == np.float32 or mask_np.dtype == np.float64:
                alpha = (mask_np * 255).astype(np.uint8)
            else:
                alpha = mask_np.astype(np.uint8)
            if alpha.ndim == 2:
                alpha = alpha[:, :, np.newaxis]
        else:
            alpha = np.full((image_np.shape[0], image_np.shape[1], 1), 255, dtype=np.uint8)
        image_np = np.concatenate([image_np, alpha], axis=-1)

    # Get preprocessor and preprocess image
    log.info("Preprocessing image for SLAT...")
    slat_preprocessor = _get_preprocessor(config_path, 'slat')
    slat_input_dict = preprocess_image_lazy(image_np, mask_np, slat_preprocessor)

    device = comfy.model_management.get_torch_device()

    # Get coords from stage1_output
    coords = stage1_output.get("coords")
    if isinstance(coords, str):
        coords = pickle.loads(base64.b64decode(coords))
    if isinstance(coords, np.ndarray):
        coords = torch.from_numpy(coords).int()
    coords = coords.to(device)

    # Force-evict stale Stage 1 models before loading Stage 2.
    import gc
    vram("Stage2: before unload Stage1")
    comfy.model_management.unload_all_models()
    gc.collect()
    comfy.model_management.soft_empty_cache()
    vram("Stage2: after unload Stage1")

    # Load models via ModelPatcher (ComfyUI manages VRAM, per-layer offloading)
    log.info("Loading Stage 2 models...")
    slat_gen_patcher = _load_generator(config_path, 'slat', precision)
    slat_emb_patcher = _load_condition_embedder(config_path, 'slat', precision)

    vram("Stage2: before load_models_gpu")
    comfy.model_management.load_models_gpu([slat_gen_patcher, slat_emb_patcher])
    vram("Stage2: after load_models_gpu")

    slat_generator = slat_gen_patcher.model
    slat_embedder = slat_emb_patcher.model

    # Configure generator (match original override_slat_generator_cfg_config)
    slat_generator.no_shortcut = True
    # Read cfg_strength from config if available (config may override the default)
    slat_cfg = getattr(config, 'slat_cfg_strength', cfg_strength)
    slat_generator.reverse_fn.strength = slat_cfg
    slat_generator.inference_steps = inference_steps
    # Critical settings from original that were missing:
    slat_generator.reverse_fn.interval = getattr(config, 'slat_cfg_interval', [0, 500])
    slat_generator.rescale_t = getattr(config, 'slat_rescale_t', 3)

    log.info("Running SLAT generation...")
    vram("Stage2: before inference")

    # Get SLAT normalization stats
    slat_mean = torch.tensor(getattr(config, 'slat_mean', [0.0] * 8), device=device)
    slat_std = torch.tensor(getattr(config, 'slat_std', [1.0] * 8), device=device)

    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=dtype):
            image_tensor = slat_input_dict["image"]
            DEVICE = image_tensor.device
            latent_shape = (image_tensor.shape[0],) + (coords.shape[0], 8)

            # Get condition input mapping from config
            slat_condition_input_mapping = getattr(config, 'slat_condition_input_mapping', ['image'])

            # Get condition embeddings
            condition_args = [slat_input_dict[k] for k in slat_condition_input_mapping if k in slat_input_dict]
            condition_kwargs = {k: v for k, v in slat_input_dict.items() if k not in slat_condition_input_mapping}

            # Embed conditions
            if slat_embedder is not None and len(condition_args) > 0:
                tokens = slat_embedder(*condition_args, **condition_kwargs)
                condition_args = (tokens,)
                condition_kwargs = {}
            elif slat_embedder is not None:
                tokens = slat_embedder(**condition_kwargs)
                condition_args = (tokens,)
                condition_kwargs = {}

            # Add coords to condition args
            condition_args = condition_args + (coords.cpu().numpy(),)

            # Run generator with progress reporting
            steps = slat_generator.inference_steps
            pbar = comfy.utils.ProgressBar(steps)
            gen_iter = slat_generator.generate_iter(
                latent_shape, DEVICE, *condition_args, **condition_kwargs
            )
            try:
                from tqdm import tqdm
                gen_iter = tqdm(gen_iter, total=steps, desc="Stage 2 (SLAT generation)")
            except ImportError:
                pass
            for _, xt, _ in gen_iter:
                pbar.update(1)
            slat_feats = xt

            # Create SparseTensor
            slat = SparseTensor(
                coords=coords,
                feats=slat_feats[0],
            ).to(DEVICE)

            # Apply mean/std normalization
            slat = slat * slat_std + slat_mean

    vram("Stage2: after inference")
    log.info("Stage 2 complete")

    # Build output dict with SLAT for saving.
    # Only carry pose data forward — decode only needs rotation/translation/scale.
    output_dict = {
        "slat": slat,
        "stage1_data": {
            "rotation": stage1_output.get("rotation"),
            "translation": stage1_output.get("translation"),
            "scale": stage1_output.get("scale"),
        },
    }

    # Save output to disk
    if output_dir:
        saved_output = save_output_to_disk(output_dict, Path(output_dir))
    else:
        import tempfile
        temp_dir = Path(tempfile.mkdtemp())
        slat_path = temp_dir / "slat.pt"
        torch.save(output_dict, slat_path)
        saved_output = {
            "output_dir": str(temp_dir),
            "files": {"slat": str(slat_path)},
            "metadata": {}
        }

    return {
        "status": "success",
        "stage2_mode": True,
        "output": saved_output,
        "data": output_dict,
    }


# =============================================================================
# Stage 3: Gaussian/Mesh Decoding
# =============================================================================

def run_decode(
    config_path: str,
    slat_data: Dict,
    decode_format: str = "gaussian",
    output_dir: str = None,
    with_postprocess: bool = False,
    simplify: float = 0.95,
    up_axis: str = "Y-up (standard)",
    world_coordinates: bool = False,
    precision: str = "bf16",
    use_sparse_flexicubes: bool = True,
) -> Dict[str, Any]:
    """
    Run Stage 3 (Gaussian or Mesh decoding).

    Models loaded: slat_decoder_gs (~170 MB) or slat_decoder_mesh (~364 MB)
    Peak VRAM: ~1-2 GB
    """
    import trimesh
    from ..sam3d.sparse import SparseTensor as _SparseTensor

    log.info("Running decode (%s)", decode_format)
    pbar = comfy.utils.ProgressBar(4)

    # Extract slat - handle different input formats
    slat = None

    if isinstance(slat_data, _SparseTensor):
        slat = slat_data
    elif isinstance(slat_data, dict) and "slat" in slat_data:
        slat_inner = slat_data["slat"]
        if isinstance(slat_inner, _SparseTensor):
            slat = slat_inner
        elif isinstance(slat_inner, str):
            slat_inner = pickle.loads(base64.b64decode(slat_inner))
            if isinstance(slat_inner, _SparseTensor):
                slat = slat_inner
            else:
                coords = slat_inner["coords"]
                feats = slat_inner["feats"]
        elif isinstance(slat_inner, dict):
            coords = slat_inner["coords"]
            feats = slat_inner["feats"]
    elif isinstance(slat_data, dict):
        coords = slat_data.get("coords")
        feats = slat_data.get("feats")

    # If we don't have slat yet, reconstruct from coords/feats
    if slat is None:
        if isinstance(coords, str):
            coords = pickle.loads(base64.b64decode(coords))
        if isinstance(feats, str):
            feats = pickle.loads(base64.b64decode(feats))

        device = comfy.model_management.get_torch_device()
        coords = torch.from_numpy(coords).int().to(device) if isinstance(coords, np.ndarray) else coords.int().to(device)
        feats = torch.from_numpy(feats).to(device) if isinstance(feats, np.ndarray) else feats.to(device)
        slat = _SparseTensor(coords=coords, feats=feats)
    else:
        device = comfy.model_management.get_torch_device()
        slat = slat.to(device)

    # Don't cast feats here — the decoder's input_layer is fp32 (not touched by
    # convert_to_fp16), and the model casts to fp16 internally after input_layer.

    # Load decoder
    if decode_format == "gaussian":
        decoder_name = 'slat_decoder_gs'
    else:
        decoder_name = 'slat_decoder_mesh'

    pbar.update(1)  # SLAT loaded

    log.info("Loading decoder (%s)...", decoder_name)
    dec_patcher = _load_decoder(config_path, decoder_name, precision)
    vram(f"Decode({decode_format}): before load_models_gpu")
    comfy.model_management.load_models_gpu([dec_patcher])
    vram(f"Decode({decode_format}): after load_models_gpu")
    decoder = dec_patcher.model

    log.info("Running decoder... slat.feats.dtype=%s, slat.feats.shape=%s, slat.coords.shape=%s",
             slat.feats.dtype, slat.feats.shape, slat.coords.shape)
    log.info("Decoder type: %s, model.dtype=%s", type(decoder).__name__, getattr(decoder, 'dtype', 'N/A'))

    # Log parameter dtypes for debugging
    param_dtypes = set()
    for p in decoder.parameters():
        param_dtypes.add(str(p.dtype))
    log.info("Decoder parameter dtypes: %s", param_dtypes)

    def _vram_mb():
        return torch.cuda.memory_allocated() / 1024**2

    def _vram_peak_mb():
        return torch.cuda.max_memory_allocated() / 1024**2

    torch.cuda.reset_peak_memory_stats()
    log.info("[VRAM] before decoder forward: %.0f MB (peak %.0f MB)", _vram_mb(), _vram_peak_mb())

    pbar.update(1)  # Decoder loaded

    # Cast SLAT features to match decoder precision
    dtype = _resolve_dtype(precision)
    if slat.feats.dtype != dtype:
        slat.feats = slat.feats.to(dtype=dtype)

    # (sparse FlexiCubes is now the only path — no toggle needed)

    with torch.no_grad():
        output = decoder(slat)

    log.info("[VRAM] after decoder forward: %.0f MB (peak %.0f MB)", _vram_mb(), _vram_peak_mb())
    pbar.update(1)  # Decode complete
    log.info("Decode complete")

    # Free decoder + SLAT before saving — same pattern as lines 1307-1314
    del slat
    comfy.model_management.unload_all_models()
    gc.collect()
    comfy.model_management.soft_empty_cache()
    vram(f"Decode({decode_format}): after cleanup")

    # Determine output directory
    if output_dir:
        save_dir = Path(output_dir)
    else:
        import tempfile
        save_dir = Path(tempfile.mkdtemp())
    save_dir.mkdir(parents=True, exist_ok=True)

    saved_files = {"output_dir": str(save_dir), "files": {}}

    if decode_format == "gaussian":
        gaussian = output[0] if isinstance(output, (list, tuple)) else output

        # Apply pose transformation
        pose_data = None
        if isinstance(slat_data, dict) and "stage1_data" in slat_data:
            stage1 = slat_data["stage1_data"]
            if isinstance(stage1, dict):
                pose_data = {
                    "rotation": stage1.get("rotation"),
                    "translation": stage1.get("translation"),
                    "scale": stage1.get("scale"),
                }

        if world_coordinates and pose_data is not None and pose_data.get("rotation") is not None:
            log.info("Applying pose transformation to Gaussian...")
            gaussian = _apply_pose_to_gaussian(gaussian, pose_data)
        elif not world_coordinates:
            log.info("Skipping pose (world_coordinates=False)")

        ply_path = save_dir / "gaussian.ply"
        try:
            transform = None
            if up_axis == "Y-up (standard)":
                transform = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
            gaussian.save_ply(str(ply_path), transform=transform)
            saved_files["files"]["ply"] = str(ply_path)
            log.info("Saved Gaussian PLY: %s", ply_path)
        except Exception as e:
            log.warning("Failed to save Gaussian PLY: %s", e)

        pbar.update(1)  # Output saved
        return {
            "status": "success",
            "stage2_mode": True,
            "output": saved_files,
            "file_output": saved_files,
        }
    else:
        # Mesh output
        mesh = output[0] if isinstance(output, (list, tuple)) else output

        try:
            vertices = mesh.vertices.cpu().numpy() if hasattr(mesh.vertices, 'cpu') else mesh.vertices
            faces = mesh.faces.cpu().numpy() if hasattr(mesh.faces, 'cpu') else mesh.faces

            # Get vertex colors BEFORE postprocessing
            original_vertex_colors = None
            if hasattr(mesh, 'vertex_attrs') and mesh.vertex_attrs is not None:
                if isinstance(mesh.vertex_attrs, torch.Tensor):
                    attrs = mesh.vertex_attrs.cpu().numpy()
                    original_vertex_colors = attrs[:, :3] if attrs.shape[-1] >= 3 else attrs
                elif isinstance(mesh.vertex_attrs, dict) and 'color' in mesh.vertex_attrs:
                    vc = mesh.vertex_attrs['color']
                    if hasattr(vc, 'cpu'):
                        original_vertex_colors = vc.cpu().numpy()
                    else:
                        original_vertex_colors = vc

            # Apply postprocessing if requested
            if with_postprocess:
                log.info("Applying mesh postprocessing (simplify=%s)...", simplify)
                from ..sam3d.postprocessing import postprocess_mesh
                pp_result = postprocess_mesh(
                    vertices,
                    faces,
                    simplify=True,
                    simplify_ratio=simplify,
                    fill_holes=True,
                    verbose=True,
                    vertex_colors=original_vertex_colors,
                )
                if len(pp_result) == 3:
                    vertices, faces, original_vertex_colors = pp_result
                else:
                    vertices, faces = pp_result
                    original_vertex_colors = None
                log.info("Postprocessing complete: %d vertices, %d faces", len(vertices), len(faces))

            # Process vertex colors if available
            vertex_colors = None
            if original_vertex_colors is not None:
                vc = original_vertex_colors
                if vc.max() <= 1.0:
                    vc = (vc * 255).astype(np.uint8)
                if vc.shape[-1] == 3:
                    alpha = np.full((vc.shape[0], 1), 255, dtype=np.uint8)
                    vc = np.concatenate([vc, alpha], axis=-1)
                vertex_colors = vc

            # Apply pose transformation
            pose_data = None
            if isinstance(slat_data, dict) and "stage1_data" in slat_data:
                stage1 = slat_data["stage1_data"]
                if isinstance(stage1, dict):
                    pose_data = {
                        "rotation": stage1.get("rotation"),
                        "translation": stage1.get("translation"),
                        "scale": stage1.get("scale"),
                    }

            if world_coordinates and pose_data is not None and pose_data.get("rotation") is not None:
                log.info("Applying pose transformation to world coordinates...")
                vertices = _apply_pose_to_vertices(vertices, pose_data)
            elif not world_coordinates:
                log.info("Skipping pose (world_coordinates=False)")
            else:
                log.info("No pose data found, mesh will be in normalized coordinates")

            # Transform from Z-up to Y-up if requested
            if up_axis == "Y-up (standard)":
                z_to_y_up = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
                vertices = vertices @ z_to_y_up

            # Create trimesh with vertex colors
            trimesh_mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                vertex_colors=vertex_colors,
                process=False
            )

            # Save GLB
            glb_path = save_dir / "mesh.glb"
            glb_bytes = trimesh_mesh.export(file_type="glb")
            with open(glb_path, 'wb') as f:
                f.write(glb_bytes)
            saved_files["files"]["glb"] = str(glb_path)
            log.info("Saved Mesh GLB: %s", glb_path)

        except Exception as e:
            log.error("Failed to save Mesh GLB: %s", e)
            import traceback
            traceback.print_exc()
            mesh_path = save_dir / "mesh.pt"
            mesh_data = {
                "vertices": mesh.vertices.cpu() if hasattr(mesh.vertices, 'cpu') else mesh.vertices,
                "faces": mesh.faces.cpu() if hasattr(mesh.faces, 'cpu') else mesh.faces,
            }
            torch.save(mesh_data, mesh_path)
            saved_files["files"]["mesh_pt"] = str(mesh_path)

        pbar.update(1)  # Output saved
        return {
            "status": "success",
            "stage2_mode": True,
            "output": saved_files,
            "file_output": saved_files,
        }


# =============================================================================
# Texture Baking
# =============================================================================

def run_texture_bake_direct(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run texture baking directly without loading any models.

    Loads Gaussian from PLY and Mesh from GLB, then calls to_glb() for texture baking.
    """
    import trimesh

    log.info("Running direct texture baking")

    # Extract parameters
    ply_path = request["ply_path"]
    glb_path = request["glb_path"]
    output_dir = request["output_dir"]
    texture_mode = request.get("texture_mode", "opt")
    texture_size = request.get("texture_size", 1024)
    rendering_engine = request.get("rendering_engine", "nvdiffrast")

    log.info("PLY: %s", ply_path)
    log.info("GLB: %s", glb_path)
    log.info("Mode: %s, Size: %d", texture_mode, texture_size)

    from ..sam3d.representations import Gaussian, MeshExtractResult
    from ..sam3d.postprocessing import to_glb

    device = comfy.model_management.get_torch_device()

    # Load Gaussian from PLY
    log.info("Loading Gaussian from PLY...")
    gaussian = Gaussian(
        aabb=[-1, -1, -1, 2, 2, 2],
        sh_degree=0,
        device=device
    )
    gaussian.load_ply(ply_path)
    log.info("Loaded Gaussian with %d points", gaussian._xyz.shape[0])

    # Load Mesh from GLB
    log.info("Loading Mesh from GLB...")
    loaded = trimesh.load(glb_path)

    if isinstance(loaded, trimesh.Scene):
        meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise RuntimeError("No mesh geometries found in GLB")
        trimesh_mesh = meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)
    else:
        trimesh_mesh = loaded

    log.info("Loaded mesh with %d vertices", len(trimesh_mesh.vertices))

    # Convert to MeshExtractResult format
    vertices_np = np.array(trimesh_mesh.vertices)
    vertices_tensor = torch.tensor(vertices_np, dtype=torch.float32, device=device)
    faces_tensor = torch.tensor(np.array(trimesh_mesh.faces), dtype=torch.long, device=device)

    if trimesh_mesh.visual is not None and hasattr(trimesh_mesh.visual, 'vertex_colors') and trimesh_mesh.visual.vertex_colors is not None:
        vertex_colors = np.array(trimesh_mesh.visual.vertex_colors)[:, :3] / 255.0
    else:
        vertex_colors = np.ones((len(trimesh_mesh.vertices), 3), dtype=np.float32)

    vertex_attrs_tensor = torch.tensor(vertex_colors, dtype=torch.float32, device=device)

    mesh = MeshExtractResult(
        vertices=vertices_tensor,
        faces=faces_tensor,
        vertex_attrs=vertex_attrs_tensor,
        res=64
    )

    # Run texture baking
    log.info("Running texture baking...")
    result_mesh = to_glb(
        gaussian,
        mesh,
        simplify=0,
        fill_holes=False,
        texture_size=texture_size,
        verbose=False,
        with_mesh_postprocess=False,
        with_texture_baking=True,
        rendering_engine=rendering_engine,
        texture_mode=texture_mode,
    )

    # Undo to_glb's Z->Y transform
    undo_transform = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    result_mesh.vertices = result_mesh.vertices @ undo_transform

    # Save textured GLB
    output_path = Path(output_dir) / "mesh_textured.glb"
    glb_bytes = result_mesh.export(file_type="glb")
    with open(output_path, 'wb') as f:
        f.write(glb_bytes)

    log.info("Saved textured GLB: %s", output_path)

    return {
        "status": "success",
        "output": {
            "glb_path": str(output_path),
        }
    }


# =============================================================================
# Legacy Aliases (for backward compatibility during transition)
# =============================================================================

def run_stage1_lazy(lazy_manager, *args, **kwargs):
    """Legacy wrapper - extracts config_path from lazy_manager."""
    return run_stage1(lazy_manager.config_path, *args, **kwargs)

def run_stage2_lazy(lazy_manager, *args, **kwargs):
    """Legacy wrapper - extracts config_path from lazy_manager."""
    return run_stage2(lazy_manager.config_path, *args, **kwargs)

def run_decode_lazy(lazy_manager, *args, **kwargs):
    """Legacy wrapper - extracts config_path from lazy_manager."""
    return run_decode(lazy_manager.config_path, *args, **kwargs)

def run_depth_only(model_manager, image, unload_after=True, depth_backend="moge2"):
    """Legacy wrapper - depth is now handled directly in the node."""
    raise NotImplementedError("run_depth_only is deprecated. Load MoGe directly in the node.")
