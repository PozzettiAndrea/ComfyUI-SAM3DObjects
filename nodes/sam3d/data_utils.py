# Copyright (c) Meta Platforms, Inc. and affiliates.
# Consolidated from sam3d_objects/data/utils.py and sam3d_objects/model/io.py
import logging
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
import optree
import torch
from collections.abc import Iterable as IterableABC
import inspect
import ast
from torch.utils import _pytree
import comfy.utils
from pathlib import Path
import os
import re
from loguru import logger
from glob import glob

log = logging.getLogger("sam3dobjects")

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
ChildPathType = Union[None, Iterable[Any], Any]
ArgsType = Iterable[ChildPathType]
KwargsType = Mapping[str, ChildPathType]
ArgsKwargsType = Tuple[ArgsType, KwargsType]
MappingType = Union[None, ArgsKwargsType, ArgsType, KwargsType]


# ---------------------------------------------------------------------------
# Tree utilities (from data/utils.py)
# ---------------------------------------------------------------------------

def tree_transpose_level_one(
    structure,
    check_children=False,
    map_fn=None,
    is_leaf=None,
):
    _, outer_spec = optree.tree_flatten(
        structure,
        is_leaf=lambda x: x is not structure,
        none_is_leaf=True,
    )

    spec = optree.tree_structure(structure, none_is_leaf=True, is_leaf=is_leaf)
    children_spec = spec.children()
    if len(children_spec) > 0:
        inner_spec = children_spec[0]
        if check_children:
            for child_spec in children_spec[1:]:
                assert (
                    inner_spec == child_spec
                ), f"one child was found having a different tree structure ({inner_spec} != {child_spec})"

        structure = optree.tree_transpose(outer_spec, inner_spec, structure)

    if map_fn is not None:
        structure = optree.tree_map(
            map_fn,
            structure,
            is_leaf=lambda x: optree.tree_structure(
                x, is_leaf=is_leaf, none_is_leaf=True
            )
            == outer_spec,
            none_is_leaf=True,
        )

    return structure


def tree_tensor_map(fn, tree, *rest):
    return optree.tree_map(
        fn,
        tree,
        *rest,
        is_leaf=lambda x: isinstance(x, torch.Tensor),
        none_is_leaf=False,
    )


def to_device(obj, device):
    to_fn = lambda x: x.to(device)
    return optree.tree_map(to_fn, obj, is_leaf=torch.is_tensor, none_is_leaf=False)


def expand_right(tensor, target_shape):
    current_shape = tensor.shape
    dims_to_add = len(target_shape) - len(current_shape)
    result = tensor
    for _ in range(dims_to_add):
        result = result.unsqueeze(-1)
    expand_shape = list(current_shape) + [-1] * dims_to_add
    for i in range(len(target_shape)):
        if i < len(expand_shape) and expand_shape[i] == -1:
            expand_shape[i] = target_shape[i]
    return result.expand(*expand_shape)


def expand_as_right(tensor, target):
    return expand_right(tensor, target.shape)


def as_keys(path: ChildPathType):
    if isinstance(path, IterableABC) and (not isinstance(path, str)):
        return tuple(path)
    elif path is None:
        return ()
    return (path,)


def get_child(obj: Any, *keys: Iterable[Any]):
    for key in keys:
        obj = obj[key]
    return obj


def set_child(obj: Any, value: Any, *keys: Iterable[Any]):
    parent = None
    for key in keys:
        parent = obj
        obj = obj[key]
    if parent is None:
        obj = value
    else:
        parent[key] = value
    return obj


def build_args_batch_extractor(args_mapping: ArgsType):
    def extract_fn(batch):
        return tuple(get_child(batch, *as_keys(path)) for path in args_mapping)
    return extract_fn


def build_kwargs_batch_extractor(kwargs_mapping: KwargsType):
    def extract_fn(batch):
        return {
            name: get_child(batch, *as_keys(path))
            for name, path in kwargs_mapping.items()
        }
    return extract_fn


empty_mapping = object()
kwargs_identity_mapping = object()


def build_batch_extractor(mapping: MappingType):
    extract_args_fn = lambda x: ()
    extract_kwargs_fn = lambda x: {}

    if mapping is None:
        def extract_args_fn(batch):
            return (batch,)
    elif mapping is empty_mapping:
        pass
    elif mapping is kwargs_identity_mapping:
        extract_kwargs_fn = lambda x: x
    elif isinstance(mapping, Sequence) and (not isinstance(mapping, str)):
        if (
            len(mapping) == 2
            and isinstance(mapping[0], Sequence)
            and isinstance(mapping[1], Dict)
        ):
            extract_args_fn = build_args_batch_extractor(mapping[0])
            extract_kwargs_fn = build_kwargs_batch_extractor(mapping[1])
        else:
            extract_args_fn = build_args_batch_extractor(mapping)
    elif isinstance(mapping, Mapping):
        extract_kwargs_fn = build_kwargs_batch_extractor(mapping)
    else:
        def extract_args_fn(batch):
            return (get_child(batch, *as_keys(mapping)),)

    def extract_fn(batch):
        return extract_args_fn(batch), extract_kwargs_fn(batch)

    return extract_fn


def right_broadcasting(arr, target):
    return arr.reshape(arr.shape + (1,) * (target.ndim - arr.ndim))


def get_stats(tensor: torch.Tensor):
    float_tensor = tensor.float()
    return {
        "shape": tuple(tensor.shape),
        "min": tensor.min().item(),
        "max": tensor.max().item(),
        "mean": float_tensor.mean().item(),
        "median": tensor.median().item(),
        "std": float_tensor.std().item(),
    }


def _get_caller_arg_name(argnum=0, parent_frame=1):
    try:
        frame = inspect.currentframe()
        frame = inspect.getouterframes(frame)[1 + parent_frame]
        code = inspect.getframeinfo(frame[0]).code_context[0].strip()

        tree = ast.parse(code)

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                args = node.args
                break

        label = ast.unparse(args[argnum]).strip()
    except:
        label = "{label}"
    return label


def print_stats(tensor, label=None):
    if label is None:
        label = _get_caller_arg_name(argnum=0)
    stats = get_stats(tensor)
    string = f"{label}:\n" + "\n".join(f"- {k}: {v}" for k, v in stats.items())
    log.debug("%s", string)


def tree_reduce_unique(fn, tree, ensure_unique=True, **kwargs):
    values = _pytree.tree_flatten(tree, **kwargs)[0]
    values = tuple(map(fn, values))
    first = values[0]
    if ensure_unique:
        for value in values[1:]:
            if value != first:
                raise RuntimeError(
                    f"different values found, {value} and {first} should be the same"
                )
    return first


# ---------------------------------------------------------------------------
# Checkpoint I/O utilities (from model/io.py)
# ---------------------------------------------------------------------------

def rename_checkpoint_weights_using_suffix_matching(
    checkpoint_path_in,
    checkpoint_path_out,
    model: torch.nn.Module,
    strict: bool = True,
    keys: Optional[List[Any]] = (),
):
    param_names = [n for n, _ in model.named_parameters()]
    buffer_names = [n for n, _ in model.named_buffers()]
    model_names = param_names + buffer_names

    state = comfy.utils.load_torch_file(checkpoint_path_in)

    model_state = get_child(state, *keys)
    model_state_names = list(model_state.keys())

    model_names_rev = sorted([n[::-1] for n in model_names])
    model_state_names_rev = sorted([n[::-1] for n in model_state_names])

    if strict and len(model_names) != len(model_state_names):
        raise RuntimeError(
            f"model and state don't have the same number of parameters ({len(model_names)} != {len(model_state_names)}), cannot match them (set strict = False to relax constraint)"
        )

    def common_prefix_length(str_0: str, str_1: str):
        for count in range(min(len(str_0), len(str_1))):
            if str_0[count] != str_1[count]:
                break
        return count

    name_mapping = {}
    i, j = 0, 0
    last_n = 0
    while i < len(model_names_rev):
        if j < len(model_state_names_rev):
            n = common_prefix_length(model_names_rev[i], model_state_names_rev[j])
        else:
            n = 0

        if n >= last_n:
            last_n = n
            j += 1
        else:
            last_n = 0
            name_mapping[model_names_rev[i][::-1]] = model_state_names_rev[j - 1][::-1]
            i += 1

        if not j < len(model_state_names_rev) + 1:
            break

    if i < len(model_names):
        raise RuntimeError("could not suffix match parameter names")

    for k, v in name_mapping.items():
        logger.debug(f"{k} <- {v}")

    model_state_out = {k: model_state[v] for k, v in name_mapping.items()}
    set_child(state, model_state_out, *keys)
    torch.save(state, checkpoint_path_out)


def remove_prefix_state_dict_fn(prefix: str):
    n = len(prefix)
    def state_dict_fn(state_dict):
        return {
            (key[n:] if key.startswith(prefix) else key): value
            for key, value in state_dict.items()
        }
    return state_dict_fn


def add_prefix_state_dict_fn(prefix: str):
    def state_dict_fn(state_dict):
        return {prefix + key: value for key, value in state_dict.items()}
    return state_dict_fn


def filter_and_remove_prefix_state_dict_fn(prefix: str):
    n = len(prefix)
    def state_dict_fn(state_dict):
        return {
            key[n:]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
    return state_dict_fn


def get_last_checkpoint(path: str):
    checkpoints = glob(os.path.join(path, "epoch=*-step=*.ckpt"))
    prog = re.compile(r"epoch=(\d+)-step=(\d+).ckpt")

    checkpoints_to_sort = []
    for checkpoint in checkpoints:
        checkpoint_name = os.path.basename(checkpoint)
        match = prog.match(checkpoint_name)
        if match is not None:
            n_epoch, n_step = prog.match(checkpoint_name).groups()
            n_epoch, n_step = int(n_epoch), int(n_step)
            checkpoints_to_sort.append((n_epoch, n_step, checkpoint))

    sorted_checkpoints = sorted(checkpoints_to_sort)
    if not len(sorted_checkpoints) > 0:
        raise RuntimeError(f"no checkpoint has been found at path : {path}")
    return sorted_checkpoints[-1][2]


def load_sharded_checkpoint(path: str, device: Optional[str]):
    from lightning.pytorch.utilities.consolidate_checkpoint import (
        _format_checkpoint,
        _load_distributed_checkpoint,
    )
    if device != "cpu":
        raise RuntimeError(
            f'loading sharded weights on device "{device}" is not available, please use the "cpu" device instead'
        )
    checkpoint = _load_distributed_checkpoint(Path(path))
    checkpoint = _format_checkpoint(checkpoint)
    return checkpoint


def load_model_from_checkpoint(
    model,
    checkpoint_path: str,
    strict: bool = True,
    device: Optional[str] = None,
    freeze: bool = False,
    eval: bool = False,
    map_name: Union[Dict[str, str], None] = None,
    remove_name: Union[List[str], None] = None,
    state_dict_key: Union[None, str, Iterable[str]] = "state_dict",
    state_dict_fn: Optional[Callable[[Any], Any]] = None,
):
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    if os.path.isfile(checkpoint_path):
        if checkpoint_path.endswith('.safetensors'):
            from safetensors.torch import load_file
            checkpoint = load_file(checkpoint_path, device=device or "cpu")
            state_dict_key = None
        else:
            checkpoint = comfy.utils.load_torch_file(checkpoint_path)
            state_dict_key = None
    elif os.path.isdir(checkpoint_path):
        checkpoint = load_sharded_checkpoint(checkpoint_path, device=device)
    else:
        raise FileNotFoundError(checkpoint_path)

    try:
        import lightning.pytorch as pl
        if isinstance(model, pl.LightningModule):
            model.on_load_checkpoint(checkpoint)
    except ImportError:
        pass

    state_dict = checkpoint
    if state_dict_key is not None:
        if isinstance(state_dict_key, str):
            state_dict_key = (state_dict_key,)
        state_dict = get_child(state_dict, *state_dict_key)

    if remove_name is not None:
        for name in remove_name:
            del state_dict[name]

    if map_name is not None:
        for src, dst in map_name.items():
            if src not in state_dict:
                continue
            state_dict[dst] = state_dict[src]
            del state_dict[src]

    if state_dict_fn is not None:
        state_dict = state_dict_fn(state_dict)

    model.load_state_dict(state_dict, strict=strict)

    if device is not None:
        model = model.to(device)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False
        eval = True

    if eval:
        model.eval()

    return model
