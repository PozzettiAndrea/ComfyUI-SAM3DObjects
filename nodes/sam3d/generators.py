# Copyright (c) Meta Platforms, Inc. and affiliates.
# Consolidated from sam3d_objects/model/backbone/generator/

import copy
import math
import os
import random
from functools import partial
from numbers import Number
from typing import Callable, Optional, Sequence, Union

import numpy as np
import optree
import torch
from loguru import logger
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils import _pytree
from torch.utils._pytree import tree_map_only
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from .data_utils import right_broadcasting
from .data_utils import tree_tensor_map, tree_reduce_unique


# --- base.py ---


class Base(torch.nn.Module):
    def __init__(self, seed_or_generator: Optional[Union[int, torch.Generator]] = None):
        super().__init__()

        if isinstance(seed_or_generator, torch.Generator):
            self.random_generator = seed_or_generator
        elif isinstance(seed_or_generator, int):
            self.seed = seed_or_generator
        elif seed_or_generator is None:
            self.random_generator = torch.default_generator
        else:
            raise RuntimeError(
                f"cannot use argument of type {type(seed_or_generator)} to set random generator"
            )

    @property
    def seed(self):
        raise AttributeError(f"Cannot read attribute 'seed'.")

    @seed.setter
    def seed(self, value: int):
        self._random_generator = torch.Generator().manual_seed(value)

    @property
    def random_generator(self):
        return self._random_generator

    @random_generator.setter
    def random_generator(self, generator: torch.Generator):
        self._random_generator = generator

    def forward(self, x_shape, x_device, *args_conditionals, **kwargs_conditionals):
        return self.generate(
            x_shape,
            x_device,
            *args_conditionals,
            **kwargs_conditionals,
        )

    def generate(self, x_shape, x_device, *args_conditionals, **kwargs_conditionals):
        for _, xt, _ in self.generate_iter(
            x_shape,
            x_device,
            *args_conditionals,
            **kwargs_conditionals,
        ):
            pass
        return xt

    def generate_iter(
        self,
        x_shape,
        x_device,
        *args_conditionals,
        **kwargs_conditionals,
    ):
        raise NotImplementedError

    def loss(self, x, *args_conditionals, **kwargs_conditionals):
        raise NotImplementedError


# --- solver.py ---


def linear_approximation_step(x_t, dt, velocity):
    # x_tp1 = x_t + velocity * dt
    x_tp1 = tree_tensor_map(lambda x, v: x + v * dt, x_t, velocity)
    return x_tp1


def gradient(output, x, create_graph: bool = False):
    tensors, pyspec = optree.tree_flatten(
        x, is_leaf=lambda x: isinstance(x, torch.Tensor)
    )
    grad_outputs = [torch.ones_like(output).detach() for _ in tensors]
    grads = torch.autograd.grad(
        output,
        tensors,
        grad_outputs=grad_outputs,
        create_graph=create_graph,
    )
    return optree.tree_unflatten(pyspec, grads)


class ODESolver:
    def step(self, dynamics_fn, x_t, t, dt, *args, **kwargs):
        raise NotImplementedError

    def solve_iter(self, dynamics_fn, x_init, times, *args, **kwargs):
        x_t = x_init
        for t0, t1 in zip(times[:-1], times[1:]):
            dt = t1 - t0
            x_t = self.step(dynamics_fn, x_t, t0, dt, *args, **kwargs)
            yield x_t, t0

    def solve(self, dynamics_fn, x_init, times, *args, **kwargs):
        for x_t, _ in self.solve_iter(dynamics_fn, x_init, times, *args, **kwargs):
            pass
        return x_t


# https://en.wikipedia.org/wiki/Euler_method
class Euler(ODESolver):
    def step(self, dynamics_fn, x_t, t, dt, *args, **kwargs):
        velocity = dynamics_fn(x_t, t, *args, **kwargs)
        x_tp1 = linear_approximation_step(x_t, dt, velocity)
        return x_tp1


# https://arxiv.org/abs/2505.05470
class SDE(ODESolver):
    def __init__(self, **kwargs):
        super().__init__()
        self.sde_strength = kwargs.get("sde_strength", 0.1)

    def step(self, dynamics_fn, x_t, t, dt, *args, **kwargs):
        velocity = dynamics_fn(x_t, t, *args, **kwargs)
        sigma = 1 - t
        var_t = sigma / (1 - torch.tensor(sigma).clamp(min=dt))
        std_dev_t = (
            torch.sqrt(variance) * self.sde_strength
        )  # self.sde_strength = alpha

        def compute_mean(x, v):
            drift_term = x * (std_dev_t**2 / (2 * sigma) * dt)
            velocity_term = v * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
            return x + drift_term + velocity_term

        prev_sample_mean = tree_tensor_map(compute_mean, x_t, velocity)

        # Generate noise and compute final sample using tree_tensor_map
        def add_noise(mean_val):
            variance_noise = torch.randn_like(mean_val)
            return mean_val + std_dev_t * torch.sqrt(torch.tensor(dt)) * variance_noise

        prev_sample = tree_tensor_map(add_noise, prev_sample_mean)

        return prev_sample


# https://en.wikipedia.org/wiki/Midpoint_method
class Midpoint(ODESolver):
    def step(self, dynamics_fn, x_t, t, dt, *args, **kwargs):
        half_dt = 0.5 * dt

        x_mid = Euler.step(self, dynamics_fn, x_t, t, half_dt, *args, **kwargs)

        velocity_mid = dynamics_fn(x_mid, t + half_dt, *args, **kwargs)
        x_tp1 = linear_approximation_step(x_t, dt, velocity_mid)
        return x_tp1


# https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
class RungeKutta4(ODESolver):

    def k1(self, dynamics_fn, x_t, t, dt, *args, **kwargs):
        return dynamics_fn(x_t, t, *args, **kwargs)

    def k2(self, dynamics_fn, x_t, t, dt, k1, *args, **kwargs):
        x_k1 = linear_approximation_step(x_t, dt * 0.5, k1)
        return dynamics_fn(x_k1, t + dt * 0.5, *args, **kwargs)

    def k3(self, dynamics_fn, x_t, t, dt, k2, *args, **kwargs):
        x_k2 = linear_approximation_step(x_t, dt * 0.5, k2)
        return dynamics_fn(x_k2, t + dt * 0.5, *args, **kwargs)

    def k4(self, dynamics_fn, x_t, t, dt, k3, *args, **kwargs):
        x_k3 = linear_approximation_step(x_t, dt, k3)
        return dynamics_fn(x_k3, t + dt, *args, **kwargs)

    def step(self, dynamics_fn, x_t, t, dt, *args, **kwargs):
        k1 = self.k1(dynamics_fn, x_t, t, dt, *args, **kwargs)
        k2 = self.k2(dynamics_fn, x_t, t, dt, k1, *args, **kwargs)
        k3 = self.k3(dynamics_fn, x_t, t, dt, k2, *args, **kwargs)
        k4 = self.k4(dynamics_fn, x_t, t, dt, k3, *args, **kwargs)

        def compute_velocity(k1, k2, k3, k4):
            return (k1 + 2 * k2 + 2 * k3 + k4) / 6

        velocity_k = tree_tensor_map(compute_velocity, k1, k2, k3, k4)
        x_tp1 = linear_approximation_step(x_t, dt, velocity_k)
        return x_tp1


# --- model.py ---


# default sampler in flow matching
uniform_sampler = torch.rand


# https://arxiv.org/pdf/2403.03206
def lognorm_sampler(mean=0.0, std=1.0, **kwargs):
    logit = torch.randn(**kwargs) * std + mean
    return torch.nn.functional.sigmoid(logit)


# for backwards compatibility; please do not use this
def rev_lognorm_sampler(mean=0.0, std=1.0, **kwargs):
    logit = torch.randn(**kwargs) * std + mean
    return 1 - torch.nn.functional.sigmoid(logit)


# https://arxiv.org/pdf/2210.02747
class FlowMatching(Base):
    SOLVER_METHODS = {
        "euler": Euler,
        "midpoint": Midpoint,
        "rk4": RungeKutta4,
        "sde": SDE,
    }

    def __init__(
        self,
        reverse_fn: Callable,
        sigma_min: float = 0.0,  # 0. = rectifier flow
        inference_steps: int = 100,
        time_scale: float = 1000.0,  # scale [0,1]-time before passing to `reverse_fn`
        training_time_sampler_fn: Callable = partial(
            lognorm_sampler,
            mean=0,
            std=1,
        ),
        reversed_timestamp=False,
        rescale_t=1.0,
        loss_fn=partial(torch.nn.functional.mse_loss, reduction="mean"),
        loss_weights=1.0,
        solver_method: Union[str, ODESolver] = "euler",
        solver_kwargs: dict = {},
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.reverse_fn = reverse_fn
        self.sigma_min = sigma_min
        self.inference_steps = inference_steps
        self.time_scale = time_scale
        self.training_time_sampler_fn = training_time_sampler_fn
        self.reversed_timestamp = reversed_timestamp
        self.rescale_t = rescale_t
        self.loss_fn = loss_fn
        self.loss_weights = loss_weights
        self._solver_method, self._solver = self._get_solver(
            solver_method, solver_kwargs
        )

    def _get_solver(self, solver_method, solver_kwargs):
        if solver_method in FlowMatching.SOLVER_METHODS:
            solver = FlowMatching.SOLVER_METHODS[solver_method](**solver_kwargs)
        elif isinstance(solver_method, ODESolver):
            solver_method = f"custom[{solver_method.__class__.__name__}]"
            solver = solver_method
        else:
            raise ValueError(
                f"Invalid solver `{solver_method}`, should be in {set(self.SOLVER_METHODS.keys())} or an ODESolver instance"
            )
        return solver_method, solver

    def _generate_noise_tensor(self, x_shape, x_device):
        dtype = getattr(self, '_compute_dtype', None)
        print(f"[DTYPE_DBG] _generate_noise_tensor: _compute_dtype={dtype} shape={x_shape} device={x_device}", flush=True)
        return torch.randn(
            x_shape,
            # generator=self.random_generator,
            device=x_device,
            dtype=dtype,
        )

    def _generate_noise(self, x_shape, x_device):
        def is_shape(maybe_shape):
            return isinstance(maybe_shape, Sequence) and all(
                (isinstance(s, int) and s >= 0) for s in maybe_shape
            )

        return optree.tree_map(
            partial(self._generate_noise_tensor, x_device=x_device),
            x_shape,
            is_leaf=is_shape,
            none_is_leaf=False,
        )

    def _generate_x0_tensor(self, x1: torch.Tensor):
        x0 = self._generate_noise_tensor(x1.shape, x1.device)
        return x0

    def _generate_xt_tensor(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor):
        # equation (22)
        tb = right_broadcasting(t.to(x1.device), x1)
        x_t = (1 - (1 - self.sigma_min) * tb) * x0 + tb * x1

        return x_t

    def _generate_target_tensor(self, x0: torch.Tensor, x1: torch.Tensor):
        # equation (23)
        target = x1 - (1 - self.sigma_min) * x0

        return target

    def _generate_x0(self, x1):
        return tree_tensor_map(self._generate_x0_tensor, x1)

    def _generate_xt(self, x0, x1, t):
        return tree_tensor_map(
            partial(self._generate_xt_tensor, t=t),
            x0,
            x1,
        )

    def _generate_target(self, x0, x1):
        return tree_tensor_map(
            self._generate_target_tensor,
            x0,
            x1,
        )

    def _generate_t(self, x1):
        first_tensor = optree.tree_flatten(x1)[0][0]
        batch_size = first_tensor.shape[0]
        device = first_tensor.device

        t = self.training_time_sampler_fn(
            size=(batch_size,),
            generator=self.random_generator,
        ).to(device)

        return t

    def loss(self, x1: torch.Tensor, *args_conditionals, **kwargs_conditionals):
        t = self._generate_t(x1)
        x0 = self._generate_x0(x1)
        x_t = self._generate_xt(x0, x1, t)
        target = self._generate_target(x0, x1)

        prediction = self.reverse_fn(
            x_t,
            t * self.time_scale,
            *args_conditionals,
            **kwargs_conditionals,
        )

        # broadcast & and compute loss
        loss = optree.tree_broadcast_map(
            lambda fn, weight, pred, targ: weight * fn(pred, targ),
            self.loss_fn,
            self.loss_weights,
            prediction,
            target,
        )

        total_loss = sum(optree.tree_flatten(loss)[0])

        # Create detailed loss breakdown
        detail_losses = {
            "flow_matching_loss": total_loss,
        }
        if isinstance(loss, dict):
            detail_losses.update(loss)
        return total_loss, detail_losses

    def _prepare_t(self, steps=None):
        steps = self.inference_steps if steps is None else steps
        t_seq = torch.linspace(0, 1, steps + 1)

        if self.rescale_t:
            t_seq = t_seq / (1 + (self.rescale_t - 1) * (1 - t_seq))

        if self.reversed_timestamp:
            t_seq = 1 - t_seq

        return t_seq

    def generate_iter(
        self,
        x_shape,
        x_device,
        *args_conditionals,
        **kwargs_conditionals,
    ):
        x_0 = self._generate_noise(x_shape, x_device)
        t_seq = self._prepare_t().to(x_device)

        for x_t, t in self._solver.solve_iter(
            self._generate_dynamics,
            x_0,
            t_seq,
            *args_conditionals,
            **kwargs_conditionals,
        ):
            yield t, x_t, ()

    def _generate_dynamics(
        self,
        x_t,
        t,
        *args_conditionals,
        **kwargs_conditionals,
    ):
        return self.reverse_fn(x_t, t * self.time_scale, *args_conditionals, **kwargs_conditionals)

    def _log_p0(self, x0):
        x0 = self._tree_flatten(x0)
        inside_exp = -(x0**2).sum(dim=1) / 2
        return inside_exp - math.log(2 * math.pi) / 2 * x0.shape[1]

    def log_likelihood(
        self,
        x1,
        solver=None,
        steps=None,
        z_samples=1,
        *args_conditionals,
        **kwargs_conditionals,
    ):
        device = tree_reduce_unique(lambda tensor: tensor.device, x1)
        # device = "cuda"
        t_seq = self._prepare_t(steps).to(device)
        t_seq = 1 - t_seq  # from x1 to x0
        solver = self._solver if solver is None else self._get_solver(solver)[1]

        x_0 = solver.solve(
            partial(self._log_likelihood_dynamics, device=device, z_samples=z_samples),
            {"x": x1, "log_p": 0.0},
            t_seq,
            *args_conditionals,
            **kwargs_conditionals,
        )

        log_p1 = x_0["log_p"] + self._log_p0(x_0["x"])

        return log_p1

    def _log_likelihood_dynamics(
        self,
        state,
        t,
        device,
        z_samples,
        *args_conditionals,
        **kwargs_conditionals,
    ):
        t = torch.tensor([t * self.time_scale], device=device, dtype=torch.float32)
        x_t = state["x"]

        with torch.set_grad_enabled(True):
            tree_tensor_map(lambda x,: x.requires_grad_(True), x_t)
            velocity = self.reverse_fn(
                x_t,
                t,
                *args_conditionals,
                **kwargs_conditionals,
            )

            # compute the divergence estimate
            div = self._compute_hutchinson_divergence(velocity, x_t, z_samples)

        tree_tensor_map(lambda x,: x.requires_grad_(False), x_t)
        velocity = tree_tensor_map(lambda x: x.detach(), velocity)

        return {"x": velocity, "log_p": div.detach()}

    def _tree_flatten(self, tree):
        flat_x = tree_tensor_map(lambda x: x.flatten(start_dim=1), tree)
        flat_x, _ = optree.tree_flatten(
            flat_x,
            is_leaf=lambda x: isinstance(x, torch.Tensor),
        )
        flat_x = torch.cat(flat_x, dim=1)
        return flat_x

    def _compute_hutchinson_divergence(self, velocity, x_t, z_samples):
        flat_velocity = self._tree_flatten(velocity)
        flat_velocity = flat_velocity.unsqueeze(-1)

        z = torch.randn(
            flat_velocity.shape[:-1] + (z_samples,),
            dtype=flat_velocity.dtype,
            device=flat_velocity.device,
        )
        z = z < 0
        z = z * 2.0 - 1.0
        z = z / math.sqrt(z_samples)

        # compute Hutchinson divergence estimator E[z^T D_x(vt) z] = E[D_x(z^T vt) z)]
        vt_dot_z = torch.einsum("ijk,ijk->ik", flat_velocity, z)
        grad_vt_dot_z = [
            gradient(vt_dot_z[..., i], x_t, create_graph=(z_samples > 1))
            for i in range(z_samples)
        ]
        grad_vt_dot_z = [self._tree_flatten(g) for g in grad_vt_dot_z]
        grad_vt_dot_z = torch.stack(grad_vt_dot_z, dim=-1)
        div = torch.einsum("ijk,ijk->i", grad_vt_dot_z, z)
        return div


def _get_device(x):
    device = tree_reduce_unique(lambda tensor: tensor.device, x)
    return device


class ConditionalFlowMatching(FlowMatching):
    def generate_iter(
        self,
        x_shape,
        x_device,
        *args_conditionals,
        **kwargs_conditionals,
    ):
        x_0 = self._generate_noise(x_shape, x_device)
        t_seq = self._prepare_t().to(x_device)

        noise_override = None
        if "noise_override" in kwargs_conditionals:
            noise_override = kwargs_conditionals["noise_override"]
            del kwargs_conditionals["noise_override"]
            if noise_override is not None:
                if type(x_0) == dict:
                    x_0.update(noise_override)
                else:
                    x_0 = noise_override

        for x_t, t in self._solver.solve_iter(
            self._generate_dynamics,
            x_0,
            t_seq,
            *args_conditionals,
            **kwargs_conditionals,
        ):
            if noise_override is not None:
                if type(noise_override) == dict:
                    x_t.update(noise_override)
                else:
                    x_t = noise_override
            yield t, x_t, ()


# --- classifier_free_guidance.py ---


def _zeros_like(struct):
    def make_zeros(x):
        if isinstance(x, torch.Tensor):
            return torch.zeros_like(x)
        return x

    return _pytree.tree_map(make_zeros, struct)


def zero_out(args, kwargs):
    args = _zeros_like(args)
    kwargs = _zeros_like(kwargs)
    return args, kwargs


def discard(args, kwargs):
    return (), {}


def _drop_tensors(struct):
    """
    Drop any conditioning that are tensors
    Not using _pytree since we actually want to throw them instead of keeping them.
    """
    if isinstance(struct, dict):
        return {
            k: _drop_tensors(v)
            for k, v in struct.items()
            if not isinstance(v, torch.Tensor)
        }
    elif isinstance(struct, (list, tuple)):
        filtered = [_drop_tensors(x) for x in struct if not isinstance(x, torch.Tensor)]
        return tuple(filtered) if isinstance(struct, tuple) else filtered
    else:
        return struct


def drop_tensors(args, kwargs):
    args = _drop_tensors(args)
    kwargs = _drop_tensors(kwargs)
    return args, kwargs


def add_flag(args, kwargs):
    kwargs["cfg"] = True
    return args, kwargs


class ClassifierFreeGuidance(torch.nn.Module):
    UNCONDITIONAL_HANDLING_TYPES = {
        "zeros": zero_out,
        "discard": discard,
        "drop_tensors": drop_tensors,
        "add_flag": add_flag,
    }

    def __init__(
        self,
        backbone,  # backbone should be a backbone/generator (e.g. DDPM/DDIM/FlowMatching)
        p_unconditional=0.1,
        strength=3.0,
        # "zeros" = set cond tensors to 0,
        # "discard" = remove cond arguments and let underlying model handle it
        # "drop_tensors" = drop all tensors but leave non-tensors
        # "add_flag" = add an argument in kwargs as "cfg" and defer the handling to generator backbone
        unconditional_handling="zeros",
        interval=None,  # only perform cfg if t within interval
    ):
        super().__init__()

        if not (
            unconditional_handling
            in ClassifierFreeGuidance.UNCONDITIONAL_HANDLING_TYPES
        ):
            raise RuntimeError(
                f"'{unconditional_handling}' is not valid for `unconditional_handling`, should be in {ClassifierFreeGuidance.UNCONDITIONAL_HANDLING_TYPES}"
            )

        self.backbone = backbone
        self.p_unconditional = p_unconditional
        self.strength = strength
        self.unconditional_handling = unconditional_handling
        self.interval = interval
        self._make_unconditional_args = (
            ClassifierFreeGuidance.UNCONDITIONAL_HANDLING_TYPES[
                self.unconditional_handling
            ]
        )

    def _cfg_step_tensor(self, y_cond, y_uncond, strength):
        return (1 + strength) * y_cond - strength * y_uncond

    def _cfg_step(self, y_cond, y_uncond, strength):
        if isinstance(strength, dict):
            return _pytree.tree_map(self._cfg_step_tensor, y_cond, y_uncond, strength)
        else:
            return _pytree.tree_map(partial(self._cfg_step_tensor, strength=strength), y_cond, y_uncond)

    def inner_forward(self, x, t, is_cond, strength, *args_cond, **kwargs_cond):
        y_cond = self.backbone(x, t, *args_cond, **kwargs_cond)
        if is_cond:
            return y_cond
        else:
            args_cond, kwargs_cond = self._make_unconditional_args(
                args_cond,
                kwargs_cond,
            )
            y_uncond = self.backbone(x, t, *args_cond, **kwargs_cond)
            return self._cfg_step(y_cond, y_uncond, strength)

    def forward(self, x, t, *args_cond, **kwargs_cond):
        # handle case when no conditional arguments are provided
        if len(args_cond) + len(kwargs_cond) == 0:  # unconditional
            if self.unconditional_handling != "discard":
                raise RuntimeError(
                    f"cannot call `ClassifierFreeGuidance` module without condition"
                )
            return self.backbone(x, t)
        else:  # conditional arguments are provided
            # training mode
            if self.training:
                coin_flip = random.random() < self.p_unconditional
                if coin_flip:  # unconditional
                    args_cond, kwargs_cond = self._make_unconditional_args(
                        args_cond,
                        kwargs_cond,
                    )
                return self.backbone(x, t, *args_cond, **kwargs_cond)
            else:  # inference mode
                strength = get_strength(self.strength, self.interval, t)
                is_cond = not any(x > 0.0 for x in _pytree.tree_flatten(strength)[0])
                return self.inner_forward(
                    x, t, is_cond, strength, *args_cond, **kwargs_cond
                )

def get_strength(strength, interval, t):
    if interval is None:
        return _pytree.tree_map(lambda x: 0.0, strength)

    # If interval is not a dict (single tuple), broadcast it
    if not isinstance(interval, dict):
        return _pytree.tree_map(
            lambda x: x if interval[0] <= t <= interval[1] else 0.0,
            strength
        )

    return _pytree.tree_map(
        lambda x, iv: x if iv[0] <= t <= iv[1] else 0.0,
        strength,
        interval
    )

class PointmapCFG(ClassifierFreeGuidance):

    def __init__(self, *args, strength_pm=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.strength_pm = strength_pm

    def _cfg_step_tensor(self, y_cond, y_uncond, y_unpm, strength, strength_pm):
        # https://arxiv.org/abs/2411.18613
        return y_cond \
            + strength_pm * (y_cond - y_unpm) \
            + strength * (y_unpm - y_uncond)

    def _cfg_step(self, y_cond, y_uncond, y_pm, strength, strength_pm):
        if isinstance(strength, dict):
            return _pytree.tree_map(self._cfg_step_tensor, y_cond, y_uncond, y_pm, strength, strength_pm)
        else:
            return _pytree.tree_map(partial(self._cfg_step_tensor, strength=strength, strength_pm=strength_pm), y_cond, y_uncond, y_pm)

    def inner_forward(self, x, t, is_cond, strength, strength_pm, *args_cond, **kwargs_cond):
        y_cond = self.backbone(x, t, *args_cond, **kwargs_cond)

        if is_cond:
            return y_cond
        else:
            force_drop_modalities = self.backbone.condition_embedder.force_drop_modalities
            self.backbone.condition_embedder.force_drop_modalities = ['pointmap', 'rgb_pointmap']
            y_pm = self.backbone(x, t, *args_cond, **kwargs_cond)
            self.backbone.condition_embedder.force_drop_modalities = force_drop_modalities

            args_cond, kwargs_cond = self._make_unconditional_args(
                args_cond,
                kwargs_cond,
            )
            y_uncond = self.backbone(x, t, *args_cond, **kwargs_cond)
            return self._cfg_step(y_cond, y_uncond, y_pm, strength, strength_pm)

    def forward(self, x, t, *args_cond, **kwargs_cond):
        # handle case when no conditional arguments are provided
        if len(args_cond) + len(kwargs_cond) == 0:  # unconditional
            if self.unconditional_handling != "discard":
                raise RuntimeError(
                    f"cannot call `ClassifierFreeGuidance` module without condition"
                )
            return self.backbone(x, t)
        else:  # conditional arguments are provided
            # training mode
            if self.training:
                coin_flip = random.random() < self.p_unconditional
                if coin_flip:  # unconditional
                    args_cond, kwargs_cond = self._make_unconditional_args(
                        args_cond,
                        kwargs_cond,
                    )
                return self.backbone(x, t, *args_cond, **kwargs_cond)
            else:  # inference mode
                strength = get_strength(self.strength, self.interval, t)
                is_cond = not any(x > 0.0 for x in _pytree.tree_flatten(strength)[0])
                strength_pm = get_strength(self.strength_pm, self.interval, t)
                return self.inner_forward(
                    x, t, is_cond, strength, strength_pm, *args_cond, **kwargs_cond
                )

class ClassifierFreeGuidanceWithExternalUnconditionalProbability(ClassifierFreeGuidance):

    def __init__(self, *args, use_unconditional_from_flow_matching=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_unconditional_from_flow_matching = use_unconditional_from_flow_matching

    def forward(self, x, t, *args_cond, p_unconditional=None, **kwargs_cond):
        # p_unconditional should be a value in [0, 1], indicating the probability of unconditional

        if p_unconditional is None:
            coin_flip = random.random() < self.p_unconditional
        else:
            coin_flip = random.random() < p_unconditional

        # handle case when no conditional arguments are provided
        if len(args_cond) + len(kwargs_cond) == 0:  # unconditional
            if self.unconditional_handling != "discard":
                raise RuntimeError(
                    f"cannot call `ClassifierFreeGuidance` module without condition"
                )
            return self.backbone(x, t)
        else:  # conditional arguments are provided
            # training mode
            if self.training:
                if coin_flip:  # unconditional
                    args_cond, kwargs_cond = self._make_unconditional_args(
                        args_cond,
                        kwargs_cond,
                    )
                return self.backbone(x, t, *args_cond, **kwargs_cond)
            else:  # inference mode
                strength = get_strength(self.strength, self.interval, t)
                is_cond = not any(x > 0.0 for x in _pytree.tree_flatten(strength)[0])
                return self.inner_forward(
                    x, t, is_cond, strength, *args_cond, **kwargs_cond
                )


# --- model.py (shortcut) ---


# https://arxiv.org/pdf/2410.12557
class ShortCut(FlowMatching):
    def __init__(
        self,
        no_shortcut=False,
        self_consistency_prob=0.25,
        shortcut_loss_weight=1.0,
        self_consistency_cfg_strength=3.0,
        ratio_cfg_samples_in_self_consistency_target=0.5,
        fm_in_shortcut_target_prob=0.0,
        fm_eps_max=0,
        batch_mode=False,
        cfg_modalities=["shape"],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.no_shortcut = no_shortcut
        self.self_consistency_prob = self_consistency_prob
        self.shortcut_loss_weight = shortcut_loss_weight
        self.self_consistency_cfg_strength = self_consistency_cfg_strength
        self.ratio_cfg_samples_in_self_consistency_target = ratio_cfg_samples_in_self_consistency_target
        self.fm_in_shortcut_target_prob = fm_in_shortcut_target_prob
        self.fm_eps_max = fm_eps_max
        self.batch_mode = batch_mode
        self.cfg_modalities = cfg_modalities

    def _generate_d(self, x1):
        """
        Generate shortcut step sizes d with binary-time schedule.

        This method ensures deterministic behavior for distributed training:
        - Exactly self_consistency_prob fraction of samples will have d > 0 (self-consistency)
        - Remaining samples will have d = 0 (flow matching)
        - All distributed ranks will have consistent counts, preventing deadlocks

        Args:
            x1: Input tensor or tree of tensors

        Returns:
            d: Tensor of step sizes with shape [batch_size]
        """
        first_tensor = optree.tree_flatten(x1)[0][0]
        batch_size = first_tensor.shape[0]
        device = first_tensor.device

        # Use binary-time schedule: d in {1/2^i for i in range(8)}
        base = [1 / 2**i for i in range(8)]

        # Deterministic approach: exactly self_consistency_prob fraction will have d>0
        # This ensures all distributed ranks have consistent behavior
        if self.batch_mode:
            num_self_consistency_samples = int(random.random() < self.self_consistency_prob) * batch_size
        else:
            num_self_consistency_samples = int(batch_size * self.self_consistency_prob)
        num_flow_matching_samples = batch_size - num_self_consistency_samples

        # Create deterministic d values
        d = torch.zeros(batch_size, device=device)

        if num_self_consistency_samples > 0:
            # Randomly select d values for self-consistency samples
            selected_elements = random.choices(base, k=num_self_consistency_samples)
            d[:num_self_consistency_samples] = torch.FloatTensor(selected_elements).to(device)

        # Shuffle the d values to randomize which samples get which d values
        # This maintains the deterministic count while randomizing positions
        shuffle_indices = torch.randperm(batch_size, device=device)
        d = d[shuffle_indices]

        return d

    @torch.no_grad()
    def compute_self_consistency_target(self, x_t, t, d, *args_conditionals, **kwargs_conditionals):
        """
        Compute self-consistency target for shortcut model's self-consistency objective.

        This method uses a mixed approach where:
        - First 25% of samples (num_cfg_samples) use CFG blending with strength 7.0
        - Remaining 75% of samples use conditional-only targets

        Safety guarantees:
        - Ensures at least 1 sample in CFG part (num_cfg_samples >= 1 for batch_size >= 2)
        - For batch_size < 2, falls back to all conditional-only (no CFG)
        - Handles edge cases where batch size is too small for mixed approach

        The process involves:
        1. Forward all samples through conditional model to get s_t_cond and s_td_cond
        2. Forward first num_cfg_samples through unconditional model to get s_t_uncond and s_td_uncond
        3. Apply CFG blending: (1 + strength) * cond - strength * uncond for first num_cfg_samples
        4. Concatenate CFG results with conditional-only results for remaining samples
        5. Average the two velocities to get final self-consistency target
        """
        # CFG strength for self-consistency target computation
        self_consistency_cfg_strength = self.self_consistency_cfg_strength

        # Mixed approach: configurable ratio of CFG:conditional-only samples
        batch_size = x_t.shape[0] if not isinstance(x_t, dict) else next(iter(x_t.values())).shape[0]
        if self.batch_mode:
            num_cfg_samples = int(random.random() < self.ratio_cfg_samples_in_self_consistency_target) * batch_size
        else:
            num_cfg_samples = int(batch_size * self.ratio_cfg_samples_in_self_consistency_target)  # Configurable ratio for CFG
        num_cond_only_samples = batch_size - num_cfg_samples  # Remaining for conditional-only
        use_fm_in_shortcut_target = random.random() < self.fm_in_shortcut_target_prob

        # Handle edge case where batch_size < 2 (fallback to all conditional-only)
        # if batch_size < 2:
        #     num_cfg_samples = 0
        #     num_cond_only_samples = batch_size


        # ### DEBUG ###############################
        # num_cfg_samples = 0
        # num_cond_only_samples = batch_size
        # # ### DEBUG ###############################

        # Step 1: Get velocity predictions at current time t
        # Forward all samples through conditional model
        s_t_cond = self.reverse_fn(
            x_t,
            t * self.time_scale,
            *args_conditionals,
            d=d * self.time_scale if not use_fm_in_shortcut_target else d * self.time_scale * 0,
            p_unconditional=0.0,
            **kwargs_conditionals,
        )

        # Handle CFG and conditional-only parts
        if num_cfg_samples > 0:
            # Forward first num_cfg_samples through unconditional model
            if isinstance(x_t, dict):
                x_t_cfg = {k: v[:num_cfg_samples] for k, v in x_t.items()}
            else:
                x_t_cfg = x_t[:num_cfg_samples]

            s_t_uncond = self.reverse_fn(
                x_t_cfg,
                t[:num_cfg_samples] * self.time_scale,
                *(arg[:num_cfg_samples] if not self.batch_mode and torch.is_tensor(arg) else arg for arg in args_conditionals),
                d=d[:num_cfg_samples] * self.time_scale if not use_fm_in_shortcut_target else d[:num_cfg_samples] * self.time_scale * 0,
                p_unconditional=1.0,
                **{k: v[:num_cfg_samples] if not self.batch_mode and torch.is_tensor(v) else v for k, v in kwargs_conditionals.items()},
            )

            # Apply CFG blending for first num_cfg_samples using our standard formula
            s_t_cfg = tree_tensor_map(
                lambda cond, uncond: (1 + self_consistency_cfg_strength) * cond - self_consistency_cfg_strength * uncond,
                tree_tensor_map(lambda x: x[:num_cfg_samples], s_t_cond), s_t_uncond
            )

            # Combine CFG results with conditional-only results for remaining samples
            if num_cond_only_samples > 0:
                s_t = tree_tensor_map(
                    lambda cfg, cond: torch.cat([cfg, cond[num_cfg_samples:]], dim=0),
                    s_t_cfg, s_t_cond
                )
            else:
                # All samples use CFG
                s_t = s_t_cond
                if isinstance(s_t_cond, dict):
                    for modality in self.cfg_modalities:
                        s_t[modality] = s_t_cfg[modality]
                else:
                    s_t = s_t_cfg
        else:
            # All samples use conditional-only (fallback for very small batches)
            s_t = s_t_cond

        # Step 2: Take a step of size d using current velocity
        x_td = tree_tensor_map(lambda x, v: x + v * d[..., None, None], x_t, s_t)

        # Step 3: Get velocity predictions at time t+d
        # Forward all samples through conditional model at t+d
        s_td_cond = self.reverse_fn(
            x_td,
            (t + d) * self.time_scale,
            *args_conditionals,
            d=d * self.time_scale if not use_fm_in_shortcut_target else d * self.time_scale * 0,
            p_unconditional=0.0,
            **kwargs_conditionals,
        )

        # Handle CFG and conditional-only parts at t+d
        if num_cfg_samples > 0:
            # Forward first num_cfg_samples through unconditional model at t+d
            if isinstance(x_td, dict):
                x_td_cfg = {k: v[:num_cfg_samples] for k, v in x_td.items()}
            else:
                x_td_cfg = x_td[:num_cfg_samples]

            s_td_uncond = self.reverse_fn(
                x_td_cfg,
                (t + d)[:num_cfg_samples] * self.time_scale,
                *(arg[:num_cfg_samples] if not self.batch_mode and torch.is_tensor(arg) else arg for arg in args_conditionals),
                d=d[:num_cfg_samples] * self.time_scale if not use_fm_in_shortcut_target else d[:num_cfg_samples] * self.time_scale * 0,
                p_unconditional=1.0,
                **{k: v[:num_cfg_samples] if not self.batch_mode and torch.is_tensor(v) else v for k, v in kwargs_conditionals.items()},
            )

            # Apply CFG blending for first num_cfg_samples at t+d using our standard formula
            s_td_cfg = tree_tensor_map(
                lambda cond, uncond: (1 + self_consistency_cfg_strength) * cond - self_consistency_cfg_strength * uncond,
                tree_tensor_map(lambda x: x[:num_cfg_samples], s_td_cond), s_td_uncond
            )

            # Combine CFG results with conditional-only results for remaining samples at t+d
            if num_cond_only_samples > 0:
                s_td = tree_tensor_map(
                    lambda cfg, cond: torch.cat([cfg, cond[num_cfg_samples:]], dim=0),
                    s_td_cfg, s_td_cond
                )
            else:
                # All samples use CFG
                s_td = s_td_cond
                if isinstance(s_td_cond, dict):
                    for modality in self.cfg_modalities:
                        s_td[modality] = s_td_cfg[modality]
                else:
                    s_td = s_td_cfg
        else:
            # All samples use conditional-only (fallback for very small batches)
            s_td = s_td_cond

        # Step 4: Compute self-consistency target as average of two velocities
        s_target = tree_tensor_map(lambda a, b: (a + b).detach() / 2, s_t, s_td)

        return s_target

    def _generate_t_and_d(self, x1):
        """
        Generate t and d together according to shortcut models paper.

        According to the paper: "During training, we first sample d, then sample t only at the discrete
        points for which the shortcut model will be queried, i.e. multiples of d. We train the
        self-consistency objective only at these timesteps."

        This ensures that when d > 0 (self-consistency samples), t is sampled at multiples of d.
        When d = 0 (flow matching samples), t can be sampled normally.
        """
        first_tensor = optree.tree_flatten(x1)[0][0]
        batch_size = first_tensor.shape[0]
        device = first_tensor.device

        # First sample d
        d = self._generate_d(x1)

        # Then sample t based on d
        t = torch.zeros(batch_size, device=device)

        # For flow matching samples (d = 0), sample t normally
        flow_matching_mask = (d == 0)
        if flow_matching_mask.any():
            num_flow_samples = flow_matching_mask.sum().item()
            t_flow = self.training_time_sampler_fn(
                size=(num_flow_samples,),
                generator=self.random_generator,
            ).to(device)
            t[flow_matching_mask] = t_flow

        # For self-consistency samples (d > 0), sample t at multiples of d
        self_consistency_mask = (d > 0)
        if self_consistency_mask.any():
            d_nonzero = d[self_consistency_mask]
            # Sample how many multiples of d to use for each sample
            # We want t to be k*d where k is a random integer such that t in [0, 1-d]
            # This ensures t + d <= 1
            max_multiples = torch.floor((1.0 - d_nonzero) / d_nonzero).long()
            # Ensure max_multiples is at least 0 to avoid empty range
            max_multiples = torch.clamp(max_multiples, min=0)

            # For each sample, randomly choose k from [0, max_multiples] - vectorized
            # Generate random values [0, 1) for all samples
            random_vals = torch.rand_like(d_nonzero)
            # Scale to [0, max_multiples + 1) and floor to get integers [0, max_multiples]
            k_values = torch.floor(random_vals * (max_multiples.float() + 1))
            # Compute t = k * d for all samples
            t_self_consistency = k_values * d_nonzero

            t[self_consistency_mask] = t_self_consistency

        return t, d

    def loss(self, x1: torch.Tensor, *args_conditionals, **kwargs_conditionals):
        """Compute shortcut model loss with mixed flow matching and self-consistency objectives"""
        # t, d = self._generate_t_and_d(x1)
        t = self._generate_t(x1)
        d = self._generate_d(x1)
        x0 = self._generate_x0(x1)
        x_t = self._generate_xt(x0, x1, t)

        # Determine which samples use flow matching vs  self-consistency
        flow_matching_indices = (d == 0).nonzero(as_tuple=False).squeeze(-1)  # 75% of the time use d=0 (flow matching), 25% use self-consistency
        self_consistency_indices = (d > 0).nonzero(as_tuple=False).squeeze(-1)
        d[d == 0] = torch.rand_like(d[d == 0]) * self.fm_eps_max

        # Clear autocast cache for gradient computation
        torch.clear_autocast_cache()

        # Get model prediction
        s = self.reverse_fn(
            x_t,
            t * self.time_scale,
            *args_conditionals,
            d=2 * d * self.time_scale,
            **kwargs_conditionals,
        )

        # Compute component losses separately by selecting relevant indices
        flow_matching_loss_val = torch.tensor(0.0, device=d.device, dtype=torch.float32)
        self_consistency_loss_val = torch.tensor(0.0, device=d.device, dtype=torch.float32)

        # Flow matching component (for d=0 samples)
        if len(flow_matching_indices) > 0:
            # Select samples where d=0 and compute flow matching target only for these samples
            x0_flow = tree_tensor_map(lambda x: x[flow_matching_indices], x0)
            x1_flow = tree_tensor_map(lambda x: x[flow_matching_indices], x1)
            s_flow = tree_tensor_map(lambda x: x[flow_matching_indices], s)

            # Compute flow matching target only for selected samples
            flow_matching_target = self._generate_target(x0_flow, x1_flow)

            flow_matching_loss = optree.tree_broadcast_map(
                lambda fn, weight, pred, targ: weight * fn(pred, targ),
                self.loss_fn,
                self.loss_weights,
                s_flow,
                flow_matching_target,
            )
            flow_matching_loss_val = sum(optree.tree_flatten(flow_matching_loss)[0])

        # Shortcut self-consistency component (for d>0 samples)
        if len(self_consistency_indices) > 0:
            # Select samples where d>0 and compute self-consistency target only for these samples
            x_t_shortcut = tree_tensor_map(lambda x: x[self_consistency_indices], x_t)
            t_shortcut = t[self_consistency_indices]
            d_shortcut = d[self_consistency_indices]
            s_shortcut = tree_tensor_map(lambda x: x[self_consistency_indices], s)

            # Create conditional arguments for selected samples
            if self.batch_mode:
                args_conditionals_shortcut = args_conditionals
                kwargs_conditionals_shortcut = kwargs_conditionals
            else:
                args_conditionals_shortcut = tuple(
                    tree_tensor_map(lambda x: x[self_consistency_indices], arg) if torch.is_tensor(arg) else arg
                    for arg in args_conditionals
                )
                kwargs_conditionals_shortcut = {
                    k: (tree_tensor_map(lambda x: x[self_consistency_indices], v) if torch.is_tensor(v) else v)
                    for k, v in kwargs_conditionals.items()
                }

            # Compute self-consistency target only for selected samples
            self_consistency_target = self.compute_self_consistency_target(
                x_t_shortcut, t_shortcut, d_shortcut,
                *args_conditionals_shortcut, **kwargs_conditionals_shortcut
            )

            self_consistency_loss = optree.tree_broadcast_map(
                lambda fn, weight, pred, targ: weight * fn(pred, targ),
                self.loss_fn,
                self.loss_weights,
                s_shortcut,
                self_consistency_target,
            )
            self_consistency_loss_val = sum(optree.tree_flatten(self_consistency_loss)[0])

        # Total loss is the sum of both components (linear combination)
        total_loss = flow_matching_loss_val + self.shortcut_loss_weight * self_consistency_loss_val

        # Create detailed loss breakdown
        detail_losses = {
            "flow_matching_loss": flow_matching_loss_val,
            "self_consistency_loss": self_consistency_loss_val,
        }
        return total_loss, detail_losses

    def _prepare_t_and_d(self, steps=None):
        """Prepare time sequence and step size for inference"""
        steps = self.inference_steps if steps is None else steps
        t_seq = np.linspace(0, 1, steps + 1)

        if self.no_shortcut:
            d = 0
        else:
            # Use uniform step size for inference
            d = 1 / steps

        if self.rescale_t:
            t_seq = t_seq / (1 + (self.rescale_t - 1) * (1 - t_seq))

        if self.reversed_timestamp:
            t_seq = 1 - t_seq

        return t_seq, d

    def generate_iter(
        self,
        x_shape,
        x_device,
        *args_conditionals,
        **kwargs_conditionals,
    ):
        """Generate samples using shortcut model"""
        x_0 = self._generate_noise(x_shape, x_device)
        t_seq, d = self._prepare_t_and_d()

        for x_t, t in self._solver.solve_iter(
            self._generate_dynamics,
            x_0,
            t_seq,
            d,
            *args_conditionals,
            **kwargs_conditionals,
        ):
            yield t, x_t, ()

    def _generate_dynamics(
        self,
        x_t,
        t,
        d,
        *args_conditionals,
        **kwargs_conditionals,
    ):
        """Generate dynamics for ODE solver"""
        t = torch.tensor(
            [t * self.time_scale], device=_get_device(x_t), dtype=torch.float32
        )
        d = torch.tensor(
            [d * self.time_scale], device=_get_device(x_t), dtype=torch.float32
        )
        return self.reverse_fn(x_t, t, *args_conditionals, d=d, **kwargs_conditionals)
