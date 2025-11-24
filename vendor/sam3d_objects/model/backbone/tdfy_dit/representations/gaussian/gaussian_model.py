# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch
import numpy as np
from plyfile import PlyData, PlyElement
from .general_utils import inverse_sigmoid, strip_symmetric, build_scaling_rotation
from ...renderers.sh_utils import SH2RGB


class Gaussian:
    def __init__(
        self,
        aabb: list,
        sh_degree: int = 0,
        mininum_kernel_size: float = 0.0,
        scaling_bias: float = 0.01,
        opacity_bias: float = 0.1,
        scaling_activation: str = "exp",
        device="cuda",
    ):
        self.init_params = {
            "aabb": aabb,
            "sh_degree": sh_degree,
            "mininum_kernel_size": mininum_kernel_size,
            "scaling_bias": scaling_bias,
            "opacity_bias": opacity_bias,
            "scaling_activation": scaling_activation,
        }

        self.sh_degree = sh_degree
        self.active_sh_degree = sh_degree
        self.mininum_kernel_size = mininum_kernel_size
        self.scaling_bias = scaling_bias
        self.opacity_bias = opacity_bias
        self.scaling_activation_type = scaling_activation
        self.device = device
        self.aabb = torch.tensor(aabb, dtype=torch.float32, device=device)
        self.setup_functions()

        self._xyz = None
        self._features_dc = None
        self._features_rest = None
        self._scaling = None
        self._rotation = None
        self._opacity = None

    def setup_functions(self):
        if self.scaling_activation_type == "exp":
            self.scaling_activation = torch.exp
            self.inverse_scaling_activation = torch.log
        elif self.scaling_activation_type == "softplus":
            self.scaling_activation = torch.nn.functional.softplus
            self.inverse_scaling_activation = softplus_inverse_scaling_activation

        self.covariance_activation = self.build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        self.scale_bias = self.inverse_scaling_activation(
            torch.tensor(self.scaling_bias)
        ).cuda()
        self.rots_bias = torch.zeros((4)).cuda()
        self.rots_bias[0] = 1
        self.opacity_bias = self.inverse_opacity_activation(
            torch.tensor(self.opacity_bias)
        ).cuda()

    @staticmethod
    def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
        L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        symm = strip_symmetric(actual_covariance)
        return symm

    @property
    def get_scaling(self):
        scales = self.scaling_activation(self._scaling + self.scale_bias)
        scales = torch.square(scales) + self.mininum_kernel_size**2
        scales = torch.sqrt(scales)
        return scales

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation + self.rots_bias[None, :])

    @property
    def get_xyz(self):
        return self._xyz * self.aabb[None, 3:] + self.aabb[None, :3]

    @property
    def get_features(self):
        return (
            torch.cat((self._features_dc, self._features_rest), dim=2)
            if self._features_rest is not None
            else self._features_dc
        )

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity + self.opacity_bias)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation + self.rots_bias[None, :]
        )

    def from_scaling(self, scales):
        scales = torch.sqrt(torch.square(scales) - self.mininum_kernel_size**2)
        self._scaling = self.inverse_scaling_activation(scales) - self.scale_bias

    def from_rotation(self, rots):
        self._rotation = rots - self.rots_bias[None, :]

    def from_xyz(self, xyz):
        self._xyz = (xyz - self.aabb[None, :3]) / self.aabb[None, 3:]

    def from_features(self, features):
        self._features_dc = features

    def from_opacity(self, opacities):
        self._opacity = self.inverse_opacity_activation(opacities) - self.opacity_bias

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def save_ply(self, path):
        xyz = self.get_xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)

        # Prepare raw SH coefficients (for Gaussian Splatting viewers)
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )

        # Also convert SH to RGB colors (for standard PLY viewers)
        f_dc_tensor = self._features_dc[:, 0, :].detach()  # Shape: [N, 3]

        # Apply SH rendering formula: C0 * sh + 0.5
        C0 = 0.28209479177387814
        colors_linear = (f_dc_tensor * C0 + 0.5).cpu().numpy()

        # Normalize per-channel to maximize color variation
        # This normalizes R, G, B independently to use full [0, 255] range
        colors_normalized = np.zeros_like(colors_linear)
        for i in range(3):  # R, G, B channels
            channel = colors_linear[:, i]
            channel_min = channel.min()
            channel_max = channel.max()
            if channel_max > channel_min:
                colors_normalized[:, i] = (channel - channel_min) / (channel_max - channel_min)
            else:
                colors_normalized[:, i] = 0.5
        colors_rgb = np.clip(colors_normalized * 255, 0, 255).astype(np.uint8)

        print(f"[PLY Export] Saving both SH coefficients and RGB colors (ASCII format)")
        print(f"  SH range: [{f_dc_tensor.min():.3f}, {f_dc_tensor.max():.3f}]")
        print(f"  RGB range: [0, 255] uint8 (per-channel normalized)")
        print(f"  Sample RGB (first 5):")
        for i in range(min(5, colors_rgb.shape[0])):
            print(f"    Point {i}: R={colors_rgb[i,0]:3d} G={colors_rgb[i,1]:3d} B={colors_rgb[i,2]:3d}")

        opacities = inverse_sigmoid(self.get_opacity).detach().cpu().numpy()
        if opacities.ndim == 1:
            opacities = opacities[:, np.newaxis]

        scale = torch.log(self.get_scaling).detach().cpu().numpy()
        rotation = (self._rotation + self.rots_bias[None, :]).detach().cpu().numpy()

        # Build dtype with BOTH raw SH coefficients AND RGB colors
        # RGB as uint8 [0, 255] for ASCII PLY (VTK.js compatibility)
        # Gaussian Splatting viewers will use f_dc_0/1/2
        dtype_full = [("x", "f4"), ("y", "f4"), ("z", "f4"),
                      ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
                      ("red", "u1"), ("green", "u1"), ("blue", "u1")]

        # Add raw SH coefficients (f_dc_0, f_dc_1, f_dc_2)
        for i in range(f_dc.shape[1]):
            dtype_full.append(("f_dc_{}".format(i), "f4"))

        dtype_full.append(("opacity", "f4"))

        for i in range(scale.shape[1]):
            dtype_full.append(("scale_{}".format(i), "f4"))
        for i in range(rotation.shape[1]):
            dtype_full.append(("rot_{}".format(i), "f4"))

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, colors_rgb, f_dc, opacities, scale, rotation), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")

        # Write as ASCII format for VTK.js compatibility
        plydata = PlyData([el])
        plydata.text = True
        plydata.write(path)

    def load_ply(self, path):
        print(f"[Gaussian] load_ply: Loading from {path}")
        plydata = PlyData.read(path)
        print(f"[Gaussian] load_ply: PLY loaded, {len(plydata.elements[0])} vertices")
        print(f"[Gaussian] load_ply: Properties: {[p.name for p in plydata.elements[0].properties]}")

        print(f"[Gaussian] load_ply: Extracting xyz coordinates...")
        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        print(f"[Gaussian] load_ply: xyz shape={xyz.shape}, dtype={xyz.dtype}, strides={xyz.strides}")

        print(f"[Gaussian] load_ply: Extracting opacities...")
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].copy()
        print(f"[Gaussian] load_ply: opacities shape={opacities.shape}, dtype={opacities.dtype}, strides={opacities.strides}")

        print(f"[Gaussian] load_ply: Extracting features_dc...")
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
        print(f"[Gaussian] load_ply: features_dc shape={features_dc.shape}, dtype={features_dc.dtype}")

        if self.sh_degree > 0:
            extra_f_names = [
                p.name
                for p in plydata.elements[0].properties
                if p.name.startswith("f_rest_")
            ]
            extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
            assert len(extra_f_names) == 3 * (self.sh_degree + 1) ** 2 - 3
            features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
            for idx, attr_name in enumerate(extra_f_names):
                features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
            features_extra = features_extra.reshape(
                (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
            )

        print(f"[Gaussian] load_ply: Extracting scales...")
        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
        print(f"[Gaussian] load_ply: scales shape={scales.shape}, dtype={scales.dtype}")

        print(f"[Gaussian] load_ply: Extracting rotations...")
        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        print(f"[Gaussian] load_ply: rots shape={rots.shape}, dtype={rots.dtype}")

        # convert to actual gaussian attributes
        print(f"[Gaussian] load_ply: Converting to torch tensors on device={self.device}...")
        print(f"[Gaussian] load_ply: Converting xyz...")
        xyz = torch.tensor(xyz, dtype=torch.float, device=self.device)
        print(f"[Gaussian] load_ply: Converting features_dc...")
        features_dc = (
            torch.tensor(features_dc, dtype=torch.float, device=self.device)
            .transpose(1, 2)
            .contiguous()
        )
        if self.sh_degree > 0:
            print(f"[Gaussian] load_ply: Converting features_extra...")
            features_extra = (
                torch.tensor(features_extra, dtype=torch.float, device=self.device)
                .transpose(1, 2)
                .contiguous()
            )
        print(f"[Gaussian] load_ply: Converting opacities...")
        opacities = torch.sigmoid(
            torch.tensor(opacities, dtype=torch.float, device=self.device)
        )
        print(f"[Gaussian] load_ply: Converting scales...")
        scales = torch.exp(torch.tensor(scales, dtype=torch.float, device=self.device))
        print(f"[Gaussian] load_ply: Converting rots...")
        rots = torch.tensor(rots, dtype=torch.float, device=self.device)

        # convert to _hidden attributes
        print(f"[Gaussian] load_ply: Setting internal attributes...")
        self._xyz = (xyz - self.aabb[None, :3]) / self.aabb[None, 3:]
        self._features_dc = features_dc
        if self.sh_degree > 0:
            self._features_rest = features_extra
        else:
            self._features_rest = None
        self._opacity = self.inverse_opacity_activation(opacities) - self.opacity_bias
        self._scaling = (
            self.inverse_scaling_activation(
                torch.sqrt(torch.square(scales) - self.mininum_kernel_size**2)
            )
            - self.scale_bias
        )
        self._rotation = rots - self.rots_bias[None, :]
        print(f"[Gaussian] load_ply: Complete! Loaded {self._xyz.shape[0]} gaussians")

def softplus_inverse_scaling_activation(x):
    return x + torch.log(-torch.expm1(-x))