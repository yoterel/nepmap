import numpy as np
from typing import Callable, Optional
import gsoup
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,  # The number of input tensor channels.
        output_dim: int = None,  # The number of output tensor channels.
        net_depth: int = 8,  # The depth of the MLP.
        net_width: int = 256,  # The width of the MLP.
        skip_layer: int = 4,  # The layer to add skip layers to.
        hidden_init: Callable = nn.init.xavier_uniform_,
        hidden_activation: Callable = nn.LeakyReLU(),  # ReLU, LeakyReLU, GELU
        output_enabled: bool = True,
        output_init: Optional[Callable] = nn.init.xavier_uniform_,
        output_activation: Optional[Callable] = nn.Identity(),
        bias_enabled: bool = True,
        bias_init: Callable = nn.init.zeros_,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net_depth = net_depth
        self.net_width = net_width
        self.skip_layer = skip_layer
        self.hidden_init = hidden_init
        self.hidden_activation = hidden_activation
        self.output_enabled = output_enabled
        self.output_init = output_init
        self.output_activation = output_activation
        self.bias_enabled = bias_enabled
        self.bias_init = bias_init

        self.hidden_layers = nn.ModuleList()
        in_features = self.input_dim
        for i in range(self.net_depth):
            self.hidden_layers.append(
                nn.Linear(in_features, self.net_width, bias=bias_enabled)
            )
            if (
                (self.skip_layer is not None)
                and (i % self.skip_layer == 0)
                and (i > 0)
            ):
                in_features = self.net_width + self.input_dim
            else:
                in_features = self.net_width
        if self.output_enabled:
            self.output_layer = nn.Linear(
                in_features, self.output_dim, bias=bias_enabled
            )
        else:
            self.output_dim = in_features

        self.initialize()

    def initialize(self):
        def init_func_hidden(m):
            if isinstance(m, nn.Linear):
                if self.hidden_init is not None:
                    self.hidden_init(m.weight)
                if self.bias_enabled and self.bias_init is not None:
                    self.bias_init(m.bias)

        self.hidden_layers.apply(init_func_hidden)
        if self.output_enabled:

            def init_func_output(m):
                if isinstance(m, nn.Linear):
                    if self.output_init is not None:
                        self.output_init(m.weight)
                    if self.bias_enabled and self.bias_init is not None:
                        self.bias_init(m.bias)

            self.output_layer.apply(init_func_output)

    def forward(self, x):
        inputs = x
        for i in range(self.net_depth):
            x = self.hidden_layers[i](x)
            x = self.hidden_activation(x)
            if (
                (self.skip_layer is not None)
                and (i % self.skip_layer == 0)
                and (i > 0)
            ):
                x = torch.cat([x, inputs], dim=-1)
        if self.output_enabled:
            x = self.output_layer(x)
            x = self.output_activation(x)
        return x


class DenseLayer(MLP):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            net_depth=0,  # no hidden layers
            **kwargs,
        )


class VisibilityMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 3,  # The number of input tensor channels.
        condition_dim: int = 3,  # The number of condition tensor channels.
        net_depth: int = 8,  # The depth of the MLP.
        net_width: int = 256,  # The width of the MLP.
        skip_layer: int = 4,  # The layer to add skip layers to.
        net_depth_condition: int = 1,  # The depth of the second part of MLP.
        net_width_condition: int = 128,  # The width of the second part of MLP.
        output_dims: int = 1
    ):
        super().__init__()
        self.base = MLP(
            input_dim=input_dim,
            net_depth=net_depth,
            net_width=net_width,
            skip_layer=skip_layer,
            output_enabled=False,
        )
        hidden_features = self.base.output_dim
        self.bottleneck_layer = DenseLayer(hidden_features, net_width)
        self.rgb_layer = MLP(
            input_dim=net_width + condition_dim,
            output_dim=output_dims,
            net_depth=net_depth_condition,
            net_width=net_width_condition,
            skip_layer=None,
        )

    def forward(self, x, condition):
        x = self.base(x)
        if condition is not None:
            if condition.shape[:-1] != x.shape[:-1]:
                num_rays, n_dim = condition.shape
                condition = condition.view(
                    [num_rays] + [1] * (x.dim() - condition.dim()) + [n_dim]
                ).expand(list(x.shape[:-1]) + [n_dim])
            bottleneck = self.bottleneck_layer(x)
            x = torch.cat([bottleneck, condition], dim=-1)
        raw_visibility = self.rgb_layer(x)
        return raw_visibility


class NerfMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,  # The number of input tensor channels.
        condition_dim: int,  # The number of condition tensor channels.
        net_depth: int = 8,  # The depth of the MLP.
        net_width: int = 256,  # The width of the MLP.
        skip_layer: int = 4,  # The layer to add skip layers to.
        net_depth_condition: int = 1,  # The depth of the second part of MLP.
        net_width_condition: int = 128,  # The width of the second part of MLP.
        output_dims: int = 3
    ):
        super().__init__()
        self.base = MLP(
            input_dim=input_dim,
            net_depth=net_depth,
            net_width=net_width,
            skip_layer=skip_layer,
            output_enabled=False,
        )
        hidden_features = self.base.output_dim
        self.sigma_layer = DenseLayer(hidden_features, 1)

        if condition_dim > 0:
            self.bottleneck_layer = DenseLayer(hidden_features, net_width)
            self.rgb_layer = MLP(
                input_dim=net_width + condition_dim,
                output_dim=output_dims,
                net_depth=net_depth_condition,
                net_width=net_width_condition,
                skip_layer=None,
            )
        else:
            self.rgb_layer = DenseLayer(hidden_features, output_dims)

    def query_density(self, x):
        x = self.base(x)
        raw_sigma = self.sigma_layer(x)
        return raw_sigma

    def forward(self, x, condition=None):
        x = self.base(x)
        raw_sigma = self.sigma_layer(x)
        if condition is not None:
            if condition.shape[:-1] != x.shape[:-1]:
                num_rays, n_dim = condition.shape
                condition = condition.view(
                    [num_rays] + [1] * (x.dim() - condition.dim()) + [n_dim]
                ).expand(list(x.shape[:-1]) + [n_dim])
            bottleneck = self.bottleneck_layer(x)
            x = torch.cat([bottleneck, condition], dim=-1)
        raw_rgb = self.rgb_layer(x)
        return raw_rgb, raw_sigma


class SinusoidalEncoder(nn.Module):
    """Sinusoidal Positional Encoder used in Nerf."""

    def __init__(self, x_dim, min_deg, max_deg, use_identity=True, lowpass=True, final_step=None):
        super().__init__()
        self.x_dim = x_dim
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.use_identity = use_identity
        self.lowpass = lowpass
        self.final_step = final_step
        self.register_buffer(
            "scales", torch.tensor([2**i for i in range(min_deg, max_deg)])
        )

    @property
    def latent_dim(self) -> int:
        return (
            int(self.use_identity) + (self.max_deg - self.min_deg) * 2
        ) * self.x_dim

    def forward(self, x, cur_step=None) -> torch.Tensor:
        """
        Args:
            x: [..., x_dim]
        Returns:
            latent: [..., latent_dim]
        """
        if self.max_deg == self.min_deg:
            return x
        xb = x[..., None, :] * self.scales[:, None]
        xb = xb.view(*x.shape[:-1], (self.max_deg - self.min_deg) * self.x_dim) # freq1x, freq1y, freq1z, freq2y, ...
        latent = torch.sin(torch.cat([xb, xb + 0.5 * np.pi], dim=-1))
        # latent = torch.sin(torch.cat([xb[:, None, :], xb[:, None, :] + 0.5 * np.pi], dim=1)) # sin(freq1X), sin(freq2X), ... cos(freq1X), ...
        # if self.lowpass and cur_step is not None and self.final_step is not None:
        #     start, end = 0.08, 0.3
        #     L = self.max_deg - self.min_deg
        #     progress = (cur_step)/(self.final_step)
        #     alpha = (progress-start)/(end-start)
        #     k = torch.arange(L, dtype=torch.float32, device=x.device).repeat_interleave(3)
        #     lowpass_weights = torch.exp(alpha - k/L)
        #     lowpass_weights[lowpass_weights > 1] = 1
        #     # lowpass_weights = (1-torch.cos(torch.clamp(alpha-k, min=0, max=1)*np.pi))/2
        #     # lowpass_weights = torch.ones_like(k)
        #     # apply weights
        #     latent = latent * lowpass_weights[None, None, :]
        # latent = latent.view(xb.shape[0], -1)
        if self.use_identity:
            latent = torch.cat([x] + [latent], dim=-1)
        return latent


class VisibilityField(nn.Module):
    def __init__(
        self,
        net_depth: int = 8,  # The depth of the MLP.
        net_width: int = 128,  # The width of the MLP.
        skip_layer: int = 4,  # The layer to add skip layers to.
        net_depth_condition: int = 1,  # The depth of the second part of MLP.
        net_width_condition: int = 128,  # The width of the second part of MLP.
        final_step = None
    ) -> None:
        super().__init__()
        self.posi_encoder = SinusoidalEncoder(3, 0, 6, final_step=final_step)
        self.view_encoder = SinusoidalEncoder(3, 0, 5, final_step=final_step)
        self.mlp = VisibilityMLP(
            input_dim=self.posi_encoder.latent_dim,
            condition_dim=self.view_encoder.latent_dim,
            net_depth=net_depth,
            net_width=net_width,
            skip_layer=skip_layer,
            net_depth_condition=net_depth_condition,
            net_width_condition=net_width_condition,
        )

    def forward(self, x, condition):
        # return torch.ones_like(x[..., 0:1])
        x = self.posi_encoder(x)
        condition = self.view_encoder(condition)
        visibility = self.mlp(x, condition=condition)
        return F.relu(visibility)


class ProjectorRadianceField(nn.Module):
    def __init__(
        self,
        net_depth: int = 8,  # The depth of the MLP.
        net_width: int = 128,  # The width of the MLP.
        skip_layer: int = 4,  # The layer to add skip layers to.
        net_depth_condition: int = 1,  # The depth of the second part of MLP.
        net_width_condition: int = 128,  # The width of the second part of MLP.
        final_step = None,
        geo_freq: int = 7,
        mat_freq: int = 8,
    ) -> None:
        super().__init__()
        self.vis_network = VisibilityField()
        self.posi_encoder = SinusoidalEncoder(3, 0, geo_freq, final_step=final_step)
        self.mat_posi_encoder = SinusoidalEncoder(3, 0, mat_freq, final_step=final_step)
        self.mlp = NerfMLP(
            input_dim=self.posi_encoder.latent_dim,
            condition_dim=0,
            net_depth=net_depth,
            net_width=net_width,
            skip_layer=skip_layer,
            net_depth_condition=net_depth_condition,
            net_width_condition=net_width_condition,
            output_dims=3
        )
        self.mat_mlp = NerfMLP(
            input_dim=self.mat_posi_encoder.latent_dim,
            condition_dim=0,
            net_depth=net_depth,
            net_width=net_width,
            skip_layer=skip_layer,
            net_depth_condition=net_depth_condition,
            net_width_condition=net_width_condition,
            output_dims=3
        )

    def query_opacity(self, x, step_size):
        density = self.query_density(x)
        # if the density is small enough those two are the same.
        # opacity = 1.0 - torch.exp(-density * step_size)
        opacity = density * step_size
        return opacity

    def query_density(self, x):
        x = self.posi_encoder(x)
        sigma = self.mlp.query_density(x)
        return F.relu(sigma)

    def forward(self, x, views, texture_ids=None, light_field=None, calc_norms=True, cur_step=None):
        # predict shape properties
        if calc_norms and not x.requires_grad:
            x.requires_grad = True
        pos_x = self.posi_encoder(x, cur_step)
        predicted_normals, sigma = self.mlp(pos_x, None)
        # predicted_albedo, predicted_roughness, predicted_normals = others[:, :3], others[:, 3:4], others[:, 4:7]
        mat_pos_x = self.mat_posi_encoder(x, cur_step)
        predicted_albedo, predicted_roughness = self.mat_mlp(mat_pos_x, None)
        predicted_albedo = torch.sigmoid(predicted_albedo)
        predicted_roughness = torch.sigmoid(predicted_roughness)
        # predicted_albedo = torch.sigmoid(rgb_n_r[..., :3])
        # predicted_normals = rgb_n_r[..., 3:6]
        predicted_normals = torch.nn.functional.normalize(predicted_normals, dim=-1, eps=1e-6)
        # predicted_roughness = rgb_n_r[..., 6:7]
        # compute analytical normals
        if calc_norms:
            # x.requires_grad = True
            d_output = torch.ones_like(sigma[..., 0], requires_grad=False, device=sigma.device)
            normals = torch.autograd.grad(
                    outputs=sigma[..., 0],
                    inputs=x,
                    grad_outputs=d_output,
                    create_graph=False,
                    retain_graph=True)[0]
            # x.requires_grad = False
            # normals = normals.detach()
            normals = -1 * normals
            normals = torch.nn.functional.normalize(normals, dim=-1, eps=1e-5)
        else:
            normals = None
        # predict material properties
        # mat_pos_x = self.mat_posi_encoder(x)
        # predicted_albedo, predicted_roughness = self.mat_mlp(mat_pos_x, None)
        # predicted_albedo = torch.sigmoid(predicted_albedo)
        # compute sampled texture
        light_rays_d = None
        visible_texture = None
        pred_proj_transm = None
        sampled_texture = None
        if "projectors" in light_field:
            for i, projector in enumerate(light_field["projectors"]):
                light_rays_raw = projector["t"][None, :] - x  # towards projector
                light_rays_norm = light_rays_raw.norm(dim=-1, keepdim=True)
                light_rays_d = light_rays_raw / (light_rays_norm + 1e-5)
                pred_proj_transm = self.vis_network(x, -light_rays_d).detach()  # vis network expects direction towards surface
                K_proj = torch.eye(3, device=x.device)
                K_proj[0, 0] = projector["f"] * projector["W"]
                K_proj[0, 2] = projector["cx"] * projector["W"]
                K_proj[1, 1] = projector["f"] * projector["W"]
                K_proj[1, 2] = projector["cy"] * projector["W"]
                if projector["v"].shape[0] == 3:
                    R_proj = gsoup.rotvec2mat(projector["v"])
                else:
                    R_proj = gsoup.qvec2mat(projector["v"])
                T_proj = projector["t"]
                Rt = torch.cat((R_proj.T, -R_proj.T @ T_proj[:, None]), axis=1)  # now it is w2c
                KRt = K_proj @ Rt
                verts_screen = KRt @ gsoup.to_hom(x).T
                verts_screen_xy = verts_screen.T[:, :2] / verts_screen.T[:, 2:3]  # de-homogenize
                verts_screen_xy /= torch.cat((projector["W"], projector["H"]))  # go to 0:1 range
                verts_screen_xy = (verts_screen_xy * 2) - 1  # go to -1:+1 range
                verts_screen_xy = verts_screen_xy[None, None, :, :]
                if projector["textures"].ndim < 4:
                    input = projector["textures"] # * torch.clamp(projector["RGB"], 0, 1)[:, None, None]
                    input = input.unsqueeze(0)
                else:
                    input = projector["textures"].reshape(-1, *projector["textures"].shape[2:])[None, :, :, :]
                input = torch.pow(input, projector["gamma"])
                # input = torch.clamp(projector["RGB"], 0, 1).repeat(projector["textures"].shape[0])[None, :, None, None] * input
                sampled_texture = torch.nn.functional.grid_sample(input,
                                                                  verts_screen_xy[:, :, :, :2],
                                                                  mode='bilinear', padding_mode='zeros',
                                                                  align_corners=False).squeeze()
                if texture_ids is not None and projector["textures"].ndim >= 4:
                    sampled_texture = sampled_texture.reshape(projector["textures"].shape[0], 3, -1)
                    ids = texture_ids[:, i][None, None, :].repeat(1, 3, 1)
                    sampled_texture = torch.gather(sampled_texture, 0, ids).squeeze(0).permute(1, 0)
                else:
                    sampled_texture = sampled_texture.T.view(x.shape)
                visible_texture = sampled_texture * pred_proj_transm * projector["amp"]
        # compute color using BRDF (rays, normals, albedo, roughness, texture, weights)
        brdf1 = torch.zeros_like(predicted_normals)
        brdf2 = torch.zeros_like(predicted_normals)
        if "coloc_light" in light_field and light_field["coloc_light"] is not None:
            coloc_brdf = MicrofacetBRDF(-views, -views, predicted_normals, predicted_albedo, predicted_roughness) * light_field["coloc_light"]
            brdf2 = coloc_brdf
        if "projectors" in light_field:
            projectors_brdf = MicrofacetBRDF(light_rays_d, -views, predicted_normals, predicted_albedo, predicted_roughness) * visible_texture
            brdf1 = projectors_brdf
        return brdf1, brdf2, F.relu(sigma), pred_proj_transm, normals, predicted_normals, predicted_albedo, predicted_roughness, sampled_texture, visible_texture


def MicrofacetBRDF(light_dir, view_dir, normal, a, gamma, debug=False):
    """
    based on https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf
    :param light_dir: incoming light direction (B x S x 3), must be oriented towards outside of surface (i.e. same hemisphere as normal to surface)
    :param view_dir: view direction (B x S x 3), must be oriented towards outside of surface (i.e. same hemisphere as normal to surface)
    :param normal: normal (B x S x 3)
    :param a: diffuse albedo (B x S x 3)
    :param gamma: roughness (B x S x 1)
    :return: BRDF (B x S x 3)
    note: this doesnt actually return only the BRDF, but includes lighting cosine term and base color.
    """
    eps = 1e-6
    # b_size, samples, _ = normal.shape
    if debug:
        normal = torch.nan_to_num(normal)
    n = torch.nn.functional.normalize(normal, dim=-1, eps=eps)
    wi = torch.nn.functional.normalize(light_dir, dim=-1, eps=eps)
    wo = torch.nn.functional.normalize(view_dir, dim=-1, eps=eps)
    h = torch.nn.functional.normalize(wi + wo, dim=-1, eps=eps)
    n_dot_h = torch.bmm(n.view(-1, 1, 3), h.view(-1, 3, 1)).clamp(0, 1)
    wi_dot_h = torch.bmm(wi.view(-1, 1, 3), h.view(-1, 3, 1)).clamp(0, 1)
    n_dot_wo = torch.bmm(n.view(-1, 1, 3), wo.view(-1, 3, 1)).abs()
    n_dot_wi = torch.bmm(n.view(-1, 1, 3), wi.view(-1, 3, 1)).clamp(0, 1)

    gamma_cubed = gamma.contiguous().view(-1, 1, 1) ** 4
    F = get_f(wi_dot_h)
    G = get_g(n_dot_wi, n_dot_wo, gamma_cubed, eps)
    D = get_d(n_dot_h, gamma_cubed, eps)
    # diffuse = n_dot_wi * (1 - F) * (a.view(-1, 1, 3) / np.pi)
    diffuse = n_dot_wi * (a.view(-1, 1, 3)) / np.pi
    glossy = (F * G * D / (4 * n_dot_wo + eps))
    # glossy[n_dot_wi <= 0] = 0
    if debug:
        glossy = glossy.repeat(1, 1, 3)
        glossy[:, :, 1:] = 0
    # glossy[wi_dot_h <= 0] = 0
    # glossy[n_dot_wo <= 0] = 0
    R = diffuse + glossy
    # R[n_dot_wo.squeeze() < 0, :, :] = 0
    # R = n_dot_wo.broadcast_to(diffuse.shape).contiguous()
    # R = glossy.broadcast_to(diffuse.shape).contiguous()
    # R[n_dot_wi.squeeze() <= 0, :, :] = 0
    # R[n_dot_wo.squeeze() <= 0, :, :] = 0
    return R.view(a.shape)

def get_f(wi_dot_h):
    """
    fresnel dispersion
    """
    f0 = 0.04
    return f0 + (1-f0)*((1-wi_dot_h)**5)


def get_g(n_dot_wi, n_dot_wo, gamma_cubed, eps):
    """
    occlusion or shadowing
    """
    k = gamma_cubed / 2
    numerator = n_dot_wi * n_dot_wo
    denom = (n_dot_wo*(1-k)+k)*(n_dot_wi*(1-k)+k)
    return numerator / (denom + eps)


def get_d(n_dot_h, gamma_cubed, eps):
    """
    distribution of the microfacets
    """
    denom = np.pi*((n_dot_h**2)*(gamma_cubed - 1) + 1)**2
    return gamma_cubed / (denom + eps)
