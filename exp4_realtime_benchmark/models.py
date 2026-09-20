import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def _replace_bn(module, features_per_group=16):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            groups = max(1, child.num_features // features_per_group)
            setattr(module, name, nn.GroupNorm(groups, child.num_features))
        else:
            _replace_bn(child, features_per_group)
    return module


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half = self.dim // 2
        scale = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=x.device) * -scale)
        emb = x[..., None] * emb
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Conv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels), nn.SiLU())

    def forward(self, x):
        return self.block(x)


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3,
                 n_groups=8, cond_predict_scale=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups)])
        cond_channels = out_channels * (2 if cond_predict_scale else 1)
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.SiLU(), nn.Linear(cond_dim, cond_channels))
        self.residual_conv = (nn.Conv1d(in_channels, out_channels, 1)
                              if in_channels != out_channels else nn.Identity())

    def forward(self, x, cond):
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)[..., None]
        if self.cond_predict_scale:
            scale, bias = embed.reshape(embed.shape[0], 2,
                                        self.out_channels, 1).unbind(1)
            out = scale * out + bias
        else:
            out = out + embed
        return self.blocks[1](out) + self.residual_conv(x)


class ConditionalUnet1D(nn.Module):
    def __init__(self, input_dim, global_cond_dim=None,
                 diffusion_step_embed_dim=256, down_dims=(256, 512, 1024),
                 kernel_size=3, n_groups=8, cond_predict_scale=False):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.SiLU(), nn.Linear(diffusion_step_embed_dim * 4,
                                  diffusion_step_embed_dim))
        cond_dim = diffusion_step_embed_dim + (global_cond_dim or 0)
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(all_dims[-1], all_dims[-1], cond_dim,
                                       kernel_size, n_groups, cond_predict_scale),
            ConditionalResidualBlock1D(all_dims[-1], all_dims[-1], cond_dim,
                                       kernel_size, n_groups, cond_predict_scale)])
        self.down_modules = nn.ModuleList()
        for i, (dim_in, dim_out) in enumerate(in_out):
            last = i >= len(in_out) - 1
            self.down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_in, dim_out, cond_dim,
                                           kernel_size, n_groups, cond_predict_scale),
                ConditionalResidualBlock1D(dim_out, dim_out, cond_dim,
                                           kernel_size, n_groups, cond_predict_scale),
                Downsample1d(dim_out) if not last else nn.Identity()]))
        self.up_modules = nn.ModuleList()
        for i, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            last = i >= len(in_out) - 1
            self.up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_out * 2, dim_in, cond_dim,
                                           kernel_size, n_groups, cond_predict_scale),
                ConditionalResidualBlock1D(dim_in, dim_in, cond_dim,
                                           kernel_size, n_groups, cond_predict_scale),
                Upsample1d(dim_in) if not last else nn.Identity()]))
        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size, n_groups),
            nn.Conv1d(start_dim, input_dim, 1))

    def forward(self, sample, timestep, global_cond=None):
        x = sample.permute(0, 2, 1)
        if timestep.ndim == 0:
            timestep = timestep[None]
        timestep = timestep.to(x.device).expand(x.shape[0])
        cond = self.diffusion_step_encoder(timestep)
        if global_cond is not None:
            cond = torch.cat((cond, global_cond), dim=-1)
        h = []
        for block in self.down_modules:
            resnet, resnet2, downsample = block[0], block[1], block[2]
            x = resnet(x, cond)
            x = resnet2(x, cond)
            h.append(x)
            x = downsample(x)
        for i in range(len(self.mid_modules)):
            x = self.mid_modules[i](x, cond)
        for block in self.up_modules:
            resnet, resnet2, upsample = block[0], block[1], block[2]
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, cond)
            x = resnet2(x, cond)
            x = upsample(x)
        return self.final_conv(x).permute(0, 2, 1)


class FlowPolicy(nn.Module):
    def __init__(self, image_size=128, action_dim=4, hidden_dim=128,
                 history_length=1):
        super().__init__()
        self.action_dim = action_dim
        self.history_length = history_length
        squeezenet = torchvision.models.squeezenet1_1(weights=None)
        encoder = nn.Sequential(*list(squeezenet.features.children()))
        encoder.add_module("adaptive_pool", nn.AdaptiveAvgPool2d(1))
        encoder.add_module("flatten", nn.Flatten())
        encoder.add_module("downsample", nn.Sequential(nn.Linear(512, 24),
                                                        nn.ReLU()))
        self.visual_encoder = encoder
        self.sequence_encoder = nn.Sequential(
            nn.Linear(24 * history_length, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 24))
        self.time_embed_dim = 16
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(16), nn.Linear(16, 64), nn.Mish(),
            nn.Linear(64, 16))
        self.flow_net = nn.Sequential(
            nn.Linear(24 + 16 + action_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, action_dim))

    def encode(self, image_sequence):
        b, h = image_sequence.shape[:2]
        features = self.visual_encoder(image_sequence.flatten(0, 1))
        return self.sequence_encoder(features.reshape(b, h, -1).flatten(1))

    def forward_cached(self, features, action_noise, t):
        b = features.shape[0]
        time = self.time_emb(t).reshape(b, 16)
        return self.flow_net(torch.cat((features, time, action_noise), dim=1))

    def forward(self, image_sequence, action_noise, t):
        return self.forward_cached(self.encode(image_sequence), action_noise, t)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten())
        self.regressor = nn.Sequential(nn.Linear(64 * 28 * 28, 16),
                                       nn.ReLU(), nn.Linear(16, 4))

    def forward(self, x):
        return self.regressor(self.features(x))


class DiffusionPolicy(nn.Module):
    def __init__(self, history_length=1, pred_horizon=2, inference_steps=20):
        super().__init__()
        self.history_length = history_length
        self.pred_horizon = pred_horizon
        self.action_dim = 2
        self.inference_steps = inference_steps
        resnet = torchvision.models.resnet18(weights=None)
        resnet.fc = nn.Identity()
        self.visual_encoder = _replace_bn(resnet)
        self.diffusion_model = ConditionalUnet1D(
            2, global_cond_dim=512 * history_length,
            diffusion_step_embed_dim=128, down_dims=(128, 256, 512))

    def encode(self, images):
        b, h = images.shape[:2]
        return self.visual_encoder(images.flatten(0, 1)).reshape(b, h, -1).flatten(1)

    def forward_cached(self, noisy_actions, timesteps, features):
        length = noisy_actions.shape[1]
        padded = max(4, (length + 3) // 4 * 4)
        if padded != length:
            noisy_actions = F.pad(noisy_actions, (0, 0, 0, padded - length))
        output = self.diffusion_model(noisy_actions, timesteps,
                                      global_cond=features)
        return output[:, :length]

    def forward(self, noisy_actions, timesteps, images):
        return self.forward_cached(noisy_actions, timesteps, self.encode(images))
