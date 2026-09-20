import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

try:
    from .net import ConditionalUnet1D, get_resnet, replace_bn_with_gn
except ImportError:
    from src.generative_model.core.net import ConditionalUnet1D, get_resnet, replace_bn_with_gn


def _unit_vectors(values):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim > 1 and values.shape[-1] == 1:
        values = values[:, 0]
    if values.ndim == 2 and values.shape[1] == 2:
        return values / np.linalg.norm(values, axis=1, keepdims=True).clip(1e-8)
    # Existing rollout files use degrees; the kinematic controller stores radians.
    if np.nanmax(np.abs(values)) > 2 * np.pi + 1e-3:
        values = np.deg2rad(values)
    return np.stack((np.cos(values), np.sin(values)), axis=1).astype(np.float32)


class DiffusionPolicyDataset(Dataset):
    """Build direct-heading targets from an expert rollout or the GVF grid.

    A rollout directory contains ``imgs`` and either ``heading_commands.npy``
    or ``commands.npy``.  The grid mode reuses the already rendered panoramas
    and analytical ``(ux, uy)`` labels, avoiding a second Unity data collection.
    """

    def __init__(self, vector_field_file=None, img_dir=None, data_dir=None,
                 seq_length=1, pred_horizon=2, resize_shape=(256, 256),
                 crop_shape=(224, 224)):
        self.seq_length, self.pred_horizon = int(seq_length), int(pred_horizon)
        self.preprocess = transforms.Compose([
            transforms.Resize(resize_shape), transforms.CenterCrop(crop_shape)])
        self.rollout = data_dir is not None

        if self.rollout:
            self.img_dir = os.path.join(data_dir, "imgs")
            action_file = next((os.path.join(data_dir, n) for n in
                                ("heading_commands.npy", "commands.npy")
                                if os.path.exists(os.path.join(data_dir, n))), None)
            if action_file is None:
                raise FileNotFoundError("rollout needs heading_commands.npy or commands.npy")
            self.image_files = sorted(f for f in os.listdir(self.img_dir)
                                      if f.lower().endswith((".png", ".jpg", ".jpeg")))
            self.actions = _unit_vectors(np.load(action_file))
            n = min(len(self.image_files), len(self.actions))
            self.image_files, self.actions = self.image_files[:n], self.actions[:n]
            self.indices = range(max(0, n - self.seq_length - self.pred_horizon + 1))
        else:
            if vector_field_file is None or img_dir is None:
                raise ValueError("provide data_dir or vector_field_file + img_dir")
            table = pd.read_csv(vector_field_file)
            self.img_dir = img_dir
            self.image_files = [f"rectified_color_panorama_{int(i):04d}.jpg"
                                for i in table.frame_id]
            self.actions = _unit_vectors(table[["ux", "uy"]].to_numpy())
            self.indices = range(len(self.actions))

    def __len__(self):
        return len(self.indices)

    def _image(self, index):
        image = self.preprocess(Image.open(os.path.join(
            self.img_dir, self.image_files[index])).convert("RGB"))
        return torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 255.0

    def __getitem__(self, item):
        start = self.indices[item]
        ids = [min(start + i, len(self.image_files) - 1)
               for i in range(self.seq_length)]
        images = torch.stack([self._image(i) for i in ids])
        if self.rollout:
            target = self.actions[start + self.seq_length:
                                  start + self.seq_length + self.pred_horizon]
        else:
            target = np.repeat(self.actions[start][None], self.pred_horizon, axis=0)
        return images, torch.from_numpy(np.asarray(target, dtype=np.float32))


class DiffusionPolicy(nn.Module):
    """Compact image-conditioned DDPM policy."""

    def __init__(self, history_length=1, pred_horizon=2, action_dim=2,
                 diffusion_step_embed_dim=128, down_dims=(128, 256, 512),
                 num_train_timesteps=100, inference_steps=20):
        super().__init__()
        self.history_length, self.pred_horizon = history_length, pred_horizon
        self.action_dim, self.inference_steps = action_dim, inference_steps
        self.visual_encoder = replace_bn_with_gn(get_resnet("resnet18"))
        self.diffusion_model = ConditionalUnet1D(
            input_dim=action_dim, global_cond_dim=512 * history_length,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=list(down_dims))
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps, beta_start=1e-4,
            beta_end=0.02, beta_schedule="squaredcos_cap_v2",
            variance_type="fixed_small", clip_sample=True,
            prediction_type="epsilon")
        # DDIM with eta=0 keeps a locked latent deterministic at inference.
        self.ddim_scheduler = DDIMScheduler.from_config(self.noise_scheduler.config)

    def encode(self, images):
        b, h = images.shape[:2]
        feat = self.visual_encoder(images.flatten(0, 1))
        return feat.reshape(b, h, -1).flatten(1)

    def forward(self, noisy_actions, timesteps, images):
        # The shared U-Net downsamples the temporal axis twice.  Pad short or
        # non-multiple-of-four horizons and crop back so ``pred_horizon=2``
        # remains a valid, lightweight direct-heading policy.
        length = noisy_actions.shape[1]
        padded = max(4, ((length + 3) // 4) * 4)
        if padded != length:
            noisy_actions = F.pad(noisy_actions, (0, 0, 0, padded - length))
        output = self.diffusion_model(noisy_actions, timesteps,
                                      global_cond=self.encode(images))
        return output[:, :length]

    @torch.no_grad()
    def predict_action(self, images, num_inference_steps=None, initial_noise=None,
                       deterministic=False):
        b, device = images.shape[0], images.device
        actions = (torch.randn(b, self.pred_horizon, self.action_dim, device=device)
                   if initial_noise is None else initial_noise.to(device).clone())
        scheduler = self.ddim_scheduler if deterministic else self.noise_scheduler
        scheduler.set_timesteps(num_inference_steps or self.inference_steps)
        for t in scheduler.timesteps:
            ts = torch.full((b,), t, device=device, dtype=torch.long)
            actions = scheduler.step(self(actions, ts, images), t, actions,
                                     eta=0.0).prev_sample if deterministic else scheduler.step(
                                         self(actions, ts, images), t, actions).prev_sample
        return F.normalize(actions, dim=-1)


def diffusion_policy_loss(model, images, targets):
    b = images.shape[0]
    ts = torch.randint(0, model.noise_scheduler.config.num_train_timesteps,
                       (b,), device=images.device).long()
    noise = torch.randn_like(targets)
    noisy = model.noise_scheduler.add_noise(targets, noise, ts)
    return F.mse_loss(model(noisy, ts, images), noise)


def train_diffusion_policy(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = DiffusionPolicyDataset(
        vector_field_file=getattr(args, "vector_field_file", None),
        img_dir=getattr(args, "img_dir", None), data_dir=getattr(args, "data_dir", None),
        seq_length=args.seq_length, pred_horizon=args.pred_horizon)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=getattr(args, "num_workers", 0),
                        pin_memory=device.type == "cuda")
    model = DiffusionPolicy(args.seq_length, args.pred_horizon).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(
        args.save_dir, "logs_" + datetime.now().strftime("%Y%m%d_%H%M%S")))
    best = float("inf")
    for epoch in range(args.epochs):
        model.train(); total = 0.0
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            loss = diffusion_policy_loss(model, images, targets)
            optimizer.zero_grad(); loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step(); total += loss.item()
        avg = total / max(1, len(loader)); writer.add_scalar("loss/train", avg, epoch + 1)
        print(f"epoch {epoch + 1:03d}/{args.epochs}: {avg:.5f}")
        if avg < best:
            best = avg
            torch.save(model.state_dict(), os.path.join(
                args.save_dir, "diffusion_policy_best.pth"))
        if (epoch + 1) % args.checkpoint_interval == 0:
            torch.save(model.state_dict(), os.path.join(
                args.save_dir, f"diffusion_policy_{epoch + 1}.pth"))
    torch.save(model.state_dict(), os.path.join(
        args.save_dir, "diffusion_policy_final.pth"))
    writer.close()
    return best


if __name__ == "__main__":
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    p = argparse.ArgumentParser()
    p.add_argument("--path_type", default="cassini")
    p.add_argument("--data_dir", default=None)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--seq_length", type=int, default=1)
    p.add_argument("--pred_horizon", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--checkpoint_interval", type=int, default=50)
    a = p.parse_args()
    if a.data_dir is None:
        a.vector_field_file = os.path.join(root, "src", "path_following",
                                           f"vector_field_with_gradients_{a.path_type}.csv")
        a.img_dir = os.path.join(root, "src", "path_following",
                                 f"rectified_panorama_{a.path_type}")
    a.save_dir = os.path.join(root, "src", "generative_model", "models",
                              f"diffusion_policy_{a.path_type}_models")
    train_diffusion_policy(a)
