#!/usr/bin/env python3
"""Train a simple DDPM on MNIST and sample generated digits."""

import argparse
import math
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


@dataclass
class DiffusionConfig:
    timesteps: int = 200
    beta_start: float = 1e-4
    beta_end: float = 2e-2


class TimeEmbedding(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.emb_dim = emb_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.emb_dim // 2
        exponent = -math.log(10000) / max(half_dim - 1, 1)
        freqs = torch.exp(torch.arange(half_dim, device=t.device) * exponent)
        angles = t[:, None].float() * freqs[None, :]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if self.emb_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class SimpleDenoiser(nn.Module):
    def __init__(self, time_dim: int = 128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            TimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU(),
        )

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 32, 3, padding=1)
        self.out = nn.Conv2d(32, 1, 1)

        self.to_scale1 = nn.Linear(time_dim, 32)
        self.to_scale2 = nn.Linear(time_dim, 64)
        self.to_scale3 = nn.Linear(time_dim, 64)
        self.to_scale4 = nn.Linear(time_dim, 32)

    def _apply_time(self, x: torch.Tensor, t_emb: torch.Tensor, proj: nn.Linear) -> torch.Tensor:
        scale = proj(t_emb)[:, :, None, None]
        return x + scale

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)

        x = F.relu(self._apply_time(self.conv1(x), t_emb, self.to_scale1))
        x = F.relu(self._apply_time(self.conv2(x), t_emb, self.to_scale2))
        x = F.relu(self._apply_time(self.conv3(x), t_emb, self.to_scale3))
        x = F.relu(self._apply_time(self.conv4(x), t_emb, self.to_scale4))
        return self.out(x)


class DDPM:
    def __init__(self, cfg: DiffusionConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.timesteps, device=device)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bar = alpha_bar
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        a = self.sqrt_alpha_bar[t][:, None, None, None]
        b = self.sqrt_one_minus_alpha_bar[t][:, None, None, None]
        return a * x0 + b * noise

    @torch.no_grad()
    def sample(self, model: nn.Module, num_samples: int, image_size: int = 28) -> torch.Tensor:
        model.eval()
        x = torch.randn(num_samples, 1, image_size, image_size, device=self.device)

        for i in reversed(range(self.cfg.timesteps)):
            t = torch.full((num_samples,), i, device=self.device, dtype=torch.long)
            pred_noise = model(x, t)

            alpha_t = self.alphas[i]
            alpha_bar_t = self.alpha_bar[i]
            beta_t = self.betas[i]

            mean = (1.0 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * pred_noise)

            if i > 0:
                z = torch.randn_like(x)
                sigma = torch.sqrt(beta_t)
                x = mean + sigma * z
            else:
                x = mean

        model.train()
        return x.clamp(-1, 1)


def save_grid(images: torch.Tensor, out_path: str, nrow: int = 8) -> None:
    images = (images + 1) / 2.0
    images = images.cpu()
    total = images.shape[0]
    ncol = math.ceil(total / nrow)

    fig, axes = plt.subplots(ncol, nrow, figsize=(nrow, ncol))
    axes = axes.flatten()

    for idx, ax in enumerate(axes):
        ax.axis("off")
        if idx < total:
            ax.imshow(images[idx, 0], cmap="gray")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    os.makedirs(args.outdir, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    train_ds = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transform)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    model = SimpleDenoiser(time_dim=args.time_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    diffusion = DDPM(
        DiffusionConfig(timesteps=args.timesteps, beta_start=args.beta_start, beta_end=args.beta_end),
        device=device,
    )

    global_step = 0
    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for x0, _ in pbar:
            x0 = x0.to(device)
            t = torch.randint(0, args.timesteps, (x0.shape[0],), device=device)
            noise = torch.randn_like(x0)
            xt = diffusion.q_sample(x0, t, noise)
            pred = model(xt, t)

            loss = F.mse_loss(pred, noise)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", step=global_step)

        ckpt = os.path.join(args.outdir, f"mnist_ddpm_epoch_{epoch + 1}.pt")
        torch.save({"model": model.state_dict(), "epoch": epoch + 1, "args": vars(args)}, ckpt)

    samples = diffusion.sample(model, num_samples=args.num_samples)
    out_img = os.path.join(args.outdir, "mnist_samples.png")
    save_grid(samples, out_img)
    print(f"Training complete on {device}. Saved samples to {out_img}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train simple DDPM for MNIST")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--timesteps", type=int, default=200)
    p.add_argument("--beta-start", type=float, default=1e-4)
    p.add_argument("--beta-end", type=float, default=2e-2)
    p.add_argument("--time-dim", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--num-samples", type=int, default=64)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--outdir", default="./outputs")
    p.add_argument("--force-cpu", action="store_true")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    train(args)
