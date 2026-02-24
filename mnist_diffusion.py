#!/usr/bin/env python3
"""Train a DDPM on MNIST and generate plots/dashboard artifacts."""

import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone

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


def build_group_norm(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.norm1 = build_group_norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = build_group_norm(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class UNetDenoiser(nn.Module):
    """A compact U-Net style denoiser with residual blocks and time conditioning."""

    def __init__(self, time_dim: int = 128):
        super().__init__()
        time_hidden = time_dim * 4
        self.time_mlp = nn.Sequential(
            TimeEmbedding(time_dim),
            nn.Linear(time_dim, time_hidden),
            nn.SiLU(),
            nn.Linear(time_hidden, time_hidden),
        )

        self.stem = nn.Conv2d(1, 64, 3, padding=1)

        self.down1 = ResidualBlock(64, 64, time_hidden)
        self.downsample1 = nn.Conv2d(64, 128, 4, stride=2, padding=1)  # 28 -> 14
        self.down2 = ResidualBlock(128, 128, time_hidden)
        self.downsample2 = nn.Conv2d(128, 256, 4, stride=2, padding=1)  # 14 -> 7

        self.mid1 = ResidualBlock(256, 256, time_hidden)
        self.mid2 = ResidualBlock(256, 256, time_hidden)

        self.up1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)  # 7 -> 14
        self.up_block1 = ResidualBlock(256, 128, time_hidden)
        self.up2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)  # 14 -> 28
        self.up_block2 = ResidualBlock(128, 64, time_hidden)

        self.out_norm = build_group_norm(64)
        self.out_conv = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)

        h0 = self.stem(x)
        h1 = self.down1(h0, t_emb)
        h2 = self.downsample1(h1)
        h3 = self.down2(h2, t_emb)
        h4 = self.downsample2(h3)

        h = self.mid1(h4, t_emb)
        h = self.mid2(h, t_emb)

        h = self.up1(h)
        if h.shape[-2:] != h3.shape[-2:]:
            h = F.interpolate(h, size=h3.shape[-2:], mode="nearest")
        h = self.up_block1(torch.cat([h, h3], dim=1), t_emb)

        h = self.up2(h)
        if h.shape[-2:] != h1.shape[-2:]:
            h = F.interpolate(h, size=h1.shape[-2:], mode="nearest")
        h = self.up_block2(torch.cat([h, h1], dim=1), t_emb)

        return self.out_conv(F.silu(self.out_norm(h)))


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


def save_loss_plots(step_losses: list[float], epoch_losses: list[float], outdir: str) -> tuple[str, str]:
    step_plot = os.path.join(outdir, "loss_curve_step.png")
    epoch_plot = os.path.join(outdir, "loss_curve_epoch.png")

    plt.figure(figsize=(10, 4))
    plt.plot(range(1, len(step_losses) + 1), step_losses, linewidth=1.0)
    plt.title("Training loss per step")
    plt.xlabel("Step")
    plt.ylabel("MSE loss")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(step_plot, dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker="o", linewidth=1.5)
    plt.title("Mean training loss per epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Mean MSE loss")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(epoch_plot, dpi=160)
    plt.close()

    return step_plot, epoch_plot


def save_architecture_schematic(outdir: str) -> str:
    schematic_path = os.path.join(outdir, "architecture_schematic.png")
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.axis("off")

    blocks = [
        (0.02, 0.60, 0.16, 0.18, "Input x_t\\n1x28x28"),
        (0.22, 0.60, 0.16, 0.18, "Stem Conv\\n1->64"),
        (0.42, 0.60, 0.16, 0.18, "Down Block 1\\nRes(64)") ,
        (0.62, 0.60, 0.16, 0.18, "Down Block 2\\nRes(128)") ,
        (0.82, 0.60, 0.16, 0.18, "Bottleneck\\nRes(256) x2"),
        (0.62, 0.28, 0.16, 0.18, "Up Block 1\\nRes(256->128)"),
        (0.42, 0.28, 0.16, 0.18, "Up Block 2\\nRes(128->64)"),
        (0.22, 0.28, 0.16, 0.18, "Output Conv\\n64->1"),
    ]

    for x, y, w, h, label in blocks:
        rect = plt.Rectangle((x, y), w, h, edgecolor="#1f2937", facecolor="#e5edff", linewidth=2.0)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=14)

    flow = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]
    for i, j in flow:
        x1 = blocks[i][0] + blocks[i][2]
        y1 = blocks[i][1] + blocks[i][3] / 2
        x2 = blocks[j][0]
        y2 = blocks[j][1] + blocks[j][3] / 2
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=2.2, color="#111827"))

    ax.text(0.50, 0.92, "Sinusoidal t embedding -> MLP -> injected into all residual blocks", ha="center", fontsize=15)
    for idx in [2, 3, 4, 5, 6]:
        bx, by, bw, bh, _ = blocks[idx]
        ax.annotate("", xy=(bx + bw / 2, by + bh), xytext=(0.50, 0.88), arrowprops=dict(arrowstyle="->", lw=1.5, color="#374151"))

    ax.text(0.52, 0.50, "Skip connection", fontsize=13, color="#374151")
    ax.annotate("", xy=(0.68, 0.37), xytext=(0.50, 0.66), arrowprops=dict(arrowstyle="->", lw=1.8, linestyle="--", color="#374151"))
    ax.annotate("", xy=(0.48, 0.37), xytext=(0.30, 0.66), arrowprops=dict(arrowstyle="->", lw=1.8, linestyle="--", color="#374151"))

    ax.set_title("MNIST DDPM U-Net Denoiser Architecture (schematic)", fontsize=18)
    plt.tight_layout(pad=2.0)
    plt.savefig(schematic_path, dpi=220)
    plt.close(fig)
    return schematic_path


def write_dashboard(metrics_path: str, outdir: str) -> str:
    dashboard_path = os.path.join(outdir, "dashboard.html")
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>MNIST Diffusion Dashboard</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 2rem; background: #f5f7fb; color: #111827; }}
    .card {{ background: #fff; border-radius: 12px; padding: 1rem 1.25rem; box-shadow: 0 4px 12px rgba(0,0,0,0.08); margin-bottom: 1rem; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 1rem; }}
    img {{ max-width: 100%; border-radius: 8px; border: 1px solid #e5e7eb; }}
    code {{ background: #eef2ff; padding: 0.15rem 0.35rem; border-radius: 6px; }}
  </style>
</head>
<body>
  <h1>MNIST Diffusion Dashboard</h1>
  <div class=\"card\">
    <p><strong>Run tag:</strong> {metrics['run_tag']}</p>
    <p><strong>Started (UTC):</strong> {metrics['started_utc']}</p>
    <p><strong>Finished (UTC):</strong> {metrics['finished_utc']}</p>
    <p><strong>Device:</strong> {metrics['device']}</p>
    <p><strong>Total steps:</strong> {metrics['total_steps']}</p>
    <p><strong>Final epoch mean loss:</strong> {metrics['final_epoch_loss']:.6f}</p>
    <p><strong>Best epoch mean loss:</strong> {metrics['best_epoch_loss']:.6f}</p>
  </div>
  <div class=\"grid\">
    <div class=\"card\">
      <h3>Generated samples</h3>
      <img src=\"mnist_samples.png\" alt=\"MNIST samples\" />
    </div>
    <div class=\"card\">
      <h3>Step loss</h3>
      <img src=\"loss_curve_step.png\" alt=\"Step loss\" />
    </div>
    <div class=\"card\">
      <h3>Epoch loss</h3>
      <img src=\"loss_curve_epoch.png\" alt=\"Epoch loss\" />
    </div>
    <div class=\"card\">
      <h3>Model architecture</h3>
      <img src=\"architecture_schematic.png\" alt=\"Architecture schematic\" />
    </div>
  </div>
  <div class=\"card\">
    <h3>Raw outputs</h3>
    <p><code>metrics.json</code>, <code>loss_history.csv</code>, <code>architecture_schematic.png</code>, checkpoints per epoch.</p>
  </div>
</body>
</html>
"""
    with open(dashboard_path, "w", encoding="utf-8") as f:
        f.write(html)
    return dashboard_path


def train(args: argparse.Namespace) -> None:
    run_tag = args.run_tag or datetime.now(timezone.utc).strftime("mnist_ddpm_%Y%m%d_%H%M%S")
    outdir = os.path.join(args.outdir, run_tag)
    os.makedirs(outdir, exist_ok=True)

    started = datetime.now(timezone.utc)
    wall_start = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    train_ds = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transform)
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = UNetDenoiser(time_dim=args.time_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    diffusion = DDPM(
        DiffusionConfig(timesteps=args.timesteps, beta_start=args.beta_start, beta_end=args.beta_end),
        device=device,
    )

    step_losses: list[float] = []
    epoch_losses: list[float] = []
    csv_path = os.path.join(outdir, "loss_history.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_f:
        writer = csv.writer(csv_f)
        writer.writerow(["step", "epoch", "loss"])

        global_step = 0
        for epoch in range(args.epochs):
            pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}")
            running_loss = 0.0
            epoch_steps = 0
            for x0, _ in pbar:
                x0 = x0.to(device)
                t = torch.randint(0, args.timesteps, (x0.shape[0],), device=device)
                noise = torch.randn_like(x0)
                xt = diffusion.q_sample(x0, t, noise)
                pred = model(xt, t)

                loss = F.mse_loss(pred, noise)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                global_step += 1
                epoch_steps += 1
                loss_value = float(loss.item())
                running_loss += loss_value
                step_losses.append(loss_value)
                writer.writerow([global_step, epoch + 1, f"{loss_value:.8f}"])
                pbar.set_postfix(loss=f"{loss_value:.4f}", step=global_step)

            mean_epoch_loss = running_loss / max(epoch_steps, 1)
            epoch_losses.append(mean_epoch_loss)
            ckpt = os.path.join(outdir, f"mnist_ddpm_epoch_{epoch + 1}.pt")
            torch.save({"model": model.state_dict(), "epoch": epoch + 1, "args": vars(args)}, ckpt)
            print(f"Epoch {epoch + 1} mean loss: {mean_epoch_loss:.6f}")

    samples = diffusion.sample(model, num_samples=args.num_samples)
    sample_img_path = os.path.join(outdir, "mnist_samples.png")
    save_grid(samples, sample_img_path)

    step_plot, epoch_plot = save_loss_plots(step_losses=step_losses, epoch_losses=epoch_losses, outdir=outdir)
    architecture_plot = save_architecture_schematic(outdir=outdir)

    finished = datetime.now(timezone.utc)
    metrics = {
        "run_tag": run_tag,
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "elapsed_seconds": round(time.time() - wall_start, 2),
        "device": str(device),
        "total_steps": len(step_losses),
        "epochs": args.epochs,
        "final_epoch_loss": float(epoch_losses[-1]) if epoch_losses else None,
        "best_epoch_loss": float(min(epoch_losses)) if epoch_losses else None,
        "sample_image": os.path.basename(sample_img_path),
        "step_loss_plot": os.path.basename(step_plot),
        "epoch_loss_plot": os.path.basename(epoch_plot),
        "architecture_schematic": os.path.basename(architecture_plot),
        "loss_csv": os.path.basename(csv_path),
    }

    metrics_path = os.path.join(outdir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    dashboard_path = write_dashboard(metrics_path=metrics_path, outdir=outdir)

    latest_path = os.path.join(args.outdir, "LATEST_RUN.txt")
    with open(latest_path, "w", encoding="utf-8") as f:
        f.write(run_tag + "\n")

    print(f"Training complete on {device}")
    print(f"Run output directory: {outdir}")
    print(f"Dashboard: {dashboard_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train simple DDPM for MNIST")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--timesteps", type=int, default=200)
    p.add_argument("--beta-start", type=float, default=1e-4)
    p.add_argument("--beta-end", type=float, default=2e-2)
    p.add_argument("--time-dim", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--num-samples", type=int, default=64)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--outdir", default="./outputs")
    p.add_argument("--run-tag", default="")
    p.add_argument("--force-cpu", action="store_true")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    train(args)
