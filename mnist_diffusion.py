#!/usr/bin/env python3
"""Train an EDM-style diffusion model on MNIST and generate dashboard artifacts."""

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
class EDMConfig:
    num_steps: int = 100
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: float = 7.0
    sigma_data: float = 0.5
    p_mean: float = -1.2
    p_std: float = 1.2


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
        self.downsample1 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.down2 = ResidualBlock(128, 128, time_hidden)
        self.downsample2 = nn.Conv2d(128, 256, 4, stride=2, padding=1)

        self.mid1 = ResidualBlock(256, 256, time_hidden)
        self.mid2 = ResidualBlock(256, 256, time_hidden)

        self.up1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.up_block1 = ResidualBlock(256, 128, time_hidden)
        self.up2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
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


class EDM:
    def __init__(self, cfg: EDMConfig, device: torch.device):
        self.cfg = cfg
        self.device = device

    def _sigma_to_tensor(self, sigma, batch_size: int) -> torch.Tensor:
        if isinstance(sigma, float):
            return torch.full((batch_size, 1, 1, 1), sigma, device=self.device)
        if sigma.ndim == 1:
            return sigma[:, None, None, None]
        return sigma

    def sample_training_sigmas(self, batch_size: int) -> torch.Tensor:
        rnd = torch.randn(batch_size, device=self.device)
        return torch.exp(self.cfg.p_mean + self.cfg.p_std * rnd)

    def preconditioned_denoise(self, model: nn.Module, x: torch.Tensor, sigma) -> torch.Tensor:
        sigma = self._sigma_to_tensor(sigma, x.shape[0])
        sigma_data = self.cfg.sigma_data
        sigma_sq = sigma * sigma
        data_sq = sigma_data * sigma_data
        denom = torch.sqrt(sigma_sq + data_sq)

        c_in = 1.0 / denom
        c_skip = data_sq / (sigma_sq + data_sq)
        c_out = sigma * sigma_data / denom
        c_noise = 0.25 * torch.log(torch.clamp(sigma.squeeze(-1).squeeze(-1).squeeze(-1), min=1e-8))

        model_out = model(c_in * x, c_noise)
        return c_skip * x + c_out * model_out

    def loss(self, model: nn.Module, x0: torch.Tensor) -> torch.Tensor:
        sigma = self.sample_training_sigmas(x0.shape[0])[:, None, None, None]
        noise = torch.randn_like(x0) * sigma
        x_noisy = x0 + noise
        denoised = self.preconditioned_denoise(model, x_noisy, sigma)

        sigma_data = self.cfg.sigma_data
        weight = (sigma * sigma + sigma_data * sigma_data) / torch.clamp((sigma * sigma_data) ** 2, min=1e-8)
        return (weight * (denoised - x0) ** 2).mean()

    def sigma_schedule(self) -> torch.Tensor:
        ramp = torch.linspace(0.0, 1.0, self.cfg.num_steps, device=self.device)
        rho_inv = 1.0 / self.cfg.rho
        sigmas = (self.cfg.sigma_max ** rho_inv + ramp * (self.cfg.sigma_min ** rho_inv - self.cfg.sigma_max ** rho_inv)) ** self.cfg.rho
        return torch.cat([sigmas, torch.zeros(1, device=self.device)], dim=0)

    @torch.no_grad()
    def sample(self, model: nn.Module, num_samples: int, image_size: int = 28) -> torch.Tensor:
        model.eval()
        sigmas = self.sigma_schedule()
        x = torch.randn(num_samples, 1, image_size, image_size, device=self.device) * sigmas[0]

        for i in range(len(sigmas) - 1):
            sigma = float(sigmas[i].item())
            sigma_next = float(sigmas[i + 1].item())
            denoised = self.preconditioned_denoise(model, x, sigma)
            d_cur = (x - denoised) / max(sigma, 1e-8)
            x_euler = x + (sigma_next - sigma) * d_cur

            if sigma_next > 0.0:
                denoised_next = self.preconditioned_denoise(model, x_euler, sigma_next)
                d_next = (x_euler - denoised_next) / max(sigma_next, 1e-8)
                x = x + (sigma_next - sigma) * 0.5 * (d_cur + d_next)
            else:
                x = x_euler

        model.train()
        return x.clamp(-1, 1)

    @torch.no_grad()
    def posterior_sample(
        self,
        model: nn.Module,
        observed_x0: torch.Tensor,
        observed_mask: torch.Tensor,
        num_samples: int,
        guidance_scale: float,
        likelihood_sigma: float,
        guidance_min_frac: float,
        guidance_power: float,
        noise_aware_coeff: float,
        hard_data_consistency: bool,
    ) -> torch.Tensor:
        model.eval()
        sigmas = self.sigma_schedule()
        x = torch.randn(num_samples, 1, observed_x0.shape[-2], observed_x0.shape[-1], device=self.device) * sigmas[0]
        y = observed_x0.expand(num_samples, -1, -1, -1)
        m = observed_mask.expand(num_samples, -1, -1, -1)
        observed_frac = float(torch.clamp(m.mean(), min=1e-6).item())
        base_sigma_sq = max(likelihood_sigma * likelihood_sigma, 1e-8)
        total_steps = max(len(sigmas) - 2, 1)

        def guided_direction(x_in: torch.Tensor, sigma_val: float, progress: float) -> torch.Tensor:
            x0_hat = self.preconditioned_denoise(model, x_in, sigma_val)
            if hard_data_consistency:
                x0_hat = m * y + (1.0 - m) * x0_hat

            sigma_eff_sq = base_sigma_sq + noise_aware_coeff * (sigma_val * sigma_val)
            inv_var = 1.0 / max(sigma_eff_sq, 1e-8)
            grad_x0 = (m * (y - x0_hat) * inv_var) / observed_frac

            guide_factor = guidance_min_frac + (1.0 - guidance_min_frac) * (progress**guidance_power)
            step_guidance = guidance_scale * guide_factor
            x0_guided = x0_hat + step_guidance * sigma_val * grad_x0
            return (x_in - x0_guided) / max(sigma_val, 1e-8)

        for i in range(len(sigmas) - 1):
            sigma = float(sigmas[i].item())
            sigma_next = float(sigmas[i + 1].item())
            progress = i / total_steps

            d_cur = guided_direction(x, sigma, progress)
            x_euler = x + (sigma_next - sigma) * d_cur

            if hard_data_consistency and sigma_next > 0.0:
                x_obs = y + torch.randn_like(x_euler) * sigma_next
                x_euler = m * x_obs + (1.0 - m) * x_euler

            if sigma_next > 0.0:
                d_next = guided_direction(x_euler, sigma_next, min((i + 1) / total_steps, 1.0))
                x = x + (sigma_next - sigma) * 0.5 * (d_cur + d_next)
                if hard_data_consistency:
                    x_obs = y + torch.randn_like(x) * sigma_next
                    x = m * x_obs + (1.0 - m) * x
            else:
                x = x_euler
                if hard_data_consistency:
                    x = m * y + (1.0 - m) * x

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


def save_posterior_overview(
    ground_truth: torch.Tensor,
    observed_mask: torch.Tensor,
    observed_image: torch.Tensor,
    posterior_samples: torch.Tensor,
    out_path: str,
) -> None:
    gt = ((ground_truth + 1) / 2.0).cpu().squeeze().numpy()
    mask = observed_mask.cpu().squeeze().numpy()
    obs = ((observed_image + 1) / 2.0).cpu().squeeze().numpy()
    post = ((posterior_samples + 1) / 2.0).cpu().squeeze(1).numpy()

    cols = 3 + post.shape[0]
    fig, axes = plt.subplots(1, cols, figsize=(2.2 * cols, 2.8))
    axes[0].imshow(gt, cmap="gray")
    axes[0].set_title("Ground truth")
    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Observed mask")
    axes[2].imshow(obs, cmap="gray")
    axes[2].set_title("Observed pixels")

    for i in range(post.shape[0]):
        axes[3 + i].imshow(post[i], cmap="gray")
        axes[3 + i].set_title(f"Posterior {i + 1}")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def save_architecture_schematic(outdir: str) -> str:
    schematic_path = os.path.join(outdir, "architecture_schematic.png")
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.axis("off")

    blocks = [
        (0.02, 0.60, 0.16, 0.18, "Input x_sigma\\n1x28x28"),
        (0.22, 0.60, 0.16, 0.18, "Stem Conv\\n1->64"),
        (0.42, 0.60, 0.16, 0.18, "Down Block 1\\nRes(64)"),
        (0.62, 0.60, 0.16, 0.18, "Down Block 2\\nRes(128)"),
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

    ax.set_title("MNIST EDM U-Net Denoiser Architecture (schematic)", fontsize=18)
    plt.tight_layout(pad=2.0)
    plt.savefig(schematic_path, dpi=220)
    plt.close(fig)
    return schematic_path


def write_run_dashboard(metrics_path: str, outdir: str, refresh_seconds: int) -> str:
    dashboard_path = os.path.join(outdir, "dashboard.html")
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    posterior_cards = ""
    posterior_outputs = ""
    if metrics.get("posterior_overview_image"):
        posterior_cards = (
            '<div class="card"><h3>Posterior conditioning overview</h3>'
            '<img class="zoomable" src="posterior_conditioning_overview.png" alt="Posterior overview" /></div>'
            '<div class="card"><h3>Posterior samples</h3>'
            '<img class="zoomable" src="posterior_samples.png" alt="Posterior samples" /></div>'
        )
        posterior_outputs = ", <code>posterior_conditioning_overview.png</code>, <code>posterior_samples.png</code>"

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
    .zoomable {{ max-width: 100%; border-radius: 8px; border: 1px solid #e5e7eb; cursor: zoom-in; }}
    code {{ background: #eef2ff; padding: 0.15rem 0.35rem; border-radius: 6px; }}
    .modal {{ display:none; position: fixed; inset: 0; background: rgba(0,0,0,0.85); z-index: 2000; }}
    .modal-content {{ position:absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); max-width: 92vw; max-height: 88vh; overflow: auto; }}
    .modal img {{ transform-origin: center center; }}
    .controls {{ position: absolute; top: 1rem; right: 1rem; display: flex; gap: .5rem; }}
    .btn {{ border: none; border-radius: 8px; padding: .45rem .6rem; background: #fff; cursor: pointer; }}
  </style>
</head>
<body>
  <h1>MNIST Diffusion Dashboard</h1>
  <div class=\"card\">
    <p><strong>Run tag:</strong> {metrics.get('run_tag')}</p>
    <p><strong>Status:</strong> <span id=\"run-status\">{metrics.get('status', 'unknown')}</span></p>
    <p><strong>Current epoch:</strong> <span id=\"current-epoch\">{metrics.get('current_epoch', 0)}</span> / {metrics.get('epochs')}</p>
    <p><strong>Started (UTC):</strong> {metrics.get('started_utc')}</p>
    <p><strong>Finished (UTC):</strong> <span id=\"finished-at\">{metrics.get('finished_utc')}</span></p>
    <p><strong>Device:</strong> {metrics.get('device')}</p>
    <p><strong>Total steps:</strong> <span id=\"total-steps\">{metrics.get('total_steps')}</span></p>
    <p><strong>Final epoch mean loss:</strong> <span id=\"final-loss\">{metrics.get('final_epoch_loss')}</span></p>
    <p><strong>Best epoch mean loss:</strong> <span id=\"best-loss\">{metrics.get('best_epoch_loss')}</span></p>
  </div>
  <div class=\"grid\">
    <div class=\"card\"><h3>Generated samples</h3><img class=\"zoomable\" src=\"mnist_samples.png\" alt=\"MNIST samples\" /></div>
    <div class=\"card\"><h3>Step loss</h3><img class=\"zoomable\" src=\"loss_curve_step.png\" alt=\"Step loss\" /></div>
    <div class=\"card\"><h3>Epoch loss</h3><img class=\"zoomable\" src=\"loss_curve_epoch.png\" alt=\"Epoch loss\" /></div>
    <div class=\"card\"><h3>Model architecture</h3><img class=\"zoomable\" src=\"architecture_schematic.png\" alt=\"Architecture schematic\" /></div>
    {posterior_cards}
  </div>
  <div class=\"card\"><h3>Raw outputs</h3><p><code>metrics.json</code>, <code>loss_history.csv</code>, <code>architecture_schematic.png</code>{posterior_outputs}, checkpoints per epoch.</p></div>

  <div id=\"zoom-modal\" class=\"modal\">
    <div class=\"controls\">
      <button class=\"btn\" id=\"zoom-in\">+</button>
      <button class=\"btn\" id=\"zoom-out\">-</button>
      <button class=\"btn\" id=\"zoom-reset\">reset</button>
      <button class=\"btn\" id=\"zoom-close\">close</button>
    </div>
    <div class=\"modal-content\"><img id=\"zoom-img\" src=\"\" alt=\"zoom\" /></div>
  </div>

  <script>
    (function() {{
      const refreshMs = {refresh_seconds} * 1000;
      const imgs = document.querySelectorAll('.zoomable');
      const modal = document.getElementById('zoom-modal');
      const zoomImg = document.getElementById('zoom-img');
      let scale = 1;

      function applyScale() {{ zoomImg.style.transform = `scale(${{scale}})`; }}
      imgs.forEach((img) => img.addEventListener('click', () => {{ zoomImg.src = img.src; scale = 1; applyScale(); modal.style.display = 'block'; }}));
      document.getElementById('zoom-close').onclick = () => modal.style.display = 'none';
      document.getElementById('zoom-in').onclick = () => {{ scale = Math.min(6, scale + 0.2); applyScale(); }};
      document.getElementById('zoom-out').onclick = () => {{ scale = Math.max(0.3, scale - 0.2); applyScale(); }};
      document.getElementById('zoom-reset').onclick = () => {{ scale = 1; applyScale(); }};
      modal.onclick = (e) => {{ if (e.target === modal) modal.style.display = 'none'; }};

      let runTimer = null;
      function stopRunRefreshIfDone(status) {{
        if ((status || '').toLowerCase() === 'completed' && runTimer) {{
          clearInterval(runTimer);
          runTimer = null;
        }}
      }}
      function refreshRunArtifacts() {{
        const t = Date.now();
        imgs.forEach((img) => {{
          const clean = img.src.split('?')[0];
          img.src = clean + '?t=' + t;
        }});
        fetch('metrics.json?t=' + t).then(r => r.json()).then(m => {{
          document.getElementById('run-status').textContent = m.status;
          document.getElementById('current-epoch').textContent = m.current_epoch;
          document.getElementById('finished-at').textContent = m.finished_utc || '';
          document.getElementById('total-steps').textContent = m.total_steps;
          document.getElementById('final-loss').textContent = (m.final_epoch_loss ?? '').toString();
          document.getElementById('best-loss').textContent = (m.best_epoch_loss ?? '').toString();
          stopRunRefreshIfDone(m.status);
        }}).catch(() => {{}});
      }}
      runTimer = setInterval(refreshRunArtifacts, refreshMs);
      stopRunRefreshIfDone(document.getElementById('run-status').textContent);
    }})();
  </script>
</body>
</html>
"""
    with open(dashboard_path, "w", encoding="utf-8") as f:
        f.write(html)
    return dashboard_path


def write_root_dashboard(outputs_root: str, refresh_seconds: int) -> str:
    root_dashboard = os.path.join(outputs_root, "dashboard.html")
    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>MNIST Diffusion Live Dashboard</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 1rem; background: #f5f7fb; color: #111827; }}
    .bar {{ display:flex; gap:1rem; align-items:center; margin-bottom: 0.75rem; flex-wrap: wrap; }}
    .pill {{ background:#e5edff; border-radius:999px; padding: .25rem .7rem; }}
    iframe {{ width: 100%; height: calc(100vh - 100px); border: 1px solid #d1d5db; border-radius: 12px; background: #fff; }}
  </style>
</head>
<body>
  <div class=\"bar\">
    <strong>Live Dashboard</strong>
    <span class=\"pill\">refresh: every {refresh_seconds}s</span>
    <span>latest run: <code id=\"run-tag\">(loading)</code></span>
  </div>
  <iframe id=\"dash\" src=\"current/dashboard.html\"></iframe>
  <script>
    (function() {{
      const refreshMs = {refresh_seconds} * 1000;
      const frame = document.getElementById('dash');
      const tagEl = document.getElementById('run-tag');
      let lastTag = '';

      let rootTimer = null;
      function stopRootRefreshIfDone(status) {{
        if ((status || '').toLowerCase() === 'completed' && rootTimer) {{
          clearInterval(rootTimer);
          rootTimer = null;
        }}
      }}
      function refresh() {{
        const t = Date.now();
        fetch('LATEST_RUN.txt?t=' + t).then(r => r.text()).then(txt => {{
          const tag = txt.trim();
          if (tag) {{
            tagEl.textContent = tag;
            if (tag !== lastTag) {{
              frame.src = 'current/dashboard.html?t=' + t;
              lastTag = tag;
            }} else {{
              frame.contentWindow.location.reload();
            }}
            fetch('current/metrics.json?t=' + t).then(r => r.json()).then(m => {{
              stopRootRefreshIfDone(m.status);
            }}).catch(() => {{}});
          }}
        }}).catch(() => {{
          frame.src = 'current/dashboard.html?t=' + t;
        }});
      }}

      refresh();
      rootTimer = setInterval(refresh, refreshMs);
    }})();
  </script>
</body>
</html>
"""
    with open(root_dashboard, "w", encoding="utf-8") as f:
        f.write(html)
    return root_dashboard


def write_latest_run(outputs_root: str, run_tag: str) -> str:
    latest_path = os.path.join(outputs_root, "LATEST_RUN.txt")
    with open(latest_path, "w", encoding="utf-8") as f:
        f.write(run_tag + "\n")
    return latest_path


def update_current_symlink(outputs_root: str, run_tag: str) -> None:
    target = run_tag
    link_path = os.path.join(outputs_root, "current")
    try:
        if os.path.islink(link_path) or os.path.exists(link_path):
            os.unlink(link_path)
    except FileNotFoundError:
        pass
    os.symlink(target, link_path)


def write_metrics(outdir: str, metrics: dict) -> str:
    metrics_path = os.path.join(outdir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics_path


def find_digit_example(dataset: datasets.MNIST, target_digit: int) -> torch.Tensor:
    for i in range(len(dataset)):
        img, label = dataset[i]
        if int(label) == target_digit:
            return img.unsqueeze(0)
    raise RuntimeError(f"Digit {target_digit} not found in dataset")


def train(args: argparse.Namespace) -> None:
    run_tag = args.run_tag or datetime.now(timezone.utc).strftime("mnist_edm_%Y%m%d_%H%M%S")
    outdir = os.path.join(args.outdir, run_tag)
    outputs_root = args.outdir
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(outputs_root, exist_ok=True)

    write_latest_run(outputs_root=outputs_root, run_tag=run_tag)
    update_current_symlink(outputs_root=outputs_root, run_tag=run_tag)
    write_root_dashboard(outputs_root=outputs_root, refresh_seconds=args.dashboard_refresh_seconds)

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
    diffusion = EDM(
        EDMConfig(
            num_steps=args.timesteps,
            sigma_min=args.edm_sigma_min,
            sigma_max=args.edm_sigma_max,
            rho=args.edm_rho,
            sigma_data=args.edm_sigma_data,
            p_mean=args.edm_p_mean,
            p_std=args.edm_p_std,
        ),
        device=device,
    )

    step_losses: list[float] = []
    epoch_losses: list[float] = []
    csv_path = os.path.join(outdir, "loss_history.csv")
    sample_img_path = os.path.join(outdir, "mnist_samples.png")
    architecture_plot = save_architecture_schematic(outdir=outdir)

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_f:
        writer = csv.writer(csv_f)
        writer.writerow(["step", "epoch", "loss"])

        global_step = 0
        latest_sample_epoch = 0

        for epoch in range(args.epochs):
            pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}")
            running_loss = 0.0
            epoch_steps = 0

            for x0, _ in pbar:
                x0 = x0.to(device)
                loss = diffusion.loss(model, x0)
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

            csv_f.flush()
            mean_epoch_loss = running_loss / max(epoch_steps, 1)
            epoch_losses.append(mean_epoch_loss)

            ckpt = os.path.join(outdir, f"mnist_edm_epoch_{epoch + 1}.pt")
            torch.save({"model": model.state_dict(), "epoch": epoch + 1, "args": vars(args)}, ckpt)

            step_plot, epoch_plot = save_loss_plots(step_losses=step_losses, epoch_losses=epoch_losses, outdir=outdir)

            should_sample = (epoch == 0) or ((epoch + 1) % args.sample_every == 0) or ((epoch + 1) == args.epochs)
            if should_sample:
                samples = diffusion.sample(model, num_samples=args.num_samples)
                save_grid(samples, sample_img_path)
                latest_sample_epoch = epoch + 1

            metrics = {
                "run_tag": run_tag,
                "status": "running" if (epoch + 1) < args.epochs else "completed",
                "current_epoch": epoch + 1,
                "latest_sample_epoch": latest_sample_epoch,
                "started_utc": started.isoformat(),
                "finished_utc": None,
                "elapsed_seconds": round(time.time() - wall_start, 2),
                "device": str(device),
                "total_steps": len(step_losses),
                "epochs": args.epochs,
                "diffusion_model": "edm_score_preconditioned",
                "final_epoch_loss": float(epoch_losses[-1]) if epoch_losses else None,
                "best_epoch_loss": float(min(epoch_losses)) if epoch_losses else None,
                "sample_image": os.path.basename(sample_img_path),
                "step_loss_plot": os.path.basename(step_plot),
                "epoch_loss_plot": os.path.basename(epoch_plot),
                "architecture_schematic": os.path.basename(architecture_plot),
                "loss_csv": os.path.basename(csv_path),
            }

            metrics_path = write_metrics(outdir=outdir, metrics=metrics)
            write_run_dashboard(metrics_path=metrics_path, outdir=outdir, refresh_seconds=args.dashboard_refresh_seconds)
            print(f"Epoch {epoch + 1} mean loss: {mean_epoch_loss:.6f}")

    posterior_overview_name = None
    posterior_samples_name = None
    if args.num_posterior_samples > 0:
        generator = torch.Generator(device=device)
        generator.manual_seed(args.posterior_seed)

        gt = find_digit_example(train_ds, args.posterior_digit).to(device)
        observed_mask = (torch.rand(gt.shape, device=device, generator=generator) < args.posterior_observed_fraction).float()
        observed_x0 = gt * observed_mask

        posterior = diffusion.posterior_sample(
            model=model,
            observed_x0=observed_x0,
            observed_mask=observed_mask,
            num_samples=args.num_posterior_samples,
            guidance_scale=args.posterior_guidance_scale,
            likelihood_sigma=args.posterior_likelihood_sigma,
            guidance_min_frac=args.posterior_guidance_min_frac,
            guidance_power=args.posterior_guidance_power,
            noise_aware_coeff=args.posterior_noise_aware_coeff,
            hard_data_consistency=(not args.posterior_disable_hard_consistency),
        )

        posterior_samples_name = "posterior_samples.png"
        posterior_overview_name = "posterior_conditioning_overview.png"
        posterior_samples_path = os.path.join(outdir, posterior_samples_name)
        posterior_overview_path = os.path.join(outdir, posterior_overview_name)
        save_grid(posterior, posterior_samples_path, nrow=min(8, args.num_posterior_samples))
        save_posterior_overview(gt, observed_mask, observed_x0, posterior, posterior_overview_path)

    finished = datetime.now(timezone.utc)
    final_metrics_path = os.path.join(outdir, "metrics.json")
    with open(final_metrics_path, "r", encoding="utf-8") as f:
        final_metrics = json.load(f)
    final_metrics["status"] = "completed"
    final_metrics["finished_utc"] = finished.isoformat()
    final_metrics["elapsed_seconds"] = round(time.time() - wall_start, 2)
    final_metrics["posterior_digit"] = args.posterior_digit if posterior_samples_name else None
    final_metrics["posterior_observed_fraction"] = args.posterior_observed_fraction if posterior_samples_name else None
    final_metrics["posterior_guidance_scale"] = args.posterior_guidance_scale if posterior_samples_name else None
    final_metrics["posterior_likelihood_sigma"] = args.posterior_likelihood_sigma if posterior_samples_name else None
    final_metrics["posterior_guidance_min_frac"] = args.posterior_guidance_min_frac if posterior_samples_name else None
    final_metrics["posterior_guidance_power"] = args.posterior_guidance_power if posterior_samples_name else None
    final_metrics["posterior_noise_aware_coeff"] = args.posterior_noise_aware_coeff if posterior_samples_name else None
    final_metrics["posterior_hard_data_consistency"] = (not args.posterior_disable_hard_consistency) if posterior_samples_name else None
    final_metrics["diffusion_model"] = "edm_score_preconditioned"
    final_metrics["edm_sigma_min"] = args.edm_sigma_min
    final_metrics["edm_sigma_max"] = args.edm_sigma_max
    final_metrics["edm_rho"] = args.edm_rho
    final_metrics["edm_sigma_data"] = args.edm_sigma_data
    final_metrics["edm_p_mean"] = args.edm_p_mean
    final_metrics["edm_p_std"] = args.edm_p_std
    final_metrics["posterior_samples_count"] = args.num_posterior_samples if posterior_samples_name else 0
    final_metrics["posterior_samples_image"] = posterior_samples_name
    final_metrics["posterior_overview_image"] = posterior_overview_name
    write_metrics(outdir=outdir, metrics=final_metrics)
    write_run_dashboard(metrics_path=final_metrics_path, outdir=outdir, refresh_seconds=args.dashboard_refresh_seconds)
    write_latest_run(outputs_root=outputs_root, run_tag=run_tag)
    update_current_symlink(outputs_root=outputs_root, run_tag=run_tag)
    write_root_dashboard(outputs_root=outputs_root, refresh_seconds=args.dashboard_refresh_seconds)

    print(f"Training complete on {device}")
    print(f"Run output directory: {outdir}")
    print(f"Run dashboard: {os.path.join(outdir, 'dashboard.html')}")
    print(f"Live dashboard: {os.path.join(outputs_root, 'dashboard.html')}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train EDM-style score diffusion model for MNIST")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--timesteps", type=int, default=100)
    p.add_argument("--beta-start", type=float, default=1e-4)
    p.add_argument("--beta-end", type=float, default=2e-2)
    p.add_argument("--time-dim", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--num-samples", type=int, default=64)
    p.add_argument("--sample-every", type=int, default=5)
    p.add_argument("--dashboard-refresh-seconds", type=int, default=10)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--outdir", default="./outputs")
    p.add_argument("--run-tag", default="")
    p.add_argument("--force-cpu", action="store_true")
    p.add_argument("--edm-sigma-min", type=float, default=0.002)
    p.add_argument("--edm-sigma-max", type=float, default=80.0)
    p.add_argument("--edm-rho", type=float, default=7.0)
    p.add_argument("--edm-sigma-data", type=float, default=0.5)
    p.add_argument("--edm-p-mean", type=float, default=-1.2)
    p.add_argument("--edm-p-std", type=float, default=1.2)
    p.add_argument("--posterior-digit", type=int, default=7)
    p.add_argument("--posterior-observed-fraction", type=float, default=0.7)
    p.add_argument("--posterior-guidance-scale", type=float, default=1.5)
    p.add_argument("--posterior-guidance-min-frac", type=float, default=0.25)
    p.add_argument("--posterior-guidance-power", type=float, default=1.5)
    p.add_argument("--posterior-likelihood-sigma", type=float, default=0.1)
    p.add_argument("--posterior-noise-aware-coeff", type=float, default=0.05)
    p.add_argument("--posterior-disable-hard-consistency", action="store_true")
    p.add_argument("--num-posterior-samples", type=int, default=8)
    p.add_argument("--posterior-seed", type=int, default=123)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    train(args)
