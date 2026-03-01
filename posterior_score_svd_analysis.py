#!/usr/bin/env python3
"""Posterior intrinsic-dimensionality analysis using score snapshots."""

import argparse
import json
import os

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

import mnist_diffusion as md


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Posterior score-snapshot SVD analysis")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--observed-fraction", type=float, required=True)
    p.add_argument("--num-snapshots", type=int, default=1024)
    p.add_argument("--chunk-size", type=int, default=128)
    p.add_argument("--posterior-digit", type=int, default=7)
    p.add_argument("--posterior-seed", type=int, default=123)
    p.add_argument("--posterior-guidance-scale", type=float, default=1.5)
    p.add_argument("--posterior-guidance-min-frac", type=float, default=0.25)
    p.add_argument("--posterior-guidance-power", type=float, default=1.5)
    p.add_argument("--posterior-likelihood-sigma", type=float, default=0.1)
    p.add_argument("--posterior-noise-aware-coeff", type=float, default=0.05)
    p.add_argument("--posterior-disable-hard-consistency", action="store_true")
    p.add_argument("--score-eval-sigma", type=float, default=0.1)
    p.add_argument("--force-cpu", action="store_true")
    return p


def svd_metrics(svals: torch.Tensor) -> dict:
    svals = svals.detach()
    energy = svals * svals
    cum_energy = torch.cumsum(energy, dim=0) / torch.clamp(energy.sum(), min=1e-12)
    k95 = int(torch.searchsorted(cum_energy, torch.tensor(0.95)).item() + 1)
    k99 = int(torch.searchsorted(cum_energy, torch.tensor(0.99)).item() + 1)
    part_ratio = float((energy.sum() ** 2 / torch.clamp((energy * energy).sum(), min=1e-12)).item())
    return {
        "k95_energy": k95,
        "k99_energy": k99,
        "participation_ratio": part_ratio,
        "largest_singular_value": float(svals[0].item()),
        "smallest_singular_value": float(svals[-1].item()),
    }


def save_single_spectrum(svals: torch.Tensor, out_path: str, title: str) -> None:
    s = svals.detach().cpu().numpy()
    s = s / max(float(s[0]), 1e-12)
    idx = range(1, len(s) + 1)
    plt.figure(figsize=(8, 4.5))
    plt.semilogy(idx, s, linewidth=1.4)
    plt.xlabel("Mode index")
    plt.ylabel("Normalized singular value")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def save_overlay_spectrum(svals_a: torch.Tensor, svals_b: torch.Tensor, out_path: str, title: str, label_a: str, label_b: str) -> None:
    a = svals_a.detach().cpu().numpy()
    b = svals_b.detach().cpu().numpy()
    a = a / max(float(a[0]), 1e-12)
    b = b / max(float(b[0]), 1e-12)
    idx = range(1, len(a) + 1)
    plt.figure(figsize=(8, 4.5))
    plt.semilogy(idx, a, linewidth=1.5, label=label_a)
    plt.semilogy(idx, b, linewidth=1.5, label=label_b)
    plt.xlabel("Mode index")
    plt.ylabel("Normalized singular value")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def main() -> None:
    args = build_parser().parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = ckpt.get("args", {})

    model = md.UNetDenoiser(time_dim=int(ckpt_args.get("time_dim", 128))).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    edm = md.EDM(
        md.EDMConfig(
            num_steps=int(ckpt_args.get("timesteps", 100)),
            sigma_min=float(ckpt_args.get("edm_sigma_min", 0.002)),
            sigma_max=float(ckpt_args.get("edm_sigma_max", 80.0)),
            rho=float(ckpt_args.get("edm_rho", 7.0)),
            sigma_data=float(ckpt_args.get("edm_sigma_data", 0.5)),
            p_mean=float(ckpt_args.get("edm_p_mean", -1.2)),
            p_std=float(ckpt_args.get("edm_p_std", 1.2)),
        ),
        device=device,
    )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    train_ds = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transform)

    generator = torch.Generator(device=device)
    generator.manual_seed(args.posterior_seed)
    gt = md.find_digit_example(train_ds, args.posterior_digit).to(device)
    observed_mask = (torch.rand(gt.shape, device=device, generator=generator) < args.observed_fraction).float()
    observed_x0 = gt * observed_mask

    snapshots = []
    remaining = args.num_snapshots
    with torch.no_grad():
        while remaining > 0:
            n = min(args.chunk_size, remaining)
            post = edm.posterior_sample(
                model=model,
                observed_x0=observed_x0,
                observed_mask=observed_mask,
                num_samples=n,
                guidance_scale=args.posterior_guidance_scale,
                likelihood_sigma=args.posterior_likelihood_sigma,
                guidance_min_frac=args.posterior_guidance_min_frac,
                guidance_power=args.posterior_guidance_power,
                noise_aware_coeff=args.posterior_noise_aware_coeff,
                hard_data_consistency=(not args.posterior_disable_hard_consistency),
            )
            snapshots.append(post.cpu())
            remaining -= n

    posterior = torch.cat(snapshots, dim=0)
    n_snap = posterior.shape[0]
    n_pix = posterior.shape[2] * posterior.shape[3]
    if n_snap <= n_pix:
        raise RuntimeError(f"Need num_snapshots > num_pixels ({n_snap} <= {n_pix})")

    # Baseline from previous experiment: sample snapshots matrix.
    sample_matrix = posterior.view(n_snap, -1).T  # (pixels, snapshots)
    svals_samples = torch.linalg.svdvals(sample_matrix)

    # New experiment: score snapshots matrix from approximated score function.
    sigma_eval = float(args.score_eval_sigma)
    with torch.no_grad():
        x_eval = posterior.to(device)
        denoised_eval = edm.preconditioned_denoise(model, x_eval, sigma_eval)
        score = (denoised_eval - x_eval) / max(sigma_eval * sigma_eval, 1e-8)
    score_matrix = score.view(n_snap, -1).T.cpu()
    svals_scores = torch.linalg.svdvals(score_matrix)

    torch.save(svals_samples, os.path.join(args.outdir, "singular_values_samples.pt"))
    torch.save(svals_scores, os.path.join(args.outdir, "singular_values_scores.pt"))
    md.save_grid(posterior[:64], os.path.join(args.outdir, "posterior_samples_64.png"), nrow=8)
    md.save_posterior_overview(
        ground_truth=gt.cpu(),
        observed_mask=observed_mask.cpu(),
        observed_image=observed_x0.cpu(),
        posterior_samples=posterior[:8],
        out_path=os.path.join(args.outdir, "posterior_conditioning_overview.png"),
    )

    pct = int(round(args.observed_fraction * 100))
    save_single_spectrum(
        svals=svals_samples,
        out_path=os.path.join(args.outdir, "singular_value_spectrum_samples.png"),
        title=f"Sample Snapshot Spectrum ({pct}% observed)",
    )
    save_single_spectrum(
        svals=svals_scores,
        out_path=os.path.join(args.outdir, "singular_value_spectrum_scores.png"),
        title=f"Score Snapshot Spectrum ({pct}% observed, sigma={sigma_eval})",
    )
    save_overlay_spectrum(
        svals_a=svals_samples,
        svals_b=svals_scores,
        out_path=os.path.join(args.outdir, "singular_value_spectrum_overlay.png"),
        title=f"Overlay: Sample vs Score Snapshot Spectra ({pct}% observed)",
        label_a="Sample snapshots (previous)",
        label_b="Score snapshots (current)",
    )

    summary = {
        "checkpoint": args.checkpoint,
        "device": str(device),
        "num_pixels": n_pix,
        "num_snapshots": n_snap,
        "observed_fraction": args.observed_fraction,
        "score_eval_sigma": sigma_eval,
        "sample_snapshot_metrics": svd_metrics(svals_samples),
        "score_snapshot_metrics": svd_metrics(svals_scores),
    }
    with open(os.path.join(args.outdir, "svd_score_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
