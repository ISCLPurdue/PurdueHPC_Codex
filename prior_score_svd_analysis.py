#!/usr/bin/env python3
"""Prior (unconditional) intrinsic-dimensionality analysis using sample/score SVD."""

import argparse
import json
import os

import matplotlib.pyplot as plt
import torch

import mnist_diffusion as md


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prior sample/score SVD analysis")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--num-snapshots", type=int, default=1024)
    p.add_argument("--chunk-size", type=int, default=128)
    p.add_argument("--score-eval-sigma", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--force-cpu", action="store_true")
    return p


def svd_metrics(svals: torch.Tensor) -> dict:
    svals = svals.detach()
    energy = svals * svals
    cum_energy = torch.cumsum(energy, dim=0) / torch.clamp(energy.sum(), min=1e-12)
    d95 = int(torch.searchsorted(cum_energy, torch.tensor(0.95)).item() + 1)
    d99 = int(torch.searchsorted(cum_energy, torch.tensor(0.99)).item() + 1)
    pr = float((energy.sum() ** 2 / torch.clamp((energy * energy).sum(), min=1e-12)).item())
    return {
        "d95_energy": d95,
        "d99_energy": d99,
        "participation_ratio": pr,
        "largest_singular_value": float(svals[0].item()),
        "smallest_singular_value": float(svals[-1].item()),
    }


def save_spectrum_plot(svals: torch.Tensor, out_path: str, title: str) -> None:
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


def main() -> None:
    args = build_parser().parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = ckpt.get("args", {})
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

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

    snapshots = []
    remaining = args.num_snapshots
    with torch.no_grad():
        while remaining > 0:
            n = min(args.chunk_size, remaining)
            samples = edm.sample(model=model, num_samples=n, image_size=28)
            snapshots.append(samples.cpu())
            remaining -= n

    prior = torch.cat(snapshots, dim=0)
    n_snap = prior.shape[0]
    n_pix = prior.shape[2] * prior.shape[3]
    if n_snap <= n_pix:
        raise RuntimeError(f"Need num_snapshots > num_pixels ({n_snap} <= {n_pix})")

    sample_matrix = prior.view(n_snap, -1).T
    svals_samples = torch.linalg.svdvals(sample_matrix)

    sigma_eval = float(args.score_eval_sigma)
    with torch.no_grad():
        x_eval = prior.to(device)
        denoised_eval = edm.preconditioned_denoise(model, x_eval, sigma_eval)
        score = (denoised_eval - x_eval) / max(sigma_eval * sigma_eval, 1e-8)
    score_matrix = score.view(n_snap, -1).T.cpu()
    svals_scores = torch.linalg.svdvals(score_matrix)

    torch.save(svals_samples, os.path.join(args.outdir, "singular_values_samples_prior.pt"))
    torch.save(svals_scores, os.path.join(args.outdir, "singular_values_scores_prior.pt"))
    md.save_grid(prior[:64], os.path.join(args.outdir, "prior_samples_64.png"), nrow=8)

    save_spectrum_plot(
        svals=svals_samples,
        out_path=os.path.join(args.outdir, "singular_value_spectrum_samples_prior.png"),
        title="Prior Sample Snapshot Spectrum",
    )
    save_spectrum_plot(
        svals=svals_scores,
        out_path=os.path.join(args.outdir, "singular_value_spectrum_scores_prior.png"),
        title=f"Prior Score Snapshot Spectrum (sigma={sigma_eval})",
    )

    summary = {
        "checkpoint": args.checkpoint,
        "device": str(device),
        "num_pixels": n_pix,
        "num_snapshots": n_snap,
        "score_eval_sigma": sigma_eval,
        "sample_snapshot_metrics": svd_metrics(svals_samples),
        "score_snapshot_metrics": svd_metrics(svals_scores),
    }
    with open(os.path.join(args.outdir, "prior_svd_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
