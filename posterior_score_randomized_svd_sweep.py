#!/usr/bin/env python3
"""Randomized top-k score-snapshot SVD sweep over observed fractions."""

import argparse
import json
import os

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

import mnist_diffusion as md


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Randomized top-k score-snapshot SVD sweep")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--observed-fractions", nargs="+", type=float, default=[0.2, 0.4, 0.6, 0.8])
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
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--oversample", type=int, default=10)
    p.add_argument("--niter", type=int, default=2)
    p.add_argument("--force-cpu", action="store_true")
    return p


def randomized_topk_svals(a: torch.Tensor, top_k: int, oversample: int, niter: int) -> torch.Tensor:
    """Approximate leading singular values of matrix a using randomized range finding."""
    k = min(top_k + oversample, min(a.shape))
    omega = torch.randn(a.shape[1], k, device=a.device, dtype=a.dtype)
    y = a @ omega
    for _ in range(max(niter, 0)):
        y = a @ (a.T @ y)
    q, _ = torch.linalg.qr(y, mode="reduced")
    b = q.T @ a
    svals = torch.linalg.svdvals(b)
    return svals[: min(top_k, svals.numel())].detach().cpu()


def save_overlay_topk(svals_by_frac: dict[float, torch.Tensor], out_path: str, top_k: int) -> None:
    plt.figure(figsize=(8.4, 4.8))
    idx = list(range(1, top_k + 1))
    for frac in sorted(svals_by_frac):
        s = svals_by_frac[frac].numpy()
        s = s / max(float(s[0]), 1e-12)
        if abs(frac) < 1e-12:
            label = "0% (prior)"
        else:
            label = f"{int(round(frac * 100))}% observed"
        plt.semilogy(idx[: len(s)], s, linewidth=1.6, marker="o", markersize=3.2, label=label)
    plt.xlabel("Mode index (top-20 only)")
    plt.ylabel("Normalized singular value")
    plt.title("Randomized Score-Snapshot Spectrum vs Visible-Pixel Fraction")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
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
    gt = md.find_digit_example(train_ds, args.posterior_digit).to(device)

    svals_by_frac: dict[float, torch.Tensor] = {}
    metrics = {}

    for frac in args.observed_fractions:
        snapshots = []
        remaining = args.num_snapshots
        if frac <= 0.0:
            # True prior case: unconditional EDM sampling with no observations.
            with torch.no_grad():
                while remaining > 0:
                    n = min(args.chunk_size, remaining)
                    prior = edm.sample(model=model, num_samples=n, image_size=28)
                    snapshots.append(prior.cpu())
                    remaining -= n
            generation_mode = "prior_unconditional"
        else:
            generator = torch.Generator(device=device)
            generator.manual_seed(args.posterior_seed + int(round(frac * 1000)))
            observed_mask = (torch.rand(gt.shape, device=device, generator=generator) < frac).float()
            observed_x0 = gt * observed_mask

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
            generation_mode = "posterior_conditioned"

        posterior = torch.cat(snapshots, dim=0)
        n_snap = posterior.shape[0]
        n_pix = posterior.shape[2] * posterior.shape[3]
        if n_snap <= n_pix:
            raise RuntimeError(f"Need num_snapshots > num_pixels ({n_snap} <= {n_pix})")

        sigma_eval = float(args.score_eval_sigma)
        with torch.no_grad():
            x_eval = posterior.to(device)
            denoised_eval = edm.preconditioned_denoise(model, x_eval, sigma_eval)
            score = (denoised_eval - x_eval) / max(sigma_eval * sigma_eval, 1e-8)
        score_matrix = score.view(n_snap, -1).T.to(device=device, dtype=torch.float32)

        s_top = randomized_topk_svals(score_matrix, args.top_k, args.oversample, args.niter)
        svals_by_frac[float(frac)] = s_top
        torch.save(s_top, os.path.join(args.outdir, f"singular_values_scores_top{args.top_k}_{int(round(frac * 100))}pct.pt"))

        energy = s_top * s_top
        cum = torch.cumsum(energy, dim=0) / torch.clamp(energy.sum(), min=1e-12)
        d95 = int(torch.searchsorted(cum, torch.tensor(0.95)).item() + 1)
        d99 = int(torch.searchsorted(cum, torch.tensor(0.99)).item() + 1)
        metrics[f"{int(round(frac * 100))}pct"] = {
            "observed_fraction": float(frac),
            "generation_mode": generation_mode,
            "d95_topk": d95,
            "d99_topk": d99,
            "largest_singular_value": float(s_top[0].item()),
            "smallest_singular_value_topk": float(s_top[-1].item()),
        }

    plot_path = os.path.join(args.outdir, f"randomized_score_spectrum_top{args.top_k}_overlay.png")
    save_overlay_topk(svals_by_frac=svals_by_frac, out_path=plot_path, top_k=args.top_k)

    summary = {
        "checkpoint": args.checkpoint,
        "device": str(device),
        "num_snapshots": int(args.num_snapshots),
        "num_pixels": int(gt.numel()),
        "score_eval_sigma": float(args.score_eval_sigma),
        "top_k": int(args.top_k),
        "oversample": int(args.oversample),
        "niter": int(args.niter),
        "metrics_by_fraction_topk": metrics,
        "plot": os.path.basename(plot_path),
    }
    with open(os.path.join(args.outdir, "randomized_score_svd_sweep_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
