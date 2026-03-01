#!/usr/bin/env python3
"""Sampling-free posterior intrinsic-dimensionality analysis via local geometry."""

import argparse
import json
import os

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

import mnist_diffusion as md


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sampling-free posterior geometry ID analysis")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--observed-fraction", type=float, required=True)
    p.add_argument("--posterior-digit", type=int, default=7)
    p.add_argument("--posterior-seed", type=int, default=123)
    p.add_argument("--num-masks", type=int, default=3)
    p.add_argument("--mask-seed-stride", type=int, default=101)
    p.add_argument("--score-eval-sigma", type=float, default=0.1)
    p.add_argument("--likelihood-sigma", type=float, default=0.1)
    p.add_argument("--map-steps", type=int, default=350)
    p.add_argument("--map-step-size", type=float, default=0.07)
    p.add_argument("--map-init-sigma", type=float, default=0.5)
    p.add_argument("--covariance-ridge", type=float, default=1.0)
    p.add_argument("--force-cpu", action="store_true")
    return p


def spectrum_metrics(eigs_desc: torch.Tensor) -> dict:
    eigs = eigs_desc.detach()
    energy = eigs / torch.clamp(eigs.sum(), min=1e-12)
    cum = torch.cumsum(energy, dim=0)
    k95 = int(torch.searchsorted(cum, torch.tensor(0.95)).item() + 1)
    k99 = int(torch.searchsorted(cum, torch.tensor(0.99)).item() + 1)
    pr = float((eigs.sum() ** 2 / torch.clamp((eigs * eigs).sum(), min=1e-12)).item())
    return {
        "k95_energy": k95,
        "k99_energy": k99,
        "participation_ratio": pr,
        "largest_eigenvalue": float(eigs[0].item()),
        "smallest_eigenvalue": float(eigs[-1].item()),
    }


def codimension_metrics(precision_eigs: torch.Tensor) -> dict:
    vals = torch.clamp(precision_eigs.detach(), min=0.0)
    vals_desc = torch.flip(vals, dims=[0])
    n = int(vals_desc.numel())
    energy = vals_desc / torch.clamp(vals_desc.sum(), min=1e-12)
    cum = torch.cumsum(energy, dim=0)
    c95 = int(torch.searchsorted(cum, torch.tensor(0.95)).item() + 1)
    c99 = int(torch.searchsorted(cum, torch.tensor(0.99)).item() + 1)
    pr = float((vals_desc.sum() ** 2 / torch.clamp((vals_desc * vals_desc).sum(), min=1e-12)).item())
    return {
        "codim95_precision_energy": c95,
        "codim99_precision_energy": c99,
        "intrinsic_dim95_proxy": n - c95,
        "intrinsic_dim99_proxy": n - c99,
        "precision_participation_ratio": pr,
    }


def save_spectrum_plot(eigs_desc: torch.Tensor, out_path: str, title: str) -> None:
    vals = torch.sqrt(torch.clamp(eigs_desc, min=1e-12)).cpu().numpy()
    vals = vals / max(float(vals[0]), 1e-12)
    idx = range(1, len(vals) + 1)
    plt.figure(figsize=(8, 4.5))
    plt.semilogy(idx, vals, linewidth=1.5)
    plt.xlabel("Mode index")
    plt.ylabel("Normalized local covariance scale")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def score_from_model(edm: md.EDM, model: torch.nn.Module, x: torch.Tensor, sigma_eval: float) -> torch.Tensor:
    denoised = edm.preconditioned_denoise(model, x, sigma_eval)
    return (denoised - x) / max(sigma_eval * sigma_eval, 1e-8)


def estimate_map(
    edm: md.EDM,
    model: torch.nn.Module,
    observed_x0: torch.Tensor,
    observed_mask: torch.Tensor,
    sigma_eval: float,
    likelihood_sigma: float,
    steps: int,
    step_size: float,
    init_sigma: float,
    seed: int,
) -> tuple[torch.Tensor, float]:
    device = observed_x0.device
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    x = observed_x0 + (1.0 - observed_mask) * torch.randn(observed_x0.shape, generator=g, device=device) * init_sigma
    x = x.clamp(-1.0, 1.0)

    observed_frac = float(torch.clamp(observed_mask.mean(), min=1e-6).item())
    inv_var = 1.0 / max(likelihood_sigma * likelihood_sigma, 1e-8)
    grad_norm = 0.0
    for i in range(steps):
        with torch.no_grad():
            score = score_from_model(edm, model, x, sigma_eval)
            like_grad = (observed_mask * (observed_x0 - x) * inv_var) / observed_frac
            grad = score + like_grad
            alpha = step_size * (1.0 - 0.75 * (i / max(steps - 1, 1)))
            x = (x + alpha * grad).clamp(-1.0, 1.0)
            grad_norm = float(grad.square().mean().sqrt().item())

    return x.detach(), grad_norm


def local_precision(
    edm: md.EDM,
    model: torch.nn.Module,
    x_map: torch.Tensor,
    observed_mask: torch.Tensor,
    sigma_eval: float,
    likelihood_sigma: float,
) -> torch.Tensor:
    x_dim = x_map.numel()
    x_flat = x_map.reshape(-1).detach().clone().requires_grad_(True)

    def score_flattened(x_flattened: torch.Tensor) -> torch.Tensor:
        x = x_flattened.view_as(x_map)
        score = score_from_model(edm, model, x, sigma_eval)
        return score.reshape(-1)

    jac = torch.autograd.functional.jacobian(score_flattened, x_flat, vectorize=True)
    jac = jac.detach()
    precision = -jac

    inv_var = 1.0 / max(likelihood_sigma * likelihood_sigma, 1e-8)
    mask_diag = observed_mask.reshape(-1).detach() * inv_var
    precision = precision + torch.diag(mask_diag)
    precision = 0.5 * (precision + precision.T)

    eps = 1e-6
    precision = precision + eps * torch.eye(x_dim, device=precision.device, dtype=precision.dtype)
    return precision


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

    cov_eig_spectra = []
    precision_eig_spectra = []
    map_grad_norms = []
    map_images = []
    observed_images = []
    observed_masks = []
    for i in range(args.num_masks):
        mask_seed = args.posterior_seed + i * args.mask_seed_stride
        g = torch.Generator(device=device)
        g.manual_seed(mask_seed)
        observed_mask = (torch.rand(gt.shape, device=device, generator=g) < args.observed_fraction).float()
        observed_x0 = gt * observed_mask

        x_map, grad_norm = estimate_map(
            edm=edm,
            model=model,
            observed_x0=observed_x0,
            observed_mask=observed_mask,
            sigma_eval=float(args.score_eval_sigma),
            likelihood_sigma=float(args.likelihood_sigma),
            steps=int(args.map_steps),
            step_size=float(args.map_step_size),
            init_sigma=float(args.map_init_sigma),
            seed=mask_seed + 17,
        )

        precision = local_precision(
            edm=edm,
            model=model,
            x_map=x_map,
            observed_mask=observed_mask,
            sigma_eval=float(args.score_eval_sigma),
            likelihood_sigma=float(args.likelihood_sigma),
        )
        eig_precision = torch.linalg.eigvalsh(precision).real
        eig_precision_psd = torch.clamp(eig_precision, min=0.0)
        cov_eig_desc = torch.flip(1.0 / (eig_precision_psd + float(args.covariance_ridge)), dims=[0]).cpu()

        cov_eig_spectra.append(cov_eig_desc)
        precision_eig_spectra.append(eig_precision_psd.cpu())
        map_grad_norms.append(grad_norm)
        map_images.append(x_map.cpu())
        observed_images.append(observed_x0.cpu())
        observed_masks.append(observed_mask.cpu())

    cov_eig_stack = torch.stack(cov_eig_spectra, dim=0)
    cov_eig_mean = cov_eig_stack.mean(dim=0)
    precision_eig_stack = torch.stack(precision_eig_spectra, dim=0)
    precision_eig_mean = precision_eig_stack.mean(dim=0)

    cov_metrics = spectrum_metrics(cov_eig_mean)
    codim = codimension_metrics(precision_eig_mean)
    summary = {
        "checkpoint": args.checkpoint,
        "device": str(device),
        "num_pixels": int(gt.numel()),
        "posterior_digit": int(args.posterior_digit),
        "observed_fraction": float(args.observed_fraction),
        "num_masks": int(args.num_masks),
        "mask_seed_stride": int(args.mask_seed_stride),
        "score_eval_sigma": float(args.score_eval_sigma),
        "likelihood_sigma": float(args.likelihood_sigma),
        "map_steps": int(args.map_steps),
        "map_step_size": float(args.map_step_size),
        "map_init_sigma": float(args.map_init_sigma),
        "covariance_ridge": float(args.covariance_ridge),
        "map_grad_norm_mean": float(sum(map_grad_norms) / max(len(map_grad_norms), 1)),
        "map_grad_norm_max": float(max(map_grad_norms)),
        "local_covariance_spectrum_metrics": cov_metrics,
        "precision_codimension_metrics": codim,
    }

    map_batch = torch.cat(map_images, dim=0)
    obs_batch = torch.cat(observed_images, dim=0)
    mask_batch = torch.cat(observed_masks, dim=0)

    torch.save(cov_eig_mean, os.path.join(args.outdir, "local_cov_eigenvalues_mean.pt"))
    torch.save(cov_eig_stack, os.path.join(args.outdir, "local_cov_eigenvalues_all.pt"))
    torch.save(precision_eig_mean, os.path.join(args.outdir, "precision_eigenvalues_mean.pt"))
    torch.save(precision_eig_stack, os.path.join(args.outdir, "precision_eigenvalues_all.pt"))
    torch.save(map_batch, os.path.join(args.outdir, "map_images.pt"))
    torch.save(obs_batch, os.path.join(args.outdir, "observed_images.pt"))
    torch.save(mask_batch, os.path.join(args.outdir, "observed_masks.pt"))

    pct = int(round(args.observed_fraction * 100))
    save_spectrum_plot(
        eigs_desc=cov_eig_mean,
        out_path=os.path.join(args.outdir, "local_covariance_spectrum.png"),
        title=f"Sampling-free Local Covariance Spectrum ({pct}% observed)",
    )
    md.save_posterior_overview(
        ground_truth=gt.cpu(),
        observed_mask=observed_masks[0],
        observed_image=observed_images[0],
        posterior_samples=map_batch[:8],
        out_path=os.path.join(args.outdir, "map_conditioning_overview.png"),
    )

    with open(os.path.join(args.outdir, "geometry_id_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
