#!/usr/bin/env python3
"""Create combined no-sampling intrinsic-dimension spectrum plots (70% vs 20%)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def load(path: Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu").detach()


def to_normalized_scales(eigs_desc: torch.Tensor) -> torch.Tensor:
    vals = torch.sqrt(torch.clamp(eigs_desc, min=1e-12))
    return vals / max(float(vals[0]), 1e-12)


def plot_overlay(a: torch.Tensor, b: torch.Tensor, out_path: Path) -> None:
    idx = range(1, len(a) + 1)
    plt.figure(figsize=(8, 4.6))
    plt.semilogy(idx, a.numpy(), linewidth=1.6, label="70% observed")
    plt.semilogy(idx, b.numpy(), linewidth=1.6, label="20% observed")
    plt.xlabel("Mode index")
    plt.ylabel("Normalized local covariance scale")
    plt.title("Sampling-free Local Covariance Spectrum: 70% vs 20% Observed")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    root = Path("report/assets")
    outdir = root / "gautschi_edm_geometry_id_combined"
    outdir.mkdir(parents=True, exist_ok=True)

    eig70 = load(root / "gautschi_edm_geometry_id_70pct_8213702" / "local_cov_eigenvalues_mean.pt")
    eig20 = load(root / "gautschi_edm_geometry_id_20pct_8213703" / "local_cov_eigenvalues_mean.pt")
    s70 = to_normalized_scales(eig70)
    s20 = to_normalized_scales(eig20)

    plot_overlay(s70, s20, outdir / "local_covariance_spectrum_70_vs_20.png")


if __name__ == "__main__":
    main()
