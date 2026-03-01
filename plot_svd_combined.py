#!/usr/bin/env python3
"""Create combined SVD plots for 70% vs 20% posterior and unconditional prior."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def load(path: Path):
    return torch.load(path, map_location="cpu").detach().numpy()


def plot_triplet(a, b, c, out_path: Path, title: str):
    a = a / max(float(a[0]), 1e-12)
    b = b / max(float(b[0]), 1e-12)
    c = c / max(float(c[0]), 1e-12)
    idx = range(1, len(a) + 1)
    plt.figure(figsize=(8, 4.6))
    plt.semilogy(idx, a, linewidth=1.6, label="70% observed")
    plt.semilogy(idx, b, linewidth=1.6, label="20% observed")
    plt.semilogy(idx, c, linewidth=1.6, label="Prior (unconditional)")
    plt.xlabel("Mode index")
    plt.ylabel("Normalized singular value")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    base = Path("report/assets/gautschi_edm_svd_combined")
    samples70 = load(base / "singular_values_samples_70pct.pt")
    samples20 = load(base / "singular_values_samples_20pct.pt")
    scores70 = load(base / "singular_values_scores_70pct.pt")
    scores20 = load(base / "singular_values_scores_20pct.pt")
    samples_prior = load(base / "singular_values_samples_prior.pt")
    scores_prior = load(base / "singular_values_scores_prior.pt")

    plot_triplet(
        samples70,
        samples20,
        samples_prior,
        base / "singular_value_spectrum_samples_70_vs_20.png",
        "Sample Snapshot Spectrum: 70% vs 20% Observed + Prior",
    )
    plot_triplet(
        scores70,
        scores20,
        scores_prior,
        base / "singular_value_spectrum_scores_70_vs_20.png",
        "Score Snapshot Spectrum: 70% vs 20% Observed + Prior",
    )


if __name__ == "__main__":
    main()
