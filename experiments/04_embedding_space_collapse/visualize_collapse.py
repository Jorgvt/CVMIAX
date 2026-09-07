"""
Visualization of Embedding Space Collapse vs Well-Dispersed Representations.

Generates publication-quality, pedagogical figures for:
1. 2D Unit Circle Embeddings (Healthy vs Constant Collapse vs Dimensional Collapse)
2. Covariance Heatmaps & SVD Singular Value Spectra (Rank Degeneracy Analysis)
3. Master Lecture Slide Overview Poster
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from collapse_utils import (
    generate_healthy_embeddings,
    generate_constant_collapsed_embeddings,
    generate_subspace_collapsed_embeddings,
    compute_svd_spectrum,
)


def set_plot_style():
    plt.rcParams.update({
        "font.sans-serif": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "figure.titlesize": 16,
        "figure.titleweight": "bold",
        "figure.dpi": 300,
    })


# Class colors and names for intuitive pedagogical visualization
CLASS_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
CLASS_NAMES = ["Cat", "Dog", "Car", "Plane", "Bird"]


# =========================================================================
# 1. 2D Unit Circle Embeddings: Healthy vs Complete vs Dimensional Collapse
# =========================================================================

def plot_2d_embedding_comparison(output_dir):
    num_samples = 500
    healthy_pts, labels = generate_healthy_embeddings(num_samples=num_samples, num_classes=5, dim=2)
    const_pts, _ = generate_constant_collapsed_embeddings(num_samples=num_samples, num_classes=5, dim=2)
    sub_pts, _ = generate_subspace_collapsed_embeddings(num_samples=num_samples, num_classes=5, dim=2)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    datasets = [
        ("Healthy / Well-Dispersed Space\n(High Uniformity & Alignment ✓)", healthy_pts, "#2ca02c"),
        ("Complete Constant Collapse ✗\n(All samples $\\mathbf{z} \\to \\mathbf{c}$, 0 bits information)", const_pts, "#d62728"),
        ("Dimensional / Subspace Collapse ✗\n(Embeddings collapse to a 1D line)", sub_pts, "#d62728"),
    ]

    for ax, (title, pts, title_color) in zip(axes, datasets):
        # Draw Unit Circle
        unit_circle = plt.Circle((0, 0), 1.0, color="gray", fill=False, linestyle="--", alpha=0.6, linewidth=1.5)
        ax.add_patch(unit_circle)

        # Plot coordinate axes
        ax.axhline(0, color="lightgray", linestyle=":", linewidth=1)
        ax.axvline(0, color="lightgray", linestyle=":", linewidth=1)

        # Plot samples per class
        for c_idx in range(5):
            mask = labels == c_idx
            ax.scatter(
                pts[mask, 0], pts[mask, 1],
                color=CLASS_COLORS[c_idx],
                label=CLASS_NAMES[c_idx],
                alpha=0.75,
                s=35,
                edgecolors="none",
            )

        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-1.25, 1.25)
        ax.set_aspect("equal")
        ax.set_xlabel("Latent Dimension $z_1$", fontsize=11)
        ax.set_ylabel("Latent Dimension $z_2$", fontsize=11)
        ax.set_title(title, fontsize=12, color=title_color, fontweight="bold", pad=10)
        ax.grid(True, linestyle="--", alpha=0.3)

    axes[0].legend(loc="upper right", framealpha=0.9, fontsize=9.5)

    plt.suptitle(
        "Self-Supervised Learning: Embedding Space Collapse vs Healthy Representation",
        fontsize=15,
        fontweight="bold",
        y=1.03,
    )
    plt.tight_layout()
    save_path = os.path.join(output_dir, "01_2d_embedding_space_comparison.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# =========================================================================
# 2. Covariance Matrix Heatmaps & SVD Spectra (Rank Degeneracy Analysis)
# =========================================================================

def plot_covariance_and_svd(output_dir, dim=16):
    num_samples = 1000
    healthy_pts, _ = generate_healthy_embeddings(num_samples=num_samples, num_classes=5, dim=dim, noise_std=0.3)
    const_pts, _ = generate_constant_collapsed_embeddings(num_samples=num_samples, num_classes=5, dim=dim)
    sub_pts, _ = generate_subspace_collapsed_embeddings(num_samples=num_samples, num_classes=5, dim=dim)

    cov_h, svd_h = compute_svd_spectrum(healthy_pts)
    cov_c, svd_c = compute_svd_spectrum(const_pts)
    cov_s, svd_s = compute_svd_spectrum(sub_pts)

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    # --- Top Row: Covariance Matrix Heatmaps ---
    vmax = max(np.max(cov_h), np.max(cov_s))
    im0 = axes[0, 0].imshow(cov_h, cmap="coolwarm", vmin=-0.05, vmax=0.1)
    axes[0, 0].set_title("Healthy Covariance $\\mathbf{Cov}(Z)$\n(Diagonal energy, decorrelated off-diagonals)", fontsize=11, color="green")
    axes[0, 0].set_xlabel("Latent Dim Index")
    axes[0, 0].set_ylabel("Latent Dim Index")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(cov_c, cmap="coolwarm", vmin=-0.05, vmax=0.1)
    axes[0, 1].set_title("Complete Collapse Covariance\n(Zero variance across all dimensions $\\to$ 0)", fontsize=11, color="red")
    axes[0, 1].set_xlabel("Latent Dim Index")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[0, 2].imshow(cov_s, cmap="coolwarm", vmin=-0.05, vmax=0.1)
    axes[0, 2].set_title("Subspace Collapse Covariance\n(Degenerate Rank $\\approx 1$, all dims collinear)", fontsize=11, color="red")
    axes[0, 2].set_xlabel("Latent Dim Index")
    fig.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # --- Bottom Row: SVD Singular Value Spectra (Scree Plots) ---
    dims_range = np.arange(1, dim + 1)

    axes[1, 0].plot(dims_range, svd_h, "o-", color="#2ca02c", linewidth=2.5, markersize=6)
    axes[1, 0].set_title("Healthy Singular Spectrum\n(Full Rank: All dimensions utilized)", fontsize=11, color="green")
    axes[1, 0].set_xlabel("Singular Value Index $k$")
    axes[1, 0].set_ylabel("Fraction of Variance Explained $\\sigma_k / \\sum \\sigma$")
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)
    axes[1, 0].set_ylim(-0.02, 1.05)

    axes[1, 1].plot(dims_range, svd_c, "s-", color="#d62728", linewidth=2.5, markersize=6)
    axes[1, 1].set_title("Complete Collapse Spectrum\n(Rank = 0: No variance)", fontsize=11, color="red")
    axes[1, 1].set_xlabel("Singular Value Index $k$")
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)
    axes[1, 1].set_ylim(-0.02, 1.05)

    axes[1, 2].plot(dims_range, svd_s, "d-", color="#d62728", linewidth=2.5, markersize=6)
    axes[1, 2].set_title("Subspace Collapse Spectrum\n(Rank = 1: 1st component absorbs 100% variance)", fontsize=11, color="red")
    axes[1, 2].set_xlabel("Singular Value Index $k$")
    axes[1, 2].grid(True, linestyle="--", alpha=0.5)
    axes[1, 2].set_ylim(-0.02, 1.05)

    plt.suptitle(
        "Spectral and Covariance Diagnosis of Embedding Space Collapse ($d=16$ Latent Dimensions)",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    save_path = os.path.join(output_dir, "02_spectral_analysis_and_covariance.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# =========================================================================
# 3. Master Lecture Slide Overview Poster
# =========================================================================

def plot_master_collapse_slide(output_dir):
    num_samples = 400
    healthy_pts, labels = generate_healthy_embeddings(num_samples=num_samples, num_classes=5, dim=2)
    const_pts, _ = generate_constant_collapsed_embeddings(num_samples=num_samples, num_classes=5, dim=2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))

    # Panel 1: Healthy Space
    circle1 = plt.Circle((0, 0), 1.0, color="gray", fill=False, linestyle="--", alpha=0.6, linewidth=1.5)
    axes[0].add_patch(circle1)
    for c_idx in range(5):
        mask = labels == c_idx
        axes[0].scatter(healthy_pts[mask, 0], healthy_pts[mask, 1], color=CLASS_COLORS[c_idx], label=CLASS_NAMES[c_idx], s=35, alpha=0.8)
    axes[0].set_xlim(-1.25, 1.25)
    axes[0].set_ylim(-1.25, 1.25)
    axes[0].set_aspect("equal")
    axes[0].set_title("1. Healthy Embedding Space ✓\n(High Uniformity & Alignment)", color="green", fontsize=12)
    axes[0].grid(True, linestyle="--", alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=9)

    # Panel 2: Collapsed Space
    circle2 = plt.Circle((0, 0), 1.0, color="gray", fill=False, linestyle="--", alpha=0.6, linewidth=1.5)
    axes[1].add_patch(circle2)
    for c_idx in range(5):
        mask = labels == c_idx
        axes[1].scatter(const_pts[mask, 0], const_pts[mask, 1], color=CLASS_COLORS[c_idx], s=35, alpha=0.8)
    axes[1].set_xlim(-1.25, 1.25)
    axes[1].set_ylim(-1.25, 1.25)
    axes[1].set_aspect("equal")
    axes[1].set_title("2. Collapsed Embedding Space ✗\n(Trivial Constant Solution: $\\forall x, f(x) = c$)", color="red", fontsize=12)
    axes[1].grid(True, linestyle="--", alpha=0.3)

    # Panel 3: Pedagogical Summary Box
    axes[2].text(
        0.5, 0.90, "Embedding Space Collapse: Causes & Solutions",
        fontsize=12, fontweight="bold", ha="center", va="center", color="black"
    )
    axes[2].text(
        0.5, 0.48,
        "Why Collapse Occurs:\n"
        "• In Siamese networks, minimizing distance $\\mathcal{L} = \\|z_1 - z_2\\|^2$\n"
        "  has a trivial minimum at $z_1 = z_2 = \\mathbf{c}$ (constant).\n"
        "• Without repulsion, gradients push all representations to a single point.\n\n"
        "How Modern SSL Methods Prevent Collapse:\n"
        "1. Contrastive Negative Pairs (SimCLR, MoCo):\n"
        "   Push dissimilar images apart via InfoNCE loss.\n"
        "2. Asymmetric Dynamics (BYOL, SimSiam):\n"
        "   Stop-gradient operation $(\\text{stop\\_grad})$ + Predictor head.\n"
        "3. Covariance Regularization (Barlow Twins, VICReg):\n"
        "   Explicitly enforce variance $\\text{Var}(z) \\geq 1$ and off-diagonal covariance $\\to 0$.\n"
        "4. Cluster Assignment (SwAV):\n"
        "   Enforce equal partition over learnable prototypes.",
        fontsize=9.5,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8f9fa", edgecolor="#343a40", linewidth=1.5),
    )
    axes[2].axis("off")

    plt.suptitle(
        "Self-Supervised Learning Core Concept: Embedding Space Collapse",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    save_path = os.path.join(output_dir, "slide_embedding_collapse_overview.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    set_plot_style()
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    os.makedirs(output_dir, exist_ok=True)

    print("Generating 2D Embedding Space Comparison...")
    plot_2d_embedding_comparison(output_dir)

    print("Generating Spectral Analysis & Covariance Heatmaps...")
    plot_covariance_and_svd(output_dir, dim=16)

    print("Generating Master Lecture Slide Poster...")
    plot_master_collapse_slide(output_dir)

    print(f"\n[Success] All collapse visualizations saved cleanly to: {output_dir}")


if __name__ == "__main__":
    main()
