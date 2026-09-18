"""Publication-quality visualization suite for Image Quality Assessment (IQA).

Generates:
1. 01_distortion_taxonomy_gallery.png: Multi-distortion and intensity gallery.
2. 02_classical_vs_perceptual_correlation.png: 4-panel correlation scatter plots.
3. 03_training_dynamics.png: Convergence and correlation history curves.
4. 04_distortion_wise_performance_breakdown.png: SROCC grouped bar chart across distortion types.
5. 05_perceptual_error_heatmaps.png: Spatial feature error maps from FR-IQA model.
"""

import os
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

try:
    from .dataset import DISTORTION_NAMES
    from .metrics import fit_logistic_mapping
    from .models import get_fr_spatial_diff_extractor
except ImportError:
    from dataset import DISTORTION_NAMES
    from metrics import fit_logistic_mapping
    from models import get_fr_spatial_diff_extractor



def set_matplotlib_style():
    """Apply clean, modern publication style to matplotlib figures."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 15,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "grid.color": "#dddddd",
            "grid.linestyle": "--",
            "grid.alpha": 0.7,
        }
    )


def plot_distortion_taxonomy_gallery(
    data: List[Dict],
    ref_id: int = 1,
    selected_dist_ids: Optional[List[int]] = None,
    save_path: str = "figures/01_distortion_taxonomy_gallery.png",
):
    """Visual gallery of reference image across distortion types and intensity levels."""
    set_matplotlib_style()
    if selected_dist_ids is None:
        # Gaussian noise (1), Gaussian blur (8), JPEG (10), Contrast (17)
        selected_dist_ids = [1, 8, 10, 17]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(
        len(selected_dist_ids), 5, figsize=(15, 3 * len(selected_dist_ids)), dpi=200
    )

    # Find pristine reference image
    ref_sample = next(
        item for item in data if item["reference_id"] == ref_id and item["distortion_id"] == 1
    )
    ref_img = ref_sample["reference"]

    for row_idx, dist_id in enumerate(selected_dist_ids):
        dist_name = DISTORTION_NAMES.get(dist_id, f"Distortion {dist_id}")

        # Column 0: Pristine Reference
        axes[row_idx, 0].imshow(ref_img)
        axes[row_idx, 0].axis("off")
        if row_idx == 0:
            axes[row_idx, 0].set_title("Pristine Reference\n(Ground Truth)", fontweight="bold", pad=10)
        axes[row_idx, 0].set_ylabel(dist_name, fontsize=10, fontweight="bold", labelpad=10)

        # Columns 1-4: Intensities 1 to 4
        for intensity in range(1, 5):
            sample = next(
                (
                    item
                    for item in data
                    if item["reference_id"] == ref_id
                    and item["distortion_id"] == dist_id
                    and item["distortion_intensity"] == intensity
                ),
                None,
            )
            ax = axes[row_idx, intensity]
            if sample is not None:
                ax.imshow(sample["distorted"])
                ax.axis("off")
                mos_val = float(sample["mos"])
                title_str = f"Intensity {intensity}\nMOS: {mos_val:.2f}"
                if row_idx == 0:
                    title_str = f"Level {intensity}\nMOS: {mos_val:.2f}"
                ax.set_title(title_str, fontsize=10)

    fig.suptitle(
        f"TID2008 Distortion Taxonomy & Perceived Quality (Reference #{ref_id})",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved gallery figure to {save_path}")


def plot_classical_vs_perceptual_correlation(
    test_mos: np.ndarray,
    predictions: Dict[str, np.ndarray],
    save_path: str = "figures/02_classical_vs_perceptual_correlation.png",
):
    """4-panel scatter plot comparing PSNR, SSIM, Deep FR-IQA, and Deep NR-IQA against MOS."""
    set_matplotlib_style()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13, 11), dpi=200)
    axes = axes.flatten()

    model_keys = ["PSNR", "SSIM", "Deep FR-IQA", "Deep NR-IQA"]
    colors = ["#2b5c8f", "#2e7d32", "#d32f2f", "#7b1fa2"]

    for idx, model_name in enumerate(model_keys):
        ax = axes[idx]
        if model_name not in predictions:
            ax.text(0.5, 0.5, f"{model_name}\nNot Available", ha="center", va="center")
            continue

        raw_pred = predictions[model_name]
        mapped_pred = fit_logistic_mapping(raw_pred, test_mos)

        srocc, _ = stats.spearmanr(test_mos, raw_pred)
        plcc, _ = stats.pearsonr(test_mos, mapped_pred)
        rmse = np.sqrt(np.mean((test_mos - mapped_pred) ** 2))

        # Scatter points
        ax.scatter(
            test_mos,
            mapped_pred,
            c=colors[idx],
            alpha=0.6,
            edgecolors="none",
            s=35,
            label="Test Samples",
        )

        # Ideal y=x diagonal line
        min_v = min(np.min(test_mos), np.min(mapped_pred))
        max_v = max(np.max(test_mos), np.max(mapped_pred))
        ax.plot(
            [min_v, max_v],
            [min_v, max_v],
            color="#444444",
            linestyle="--",
            linewidth=1.5,
            label="Ideal Alignment ($y=x$)",
        )

        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xlabel("Subjective Ground Truth (MOS)", fontweight="bold")
        ax.set_ylabel(f"Mapped Predicted Quality", fontweight="bold")
        ax.set_title(
            f"{model_name}\nSROCC: {srocc:.4f} | PLCC: {plcc:.4f} | RMSE: {rmse:.3f}",
            fontweight="bold",
            pad=10,
        )
        ax.legend(loc="upper left", frameon=True, framealpha=0.9)

    fig.suptitle(
        "IQA Benchmark: Classical Metrics vs Deep Perceptual Models on Unseen References",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved correlation scatter plot to {save_path}")


def plot_training_dynamics(
    fr_history: Dict,
    nr_history: Dict,
    save_path: str = "figures/03_training_dynamics.png",
):
    """Plot training and validation loss along with SROCC/PLCC evolution across epochs."""
    set_matplotlib_style()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=200)

    # 1. Loss curves
    ax = axes[0]
    if "loss" in fr_history:
        ax.plot(fr_history["loss"], label="FR-IQA Train Loss", color="#d32f2f", linewidth=2)
        ax.plot(
            fr_history["val_loss"],
            label="FR-IQA Val Loss",
            color="#d32f2f",
            linestyle="--",
            linewidth=2,
        )
    if "loss" in nr_history:
        ax.plot(nr_history["loss"], label="NR-IQA Train Loss", color="#7b1fa2", linewidth=2)
        ax.plot(
            nr_history["val_loss"],
            label="NR-IQA Val Loss",
            color="#7b1fa2",
            linestyle="--",
            linewidth=2,
        )

    ax.set_title("Training & Validation Loss (Huber Loss)", fontweight="bold")
    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel("Loss", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True)

    # 2. Validation SROCC progression
    ax = axes[1]
    if "val_srocc" in fr_history:
        ax.plot(
            fr_history["val_srocc"],
            label="FR-IQA Val SROCC",
            color="#d32f2f",
            linewidth=2,
            marker="o",
            markersize=4,
        )
    if "val_srocc" in nr_history:
        ax.plot(
            nr_history["val_srocc"],
            label="NR-IQA Val SROCC",
            color="#7b1fa2",
            linewidth=2,
            marker="s",
            markersize=4,
        )

    ax.set_title("Validation SROCC (Monotonic Ranking Correlation)", fontweight="bold")
    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel("SROCC (Higher is Better)", fontweight="bold")
    ax.set_ylim(-0.1, 1.0)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", frameon=True)

    fig.suptitle("Deep IQA Model Training Dynamics", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved training dynamics figure to {save_path}")


def plot_distortion_wise_breakdown(
    distortion_df: pd.DataFrame,
    save_path: str = "figures/04_distortion_wise_performance_breakdown.png",
):
    """Grouped bar chart showing SROCC per distortion family across models."""
    set_matplotlib_style()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    pivot_df = distortion_df.pivot(
        index="Distortion Group", columns="Model", values="SROCC"
    )

    models_order = [m for m in ["PSNR", "SSIM", "Deep FR-IQA", "Deep NR-IQA"] if m in pivot_df.columns]
    pivot_df = pivot_df[models_order]

    colors = {
        "PSNR": "#2b5c8f",
        "SSIM": "#2e7d32",
        "Deep FR-IQA": "#d32f2f",
        "Deep NR-IQA": "#7b1fa2",
    }
    bar_colors = [colors.get(col, "#555555") for col in pivot_df.columns]

    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    pivot_df.plot(
        kind="bar",
        ax=ax,
        color=bar_colors,
        width=0.75,
        edgecolor="#222222",
        linewidth=0.8,
    )

    ax.set_title(
        "SROCC Monotonic Correlation Across Distortion Categories (Unseen Test Scenes)",
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Distortion Category", fontweight="bold")
    ax.set_ylabel("SROCC Score (Higher is Better)", fontweight="bold")
    ax.set_ylim(-0.1, 1.05)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
    ax.legend(title="Method", frameon=True, framealpha=0.9, loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved distortion breakdown figure to {save_path}")


def plot_perceptual_error_heatmaps(
    fr_model,
    test_refs: np.ndarray,
    test_dists: np.ndarray,
    test_mos: np.ndarray,
    sample_indices: Optional[List[int]] = None,
    save_path: str = "figures/05_perceptual_error_heatmaps.png",
):
    """Extract and visualize multi-scale spatial difference maps |phi(ref) - phi(dist)|."""
    set_matplotlib_style()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if sample_indices is None:
        sample_indices = [0, 1, 2, 3]

    diff_extractor = get_fr_spatial_diff_extractor(fr_model)

    sub_refs = test_refs[sample_indices]
    sub_dists = test_dists[sample_indices]
    sub_mos = test_mos[sample_indices]

    diff1, diff2, diff3 = diff_extractor.predict(
        {"reference": sub_refs, "distorted": sub_dists}, verbose=0
    )

    n_samples = len(sample_indices)
    fig, axes = plt.subplots(n_samples, 5, figsize=(15, 3 * n_samples), dpi=200)

    for i in range(n_samples):
        # 1. Reference
        axes[i, 0].imshow(sub_refs[i])
        axes[i, 0].axis("off")
        if i == 0:
            axes[i, 0].set_title("Reference ($I_{ref}$)", fontweight="bold", pad=10)

        # 2. Distorted
        axes[i, 1].imshow(sub_dists[i])
        axes[i, 1].axis("off")
        if i == 0:
            axes[i, 1].set_title("Distorted ($I_{dist}$)", fontweight="bold", pad=10)
        axes[i, 1].set_ylabel(f"Sample {i+1}\nMOS: {sub_mos[i]:.2f}", fontweight="bold")

        # 3. Stage 1 Error Map (Low-level texture)
        map1 = np.mean(diff1[i], axis=-1)
        im1 = axes[i, 2].imshow(map1, cmap="magma")
        axes[i, 2].axis("off")
        if i == 0:
            axes[i, 2].set_title("Stage 1 Residual ($\Delta \phi_1$)\n[Texture/Edges]", fontweight="bold", pad=10)

        # 4. Stage 2 Error Map (Mid-level structure)
        map2 = np.mean(diff2[i], axis=-1)
        im2 = axes[i, 3].imshow(map2, cmap="magma")
        axes[i, 3].axis("off")
        if i == 0:
            axes[i, 3].set_title("Stage 2 Residual ($\Delta \phi_2$)\n[Structure]", fontweight="bold", pad=10)

        # 5. Stage 3 Error Map (High-level semantic)
        map3 = np.mean(diff3[i], axis=-1)
        im3 = axes[i, 4].imshow(map3, cmap="magma")
        axes[i, 4].axis("off")
        if i == 0:
            axes[i, 4].set_title("Stage 3 Residual ($\Delta \phi_3$)\n[High-Level]", fontweight="bold", pad=10)

    fig.suptitle(
        "Full-Reference Perceptual Error Maps Across Hierarchical CNN Stages",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved perceptual error heatmaps to {save_path}")
