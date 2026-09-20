"""Publication-Quality Visualization Suite for Time-Series Image Encodings.

Includes routines for:
1. Multi-transform comparison gallery (1D Series -> GAF -> CWT -> STFT across all 4 regimes)
2. Cone of Influence (COI) mathematical artifact demonstration
3. Multi-representation & Multi-architecture Confusion Matrices Grid
4. Transform Tournament Benchmark (Accuracy, F1, Per-Class performance)
5. Grad-CAM attention heatmap overlays.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

# Ensure experiment folder is in sys.path for sibling imports
_EXPERIMENT_DIR = str(Path(__file__).resolve().parent)
if _EXPERIMENT_DIR not in sys.path:
    sys.path.insert(0, _EXPERIMENT_DIR)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataset import REGIME_NAMES
from encoders import (
    apply_colormap,
    compute_cwt,
    compute_gaf,
    compute_stft,
)
from evaluate import compute_gradcam


def setup_plot_style() -> None:
    """Set clean, publication-ready matplotlib styling."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 14,
        "figure.autolayout": False,
    })


def plot_transformation_gallery(
    windows: np.ndarray,
    labels: np.ndarray,
    image_size: int = 128,
    save_path: Optional[str | Path] = "experiments/08_time_to_img/assets/transform_gallery.png",
) -> plt.Figure:
    """Plot representative samples from all 4 regimes across all 2D transformations."""
    setup_plot_style()
    fig, axes = plt.subplots(4, 5, figsize=(18, 14), constrained_layout=True)

    col_titles = [
        "1D Window (Price Drift)",
        "Gramian Angular Summation (GASF)",
        "Gramian Angular Difference (GADF)",
        "CWT Scalogram (Morlet)",
        "STFT Spectrogram",
    ]

    for regime_idx in range(4):
        sample_indices = np.where(labels == regime_idx)[0]
        if len(sample_indices) == 0:
            continue
        idx = sample_indices[0]
        w = windows[idx]

        # 1. 1D Price curve
        ax0 = axes[regime_idx, 0]
        cum_ret = np.cumsum(w) * 100
        ax0.plot(cum_ret, color="#1f77b4", lw=2)
        ax0.set_ylabel(f"Regime {regime_idx}\n{REGIME_NAMES[regime_idx]}", fontweight="bold", fontsize=11)
        ax0.set_xlabel("Time Step (bars)")
        if regime_idx == 0:
            ax0.set_title(col_titles[0], fontweight="bold")

        # 2. GASF
        ax1 = axes[regime_idx, 1]
        gasf = compute_gaf(np.cumsum(w), image_size=image_size, method="summation")
        ax1.imshow(gasf, cmap="viridis", origin="lower")
        ax1.axis("off")
        if regime_idx == 0:
            ax1.set_title(col_titles[1], fontweight="bold")

        # 3. GADF
        ax2 = axes[regime_idx, 2]
        gadf = compute_gaf(np.cumsum(w), image_size=image_size, method="difference")
        ax2.imshow(gadf, cmap="plasma", origin="lower")
        ax2.axis("off")
        if regime_idx == 0:
            ax2.set_title(col_titles[2], fontweight="bold")

        # 4. CWT
        ax3 = axes[regime_idx, 3]
        scalogram, _ = compute_cwt(w, image_size=image_size)
        ax3.imshow(scalogram, cmap="magma", origin="lower", aspect="auto")
        ax3.axis("off")
        if regime_idx == 0:
            ax3.set_title(col_titles[3], fontweight="bold")

        # 5. STFT
        ax4 = axes[regime_idx, 4]
        spectrogram = compute_stft(w, image_size=image_size)
        ax4.imshow(spectrogram, cmap="inferno", origin="lower", aspect="auto")
        ax4.axis("off")
        if regime_idx == 0:
            ax4.set_title(col_titles[4], fontweight="bold")

    fig.suptitle("Transform Tournament: 1D Financial Series Encoded into 2D Computer Vision Inputs", fontsize=16, fontweight="bold")

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300)

    return fig


def plot_coi_demonstration(
    window: np.ndarray,
    image_size: int = 128,
    save_path: Optional[str | Path] = "experiments/08_time_to_img/assets/coi_demonstration.png",
) -> plt.Figure:
    """Demonstrate the Cone of Influence (COI) and wavelet boundary artifacts."""
    setup_plot_style()
    scalogram, coi_mask = compute_cwt(window, image_size=image_size)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    # Raw Scalogram
    axes[0].imshow(scalogram, cmap="magma", origin="lower", aspect="auto")
    axes[0].set_title("Raw CWT Scalogram", fontweight="bold")
    axes[0].set_xlabel("Time (bars)")
    axes[0].set_ylabel("Wavelet Scale (Low -> High Freq)")

    # COI Mask Overlay
    coi_overlay = scalogram.copy()
    axes[1].imshow(coi_overlay, cmap="magma", origin="lower", aspect="auto")
    axes[1].imshow(coi_mask, cmap="Blues", alpha=0.45, origin="lower", aspect="auto")
    axes[1].set_title("Cone of Influence (COI) Shaded Zone", fontweight="bold")
    axes[1].set_xlabel("Time (bars)")

    # COI-Masked Scalogram (Valid Interior Only)
    masked_scalogram = np.where(coi_mask, np.nan, scalogram)
    axes[2].imshow(masked_scalogram, cmap="magma", origin="lower", aspect="auto")
    axes[2].set_title("Uncompromised Interior (Zero Edge Artifacts)", fontweight="bold")
    axes[2].set_xlabel("Time (bars)")

    fig.suptitle("Pedagogical Diagnostic: Cone of Influence (COI) Boundary Attenuation in CWT", fontsize=14, fontweight="bold")

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300)

    return fig


def plot_confusion_matrices(
    results_map: Dict[str, Dict[str, Any]],
    ncols: int = 4,
    save_path: Optional[str | Path] = "experiments/08_time_to_img/assets/confusion_matrices.png",
) -> plt.Figure:
    """Plot multi-class confusion matrices in a 2D grid for all evaluated models and encodings."""
    setup_plot_style()
    n_items = len(results_map)
    cols = min(ncols, n_items)
    rows = math.ceil(n_items / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.8 * rows), constrained_layout=True)
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[ax] for ax in axes])

    class_names = [REGIME_NAMES[i] for i in range(4)]

    for idx, (exp_title, res) in enumerate(results_map.items()):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]

        cm = res["confusion_matrix"]
        cm_norm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-8)

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_title(f"{exp_title}\n(Acc: {res['accuracy']:.1%}, Macro F1: {res['macro_f1']:.2f})", fontweight="bold", fontsize=11)
        ax.set_xticks(np.arange(4))
        ax.set_yticks(np.arange(4))
        ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=9)
        ax.set_yticklabels(class_names, fontsize=9)
        ax.set_xlabel("Predicted Regime", fontweight="bold", fontsize=10)
        if c == 0:
            ax.set_ylabel("True Regime", fontweight="bold", fontsize=10)

        for i in range(4):
            for j in range(4):
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.0%})", ha="center", va="center", color=color, fontsize=8)

    # Hide any unused subplots in the grid
    for idx in range(n_items, rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r, c].axis("off")

    fig.suptitle("Market Regime Classification: Confusion Matrices Across Encodings & Architectures", fontsize=15, fontweight="bold")

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300)

    return fig


def plot_tournament_summary(
    tournament_df: pd.DataFrame,
    save_path: Optional[str | Path] = "experiments/08_time_to_img/assets/tournament_benchmark.png",
) -> plt.Figure:
    """Plot comparative benchmark bar chart across Encoders and Backbones."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)

    x = np.arange(len(tournament_df))
    width = 0.35

    ax.bar(x - width / 2, tournament_df["Accuracy"] * 100, width, label="Accuracy (%)", color="#1f77b4")
    ax.bar(x + width / 2, tournament_df["Macro_F1"] * 100, width, label="Macro F1-Score (x100)", color="#2ca02c")

    ax.set_ylabel("Score (%)", fontweight="bold")
    ax.set_title("Transform Tournament: Comparative Benchmark across Representations & Architectures", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(tournament_df["Experiment"], rotation=25, ha="right", fontweight="bold")
    ax.legend(loc="upper left")
    ax.set_ylim(0, 105)

    for i in range(len(tournament_df)):
        acc = tournament_df["Accuracy"].iloc[i] * 100
        f1 = tournament_df["Macro_F1"].iloc[i] * 100
        ax.text(i - width / 2, acc + 1.5, f"{acc:.1f}%", ha="center", fontsize=9, fontweight="bold")
        ax.text(i + width / 2, f1 + 1.5, f"{f1:.1f}", ha="center", fontsize=9, fontweight="bold")

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300)

    return fig


def plot_gradcam_explanations(
    model: Any,
    images: np.ndarray,
    labels: np.ndarray,
    method_name: str = "Fusion",
    num_samples: int = 4,
    save_path: Optional[str | Path] = "experiments/08_time_to_img/assets/gradcam_explanations.png",
) -> plt.Figure:
    """Generate Grad-CAM interpretability visualizations for test images."""
    setup_plot_style()
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 3.5 * num_samples), constrained_layout=True)

    col_titles = ["Input Image Tensor", "Grad-CAM Heatmap", "Overlay (Network Focus)"]

    for i in range(num_samples):
        img = images[i]
        true_label = labels[i]

        heatmap = compute_gradcam(model, img, target_class=int(true_label))

        heatmap_colored = plt.get_cmap("jet")(heatmap)[:, :, :3]
        overlay = 0.6 * img + 0.4 * heatmap_colored
        overlay = np.clip(overlay, 0.0, 1.0)

        axes[i, 0].imshow(img)
        axes[i, 0].axis("off")
        axes[i, 0].set_ylabel(f"Sample {i+1}\nTrue: {REGIME_NAMES[true_label]}", fontweight="bold", fontsize=10)
        if i == 0:
            axes[i, 0].set_title(col_titles[0], fontweight="bold")

        axes[i, 1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
        axes[i, 1].axis("off")
        if i == 0:
            axes[i, 1].set_title(col_titles[1], fontweight="bold")

        axes[i, 2].imshow(overlay)
        axes[i, 2].axis("off")
        if i == 0:
            axes[i, 2].set_title(col_titles[2], fontweight="bold")

    fig.suptitle(f"Grad-CAM Attention Map Interpretability: {method_name.upper()} Representation", fontsize=14, fontweight="bold")

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300)

    return fig
