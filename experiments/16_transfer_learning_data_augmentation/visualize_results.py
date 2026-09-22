"""
Visualization suite for Experiment 16: Transfer Learning & Data Augmentation Regularization.

Generates publication-quality figures:
1. Augmentation sample transformations across policies.
2. Training loss & accuracy dynamics demonstrating overfitting mitigation.
3. Generalization gap and quantitative metric comparisons.
"""

import os
import json
from typing import Dict, Any, Optional, List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# Project-wide styling constants
PALETTE = {
    "none": "#E63946",         # Crimson / Red (Overfitting warning)
    "geometric": "#457B9D",    # Steel Blue
    "photometric": "#2A9D8F",  # Persian Green
    "combined": "#E76F51",     # Burnt Orange
    "mixup": "#6A4C93",        # Royal Purple
}

LABELS = {
    "none": "No Augmentation (Baseline)",
    "geometric": "Geometric Invariance",
    "photometric": "Photometric Invariance",
    "combined": "Combined Augmentation",
    "mixup": "Mixup Regularization",
}


def plot_training_curves_comparison(
    results: Dict[str, Any],
    save_path: Optional[str] = None,
):
    """
    Plots multi-panel training and validation dynamics illustrating the
    severe overfitting of the unaugmented baseline and the regularizing
    power of data augmentation.
    """
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=200)

    epochs = len(next(iter(results.values()))["train_accuracy"])
    epoch_axis = range(1, epochs + 1)

    # Panel 1: Baseline (No Aug) vs Combined Loss Curves (The Overfitting Divergence)
    ax1 = axes[0, 0]
    if "none" in results:
        ax1.plot(
            epoch_axis,
            results["none"]["train_loss"],
            color="#E63946",
            linestyle="--",
            linewidth=2.0,
            label="Baseline (No Aug) - Train Loss",
        )
        ax1.plot(
            epoch_axis,
            results["none"]["val_loss"],
            color="#E63946",
            linestyle="-",
            linewidth=2.5,
            label="Baseline (No Aug) - Val Loss (Overfitting)",
        )
    if "combined" in results:
        ax1.plot(
            epoch_axis,
            results["combined"]["train_loss"],
            color="#2A9D8F",
            linestyle="--",
            linewidth=2.0,
            label="Combined Aug - Train Loss",
        )
        ax1.plot(
            epoch_axis,
            results["combined"]["val_loss"],
            color="#2A9D8F",
            linestyle="-",
            linewidth=2.5,
            label="Combined Aug - Val Loss (Regularized)",
        )
    ax1.set_title("A. Loss Trajectories: Baseline Memorization vs. Regularization", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Categorical Crossentropy Loss", fontsize=11)
    ax1.legend(loc="upper right", frameon=True, fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Panel 2: Validation Accuracy Across All Policies
    ax2 = axes[0, 1]
    for policy, data in results.items():
        color = PALETTE.get(policy, "#333333")
        label = LABELS.get(policy, policy.capitalize())
        ax2.plot(
            epoch_axis,
            data["val_accuracy"],
            color=color,
            linewidth=2.2,
            label=f"{label} (Best: {data['best_val_acc']*100:.1f}%)",
        )
    ax2.set_title("B. Validation Accuracy Progression", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Validation Accuracy", fontsize=11)
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    ax2.legend(loc="lower right", frameon=True, fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.6)

    # Panel 3: Generalization Accuracy Gap (Train Acc - Val Acc)
    ax3 = axes[1, 0]
    for policy, data in results.items():
        color = PALETTE.get(policy, "#333333")
        label = LABELS.get(policy, policy.capitalize())
        ax3.plot(
            epoch_axis,
            data["acc_gap"],
            color=color,
            linewidth=2.2,
            label=f"{label} (Final Gap: {data['final_acc_gap']*100:.1f}%)",
        )
    ax3.axhline(0, color="gray", linestyle=":", linewidth=1.2)
    ax3.set_title("C. Generalization Gap: $\\Delta_{acc} = \\text{Acc}_{train} - \\text{Acc}_{val}$", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Epoch", fontsize=11)
    ax3.set_ylabel("Accuracy Gap", fontsize=11)
    ax3.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    ax3.legend(loc="upper left", frameon=True, fontsize=9)
    ax3.grid(True, linestyle="--", alpha=0.6)

    # Panel 4: Training vs Validation Accuracy for Baseline and Mixup
    ax4 = axes[1, 1]
    for policy in ["none", "mixup", "combined"]:
        if policy in results:
            color = PALETTE.get(policy, "#333333")
            label = LABELS.get(policy, policy.capitalize())
            ax4.plot(
                epoch_axis,
                results[policy]["train_accuracy"],
                color=color,
                linestyle="--",
                linewidth=1.8,
                label=f"{label} (Train)",
            )
            ax4.plot(
                epoch_axis,
                results[policy]["val_accuracy"],
                color=color,
                linestyle="-",
                linewidth=2.2,
                label=f"{label} (Val)",
            )
    ax4.set_title("D. Train vs. Validation Accuracy Evolution", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Epoch", fontsize=11)
    ax4.set_ylabel("Accuracy", fontsize=11)
    ax4.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    ax4.legend(loc="center right", frameon=True, fontsize=8.5)
    ax4.grid(True, linestyle="--", alpha=0.6)

    plt.suptitle(
        "Empirical Proof: Mitigating Transfer Learning Overfitting via Data Augmentation",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()

    if save_path:
        if os.path.dirname(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
        print(f"📊 Saved training curves figure to: {save_path}")
    return fig


def plot_generalization_summary_barchart(
    results: Dict[str, Any],
    save_path: Optional[str] = None,
):
    """
    Plots a multi-panel quantitative summary comparing final accuracy,
    generalization gaps, and validation loss across all tested augmentation policies.
    """
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5), dpi=200)

    policies = list(results.keys())
    display_names = [LABELS.get(p, p.capitalize()).replace(" ", "\n") for p in policies]
    colors = [PALETTE.get(p, "#457B9D") for p in policies]

    # Panel 1: Train vs Validation Accuracy
    train_accs = [results[p]["final_train_acc"] * 100 for p in policies]
    val_accs = [results[p]["final_val_acc"] * 100 for p in policies]

    x = np.arange(len(policies))
    width = 0.35

    rects1 = ax1.bar(x - width / 2, train_accs, width, label="Train Acc", color="#999999", alpha=0.8)
    rects2 = ax1.bar(x + width / 2, val_accs, width, label="Val Acc", color=colors)

    ax1.set_title("A. Final Train vs. Val Accuracy", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(display_names, fontsize=8.5)
    ax1.set_ylabel("Accuracy (%)", fontsize=10)
    ax1.set_ylim(0, 105)
    ax1.legend(loc="lower right", frameon=True, fontsize=9)
    ax1.grid(axis="y", linestyle="--", alpha=0.6)

    for r in rects2:
        h = r.get_height()
        ax1.annotate(
            f"{h:.1f}%",
            xy=(r.get_x() + r.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    # Panel 2: Generalization Gap (Train Acc - Val Acc)
    gaps = [results[p]["final_acc_gap"] * 100 for p in policies]
    rects_gap = ax2.bar(x, gaps, color=colors, width=0.55, edgecolor="black", linewidth=0.5)

    ax2.set_title("B. Generalization Gap (Overfitting Index)\n$\\Delta = \\text{Acc}_{train} - \\text{Acc}_{val}$ (Lower is Better)", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(display_names, fontsize=8.5)
    ax2.set_ylabel("Gap (%)", fontsize=10)
    ax2.grid(axis="y", linestyle="--", alpha=0.6)

    for r in rects_gap:
        h = r.get_height()
        ax2.annotate(
            f"{h:.1f}%",
            xy=(r.get_x() + r.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    # Panel 3: Final Validation Loss
    val_losses = [results[p]["final_val_loss"] for p in policies]
    rects_loss = ax3.bar(x, val_losses, color=colors, width=0.55, edgecolor="black", linewidth=0.5)

    ax3.set_title("C. Final Validation Crossentropy Loss\n(Lower is Better)", fontsize=11, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(display_names, fontsize=8.5)
    ax3.set_ylabel("Val Loss", fontsize=10)
    ax3.grid(axis="y", linestyle="--", alpha=0.6)

    for r in rects_loss:
        h = r.get_height()
        ax3.annotate(
            f"{h:.2f}",
            xy=(r.get_x() + r.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    plt.suptitle("Quantitative Regularization Benchmark Across Augmentation Schemes", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        if os.path.dirname(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
        print(f"📊 Saved generalization summary barchart to: {save_path}")
    return fig
