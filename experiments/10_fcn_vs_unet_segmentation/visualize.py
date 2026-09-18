"""
Visualization Suite for Semantic Segmentation:
Plain FCN vs U-Net Comparison.

Generates publication-quality diagnostic and qualitative figures:
1. `fcn_vs_unet_qualitative_comparison.png`: Sample predictions with error maps.
2. `fcn_vs_unet_boundary_and_corner_sharpness.png`: Boundary IoU, per-class IoU, and edge cross-section profiles.
3. `fcn_vs_unet_training_dynamics.png`: Loss and accuracy curves across training epochs.
4. `fcn_vs_unet_architecture_comparison.png`: Architectural schematic diagram.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as patches

from dataset import CLASS_NAMES, CLASS_COLORS, NUM_CLASSES


# Custom colormap for discrete segmentation masks
def get_segmentation_cmap():
    rgb_colors = [np.array(CLASS_COLORS[i]) / 255.0 for i in range(NUM_CLASSES)]
    cmap = ListedColormap(rgb_colors, name="shapes_cmap")
    bounds = list(range(NUM_CLASSES + 1))
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def plot_qualitative_comparison(
    images: np.ndarray,
    true_masks: np.ndarray,
    fcn_preds: np.ndarray,
    unet_preds: np.ndarray,
    num_samples: int = 5,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot side-by-side qualitative comparisons with pixel error maps.

    Columns:
    [Input Image | Ground Truth | Plain FCN (No Skips) | U-Net (With Skips) | FCN Error | U-Net Error]
    """
    num_samples = min(num_samples, len(images))
    cmap, norm = get_segmentation_cmap()

    fig, axes = plt.subplots(
        nrows=num_samples,
        ncols=6,
        figsize=(18, 3.1 * num_samples),
        gridspec_kw={"wspace": 0.08, "hspace": 0.15},
    )
    if num_samples == 1:
        axes = np.expand_dims(axes, 0)

    col_titles = [
        "Input Image",
        "Ground Truth",
        "Plain FCN (No Skips)\n[Blurry Boundaries]",
        "U-Net (With Skips)\n[Sharp Boundaries]",
        "FCN Error Map\n(Misclassified Pixels)",
        "U-Net Error Map\n(Misclassified Pixels)",
    ]

    for row in range(num_samples):
        img = images[row]
        gt = true_masks[row]
        p_fcn = fcn_preds[row]
        p_unet = unet_preds[row]

        # Binary error maps (1 = error, 0 = correct)
        err_fcn = (p_fcn != gt).astype(np.float32)
        err_unet = (p_unet != gt).astype(np.float32)

        # 1. Input Image
        axes[row, 0].imshow(img)
        axes[row, 0].set_ylabel(f"Sample #{row + 1}", fontsize=12, fontweight="bold")

        # 2. Ground Truth
        axes[row, 1].imshow(gt, cmap=cmap, norm=norm)

        # 3. Plain FCN Prediction
        axes[row, 2].imshow(p_fcn, cmap=cmap, norm=norm)

        # 4. U-Net Prediction
        axes[row, 3].imshow(p_unet, cmap=cmap, norm=norm)

        # 5. FCN Error Map
        axes[row, 4].imshow(err_fcn, cmap="Reds", vmin=0, vmax=1)

        # 6. U-Net Error Map
        axes[row, 5].imshow(err_unet, cmap="Reds", vmin=0, vmax=1)

        for col in range(6):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(col_titles[col], fontsize=12, fontweight="bold", pad=8)

    # Class legend on bottom
    legend_elements = [
        patches.Patch(
            facecolor=np.array(CLASS_COLORS[i]) / 255.0,
            edgecolor="black",
            label=f"{CLASS_NAMES[i]}",
        )
        for i in range(NUM_CLASSES)
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=NUM_CLASSES,
        bbox_to_anchor=(0.5, -0.01),
        fontsize=12,
        frameon=True,
    )

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved qualitative comparison to {save_path}")

    return fig


def plot_boundary_and_sharpness_analysis(
    fcn_metrics: Dict[str, float],
    unet_metrics: Dict[str, float],
    test_images: np.ndarray,
    test_masks: np.ndarray,
    fcn_probs: np.ndarray,
    unet_probs: np.ndarray,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Comprehensive quantitative and edge-sharpness profile diagnostics.
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    # Subplot 1: Overall IoU vs Boundary IoU
    ax1 = fig.add_subplot(gs[0, 0])
    categories = ["Overall mIoU", "Boundary IoU\n(Edge Precision)", "Pixel Accuracy"]
    fcn_vals = [fcn_metrics["mean_iou"], fcn_metrics["boundary_iou"], fcn_metrics["pixel_accuracy"]]
    unet_vals = [unet_metrics["mean_iou"], unet_metrics["boundary_iou"], unet_metrics["pixel_accuracy"]]

    x = np.arange(len(categories))
    width = 0.35

    rects1 = ax1.bar(x - width / 2, fcn_vals, width, label="Plain FCN (No Skips)", color="#e74c3c", alpha=0.85, edgecolor="black")
    rects2 = ax1.bar(x + width / 2, unet_vals, width, label="U-Net (With Skips)", color="#2ecc71", alpha=0.85, edgecolor="black")

    ax1.set_ylabel("Score (0.0 to 1.0)", fontsize=12, fontweight="bold")
    ax1.set_title("Overall vs Boundary-Specific Performance", fontsize=13, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=11, fontweight="bold")
    ax1.set_ylim(0.0, 1.08)
    ax1.legend(fontsize=11, loc="lower right")
    ax1.grid(axis="y", linestyle="--", alpha=0.7)

    for rect in rects1 + rects2:
        h = rect.get_height()
        ax1.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Subplot 2: Per-Class IoU Breakdown
    ax2 = fig.add_subplot(gs[0, 1])
    classes = [CLASS_NAMES[c] for c in range(NUM_CLASSES)]
    fcn_class_ious = [fcn_metrics[c] for c in classes]
    unet_class_ious = [unet_metrics[c] for c in classes]

    xc = np.arange(len(classes))
    r1 = ax2.bar(xc - width / 2, fcn_class_ious, width, label="Plain FCN", color="#e74c3c", alpha=0.85, edgecolor="black")
    r2 = ax2.bar(xc + width / 2, unet_class_ious, width, label="U-Net", color="#2ecc71", alpha=0.85, edgecolor="black")

    ax2.set_ylabel("IoU Score", fontsize=12, fontweight="bold")
    ax2.set_title("Per-Class IoU Comparison", fontsize=13, fontweight="bold")
    ax2.set_xticks(xc)
    ax2.set_xticklabels(classes, fontsize=11, fontweight="bold")
    ax2.set_ylim(0.0, 1.08)
    ax2.legend(fontsize=11, loc="lower right")
    ax2.grid(axis="y", linestyle="--", alpha=0.7)

    for rect in r1 + r2:
        h = rect.get_height()
        ax2.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Subplot 3: 1D Cross-Section Probability Profile Across Edge
    ax3 = fig.add_subplot(gs[1, 0])
    # Find a sample with a sharp shape boundary along a horizontal slice
    sample_idx = 0
    slice_y = 64  # Mid-line horizontal cross-section
    for s in range(len(test_masks)):
        diffs = np.abs(np.diff(test_masks[s][slice_y, :]))
        if np.sum(diffs > 0) >= 2:
            sample_idx = s
            break

    # Ground truth class profile along the row
    gt_row = test_masks[sample_idx, slice_y, :]
    # Let's find the dominant non-background class along this row
    non_bg_classes = gt_row[gt_row > 0]
    target_class = non_bg_classes[0] if len(non_bg_classes) > 0 else 1

    gt_target_binary = (gt_row == target_class).astype(float)
    fcn_prob_row = fcn_probs[sample_idx, slice_y, :, target_class]
    unet_prob_row = unet_probs[sample_idx, slice_y, :, target_class]

    pixel_coords = np.arange(len(gt_row))
    ax3.plot(pixel_coords, gt_target_binary, "k-", linewidth=2.5, label=f"Ground Truth ({CLASS_NAMES[target_class]})")
    ax3.plot(pixel_coords, fcn_prob_row, color="#e74c3c", linestyle="--", linewidth=2.0, label="Plain FCN Prob (Smooth / Blurry)")
    ax3.plot(pixel_coords, unet_prob_row, color="#2ecc71", linestyle="-", linewidth=2.0, label="U-Net Prob (Crisp / Step-like)")

    ax3.set_xlabel("Pixel Coordinate along Horizontal Scanline (y=64)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Predicted Class Probability", fontsize=11, fontweight="bold")
    ax3.set_title(f"1D Edge Transition Profile (Sharpness Analysis - Sample #{sample_idx + 1})", fontsize=13, fontweight="bold")
    ax3.set_ylim(-0.05, 1.1)
    ax3.legend(fontsize=10, loc="upper right")
    ax3.grid(True, linestyle="--", alpha=0.7)

    # Subplot 4: Corner & Edge Zoom-in Inspection
    ax4 = fig.add_subplot(gs[1, 1])
    # Show zoom-in error comparison
    sample_img = test_images[sample_idx]
    gt_mask = test_masks[sample_idx]
    fcn_pred = np.argmax(fcn_probs[sample_idx], axis=-1)
    unet_pred = np.argmax(unet_probs[sample_idx], axis=-1)

    # Find bounding box around first shape to zoom in
    coords = np.argwhere(gt_mask > 0)
    if len(coords) > 0:
        ymin, xmin = coords.min(axis=0)
        ymax, xmax = coords.max(axis=0)
        cy, cx = (ymin + ymax) // 2, (xmin + xmax) // 2
        crop_r = 24
        y0, y1 = max(0, cy - crop_r), min(128, cy + crop_r)
        x0, x1 = max(0, cx - crop_r), min(128, cx + crop_r)
    else:
        y0, y1, x0, x1 = 32, 96, 32, 96

    # Create composite zoomed comparison image
    cmap, norm = get_segmentation_cmap()
    crop_gt = gt_mask[y0:y1, x0:x1]
    crop_fcn = fcn_pred[y0:y1, x0:x1]
    crop_unet = unet_pred[y0:y1, x0:x1]

    # Combine 3 crops side by side
    composite_crop = np.hstack([crop_gt, crop_fcn, crop_unet])
    ax4.imshow(composite_crop, cmap=cmap, norm=norm)
    ax4.set_title("Zoom-in Boundary Contour: [Ground Truth | Plain FCN | U-Net]", fontsize=12, fontweight="bold")
    ax4.axvline(x=crop_r * 2, color="white", linewidth=2)
    ax4.axvline(x=crop_r * 4, color="white", linewidth=2)
    ax4.set_xticks([crop_r, crop_r * 3, crop_r * 5])
    ax4.set_xticklabels(["Ground Truth", "Plain FCN (Rounded)", "U-Net (Crisp)"], fontsize=10, fontweight="bold")
    ax4.set_yticks([])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved boundary & sharpness analysis to {save_path}")

    return fig


def plot_training_dynamics(
    fcn_history: Dict[str, List[float]],
    unet_history: Dict[str, List[float]],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot training and validation loss and accuracy curves for both models.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    epochs = range(1, len(fcn_history["loss"]) + 1)

    # 1. Loss Curves
    axes[0].plot(epochs, fcn_history["loss"], "r--", label="Plain FCN (Train)", linewidth=2.0)
    axes[0].plot(epochs, fcn_history["val_loss"], "r-", label="Plain FCN (Val)", linewidth=2.5)
    axes[0].plot(epochs, unet_history["loss"], "g--", label="U-Net (Train)", linewidth=2.0)
    axes[0].plot(epochs, unet_history["val_loss"], "g-", label="U-Net (Val)", linewidth=2.5)

    axes[0].set_title("Cross-Entropy Loss vs Epochs", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Sparse Categorical Cross-Entropy", fontsize=11, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.7)
    axes[0].legend(fontsize=10)

    # 2. Accuracy Curves
    axes[1].plot(epochs, fcn_history["accuracy"], "r--", label="Plain FCN (Train)", linewidth=2.0)
    axes[1].plot(epochs, fcn_history["val_accuracy"], "r-", label="Plain FCN (Val)", linewidth=2.5)
    axes[1].plot(epochs, unet_history["accuracy"], "g--", label="U-Net (Train)", linewidth=2.0)
    axes[1].plot(epochs, unet_history["val_accuracy"], "g-", label="U-Net (Val)", linewidth=2.5)

    axes[1].set_title("Pixel Accuracy vs Epochs", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Pixel Accuracy", fontsize=11, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.7)
    axes[1].legend(fontsize=10)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved training dynamics to {save_path}")

    return fig


def plot_architecture_diagram(save_path: Optional[Path] = None) -> plt.Figure:
    """
    Generate an educational architectural comparison diagram.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    def draw_model_diagram(ax, title, has_skips):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

        # Draw blocks
        # Encoder (Left column: x = 2)
        # Stage 1: y = 8.5
        ax.add_patch(patches.FancyBboxPatch((0.8, 7.8), 2.4, 1.0, boxstyle="round,pad=0.1", fc="#3498db", ec="black", lw=1.5))
        ax.text(2.0, 8.3, "Encoder Stage 1\n(128x128, 32 ch)", ha="center", va="center", color="white", fontweight="bold", fontsize=9)

        # Stage 2: y = 5.8
        ax.add_patch(patches.FancyBboxPatch((1.0, 5.2), 2.0, 1.0, boxstyle="round,pad=0.1", fc="#2980b9", ec="black", lw=1.5))
        ax.text(2.0, 5.7, "Encoder Stage 2\n(64x64, 64 ch)", ha="center", va="center", color="white", fontweight="bold", fontsize=9)

        # Stage 3: y = 3.2
        ax.add_patch(patches.FancyBboxPatch((1.2, 2.6), 1.6, 1.0, boxstyle="round,pad=0.1", fc="#1f618d", ec="black", lw=1.5))
        ax.text(2.0, 3.1, "Encoder Stage 3\n(32x32, 128 ch)", ha="center", va="center", color="white", fontweight="bold", fontsize=9)

        # Downsample arrows
        ax.annotate("", xy=(2.0, 6.3), xytext=(2.0, 7.7), arrowprops=dict(facecolor="#e74c3c", shrink=0.05, width=2, headwidth=7))
        ax.text(2.6, 7.0, "MaxPool 2x", fontsize=8, color="#c0392b", fontweight="bold")
        ax.annotate("", xy=(2.0, 3.7), xytext=(2.0, 5.1), arrowprops=dict(facecolor="#e74c3c", shrink=0.05, width=2, headwidth=7))
        ax.text(2.6, 4.4, "MaxPool 2x", fontsize=8, color="#c0392b", fontweight="bold")

        # Bottleneck (Bottom center: x = 5, y = 0.8)
        ax.add_patch(patches.FancyBboxPatch((3.8, 0.4), 2.4, 1.0, boxstyle="round,pad=0.1", fc="#8e44ad", ec="black", lw=1.5))
        ax.text(5.0, 0.9, "Bottleneck\n(16x16, 256 ch)", ha="center", va="center", color="white", fontweight="bold", fontsize=9)

        ax.annotate("", xy=(4.0, 1.4), xytext=(2.2, 2.5), arrowprops=dict(facecolor="#e74c3c", shrink=0.05, width=2, headwidth=7))
        ax.annotate("", xy=(7.8, 2.5), xytext=(6.0, 1.4), arrowprops=dict(facecolor="#27ae60", shrink=0.05, width=2, headwidth=7))

        # Decoder (Right column: x = 8)
        # Stage 3 Up: y = 3.2
        ax.add_patch(patches.FancyBboxPatch((7.2, 2.6), 1.6, 1.0, boxstyle="round,pad=0.1", fc="#27ae60", ec="black", lw=1.5))
        ax.text(8.0, 3.1, "Decoder Stage 3\n(32x32, 128 ch)", ha="center", va="center", color="white", fontweight="bold", fontsize=9)

        # Stage 2 Up: y = 5.8
        ax.add_patch(patches.FancyBboxPatch((7.0, 5.2), 2.0, 1.0, boxstyle="round,pad=0.1", fc="#2ecc71", ec="black", lw=1.5))
        ax.text(8.0, 5.7, "Decoder Stage 2\n(64x64, 64 ch)", ha="center", va="center", color="white", fontweight="bold", fontsize=9)

        # Stage 1 Up: y = 8.5
        ax.add_patch(patches.FancyBboxPatch((6.8, 7.8), 2.4, 1.0, boxstyle="round,pad=0.1", fc="#a9dfbf", ec="black", lw=1.5))
        ax.text(8.0, 8.3, "Decoder Stage 1\n(128x128, 32 ch)", ha="center", va="center", color="black", fontweight="bold", fontsize=9)

        # Upsample arrows
        ax.annotate("", xy=(8.0, 5.1), xytext=(8.0, 3.7), arrowprops=dict(facecolor="#27ae60", shrink=0.05, width=2, headwidth=7))
        ax.text(8.6, 4.4, "UpSample 2x", fontsize=8, color="#27ae60", fontweight="bold")
        ax.annotate("", xy=(8.0, 7.7), xytext=(8.0, 6.3), arrowprops=dict(facecolor="#27ae60", shrink=0.05, width=2, headwidth=7))
        ax.text(8.6, 7.0, "UpSample 2x", fontsize=8, color="#27ae60", fontweight="bold")

        if has_skips:
            # Skip Connections (Horizontal arrows)
            ax.annotate("", xy=(7.1, 3.1), xytext=(2.9, 3.1),
                        arrowprops=dict(arrowstyle="->", color="#f39c12", lw=3.0, linestyle="dashed"))
            ax.text(5.0, 3.4, "Skip Connection: Spatial Details", ha="center", color="#d35400", fontweight="bold", fontsize=9)

            ax.annotate("", xy=(6.9, 5.7), xytext=(3.1, 5.7),
                        arrowprops=dict(arrowstyle="->", color="#f39c12", lw=3.0, linestyle="dashed"))
            ax.text(5.0, 6.0, "Skip Connection: Mid-Level Features", ha="center", color="#d35400", fontweight="bold", fontsize=9)

            ax.annotate("", xy=(6.7, 8.3), xytext=(3.3, 8.3),
                        arrowprops=dict(arrowstyle="->", color="#f39c12", lw=3.0, linestyle="dashed"))
            ax.text(5.0, 8.6, "Skip Connection: High-Res Edges", ha="center", color="#d35400", fontweight="bold", fontsize=9)
        else:
            # Show information bottleneck barrier
            ax.text(5.0, 5.5, "NO Skip Connections!\nAll spatial detail must pass\nthrough 16x16 bottleneck.\n\n-> Result: Blurry / Rounded Edges",
                    ha="center", va="center", color="#c0392b", fontweight="bold", fontsize=10,
                    bbox=dict(boxstyle="square,pad=0.6", facecolor="#fcedeb", edgecolor="#e74c3c", lw=1.5))

    draw_model_diagram(ax1, "Plain FCN (No Skip Connections)\n[Information Bottleneck]", has_skips=False)
    draw_model_diagram(ax2, "U-Net (With Skip Connections)\n[Direct Spatial Detail Highways]", has_skips=True)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved architecture diagram to {save_path}")

    return fig
