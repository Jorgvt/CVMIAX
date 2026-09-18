"""
Visualization Utilities for Two-Stage Object Detection (R-CNN).

Provides publication-quality pedagogical figures for:
1. Synthetic dataset samples with GT bounding boxes.
2. Candidate region proposals and IoU matching heatmap.
3. Warped ROI patch grid fed into Stage 2 CNN.
4. Bounding box delta regression refinement (Before vs. After).
5. Non-Maximum Suppression (NMS) in action.
6. Master end-to-end R-CNN pipeline progression figure.
"""

from typing import Dict, List, Optional, Tuple
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

try:
    from .dataset import CLASS_NAMES, ID_TO_CLASS
except ImportError:
    from dataset import CLASS_NAMES, ID_TO_CLASS

# Styling palette
PALETTE = {
    "background": "#95a5a6",
    "circle": "#e74c3c",       # Red
    "rectangle": "#3498db",    # Blue
    "triangle": "#2ecc71",     # Green
    "gt": "#2c3e50",           # Dark slate for GT
    "proposal": "#f39c12",     # Orange for raw proposal
}


def _add_box(
    ax: plt.Axes,
    box: np.ndarray,
    color: str = "red",
    linestyle: str = "-",
    linewidth: float = 2.0,
    label: Optional[str] = None,
    alpha: float = 1.0,
):
    """Helper to draw a bounding box [x1, y1, x2, y2] with optional badge."""
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    rect = patches.Rectangle(
        (x1, y1),
        w,
        h,
        linewidth=linewidth,
        edgecolor=color,
        facecolor="none",
        linestyle=linestyle,
        alpha=alpha,
    )
    ax.add_patch(rect)

    if label:
        ax.text(
            x1,
            max(0, y1 - 3),
            label,
            color="white",
            fontsize=8,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.85, edgecolor="none"),
        )


def visualize_dataset_samples(
    images: np.ndarray,
    annotations: List[Dict[str, np.ndarray]],
    num_samples: int = 6,
    save_path: Optional[str] = None,
):
    """Display a grid of synthetic images with ground truth annotations."""
    cols = 3
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes = np.array(axes).flatten()

    for idx in range(num_samples):
        ax = axes[idx]
        img = images[idx]
        boxes = annotations[idx]["boxes"]
        classes = annotations[idx]["classes"]

        ax.imshow(img)
        ax.set_title(f"Sample #{idx + 1} ({len(boxes)} Objects)", fontsize=11, fontweight="bold")
        ax.axis("off")

        for box, cls_id in zip(boxes, classes):
            cls_name = ID_TO_CLASS[cls_id]
            color = PALETTE.get(cls_name, "magenta")
            _add_box(ax, box, color=color, label=cls_name)

    for j in range(num_samples, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def visualize_proposals_and_iou(
    image: np.ndarray,
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
    proposals: np.ndarray,
    matched_labels: np.ndarray,
    max_ious: np.ndarray,
    save_path: Optional[str] = None,
):
    """
    Visualize Stage 1 proposals and IoU matching:
    - Panel 1: Ground Truth boxes.
    - Panel 2: Candidate proposals color-coded by Positive vs Background.
    - Panel 3: IoU distribution of proposals.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Ground Truth
    ax1 = axes[0]
    ax1.imshow(image)
    ax1.set_title("1. Ground Truth Annotations", fontsize=12, fontweight="bold")
    ax1.axis("off")
    for box, cls_id in zip(gt_boxes, gt_classes):
        cls_name = ID_TO_CLASS[cls_id]
        _add_box(ax1, box, color=PALETTE.get(cls_name, "red"), label=cls_name, linewidth=2.5)

    # Panel 2: Proposals categorized
    ax2 = axes[1]
    ax2.imshow(image)
    ax2.set_title("2. Region Proposals & IoU Categorization", fontsize=12, fontweight="bold")
    ax2.axis("off")

    # Plot sample of background proposals (faint blue)
    bg_idx = np.where(matched_labels == 0)[0]
    for idx in bg_idx[::8]:  # Subsample for visual clarity
        _add_box(ax2, proposals[idx], color="#3498db", linestyle=":", linewidth=0.8, alpha=0.4)

    # Plot ambiguous proposals (yellow)
    ignore_idx = np.where(matched_labels == -1)[0]
    for idx in ignore_idx[::2]:
        _add_box(ax2, proposals[idx], color="#f39c12", linestyle="--", linewidth=1.0, alpha=0.6)

    # Plot positive proposals (green)
    pos_idx = np.where(matched_labels > 0)[0]
    for idx in pos_idx:
        _add_box(
            ax2,
            proposals[idx],
            color="#2ecc71",
            linestyle="-",
            linewidth=2.0,
            label=f"IoU={max_ious[idx]:.2f}",
        )

    # Custom legend for Panel 2
    legend_elements = [
        patches.Patch(edgecolor="#2ecc71", facecolor="none", linewidth=2, label="Positive (IoU >= 0.5)"),
        patches.Patch(edgecolor="#f39c12", facecolor="none", linestyle="--", linewidth=1, label="Ambiguous (0.2<=IoU<0.5)"),
        patches.Patch(edgecolor="#3498db", facecolor="none", linestyle=":", linewidth=1, label="Background (IoU < 0.2)"),
    ]
    ax2.legend(handles=legend_elements, loc="lower right", fontsize=8)

    # Panel 3: IoU Histogram
    ax3 = axes[2]
    ax3.hist(max_ious, bins=25, color="#8e44ad", edgecolor="black", alpha=0.7)
    ax3.axvline(0.5, color="#2ecc71", linestyle="--", linewidth=2, label="Positive Threshold (0.5)")
    ax3.axvline(0.2, color="#e74c3c", linestyle="--", linewidth=2, label="Negative Threshold (0.2)")
    ax3.set_title("3. Proposal IoU Distribution", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Max IoU with Ground Truth", fontsize=10)
    ax3.set_ylabel("Proposal Count", fontsize=10)
    ax3.legend(loc="upper right", fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def visualize_crops_grid(
    crops: np.ndarray,
    labels: np.ndarray,
    num_crops: int = 16,
    save_path: Optional[str] = None,
):
    """Grid showing canonical 32x32 warped patches extracted from proposals."""
    cols = 8
    rows = (num_crops + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 2 * rows))
    axes = np.array(axes).flatten()

    for i in range(num_crops):
        ax = axes[i]
        if i < len(crops):
            ax.imshow(crops[i])
            label_name = ID_TO_CLASS[labels[i]]
            color = PALETTE.get(label_name, "black")
            ax.set_title(label_name, color=color, fontsize=9, fontweight="bold")
        ax.axis("off")

    for j in range(num_crops, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Warped ROI Crops (32x32 Canonical Inputs to Stage 2 CNN)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def visualize_complete_pipeline(
    image: np.ndarray,
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
    proposals: np.ndarray,
    raw_boxes: np.ndarray,
    raw_scores: np.ndarray,
    raw_classes: np.ndarray,
    nms_boxes: np.ndarray,
    nms_scores: np.ndarray,
    nms_classes: np.ndarray,
    save_path: Optional[str] = None,
):
    """
    Pedagogical 4-panel figure illustrating the complete R-CNN execution flow:
    Stage 1 Proposals -> ROI Feature Extraction -> BBox Delta Refinement -> NMS Filtering.
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Step 1: Input & GT
    ax1 = axes[0]
    ax1.imshow(image)
    ax1.set_title("Step 1: Ground Truth Objects", fontsize=11, fontweight="bold")
    ax1.axis("off")
    for box, cls_id in zip(gt_boxes, gt_classes):
        cls_name = ID_TO_CLASS[cls_id]
        _add_box(ax1, box, color=PALETTE.get(cls_name, "red"), label=cls_name, linewidth=2.5)

    # Step 2: Stage 1 Candidate Proposals
    ax2 = axes[1]
    ax2.imshow(image)
    ax2.set_title(f"Step 2: Region Proposer ({len(proposals)} Proposals)", fontsize=11, fontweight="bold")
    ax2.axis("off")
    for box in proposals[::4]:  # Show representative subset
        _add_box(ax2, box, color="#f39c12", linestyle=":", linewidth=1.0, alpha=0.5)

    # Step 3: CNN Predictions & Delta Refinement (Before NMS)
    ax3 = axes[2]
    ax3.imshow(image)
    ax3.set_title(f"Step 3: CNN Refinement ({len(raw_boxes)} Candidates)", fontsize=11, fontweight="bold")
    ax3.axis("off")
    for box, score, cls_id in zip(raw_boxes, raw_scores, raw_classes):
        cls_name = ID_TO_CLASS[cls_id]
        _add_box(
            ax3,
            box,
            color=PALETTE.get(cls_name, "blue"),
            linestyle="--",
            linewidth=1.5,
            label=f"{cls_name} {score:.2f}",
        )

    # Step 4: Final Detections after NMS
    ax4 = axes[3]
    ax4.imshow(image)
    ax4.set_title(f"Step 4: After NMS ({len(nms_boxes)} Detections)", fontsize=11, fontweight="bold")
    ax4.axis("off")
    for box, score, cls_id in zip(nms_boxes, nms_scores, nms_classes):
        cls_name = ID_TO_CLASS[cls_id]
        _add_box(
            ax4,
            box,
            color=PALETTE.get(cls_name, "green"),
            linestyle="-",
            linewidth=2.5,
            label=f"{cls_name} ({score:.2f})",
        )

    plt.suptitle("The Two-Stage R-CNN Object Detection Pipeline from Scratch", fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
