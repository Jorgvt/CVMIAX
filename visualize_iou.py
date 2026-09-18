"""
visualize_iou.py

A standalone, publication-quality visualization suite for Intersection over Union (IoU)
in computer vision and object detection.

Generates 4 pedagogical figures:
1. `iou_01_concept_breakdown.png`: Geometric breakdown (A, B, Intersection, Union, Formula).
2. `iou_02_threshold_progression.png`: Visual spectrum from IoU = 0.00 to 1.00.
3. `iou_03_intuition_edge_cases.png`: Key edge cases (containment, aspect ratio mismatch, shift vs scale).
4. `iou_04_surface_segmentation.png`: 2D arbitrary surface / segmentation mask IoU.

Usage:
    uv run python visualize_iou.py
"""

from pathlib import Path
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


# ---------------------------------------------------------------------------
# Visual Styling Constants
# ---------------------------------------------------------------------------
COLOR_GT = "#2ECC71"       # Emerald Green (Ground Truth)
COLOR_PRED = "#3498DB"     # Blue (Prediction)
COLOR_INTER = "#E74C3C"    # Coral / Red-Orange (Intersection)
COLOR_UNION_BG = "#EAECEE" # Light Neutral Gray (Union Background)
COLOR_ACCENT = "#8E44AD"   # Purple Accent
FONT_FAMILY = "sans-serif"

plt.rcParams.update({
    "font.family": FONT_FAMILY,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.titlesize": 16,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# ---------------------------------------------------------------------------
# Core IoU Math Helpers
# ---------------------------------------------------------------------------
def compute_box_iou(box1: Tuple[float, float, float, float],
                    box2: Tuple[float, float, float, float]) -> Dict[str, Any]:
    """
    Computes exact IoU, intersection box, and area terms for two boxes in (x1, y1, x2, y2) format.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    iou = inter_area / union_area if union_area > 0 else 0.0

    return {
        "iou": iou,
        "inter_box": (x1, y1, x2, y2) if inter_area > 0 else None,
        "inter_area": inter_area,
        "union_area": union_area,
        "area1": area1,
        "area2": area2,
    }


def compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> Dict[str, Any]:
    """
    Computes exact IoU for two 2D binary segmentation masks.
    """
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    inter_count = int(np.sum(intersection))
    union_count = int(np.sum(union))
    iou = inter_count / union_count if union_count > 0 else 0.0

    return {
        "iou": iou,
        "intersection_mask": intersection,
        "union_mask": union,
        "inter_count": inter_count,
        "union_count": union_count,
    }


# ---------------------------------------------------------------------------
# Figure 1: Anatomy & Concept Breakdown
# ---------------------------------------------------------------------------
def plot_concept_breakdown(output_dir: Path) -> Path:
    """
    Creates a 4-panel visual breakdown explaining the mathematical composition of IoU.
    """
    gt_box = (1.5, 2.0, 6.0, 7.0)      # (x1, y1, x2, y2)
    pred_box = (3.5, 3.5, 8.5, 8.5)
    stats = compute_box_iou(gt_box, pred_box)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    titles = [
        "1. Input Bounding Boxes",
        r"2. Intersection Area ($A \cap B$)",
        r"3. Union Area ($A \cup B$)",
        r"4. $\mathrm{IoU} = \frac{|A \cap B|}{|A \cup B|}$"
    ]

    for ax, title in zip(axes, titles):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect("equal")
        ax.set_title(title, fontweight="bold", pad=12)
        ax.grid(True, linestyle=":", alpha=0.5, color="#BDC3C7")
        ax.set_xticks(range(0, 11, 2))
        ax.set_yticks(range(0, 11, 2))

    # Panel 1: Input Boxes
    ax1 = axes[0]
    ax1.add_patch(patches.Rectangle((gt_box[0], gt_box[1]), gt_box[2]-gt_box[0], gt_box[3]-gt_box[1],
                                    linewidth=2.5, edgecolor=COLOR_GT, facecolor=COLOR_GT, alpha=0.3,
                                    label=f"Ground Truth A (Area={stats['area1']:.1f})"))
    ax1.add_patch(patches.Rectangle((pred_box[0], pred_box[1]), pred_box[2]-pred_box[0], pred_box[3]-pred_box[1],
                                    linewidth=2.5, edgecolor=COLOR_PRED, facecolor=COLOR_PRED, alpha=0.3,
                                    label=f"Prediction B (Area={stats['area2']:.1f})"))
    ax1.legend(loc="lower left", framealpha=0.9, fontsize=9)

    # Panel 2: Intersection
    ax2 = axes[1]
    ax2.add_patch(patches.Rectangle((gt_box[0], gt_box[1]), gt_box[2]-gt_box[0], gt_box[3]-gt_box[1],
                                    linewidth=1.5, edgecolor=COLOR_GT, facecolor="none", linestyle="--"))
    ax2.add_patch(patches.Rectangle((pred_box[0], pred_box[1]), pred_box[2]-pred_box[0], pred_box[3]-pred_box[1],
                                    linewidth=1.5, edgecolor=COLOR_PRED, facecolor="none", linestyle="--"))
    ib = stats["inter_box"]
    ax2.add_patch(patches.Rectangle((ib[0], ib[1]), ib[2]-ib[0], ib[3]-ib[1],
                                    linewidth=2.5, edgecolor=COLOR_INTER, facecolor=COLOR_INTER, alpha=0.6,
                                    hatch="//", label=f"Overlap $|A \\cap B| = {stats['inter_area']:.2f}$"))
    ax2.legend(loc="lower left", framealpha=0.9, fontsize=9)

    # Panel 3: Union
    ax3 = axes[2]
    ax3.add_patch(patches.Rectangle((gt_box[0], gt_box[1]), gt_box[2]-gt_box[0], gt_box[3]-gt_box[1],
                                    linewidth=1.5, edgecolor="#7F8C8D", facecolor=COLOR_UNION_BG, alpha=0.9))
    ax3.add_patch(patches.Rectangle((pred_box[0], pred_box[1]), pred_box[2]-pred_box[0], pred_box[3]-pred_box[1],
                                    linewidth=1.5, edgecolor="#7F8C8D", facecolor=COLOR_UNION_BG, alpha=0.9))
    ax3.text(5.0, 5.25, f"Union Area $|A \\cup B|$\n$= {stats['union_area']:.2f}$",
             ha="center", va="center", fontsize=11, fontweight="bold", color="#2C3E50")

    # Panel 4: Synthesis & Ratio Calculation
    ax4 = axes[3]
    # Draw transparent full envelopes
    ax4.add_patch(patches.Rectangle((gt_box[0], gt_box[1]), gt_box[2]-gt_box[0], gt_box[3]-gt_box[1],
                                    linewidth=2, edgecolor=COLOR_GT, facecolor=COLOR_GT, alpha=0.15))
    ax4.add_patch(patches.Rectangle((pred_box[0], pred_box[1]), pred_box[2]-pred_box[0], pred_box[3]-pred_box[1],
                                    linewidth=2, edgecolor=COLOR_PRED, facecolor=COLOR_PRED, alpha=0.15))
    ax4.add_patch(patches.Rectangle((ib[0], ib[1]), ib[2]-ib[0], ib[3]-ib[1],
                                    linewidth=2.5, edgecolor=COLOR_INTER, facecolor=COLOR_INTER, alpha=0.5, hatch="//"))

    calc_text = (
        "Intersection Over Union:\n\n"
        r"$\mathrm{IoU} = \frac{|A \cap B|}{|A \cup B|}$" + "\n\n"
        rf"$\mathrm{{IoU}} = \frac{{{stats['inter_area']:.2f}}}{{{stats['union_area']:.2f}}} = \mathbf{{{stats['iou']:.3f}}}$" + "\n"
        rf"$\rightarrow$ {stats['iou']*100:.1f}% overlap"
    )
    ax4.text(5.0, 1.0, calc_text, ha="center", va="bottom",
             bbox=dict(boxstyle="round,pad=0.6", facecolor="#FDEDEC", edgecolor=COLOR_INTER, alpha=0.95),
             fontsize=10.5, color="#78281F")

    plt.suptitle("Anatomy of Intersection over Union (IoU / Jaccard Index)", fontsize=16, fontweight="bold", y=1.02)
    out_file = output_dir / "iou_01_concept_breakdown.png"
    plt.savefig(out_file)
    plt.close()
    return out_file


# ---------------------------------------------------------------------------
# Figure 2: Standard Threshold Progression (0.00 to 1.00)
# ---------------------------------------------------------------------------
def plot_threshold_progression(output_dir: Path) -> Path:
    """
    Renders a 2x3 grid displaying standard IoU landmark values (0.00, 0.25, 0.50, 0.75, 0.90, 1.00).
    """
    gt = (2.0, 2.0, 7.0, 7.0)  # Reference GT box (5x5 square at (2,2))

    # Carefully curated predictions yielding exact target IoUs
    cases = [
        {"title": "IoU = 0.00 (Disjoint / Miss)",
         "pred": (7.5, 7.5, 9.5, 9.5),
         "desc": "No spatial overlap between target and prediction."},
        {"title": "IoU = 0.25 (Poor Detection)",
         "pred": (5.0, 2.0, 10.0, 7.0),
         "desc": "Marginal overlap. Typically rejected as False Positive."},
        {"title": "IoU = 0.50 (Standard VOC / COCO Threshold)",
         "pred": (2.0 + 5.0/3.0, 2.0, 7.0 + 5.0/3.0, 7.0),
         "desc": "Classic benchmark threshold (mAP@50). Noticeable shift allowed!"},
        {"title": "IoU = 0.75 (Strict Detection Quality)",
         "pred": (2.0 + 5.0/7.0, 2.0, 7.0 + 5.0/7.0, 7.0),
         "desc": "High quality alignment (mAP@75 standard)."},
        {"title": "IoU = 0.90 (Near-Perfect Fit)",
         "pred": (2.0 + 5.0/19.0, 2.0, 7.0 + 5.0/19.0, 7.0),
         "desc": "Extremely tight boundary fit, minimal human-perceptible discrepancy."},
        {"title": "IoU = 1.00 (Exact Match)",
         "pred": (2.0, 2.0, 7.0, 7.0),
         "desc": "Perfect bounding box alignment."},
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, (ax, case) in enumerate(zip(axes, cases)):
        pred = case["pred"]
        stats = compute_box_iou(gt, pred)

        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect("equal")
        ax.grid(True, linestyle=":", alpha=0.4, color="#BDC3C7")
        ax.set_xticks(range(0, 11, 2))
        ax.set_yticks(range(0, 11, 2))

        # Ground Truth
        ax.add_patch(patches.Rectangle((gt[0], gt[1]), gt[2]-gt[0], gt[3]-gt[1],
                                        linewidth=2.5, edgecolor=COLOR_GT, facecolor=COLOR_GT, alpha=0.25,
                                        label="Ground Truth"))

        # Prediction
        ax.add_patch(patches.Rectangle((pred[0], pred[1]), pred[2]-pred[0], pred[3]-pred[1],
                                        linewidth=2.5, edgecolor=COLOR_PRED, facecolor=COLOR_PRED, alpha=0.25,
                                        label="Prediction"))

        # Overlap region
        if stats["inter_box"] is not None:
            ib = stats["inter_box"]
            ax.add_patch(patches.Rectangle((ib[0], ib[1]), ib[2]-ib[0], ib[3]-ib[1],
                                            linewidth=1.8, edgecolor=COLOR_INTER, facecolor=COLOR_INTER, alpha=0.5,
                                            hatch="//", label="Intersection"))

        iou_val = stats["iou"]
        badge_color = "#27AE60" if iou_val >= 0.75 else ("#F39C12" if iou_val >= 0.50 else "#C0392B")

        ax.set_title(case["title"], fontweight="bold", fontsize=11.5, color=badge_color, pad=8)

        # Annotation badge
        score_text = f"IoU: {iou_val:.3f} ({iou_val*100:.1f}%)"
        ax.text(0.5, 9.4, score_text, ha="left", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=badge_color, linewidth=1.5),
                fontweight="bold", fontsize=10, color=badge_color)

        ax.text(5.0, 0.5, case["desc"], ha="center", va="bottom", fontsize=8.5,
                style="italic", color="#5D6D7E", wrap=True)

        if idx == 0:
            ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

    plt.suptitle("IoU Progression Spectrum: Calibrating Visual Intuition", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_file = output_dir / "iou_02_threshold_progression.png"
    plt.savefig(out_file)
    plt.close()
    return out_file


# ---------------------------------------------------------------------------
# Figure 3: Intuition Traps and Edge Cases
# ---------------------------------------------------------------------------
def plot_intuition_edge_cases(output_dir: Path) -> Path:
    """
    Renders pedagogical edge cases where visual intuition often diverges from the IoU metric.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Case 1: Full Containment / Scale Mismatch
    gt1 = (1.5, 1.5, 8.5, 8.5)      # 7x7 box = 49 area
    pred1 = (3.5, 3.5, 6.5, 6.5)    # 3x3 box = 9 area
    stats1 = compute_box_iou(gt1, pred1)

    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.set_aspect("equal")
    ax1.grid(True, linestyle=":", alpha=0.4)
    ax1.set_title("Case A: Total Containment (Scale Mismatch)", fontweight="bold", pad=10)

    ax1.add_patch(patches.Rectangle((gt1[0], gt1[1]), gt1[2]-gt1[0], gt1[3]-gt1[1],
                                    linewidth=2.5, edgecolor=COLOR_GT, facecolor=COLOR_GT, alpha=0.2, label="GT Box (49.0)"))
    ax1.add_patch(patches.Rectangle((pred1[0], pred1[1]), pred1[2]-pred1[0], pred1[3]-pred1[1],
                                    linewidth=2.5, edgecolor=COLOR_PRED, facecolor=COLOR_INTER, alpha=0.6,
                                    hatch="//", label="Prediction (9.0)"))

    badge1 = (
        "100% of Pred is inside GT!\n"
        f"Intersection = {stats1['inter_area']:.1f}\n"
        f"Union = {stats1['union_area']:.1f}\n"
        f"IoU = {stats1['iou']:.2f} (Very low!)"
    )
    ax1.text(5.0, 0.4, badge1, ha="center", va="bottom",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#FDEDEC", edgecolor=COLOR_INTER),
             fontsize=9.5, fontweight="bold", color="#922B21")
    ax1.legend(loc="upper left", fontsize=8.5)

    # Case 2: Aspect Ratio Mismatch (Same Center & Same Area)
    gt2 = (1.0, 4.0, 9.0, 6.0)      # 8 wide x 2 tall = 16 area
    pred2 = (4.0, 1.0, 6.0, 9.0)    # 2 wide x 8 tall = 16 area
    stats2 = compute_box_iou(gt2, pred2)

    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_aspect("equal")
    ax2.grid(True, linestyle=":", alpha=0.4)
    ax2.set_title("Case B: Aspect Ratio Mismatch", fontweight="bold", pad=10)

    ax2.add_patch(patches.Rectangle((gt2[0], gt2[1]), gt2[2]-gt2[0], gt2[3]-gt2[1],
                                    linewidth=2.5, edgecolor=COLOR_GT, facecolor=COLOR_GT, alpha=0.25, label="GT (Wide: 8×2)"))
    ax2.add_patch(patches.Rectangle((pred2[0], pred2[1]), pred2[2]-pred2[0], pred2[3]-pred2[1],
                                    linewidth=2.5, edgecolor=COLOR_PRED, facecolor=COLOR_PRED, alpha=0.25, label="Pred (Tall: 2×8)"))
    ib2 = stats2["inter_box"]
    ax2.add_patch(patches.Rectangle((ib2[0], ib2[1]), ib2[2]-ib2[0], ib2[3]-ib2[1],
                                    linewidth=2.0, edgecolor=COLOR_INTER, facecolor=COLOR_INTER, alpha=0.6,
                                    hatch="//", label=f"Overlap (2×2 = {stats2['inter_area']:.1f})"))

    badge2 = (
        "Same Center (5,5) & Area (16)\n"
        f"Intersection = {stats2['inter_area']:.1f}\n"
        f"Union = {stats2['union_area']:.1f}\n"
        f"IoU = {stats2['iou']:.2f} ({stats2['iou']*100:.1f}%)"
    )
    ax2.text(5.0, 0.4, badge2, ha="center", va="bottom",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#FDEDEC", edgecolor=COLOR_INTER),
             fontsize=9.5, fontweight="bold", color="#922B21")
    ax2.legend(loc="upper right", fontsize=8.5)

    # Case 3: Shift vs Scale (Equal 20% spatial perturbation)
    gt3 = (2.0, 2.0, 7.0, 7.0)      # 5x5 box
    pred3_shift = (3.0, 2.0, 8.0, 7.0)
    pred3_scale = (1.5, 1.5, 7.5, 7.5)

    stats_shift = compute_box_iou(gt3, pred3_shift)
    stats_scale = compute_box_iou(gt3, pred3_scale)

    ax3 = axes[2]
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.set_aspect("equal")
    ax3.grid(True, linestyle=":", alpha=0.4)
    ax3.set_title("Case C: Translation vs. Scale Error", fontweight="bold", pad=10)

    ax3.add_patch(patches.Rectangle((gt3[0], gt3[1]), gt3[2]-gt3[0], gt3[3]-gt3[1],
                                    linewidth=2.5, edgecolor=COLOR_GT, facecolor=COLOR_GT, alpha=0.2, label="GT Box (5×5)"))
    ax3.add_patch(patches.Rectangle((pred3_shift[0], pred3_shift[1]), pred3_shift[2]-pred3_shift[0], pred3_shift[3]-pred3_shift[1],
                                    linewidth=2.0, edgecolor="#E67E22", facecolor="none", linestyle="--",
                                    label=f"20% Shift → IoU = {stats_shift['iou']:.2f}"))
    ax3.add_patch(patches.Rectangle((pred3_scale[0], pred3_scale[1]), pred3_scale[2]-pred3_scale[0], pred3_scale[3]-pred3_scale[1],
                                    linewidth=2.0, edgecolor="#8E44AD", facecolor="none", linestyle="-.",
                                    label=f"20% Zoom → IoU = {stats_scale['iou']:.2f}"))

    badge3 = (
        "Sensitivity Comparison:\n"
        f"• 20% Translation: IoU = {stats_shift['iou']:.2f}\n"
        f"• 20% Scale expansion: IoU = {stats_scale['iou']:.2f}"
    )
    ax3.text(5.0, 0.4, badge3, ha="center", va="bottom",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#EBF5FB", edgecolor="#2980B9"),
             fontsize=9.5, fontweight="bold", color="#1B4F72")
    ax3.legend(loc="upper left", fontsize=8.5)

    plt.suptitle("IoU Intuition Caveats: Scale, Aspect Ratio, and Sensitivity", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_file = output_dir / "iou_03_intuition_edge_cases.png"
    plt.savefig(out_file)
    plt.close()
    return out_file


# ---------------------------------------------------------------------------
# Figure 4: Surface / Semantic Segmentation Masks
# ---------------------------------------------------------------------------
def plot_surface_segmentation(output_dir: Path) -> Path:
    """
    Renders IoU for arbitrary 2D surfaces and binary segmentation masks.
    """
    import matplotlib.colors as mcolors

    grid_size = 200
    y, x = np.ogrid[:grid_size, :grid_size]

    # Ground Truth: Organic blob shape (combination of 2 circles)
    mask_gt = ((x - 85)**2 + (y - 100)**2 <= 45**2) | ((x - 115)**2 + (y - 85)**2 <= 35**2)

    # Prediction: Shifted and slightly deformed organic mask
    mask_pred = ((x - 110)**2 + (y - 115)**2 <= 42**2) | ((x - 130)**2 + (y - 95)**2 <= 30**2)

    stats = compute_mask_iou(mask_gt, mask_pred)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    # Panel 1: Masks Overlay
    ax1 = axes[0]
    rgb_overlay = np.ones((grid_size, grid_size, 3), dtype=np.float32)
    rgb_overlay[mask_gt] = [0.18, 0.80, 0.44]
    rgb_overlay[mask_pred] = [0.20, 0.60, 0.86]
    rgb_overlay[stats["intersection_mask"]] = [0.91, 0.30, 0.24]

    ax1.imshow(rgb_overlay, origin="lower")
    ax1.set_title("1. Ground Truth & Prediction Surfaces", fontweight="bold", pad=10)
    ax1.axis("off")

    patch_gt = patches.Patch(color=COLOR_GT, label=f"GT Surface ({np.sum(mask_gt):,} px)")
    patch_pred = patches.Patch(color=COLOR_PRED, label=f"Pred Surface ({np.sum(mask_pred):,} px)")
    patch_inter = patches.Patch(color=COLOR_INTER, label=f"Intersection ({stats['inter_count']:,} px)")
    ax1.legend(handles=[patch_gt, patch_pred, patch_inter], loc="lower right", fontsize=8.5, framealpha=0.9)

    # Panel 2: Intersection Mask
    ax2 = axes[1]
    cmap_inter = mcolors.ListedColormap(["#F8F9F9", "#E74C3C"])
    ax2.imshow(stats["intersection_mask"], cmap=cmap_inter, origin="lower")
    ax2.set_title(rf"2. Intersection $|A \cap B| = {stats['inter_count']:,}$ px", fontweight="bold", pad=10)
    ax2.axis("off")

    # Panel 3: Union Mask
    ax3 = axes[2]
    cmap_union = mcolors.ListedColormap(["#F8F9F9", "#34495E"])
    ax3.imshow(stats["union_mask"], cmap=cmap_union, origin="lower")
    ax3.set_title(rf"3. Union $|A \cup B| = {stats['union_count']:,}$ px", fontweight="bold", pad=10)
    ax3.axis("off")

    # Panel 4: Pixel IoU Calculation
    ax4 = axes[3]
    ax4.imshow(rgb_overlay, origin="lower", alpha=0.35)
    ax4.axis("off")
    ax4.set_title(r"4. Mask $\mathrm{IoU} = \frac{\sum (A \cap B)}{\sum (A \cup B)}$", fontweight="bold", pad=10)

    calc_text = (
        "Pixel-Level Segmentation IoU:\n\n"
        f"Intersection (TP) = {stats['inter_count']:,} px\n"
        f"Union (TP+FP+FN) = {stats['union_count']:,} px\n\n"
        rf"$\mathrm{{IoU}} = \frac{{{stats['inter_count']}}}{{{stats['union_count']}}} = \mathbf{{{stats['iou']:.4f}}}$" + "\n"
        rf"$\rightarrow$ {stats['iou']*100:.1f}% surface agreement"
    )
    ax4.text(grid_size / 2, grid_size / 2, calc_text, ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.7", facecolor="#FDEDEC", edgecolor=COLOR_INTER, linewidth=1.5),
             fontsize=10.5, color="#78281F")

    plt.suptitle("Generalizing IoU to 2D Surfaces and Semantic Segmentation Masks", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_file = output_dir / "iou_04_surface_segmentation.png"
    plt.savefig(out_file)
    plt.close()
    return out_file


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
def main():
    output_dir = Path("outputs/iou_visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Generating Pedagogical IoU Figures...")
    print("=" * 60)

    f1 = plot_concept_breakdown(output_dir)
    print(f"  [✔] Generated Figure 1: {f1}")

    f2 = plot_threshold_progression(output_dir)
    print(f"  [✔] Generated Figure 2: {f2}")

    f3 = plot_intuition_edge_cases(output_dir)
    print(f"  [✔] Generated Figure 3: {f3}")

    f4 = plot_surface_segmentation(output_dir)
    print(f"  [✔] Generated Figure 4: {f4}")

    print("=" * 60)
    print(f"All figures saved successfully to: {output_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
