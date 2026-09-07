"""
Visualizer for Self-Supervised Learning Pretext Tasks.
Generates publication-quality, pedagogical figures for:
1. Rotation Prediction (Inputs, Rotations, Discrete Class Labels)
2. Jigsaw Puzzle Solving (Grid Partitioning, Permutation Indexing, Shuffled Collage)
3. Colorization (Luminance Grayscale Input, Target Color Generation, Semantic Reasoning)
4. Master Composite Lecture Slide Poster
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from data_generators import (
    create_rotation_batch,
    extract_tiles,
    assemble_jigsaw_image,
    create_jigsaw_sample,
    create_colorization_pair,
    PREDEFINED_3x3_PERMUTATIONS,
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


def create_sample_natural_image():
    """Generates a rich, high-contrast pedagogical image with recognizable semantic parts."""
    img = Image.new("RGB", (300, 300), color=(135, 206, 235))  # Sky blue
    draw = ImageDraw.Draw(img)

    # Sun
    draw.ellipse([220, 25, 275, 80], fill=(255, 215, 0))

    # Hills
    draw.ellipse([-50, 190, 350, 450], fill=(46, 139, 87))  # SeaGreen grass

    # Tree
    draw.rectangle([45, 130, 65, 230], fill=(139, 69, 19))  # Trunk
    draw.ellipse([20, 70, 90, 150], fill=(34, 139, 34))    # Foliage

    # Dog body (Brown)
    draw.ellipse([110, 150, 220, 230], fill=(184, 115, 51))
    # Head
    draw.ellipse([185, 110, 250, 175], fill=(184, 115, 51))
    # Snout
    draw.ellipse([225, 135, 265, 170], fill=(222, 160, 110))
    # Nose
    draw.ellipse([255, 145, 265, 155], fill=(20, 20, 20))
    # Eye
    draw.ellipse([215, 130, 225, 140], fill=(20, 20, 20))
    # Ear
    draw.polygon([(195, 100), (175, 145), (205, 125)], fill=(120, 60, 20))
    # Legs
    draw.rectangle([130, 215, 148, 270], fill=(150, 85, 35))
    draw.rectangle([185, 215, 203, 270], fill=(150, 85, 35))
    # Tail
    draw.line([(115, 170), (85, 140)], fill=(150, 85, 35), width=7)

    return np.array(img)


# =========================================================================
# 1. Rotation Prediction Visualization
# =========================================================================

def visualize_rotation_prediction(output_dir):
    img = create_sample_natural_image()
    batch_img = np.expand_dims(img, axis=0)
    rotated_imgs, labels = create_rotation_batch(batch_img)

    rotation_names = [
        "0° (Upright)",
        "90° (Counter-Clockwise)",
        "180° (Inverted)",
        "270° (Clockwise)",
    ]
    colors = ["#2ca02c", "#1f77b4", "#d62728", "#ff7f0e"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))

    for i in range(4):
        axes[i].imshow(rotated_imgs[i])
        axes[i].set_title(
            f"Input: $X_{{{i}}}$ ({rotation_names[i]})\n"
            f"Target Label: $y = {labels[i]}$ (Class: {i})",
            fontsize=11,
            color=colors[i],
            fontweight="bold",
        )
        axes[i].axis("off")

        # Add a subtle colored border
        rect = patches.Rectangle(
            (0, 0), rotated_imgs[i].shape[1]-1, rotated_imgs[i].shape[0]-1,
            linewidth=3, edgecolor=colors[i], facecolor="none"
        )
        axes[i].add_patch(rect)

    plt.suptitle(
        "Self-Supervised Pretext Task 1: Rotation Prediction (Gidaris et al., 2018)\n"
        "Model must understand canonical object semantics (sky, ground, head, legs) to predict rotation angle",
        fontsize=14,
        fontweight="bold",
        y=1.05,
    )
    plt.tight_layout()
    save_path = os.path.join(output_dir, "01_rotation_prediction_task.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# =========================================================================
# 2. Jigsaw Puzzle Visualization
# =========================================================================

def visualize_jigsaw_puzzle(output_dir):
    img = create_sample_natural_image()

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.5))

    # Step 1: Original Image
    axes[0].imshow(img)
    axes[0].set_title("Step 1: Original Unlabeled Image\n(Natural Image $X$)", fontsize=11)
    axes[0].axis("off")

    # Step 2: Grid Partition with Tile IDs
    grid_img = img.copy()
    h, w, _ = grid_img.shape
    tile_h, tile_w = h // 3, w // 3
    axes[1].imshow(grid_img)
    axes[1].set_title("Step 2: 3x3 Tile Partition\nCanonical Order: (0, 1, 2, ..., 8)", fontsize=11)
    axes[1].axis("off")

    for r in range(3):
        for c in range(3):
            # Draw grid lines
            rect = patches.Rectangle(
                (c * tile_w, r * tile_h), tile_w, tile_h,
                linewidth=2, edgecolor="white", facecolor="none"
            )
            axes[1].add_patch(rect)
            tile_num = r * 3 + c
            axes[1].text(
                c * tile_w + tile_w // 2 - 8,
                r * tile_h + tile_h // 2 + 8,
                str(tile_num),
                color="yellow",
                fontsize=16,
                fontweight="bold",
                bbox=dict(boxstyle="circle,pad=0.2", facecolor="black", alpha=0.7),
            )

    # Step 3: Shuffled Jigsaw Sample (Permutation #4: [6, 3, 0, 7, 4, 1, 8, 5, 2])
    perm_idx = int(4)
    perm = PREDEFINED_3x3_PERMUTATIONS[perm_idx]
    perm_clean_list = [int(x) for x in perm]
    shuffled_comp, _, _ = create_jigsaw_sample(img, permutation_idx=perm_idx, grid_size=3)

    axes[2].imshow(shuffled_comp)
    axes[2].set_title(
        f"Step 3: Shuffled Jigsaw Input\nPermutation $\\pi$: {perm_clean_list}\nTarget Label: Class $y = {perm_idx}$",
        fontsize=11,
        color="navy",
        fontweight="bold",
    )
    axes[2].axis("off")

    # Overlay tile source IDs on the shuffled puzzle
    for i, orig_id in enumerate(perm):
        r, c = i // 3, i % 3
        rect = patches.Rectangle(
            (c * tile_w, r * tile_h), tile_w, tile_h,
            linewidth=2, edgecolor="white", facecolor="none"
        )
        axes[2].add_patch(rect)
        axes[2].text(
            c * tile_w + 10,
            r * tile_h + 25,
            f"Tile {int(orig_id)}",
            color="white",
            fontsize=10,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="blue", alpha=0.8),
        )

    # Step 4: Siamese Multi-Tower Architecture Diagram summary
    axes[3].text(
        0.5, 0.85, "Siamese Jigsaw Network",
        fontsize=12, fontweight="bold", ha="center", va="center", color="black"
    )
    axes[3].text(
        0.5, 0.65,
        "9 Shuffled Tiles $\\{t_1, ..., t_9\\}$\n"
        "       ↓ (Shared ConvNet)\n"
        "9 Feature Vectors $\\{f_1, ..., f_9\\}$\n"
        "       ↓ (Concatenate)\n"
        "Dense Feature Vector $F$\n"
        "       ↓ (Softmax)\n"
        f"Predicted Permutation: $\\hat{{y}} = {perm_idx}$",
        fontsize=10.5,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f0f8ff", edgecolor="#1f77b4", linewidth=2),
    )
    axes[3].axis("off")

    plt.suptitle(
        "Self-Supervised Pretext Task 2: Jigsaw Puzzle Solving (Noroozi & Favaro, 2016)\n"
        "Model must learn spatial context, edge continuity, and object parts to reorder shuffled tiles",
        fontsize=14,
        fontweight="bold",
        y=1.05,
    )
    plt.tight_layout()
    save_path = os.path.join(output_dir, "02_jigsaw_puzzle_task.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# =========================================================================
# 3. Image Colorization Visualization
# =========================================================================

def visualize_colorization(output_dir):
    img = create_sample_natural_image()
    input_gray, target_color = create_colorization_pair(img)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Model Input: Grayscale Luminance
    axes[0].imshow(input_gray[:, :, 0], cmap="gray")
    axes[0].set_title(
        "1. Model Input: Grayscale ($X_{\\text{gray}}$)\n"
        "Luminance Channel $L$ (Shape: $H \\times W \\times 1$)",
        fontsize=11,
        fontweight="bold",
        color="#333333",
    )
    axes[0].axis("off")

    # 2. Model Task / Target: Ground Truth Color
    axes[1].imshow(target_color)
    axes[1].set_title(
        "2. Ground Truth Target ($Y_{\\text{color}}$)\n"
        "RGB / Chrominance $(a, b)$ (Shape: $H \\times W \\times 3$)",
        fontsize=11,
        fontweight="bold",
        color="darkgreen",
    )
    axes[1].axis("off")

    # 3. Objective & Semantic Reasoning Explanation
    axes[2].text(
        0.5, 0.85, "Colorization Objective & Semantics",
        fontsize=12, fontweight="bold", ha="center", va="center", color="black"
    )
    axes[2].text(
        0.5, 0.50,
        "Supervision Loss:\n"
        "$\\mathcal{L} = \\| \\hat{Y}_{\\text{color}} - Y_{\\text{color}} \\|_2^2$\n\n"
        "Why Colorization Learns Semantics:\n"
        "• To color the sky BLUE $\\rightarrow$ detect sky texture\n"
        "• To color the grass GREEN $\\rightarrow$ segment ground\n"
        "• To color the dog BROWN $\\rightarrow$ detect quadruped\n"
        "• To color the sun YELLOW $\\rightarrow$ detect celestial sphere",
        fontsize=10.5,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#fff9e6", edgecolor="#ffbb33", linewidth=2),
    )
    axes[2].axis("off")

    plt.suptitle(
        "Self-Supervised Pretext Task 3: Colorful Image Colorization (Zhang et al., 2016)\n"
        "Predicting color channels from grayscale forces the network to learn rich semantic segmentations",
        fontsize=14,
        fontweight="bold",
        y=1.05,
    )
    plt.tight_layout()
    save_path = os.path.join(output_dir, "03_colorization_task.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# =========================================================================
# 4. Master Overview Slide Poster (All 3 Tasks Side-by-Side)
# =========================================================================

def visualize_master_overview_slide(output_dir):
    img = create_sample_natural_image()
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    # --- Row 1: Rotation Prediction ---
    batch_img = np.expand_dims(img, axis=0)
    rotated_imgs, labels = create_rotation_batch(batch_img)

    axes[0, 0].imshow(rotated_imgs[0])
    axes[0, 0].set_title("Original Image\nTarget $y = 0$ (0°)", fontsize=11)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(rotated_imgs[1])
    axes[0, 1].set_title("Input Transformed: 90° CCW\nTarget Label: $y = 1$ (90°)", fontsize=11, color="navy")
    axes[0, 1].axis("off")

    axes[0, 2].text(
        0.5, 0.5,
        "Pretext Task 1: Rotation (4-Class Classification)\n\n"
        "• Inputs: $\\{X_{0^\\circ}, X_{90^\\circ}, X_{180^\\circ}, X_{270^\\circ}\\}$\n"
        "• Labels: $y \\in \\{0, 1, 2, 3\\}$\n"
        "• Inductive Bias: Orientation requires identifying\n"
        "  canonical upright semantics (gravity, anatomy).",
        fontsize=11, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#e8f4fd", edgecolor="#2196f3", linewidth=1.5),
    )
    axes[0, 2].axis("off")

    # --- Row 2: Jigsaw Puzzle ---
    perm_idx = 4
    perm = PREDEFINED_3x3_PERMUTATIONS[perm_idx]
    shuffled_comp, _, _ = create_jigsaw_sample(img, permutation_idx=perm_idx, grid_size=3)

    axes[1, 0].imshow(img)
    axes[1, 0].set_title("Original Grid (3x3 Tiles)\nOrder: (0, 1, 2, ..., 8)", fontsize=11)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(shuffled_comp)
    axes[1, 1].set_title(f"Shuffled Jigsaw Input\nTarget: Permutation Class $y = {perm_idx}$", fontsize=11, color="navy")
    axes[1, 1].axis("off")

    axes[1, 2].text(
        0.5, 0.5,
        "Pretext Task 2: Jigsaw Puzzles ($K$-Class Classification)\n\n"
        "• Inputs: Shuffled 3x3 grid tiles $\\{t_{\\pi(1)}, ..., t_{\\pi(9)}\\}$\n"
        "• Labels: Permutation index $y = k \\in \\{0, 1, ..., K-1\\}$\n"
        "• Inductive Bias: Requires spatial reasoning and\n"
        "  object part compositionality.",
        fontsize=11, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#eafaf1", edgecolor="#2ecc71", linewidth=1.5),
    )
    axes[1, 2].axis("off")

    # --- Row 3: Colorization ---
    input_gray, target_color = create_colorization_pair(img)

    axes[2, 0].imshow(input_gray[:, :, 0], cmap="gray")
    axes[2, 0].set_title("Input: Grayscale ($X_{\\text{gray}}$)\n1-Channel Luminance", fontsize=11)
    axes[2, 0].axis("off")

    axes[2, 1].imshow(target_color)
    axes[2, 1].set_title("Target: Full Color ($Y_{\\text{RGB}}$)\n3-Channel Color Output", fontsize=11, color="navy")
    axes[2, 1].axis("off")

    axes[2, 2].text(
        0.5, 0.5,
        "Pretext Task 3: Colorization (Pixel-Level Regression)\n\n"
        "• Input: Grayscale Luminance $X_{\\text{gray}} \\in \\mathbb{R}^{H \\times W \\times 1}$\n"
        "• Label: Full Color $Y_{\\text{color}} \\in \\mathbb{R}^{H \\times W \\times 3}$\n"
        "• Inductive Bias: Predicting natural colors requires\n"
        "  recognizing semantic categories from texture alone.",
        fontsize=11, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#fef9e7", edgecolor="#f39c12", linewidth=1.5),
    )
    axes[2, 2].axis("off")

    plt.suptitle(
        "Self-Supervised Learning Pretext Tasks: How Inputs & Labels are Created Automatically",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    save_path = os.path.join(output_dir, "slide_ssl_pretext_tasks_overview.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    set_plot_style()
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    os.makedirs(output_dir, exist_ok=True)

    print("Generating Rotation Prediction figure...")
    visualize_rotation_prediction(output_dir)

    print("Generating Jigsaw Puzzle figure...")
    visualize_jigsaw_puzzle(output_dir)

    print("Generating Colorization figure...")
    visualize_colorization(output_dir)

    print("Generating Master Lecture Slide Poster...")
    visualize_master_overview_slide(output_dir)

    print(f"\n[Success] All Self-Supervised Learning figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
