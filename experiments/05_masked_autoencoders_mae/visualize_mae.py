"""
Visualizer for Masked Autoencoders (MAE).

Generates pedagogical slide-ready figures illustrating:
1. Patch Partitioning, High-Ratio Random Masking (75%), and Information Flow.
2. Comparison of Masking Ratios (25%, 50%, 75%, 90%) & The Semantic Threshold.
3. Information Routing: What the Encoder vs Decoder receives.
4. Master Lecture Slide Overview Poster.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image, ImageDraw

from mae_utils import patchify, unpatchify, random_masking, create_masked_view_image


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
    """Generates a rich, high-contrast natural scene image for patch visualization."""
    img = Image.new("RGB", (320, 320), color=(135, 206, 235))  # Sky blue
    draw = ImageDraw.Draw(img)

    # Sun
    draw.ellipse([230, 30, 290, 90], fill=(255, 215, 0))

    # Hills
    draw.ellipse([-60, 200, 380, 480], fill=(46, 139, 87))  # Grass

    # Tree
    draw.rectangle([50, 140, 70, 250], fill=(139, 69, 19))
    draw.ellipse([20, 75, 100, 160], fill=(34, 139, 34))

    # Dog body (Brown)
    draw.ellipse([120, 160, 240, 250], fill=(184, 115, 51))
    # Head
    draw.ellipse([200, 120, 270, 190], fill=(184, 115, 51))
    # Snout
    draw.ellipse([245, 145, 285, 185], fill=(222, 160, 110))
    # Eye & Nose
    draw.ellipse([235, 140, 245, 150], fill=(20, 20, 20))
    draw.ellipse([275, 155, 285, 165], fill=(20, 20, 20))
    # Ear
    draw.polygon([(210, 110), (190, 160), (220, 135)], fill=(120, 60, 20))
    # Legs
    draw.rectangle([140, 230, 160, 290], fill=(150, 85, 35))
    draw.rectangle([200, 230, 220, 290], fill=(150, 85, 35))
    # Tail
    draw.line([(125, 180), (90, 150)], fill=(150, 85, 35), width=8)

    return np.array(img).astype(np.float32) / 255.0


# =========================================================================
# 1. Masking Mechanism & Asymmetric Information Flow
# =========================================================================

def plot_masking_mechanism(output_dir):
    img = create_sample_natural_image()
    patch_size = 32  # 10x10 = 100 patches
    h, w, _ = img.shape
    num_patches_side = h // patch_size
    num_patches = num_patches_side * num_patches_side

    # Generate 75% random mask
    img_batch = np.expand_dims(img, axis=0)
    patches_arr = patchify(img_batch, patch_size=patch_size)
    vis_patches, mask, restore_idx, shuffle_idx = random_masking(patches_arr, mask_ratio=0.75, seed=42)
    masked_img, mask_vec = create_masked_view_image(img, mask_ratio=0.75, patch_size=patch_size, mask_color=(0.12, 0.12, 0.15))

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    # Panel 1: Original Image with Patch Grid
    axes[0].imshow(img)
    for r in range(num_patches_side):
        for c in range(num_patches_side):
            rect = patches.Rectangle(
                (c * patch_size, r * patch_size), patch_size, patch_size,
                linewidth=1, edgecolor="white", alpha=0.6, facecolor="none"
            )
            axes[0].add_patch(rect)
    axes[0].set_title(f"1. Original Image\n(Grid: {num_patches_side}x{num_patches_side} = {num_patches} Patches)", fontsize=11)
    axes[0].axis("off")

    # Panel 2: 75% Masked Image (What is visible vs masked)
    axes[1].imshow(masked_img)
    # Highlight visible patches in green
    num_vis = 0
    for r in range(num_patches_side):
        for c in range(num_patches_side):
            idx = r * num_patches_side + c
            if mask_vec[idx] == 0.0:
                rect = patches.Rectangle(
                    (c * patch_size, r * patch_size), patch_size, patch_size,
                    linewidth=2, edgecolor="#00ff00", facecolor="none"
                )
                axes[1].add_patch(rect)
                num_vis += 1
    axes[1].set_title(f"2. Random Masking (75% Masked)\n(Only 25% Visible = {num_vis} Patches ✓)", fontsize=11, color="navy")
    axes[1].axis("off")

    # Panel 3: Asymmetric Information Flow Diagram
    axes[2].text(
        0.5, 0.90, "Asymmetric Information Flow",
        fontsize=12, fontweight="bold", ha="center", va="center", color="black"
    )
    axes[2].text(
        0.5, 0.48,
        "• Encoder Input:\n"
        "  ONLY the 25% unmasked patches\n"
        "  (Saves ~75% compute & memory!)\n\n"
        "• Latent Representation:\n"
        "  25 encoded tokens from heavy ViT\n\n"
        "• Decoder Input:\n"
        "  [25 Encoded Tokens] +\n"
        "  [75 Shared Learnable [MASK] Tokens] +\n"
        "  [Full 100 2D Positional Embeddings]",
        fontsize=10.5,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#e8f4fd", edgecolor="#2196f3", linewidth=1.5),
    )
    axes[2].axis("off")

    # Panel 4: Expected Reconstruction
    # Simulate a realistic MAE inpainting reconstruction
    simulated_recon = img.copy()
    # Add slight blur/smooth interpolation in masked areas to represent model inpainting
    for r in range(num_patches_side):
        for c in range(num_patches_side):
            idx = r * num_patches_side + c
            if mask_vec[idx] == 1.0:
                # Reconstructed patch
                patch_area = simulated_recon[r*patch_size:(r+1)*patch_size, c*patch_size:(c+1)*patch_size]
                simulated_recon[r*patch_size:(r+1)*patch_size, c*patch_size:(c+1)*patch_size] = np.clip(
                    patch_area * 0.95 + 0.02, 0, 1
                )
    axes[3].imshow(simulated_recon)
    axes[3].set_title("3. Output: Reconstructed Image\n(Predicts missing pixels in masked patches)", fontsize=11, color="green")
    axes[3].axis("off")

    plt.suptitle(
        "Masked Autoencoder (MAE): Masking Mechanism and Asymmetric Information Routing",
        fontsize=15,
        fontweight="bold",
        y=1.03,
    )
    plt.tight_layout()
    save_path = os.path.join(output_dir, "01_mae_patch_masking_mechanism.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# =========================================================================
# 2. Masking Ratio Comparison: Why 75% is the Sweet Spot in Computer Vision
# =========================================================================

def plot_mask_ratios_comparison(output_dir):
    img = create_sample_natural_image()
    patch_size = 32
    ratios = [0.25, 0.50, 0.75, 0.90]

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.8))

    descriptions = [
        ("25% Masked (Too Easy ✗)", "Trivial local pixel interpolation.\nNo high-level semantic reasoning needed.", "#d95f02"),
        ("50% Masked (Moderate)", "Good visual continuity,\npartially redundant information.", "#1f77b4"),
        ("75% Masked (Sweet Spot ✓)", "Eliminates spatial redundancy.\nForces model to learn object semantics & holistic context!", "#2ca02c"),
        ("90% Masked (Too Hard ✗)", "Extreme degradation.\nNot enough context to reconstruct.", "#d62728"),
    ]

    for ax, ratio, (title, desc, col) in zip(axes, ratios, descriptions):
        masked_view, _ = create_masked_view_image(img, mask_ratio=ratio, patch_size=patch_size, mask_color=(0.1, 0.1, 0.12))
        ax.imshow(masked_view)
        ax.set_title(f"{title}\n{int((1-ratio)*100)}% Visible Patches", fontsize=11, color=col, fontweight="bold")
        ax.text(
            160, 310, desc,
            fontsize=9.5, ha="center", va="top", color="black",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff", edgecolor=col, linewidth=1.2)
        )
        ax.axis("off")

    plt.suptitle(
        "Why Masked Autoencoders Use 75% Masking: Images Have High Spatial Redundancy",
        fontsize=15,
        fontweight="bold",
        y=1.03,
    )
    plt.tight_layout()
    save_path = os.path.join(output_dir, "02_mae_reconstruction_vs_mask_ratio.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# =========================================================================
# 3. Information Used by Model: Breakdown of Inputs, Tokens & Embeddings
# =========================================================================

def plot_information_breakdown(output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    # Box 1: What the ENCODER uses
    axes[0].text(
        0.5, 0.88, "1. Information Used by ENCODER",
        fontsize=12, fontweight="bold", ha="center", va="center", color="#1a5276"
    )
    axes[0].text(
        0.5, 0.45,
        "• Input: ONLY Visible Patches (25%)\n"
        "  - Raw pixel patches $x_{\\text{vis}} \\in \\mathbb{R}^{N_{\\text{vis}} \\times (P^2 C)}$\n"
        "  - Projected to latent vector $D_{\\text{enc}}$\n\n"
        "• 2D Positional Embeddings:\n"
        "  - Informs the encoder of each visible\n"
        "    patch's exact spatial coordinates $(x, y)$\n\n"
        "• Note: Masked patches NEVER enter the encoder!\n"
        "  (No wasted computation on masked regions)",
        fontsize=10.5, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#eaf2f8", edgecolor="#2980b9", linewidth=1.8),
    )
    axes[0].axis("off")

    # Box 2: What the DECODER uses
    axes[1].text(
        0.5, 0.88, "2. Information Used by DECODER",
        fontsize=12, fontweight="bold", ha="center", va="center", color="#196f3d"
    )
    axes[1].text(
        0.5, 0.45,
        "• Encoded Visible Tokens:\n"
        "  - High-level semantic context vectors\n\n"
        "• Shared Learnable [MASK] Tokens:\n"
        "  - A single learned parameter vector $\\mathbf{e}_{\\text{mask}}$\n"
        "    inserted at all 75 masked positions\n\n"
        "• Full 2D Positional Embeddings:\n"
        "  - Added to ALL tokens so the decoder knows\n"
        "    which patch coordinates to reconstruct!",
        fontsize=10.5, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#eafaf1", edgecolor="#27ae60", linewidth=1.8),
    )
    axes[1].axis("off")

    # Box 3: What the LOSS uses
    axes[2].text(
        0.5, 0.88, "3. Information Used by LOSS",
        fontsize=12, fontweight="bold", ha="center", va="center", color="#7d6608"
    )
    axes[2].text(
        0.5, 0.45,
        "• Mean Squared Error (MSE):\n"
        "  $\\mathcal{L}_{\\text{MAE}} = \\frac{1}{N_{\\text{masked}}} \\sum_{i \\in \\text{Masked}} \\| x_i - \\hat{x}_i \\|_2^2$\n\n"
        "• Target Information:\n"
        "  - Evaluated ONLY on the missing masked pixels!\n"
        "  - The visible patches are already known, so\n"
        "    the model is penalized exclusively for inpainting.",
        fontsize=10.5, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#fef9e7", edgecolor="#f1c40f", linewidth=1.8),
    )
    axes[2].axis("off")

    plt.suptitle(
        "MAE Architecture: Exact Information Breakdown (Encoder vs Decoder vs Loss)",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    save_path = os.path.join(output_dir, "03_information_used_by_model.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# =========================================================================
# 4. Master Overview Lecture Slide Poster
# =========================================================================

def plot_master_mae_slide(output_dir):
    img = create_sample_natural_image()
    patch_size = 32
    masked_view, mask_vec = create_masked_view_image(img, mask_ratio=0.75, patch_size=patch_size, mask_color=(0.1, 0.1, 0.12))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))

    # 1. Input: 75% Masked View
    axes[0].imshow(masked_view)
    axes[0].set_title("1. Model Input: 75% Masked Image\n(Encoder processes ONLY 25% visible patches)", fontsize=11, color="navy", fontweight="bold")
    axes[0].axis("off")

    # 2. Output: Target Reconstruction
    axes[1].imshow(img)
    axes[1].set_title("2. Model Target: Full Pixel Inpainting\n(Loss computed strictly on masked patches)", fontsize=11, color="green", fontweight="bold")
    axes[1].axis("off")

    # 3. Core Insights Summary Box
    axes[2].text(
        0.5, 0.90, "Masked Autoencoders (MAE): Key Takeaways",
        fontsize=12, fontweight="bold", ha="center", va="center", color="black"
    )
    axes[2].text(
        0.5, 0.48,
        "Why MAE Works for Vision (He et al., 2022):\n\n"
        "1. High Masking Ratio (75% vs BERT's 15%):\n"
        "   Images have extreme spatial redundancy.\n"
        "   75% masking prevents trivial local copying and forces\n"
        "   high-level holistic scene understanding.\n\n"
        "2. Asymmetric ViT Architecture:\n"
        "   • Heavy Encoder processes ONLY visible patches (25%).\n"
        "   • Lightweight Decoder reconstructs full image from\n"
        "     encoded tokens + [MASK] tokens + 2D Positional Embeddings.\n\n"
        "3. Scalability:\n"
        "   Enables training massive Vision Transformers (ViT-Huge/Giant)\n"
        "   up to 3-4x faster with superior representation transfer.",
        fontsize=9.8,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8f9fa", edgecolor="#343a40", linewidth=1.5),
    )
    axes[2].axis("off")

    plt.suptitle(
        "Masked Autoencoder (MAE) for Vision: Masking, Asymmetry, and Reconstruction",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    save_path = os.path.join(output_dir, "slide_mae_overview.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    set_plot_style()
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    os.makedirs(output_dir, exist_ok=True)

    print("Generating Masking Mechanism & Information Flow...")
    plot_masking_mechanism(output_dir)

    print("Generating Masking Ratio Comparison...")
    plot_mask_ratios_comparison(output_dir)

    print("Generating Information Breakdown (Encoder vs Decoder vs Loss)...")
    plot_information_breakdown(output_dir)

    print("Generating Master Lecture Slide Poster...")
    plot_master_mae_slide(output_dir)

    print(f"\n[Success] All MAE visualizations saved cleanly to: {output_dir}")


if __name__ == "__main__":
    main()
