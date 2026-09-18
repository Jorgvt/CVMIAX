"""Publication-quality visualization scripts for multimodal CLIP contrastive learning experiment."""

import os
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import tensorflow as tf

# Set consistent styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 1.0,
    "grid.color": "#e0e0e0",
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
})


def plot_similarity_matrix_heatmaps(
    sim_before: np.ndarray,
    sim_after: np.ndarray,
    captions: List[str],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot contrastive similarity matrix before vs after training."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

    n = min(12, len(captions))
    labels = [c.replace("a photo of a ", "").replace("an image showing a ", "") for c in captions[:n]]

    # 1. Before training
    im0 = axes[0].imshow(sim_before[:n, :n], cmap="viridis", aspect="auto")
    axes[0].set_title("(a) Cosine Similarity: Initial Untrained Weights", fontsize=13, fontweight="bold", pad=12)
    axes[0].set_xlabel("Text Embeddings ($T_j$)", fontsize=11, labelpad=8)
    axes[0].set_ylabel("Image Embeddings ($I_i$)", fontsize=11, labelpad=8)
    axes[0].set_xticks(range(n))
    axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    axes[0].set_yticks(range(n))
    axes[0].set_yticklabels(labels, fontsize=9)
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # 2. After training
    im1 = axes[1].imshow(sim_after[:n, :n], cmap="magma", aspect="auto")
    axes[1].set_title("(b) Cosine Similarity: After InfoNCE Pre-training", fontsize=13, fontweight="bold", pad=12)
    axes[1].set_xlabel("Text Embeddings ($T_j$)", fontsize=11, labelpad=8)
    axes[1].set_ylabel("Image Embeddings ($I_i$)", fontsize=11, labelpad=8)
    axes[1].set_xticks(range(n))
    axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    axes[1].set_yticks(range(n))
    axes[1].set_yticklabels(labels, fontsize=9)
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Highlight diagonal alignment
    for ax in axes:
        for i in range(n):
            rect = plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="cyan", lw=1.2, ls=":")
            ax.add_patch(rect)

    plt.suptitle("Multimodal Alignment: Cross-Modal Cosine Similarity Matrices ($S = I \\cdot T^\\top$)", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    return fig


def plot_training_dynamics(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot symmetric InfoNCE loss, cross-modal retrieval accuracies, and temperature evolution."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), dpi=300)
    epochs = range(1, len(history["loss"]) + 1)

    # Loss
    axes[0].plot(epochs, history["loss"], "o-", color="#e63946", lw=2, label="Train Loss")
    if "val_loss" in history:
        axes[0].plot(epochs, history["val_loss"], "s--", color="#457b9d", lw=2, label="Val Loss")
    axes[0].set_title("Symmetric InfoNCE Loss", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Cross-Entropy Loss", fontsize=11)
    axes[0].grid(True)
    axes[0].legend(frameon=True)

    # Retrieval Accuracies
    axes[1].plot(epochs, history["i2t_acc"], "o-", color="#2a9d8f", lw=2, label="Image -> Text Acc")
    axes[1].plot(epochs, history["t2i_acc"], "^-.", color="#e76f51", lw=2, label="Text -> Image Acc")
    if "val_i2t_acc" in history:
        axes[1].plot(epochs, history["val_i2t_acc"], "s--", color="#264653", lw=1.5, label="Val I->T Acc")
    axes[1].set_title("Batch Retrieval Accuracies (Top-1)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Accuracy", fontsize=11)
    axes[1].set_ylim([-0.05, 1.05])
    axes[1].grid(True)
    axes[1].legend(frameon=True)

    # Temperature
    if "temperature" in history:
        axes[2].plot(epochs, history["temperature"], "d-", color="#9b5de5", lw=2, label="Scale $\\tau = \\exp(\\text{logit\\_scale})$")
        axes[2].set_title("Learnable Temperature Scale ($\\tau$)", fontsize=12, fontweight="bold")
        axes[2].set_xlabel("Epoch", fontsize=11)
        axes[2].set_ylabel("Logit Multiplier $\\tau$", fontsize=11)
        axes[2].grid(True)
        axes[2].legend(frameon=True)

    plt.suptitle("CLIP Multi-Modal Contrastive Training Dynamics", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    return fig


def plot_zero_shot_showcase(
    test_cases: List[Dict],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot zero-shot classification gallery covering seen, compositional, and novel geometric shapes.
    
    Each test_case dict contains:
    - image: np.ndarray (H, W, 3)
    - title: str category (e.g., 'Seen Pair', 'Compositional Zero-Shot', 'Novel Shape (Star)')
    - true_label: str
    - candidate_prompts: List[str]
    - probabilities: np.ndarray [num_candidates]
    """
    n_cases = len(test_cases)
    fig, axes = plt.subplots(n_cases, 2, figsize=(14, 3.2 * n_cases), dpi=300, gridspec_kw={"width_ratios": [1, 2.5]})

    if n_cases == 1:
        axes = np.expand_dims(axes, 0)

    for i, case in enumerate(test_cases):
        # 1. Display query image
        axes[i, 0].imshow(case["image"])
        axes[i, 0].axis("off")
        axes[i, 0].set_title(f"Query Image\n[{case['category']}]\nTarget: {case['true_label']}", fontsize=10, fontweight="bold", pad=6)

        # 2. Display horizontal probability bar chart
        candidates = [p.replace("a photo of a ", "").replace("a centered ", "") for p in case["candidate_prompts"]]
        probs = case["probabilities"]
        y_pos = np.arange(len(candidates))

        # Color the top-predicted bar
        colors = ["#2a9d8f" if p == np.max(probs) else "#457b9d" for p in probs]
        bars = axes[i, 1].barh(y_pos, probs * 100, color=colors, height=0.6, alpha=0.85, edgecolor="black", lw=0.5)

        axes[i, 1].set_yticks(y_pos)
        axes[i, 1].set_yticklabels(candidates, fontsize=10)
        axes[i, 1].invert_yaxis()  # top-down
        axes[i, 1].set_xlabel("Zero-Shot Softmax Probability (%)", fontsize=10)
        axes[i, 1].set_xlim([0, 105])
        axes[i, 1].grid(axis="x", alpha=0.5)

        # Add text labels on bars
        for bar in bars:
            width = bar.get_width()
            if width > 3:
                axes[i, 1].text(width + 1.5, bar.get_y() + bar.get_height() / 2, f"{width:.1f}%", va="center", ha="left", fontsize=9, fontweight="bold")

    plt.suptitle("Zero-Shot Multi-Modal Classification: Seen vs Compositional vs Novel Geometric Figures", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    return fig


def plot_cross_modal_retrieval(
    query_texts: List[str],
    retrieved_images: List[List[np.ndarray]],
    top_scores: List[List[float]],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot text-to-image retrieval results for multiple natural language queries."""
    num_queries = len(query_texts)
    top_k = len(retrieved_images[0])

    fig, axes = plt.subplots(num_queries, top_k, figsize=(3.0 * top_k, 3.2 * num_queries), dpi=300)
    if num_queries == 1:
        axes = np.expand_dims(axes, 0)

    for q_idx, (q_text, img_list, scores) in enumerate(zip(query_texts, retrieved_images, top_scores)):
        for k_idx, (img, score) in enumerate(zip(img_list, scores)):
            ax = axes[q_idx, k_idx]
            ax.imshow(img)
            ax.axis("off")
            rank_str = f"Top-{k_idx + 1}"
            ax.set_title(f"{rank_str} (Cos: {score:.3f})", fontsize=9, pad=4)

        axes[q_idx, 0].text(
            -0.2, 0.5,
            f'Query:\n"{q_text}"',
            transform=axes[q_idx, 0].transAxes,
            va="center", ha="right",
            fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f1f5f9", edgecolor="#cbd5e1")
        )

    plt.suptitle("Text-to-Image Cross-Modal Retrieval Showcase", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    return fig


def plot_joint_embedding_space(
    img_embeddings: np.ndarray,
    txt_embeddings: np.ndarray,
    labels: List[str],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot 2D PCA projection of shared multimodal embedding space."""
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    # Concatenate and reduce via PCA
    all_embeddings = np.concatenate([img_embeddings, txt_embeddings], axis=0)
    pca = PCA(n_components=2, random_state=42)
    reduced = pca.fit_transform(all_embeddings)

    n = len(img_embeddings)
    img_2d = reduced[:n]
    txt_2d = reduced[n:]

    # Distinct colors per class
    unique_labels = sorted(list(set(labels)))
    cmap = plt.cm.get_cmap("tab10", len(unique_labels))

    for idx, u_lbl in enumerate(unique_labels):
        mask = np.array([l == u_lbl for l in labels])
        c = cmap(idx)

        # Plot image points
        ax.scatter(img_2d[mask, 0], img_2d[mask, 1], color=c, marker="o", s=60, alpha=0.7, label=f"Img: {u_lbl}")
        # Plot text points
        ax.scatter(txt_2d[mask, 0], txt_2d[mask, 1], color=c, marker="^", s=130, edgecolor="black", lw=1.2, label=f"Txt: {u_lbl}")

        # Connect image centroid to text point
        img_center = np.mean(img_2d[mask], axis=0)
        txt_center = np.mean(txt_2d[mask], axis=0)
        ax.annotate(
            "", xy=txt_center, xytext=img_center,
            arrowprops=dict(arrowstyle="->", color=c, lw=1.5, ls="--")
        )

    ax.set_title("Shared Multi-Modal Metric Space Projection (PCA 2D)\n[Circles = Images, Triangles = Text Captions]", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)", fontsize=11)
    ax.set_ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)", fontsize=11)
    ax.grid(True)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", frameon=True, fontsize=9, ncol=2)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    return fig


def plot_architecture_overview(save_path: Optional[str] = None) -> plt.Figure:
    """Create a pedagogical architecture diagram of the CLIP contrastive learning paradigm."""
    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
    ax.axis("off")

    def draw_box(x, y, w, h, title, subtitle="", color="#f8fafc", edge="#64748b", lw=1.5):
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor=edge, lw=lw)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h - 0.05, title, ha="center", va="top", fontsize=11, fontweight="bold", color="#1e293b")
        if subtitle:
            ax.text(x + w / 2, y + h - 0.13, subtitle, ha="center", va="top", fontsize=9, color="#475569", multialignment="center")

    # 1. Vision Branch
    draw_box(0.04, 0.53, 0.26, 0.38, "Vision Encoder $f_I(x)$", "Input: Image $[64, 64, 3]$\nConv2D + BatchNorm + GAP\nDense Projection $\\to \\mathbb{R}^d$\n$L_2$ Normalization: $\\hat{I} = \\frac{I}{\\|I\\|_2}$", color="#e0f2fe", edge="#0284c7")

    # 2. Text Branch
    draw_box(0.04, 0.08, 0.26, 0.38, "Text Encoder $f_T(t)$", "Input: Tokens $[L]$\nEmbedding + Mini Transformer\nDense Projection $\\to \\mathbb{R}^d$\n$L_2$ Normalization: $\\hat{T} = \\frac{T}{\\|T\\|_2}$", color="#fef3c7", edge="#d97706")

    # 3. Similarity Matrix Block
    draw_box(0.38, 0.22, 0.28, 0.56, "Cosine Similarity Matrix ($S$)", "$S_{i, j} = \\exp(\\text{logit\\_scale}) \\cdot (\\hat{I}_i \\cdot \\hat{T}_j)$\n\nScaled dot-product between\nnormalized image & text features\nDiagonal: Positive Pairs $(I_i, T_i)$\nOff-Diagonal: Negative Pairs", color="#f1f5f9", edge="#475569")

    # 4. Symmetric InfoNCE Loss Block
    draw_box(0.72, 0.53, 0.24, 0.38, "Symmetric InfoNCE Loss", "$\\mathcal{L} = \\frac{1}{2}(\\mathcal{L}_{I \\to T} + \\mathcal{L}_{T \\to I})$\n\nCrossEntropy along rows (I $\\to$ T)\n+\nCrossEntropy along cols (T $\\to$ I)", color="#fee2e2", edge="#dc2626")

    # 5. Zero-Shot Inference Block
    draw_box(0.72, 0.08, 0.24, 0.38, "Zero-Shot Inference", "Query image $\\to$ Class prompts\n$P(c \\mid x) = \\text{softmax}(\\hat{I} \\cdot \\hat{T}_c)$\n\nCompositional & Novel Shapes\nwithout fine-tuning!", color="#dcfce7", edge="#16a34a")

    # Connecting arrows
    arrow_props = dict(arrowstyle="->", lw=2.0, color="#334155")
    ax.annotate("", xy=(0.38, 0.65), xytext=(0.30, 0.72), arrowprops=arrow_props)
    ax.annotate("", xy=(0.38, 0.35), xytext=(0.30, 0.27), arrowprops=arrow_props)
    ax.annotate("", xy=(0.72, 0.72), xytext=(0.66, 0.60), arrowprops=arrow_props)
    ax.annotate("", xy=(0.72, 0.27), xytext=(0.66, 0.40), arrowprops=arrow_props)

    plt.title("CLIP (Contrastive Language-Image Pre-training) Dual-Encoder Architecture", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    return fig

