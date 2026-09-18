"""
Visualization routines for Deep Computer Vision Model Architectures:
- Plain Forward CNN
- ResNet (Residual Learning)
- Inception (Multi-Scale Parallel Receptive Fields)
- EfficientNet (MBConv + Squeeze & Excitation)
- ConvNeXt (Modern Pure ConvNet)
"""

from typing import Dict, Any, Optional, List
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import keras
from keras import layers


def plot_architecture_comparison(
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Renders a clear, pedagogical side-by-side architectural diagram
    contrasting a Plain Feedforward Block vs a Residual Block with shortcut connection.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), dpi=200)
    fig.patch.set_facecolor("#ffffff")

    # --- Panel A: Plain Feedforward Block ---
    ax1 = axes[0]
    ax1.set_facecolor("#f9fafb")
    ax1.set_title("A. Plain Feedforward Block (VGG Style)\nDirect Sequential Mapping: y = sigma(F(x))", fontsize=13, fontweight="bold", pad=15)
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-0.5, 7.5)
    ax1.axis("off")

    # Nodes
    ax1.text(0, 7.0, "Input: x\n[H x W x C]", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.5", facecolor="#e0e7ff", edgecolor="#4338ca", lw=1.5), fontsize=10, fontweight="bold")
    ax1.annotate("", xy=(0, 5.8), xytext=(0, 6.5), arrowprops=dict(arrowstyle="->", lw=2, color="#374151"))
    ax1.text(0, 5.3, "Conv2D (3x3, stride s)\n+ BatchNorm + ReLU", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.5", facecolor="#fef3c7", edgecolor="#d97706", lw=1.5), fontsize=9.5)
    ax1.annotate("", xy=(0, 4.1), xytext=(0, 4.8), arrowprops=dict(arrowstyle="->", lw=2, color="#374151"))
    ax1.text(0, 3.6, "Conv2D (3x3, stride 1)\n+ BatchNorm", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.5", facecolor="#fef3c7", edgecolor="#d97706", lw=1.5), fontsize=9.5)
    ax1.annotate("", xy=(0, 2.4), xytext=(0, 3.1), arrowprops=dict(arrowstyle="->", lw=2, color="#374151"))
    ax1.text(0, 1.9, "ReLU Activation", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.5", facecolor="#fee2e2", edgecolor="#dc2626", lw=1.5), fontsize=9.5)
    ax1.annotate("", xy=(0, 0.7), xytext=(0, 1.4), arrowprops=dict(arrowstyle="->", lw=2, color="#374151"))
    ax1.text(0, 0.2, "Output: y = F(x)\n[H' x W' x C']", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.5", facecolor="#dcfce7", edgecolor="#16a34a", lw=1.5), fontsize=10, fontweight="bold")

    ax1.text(0, -0.4, "[Multiplicative Backprop] Gradients must pass through\nevery weight matrix without shortcut.", ha="center", va="center", fontsize=8.5, color="#991b1b", style="italic")

    # --- Panel B: Residual Block ---
    ax2 = axes[1]
    ax2.set_facecolor("#f9fafb")
    ax2.set_title("B. Residual Block (He et al., 2015)\nAdditive Identity Shortcut: y = sigma(F(x) + x)", fontsize=13, fontweight="bold", pad=15)
    ax2.set_xlim(-1.8, 1.8)
    ax2.set_ylim(-0.5, 7.5)
    ax2.axis("off")

    ax2.text(0, 7.0, "Input: x\n[H x W x C]", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.5", facecolor="#e0e7ff", edgecolor="#4338ca", lw=1.5), fontsize=10, fontweight="bold")
    
    # Residual Path
    ax2.annotate("", xy=(-0.5, 5.8), xytext=(0, 6.5), arrowprops=dict(arrowstyle="->", lw=2, color="#374151"))
    ax2.text(-0.5, 5.3, "Conv2D (3x3, stride s)\n+ BatchNorm + ReLU", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.5", facecolor="#fef3c7", edgecolor="#d97706", lw=1.5), fontsize=9.5)
    ax2.annotate("", xy=(-0.5, 4.1), xytext=(-0.5, 4.8), arrowprops=dict(arrowstyle="->", lw=2, color="#374151"))
    ax2.text(-0.5, 3.6, "Conv2D (3x3, stride 1)\n+ BatchNorm", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.5", facecolor="#fef3c7", edgecolor="#d97706", lw=1.5), fontsize=9.5)
    ax2.annotate("", xy=(0, 2.7), xytext=(-0.5, 3.1), arrowprops=dict(arrowstyle="->", lw=2, color="#374151"))

    # Shortcut Path
    ax2.annotate("", xy=(1.0, 4.5), xytext=(0, 6.5), arrowprops=dict(arrowstyle="->", lw=2.5, color="#2563eb", linestyle="--"))
    ax2.text(1.0, 4.5, "Identity Shortcut (s=1)\nor 1x1 Conv+BN (s>1)", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.5", facecolor="#dbeafe", edgecolor="#2563eb", lw=1.5), fontsize=9)
    ax2.annotate("", xy=(0.2, 2.7), xytext=(1.0, 3.9), arrowprops=dict(arrowstyle="->", lw=2.5, color="#2563eb", linestyle="--"))

    # Addition
    ax2.text(0, 2.5, "+ (Add)", ha="center", va="center", bbox=dict(boxstyle="circle,pad=0.4", facecolor="#c7d2fe", edgecolor="#4338ca", lw=2), fontsize=10, fontweight="bold")

    # Activation
    ax2.annotate("", xy=(0, 1.8), xytext=(0, 2.1), arrowprops=dict(arrowstyle="->", lw=2, color="#374151"))
    ax2.text(0, 1.4, "ReLU Activation", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.5", facecolor="#fee2e2", edgecolor="#dc2626", lw=1.5), fontsize=9.5)

    # Output
    ax2.annotate("", xy=(0, 0.7), xytext=(0, 1.0), arrowprops=dict(arrowstyle="->", lw=2, color="#374151"))
    ax2.text(0, 0.2, "Output: y = sigma(F(x) + x)\n[H' x W' x C']", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.5", facecolor="#dcfce7", edgecolor="#16a34a", lw=1.5), fontsize=10, fontweight="bold")

    ax2.text(0, -0.4, "[Additive Gradient Highway] dL/dx = dL/dy + dL/dy * (dF/dx)\nGradients flow cleanly through identity shortcut!", ha="center", va="center", fontsize=8.5, color="#1e40af", fontweight="bold")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved architecture comparison to: {save_path}")
    return fig


def plot_all_architectures_poster(
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Renders a comprehensive overview poster explaining the core building blocks
    of all 5 computer vision architectures.
    """
    fig, axes = plt.subplots(1, 5, figsize=(22, 9), dpi=200)
    fig.patch.set_facecolor("#ffffff")
    
    titles = [
        ("1. Plain CNN (VGG)", "#dc2626", "Sequential Conv-BN-ReLU\nDirect multiplicative flow"),
        ("2. ResNet (2015)", "#2563eb", "Residual Shortcut (F(x)+x)\nAdditive gradient highway"),
        ("3. Inception (2015)", "#059669", "Multi-Scale Parallel Branches\n1x1, 3x3, 5x5 + Pool Concat"),
        ("4. EfficientNet (2019)", "#d97706", "MBConv + SE Channel Attention\nDepthwise Separable + Swish"),
        ("5. ConvNeXt (2022)", "#7c3aed", "Modern pure ConvNet\n7x7 DW + LayerNorm + GELU"),
    ]

    for ax, (title, color, desc) in zip(axes, titles):
        ax.set_facecolor("#f9fafb")
        ax.set_title(title, fontsize=11, fontweight="bold", color=color, pad=10)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.5, 7.5)
        ax.axis("off")
        ax.text(0, -0.3, desc, ha="center", va="center", fontsize=8.5, color="#374151", style="italic")

    # 1. Plain CNN
    axes[0].text(0, 6.8, "Input x", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.3", facecolor="#e0e7ff", edgecolor="#4338ca"), fontsize=9)
    axes[0].annotate("", xy=(0, 5.7), xytext=(0, 6.4), arrowprops=dict(arrowstyle="->", lw=1.5))
    axes[0].text(0, 5.2, "Conv 3x3\n+ BN + ReLU", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.3", facecolor="#fef3c7", edgecolor="#d97706"), fontsize=8.5)
    axes[0].annotate("", xy=(0, 4.1), xytext=(0, 4.7), arrowprops=dict(arrowstyle="->", lw=1.5))
    axes[0].text(0, 3.6, "Conv 3x3\n+ BN + ReLU", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.3", facecolor="#fef3c7", edgecolor="#d97706"), fontsize=8.5)
    axes[0].annotate("", xy=(0, 2.5), xytext=(0, 3.1), arrowprops=dict(arrowstyle="->", lw=1.5))
    axes[0].text(0, 2.0, "Conv 3x3\n+ BN + ReLU", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.3", facecolor="#fef3c7", edgecolor="#d97706"), fontsize=8.5)
    axes[0].annotate("", xy=(0, 0.9), xytext=(0, 1.5), arrowprops=dict(arrowstyle="->", lw=1.5))
    axes[0].text(0, 0.5, "Output y = F(x)", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.3", facecolor="#fee2e2", edgecolor="#dc2626"), fontsize=9, fontweight="bold")

    # 2. ResNet
    axes[1].text(0, 6.8, "Input x", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.3", facecolor="#e0e7ff", edgecolor="#4338ca"), fontsize=9)
    axes[1].annotate("", xy=(-0.4, 5.7), xytext=(0, 6.4), arrowprops=dict(arrowstyle="->", lw=1.5))
    axes[1].text(-0.4, 5.2, "Conv 3x3 + BN\n+ ReLU", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.3", facecolor="#fef3c7", edgecolor="#d97706"), fontsize=8)
    axes[1].annotate("", xy=(-0.4, 4.1), xytext=(-0.4, 4.7), arrowprops=dict(arrowstyle="->", lw=1.5))
    axes[1].text(-0.4, 3.6, "Conv 3x3 + BN", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.3", facecolor="#fef3c7", edgecolor="#d97706"), fontsize=8)
    axes[1].annotate("", xy=(0, 2.5), xytext=(-0.4, 3.1), arrowprops=dict(arrowstyle="->", lw=1.5))
    axes[1].annotate("", xy=(0.6, 3.6), xytext=(0, 6.4), arrowprops=dict(arrowstyle="->", lw=2, color="#2563eb", linestyle="--"))
    axes[1].text(0.6, 4.0, "Identity Shortcut\n(or 1x1 Proj)", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.2", facecolor="#dbeafe", edgecolor="#2563eb"), fontsize=7.5)
    axes[1].annotate("", xy=(0.1, 2.5), xytext=(0.6, 3.4), arrowprops=dict(arrowstyle="->", lw=2, color="#2563eb", linestyle="--"))
    axes[1].text(0, 2.3, "+", ha="center", va="center", bbox=dict(boxstyle="circle,pad=0.3", facecolor="#c7d2fe", edgecolor="#4338ca", lw=1.5), fontsize=9, fontweight="bold")
    axes[1].annotate("", xy=(0, 1.4), xytext=(0, 2.0), arrowprops=dict(arrowstyle="->", lw=1.5))
    axes[1].text(0, 1.1, "ReLU", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.2", facecolor="#fee2e2", edgecolor="#dc2626"), fontsize=8)
    axes[1].annotate("", xy=(0, 0.6), xytext=(0, 0.9), arrowprops=dict(arrowstyle="->", lw=1.5))
    axes[1].text(0, 0.3, "Output y = F(x)+x", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.3", facecolor="#dcfce7", edgecolor="#16a34a"), fontsize=8.5, fontweight="bold")

    # 3. Inception
    axes[2].text(0, 6.8, "Input x", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.3", facecolor="#e0e7ff", edgecolor="#4338ca"), fontsize=9)
    axes[2].text(-0.9, 5.0, "1x1 Conv", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.2", facecolor="#fef3c7", edgecolor="#d97706"), fontsize=7.5)
    axes[2].text(-0.3, 5.3, "1x1 Red", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.2", facecolor="#fef3c7", edgecolor="#d97706"), fontsize=7.5)
    axes[2].text(-0.3, 4.4, "3x3 Conv", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.2", facecolor="#fef3c7", edgecolor="#d97706"), fontsize=7.5)
    axes[2].text(0.3, 5.3, "1x1 Red", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.2", facecolor="#fef3c7", edgecolor="#d97706"), fontsize=7.5)
    axes[2].text(0.3, 4.4, "5x5 Conv", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.2", facecolor="#fef3c7", edgecolor="#d97706"), fontsize=7.5)
    axes[2].text(0.9, 5.3, "3x3 Pool", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.2", facecolor="#dbeafe", edgecolor="#2563eb"), fontsize=7.5)
    axes[2].text(0.9, 4.4, "1x1 Proj", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.2", facecolor="#fef3c7", edgecolor="#d97706"), fontsize=7.5)
    axes[2].text(0, 2.5, "Concat Filters [B1, B2, B3, B4]", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.3", facecolor="#c7d2fe", edgecolor="#4338ca"), fontsize=8, fontweight="bold")
    axes[2].annotate("", xy=(0, 0.9), xytext=(0, 2.1), arrowprops=dict(arrowstyle="->", lw=1.5))
    axes[2].text(0, 0.5, "Multi-scale Output", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.3", facecolor="#dcfce7", edgecolor="#16a34a"), fontsize=8.5, fontweight="bold")

    # 4. EfficientNet
    axes[3].text(0, 6.8, "Input x", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.3", facecolor="#e0e7ff", edgecolor="#4338ca"), fontsize=9)
    axes[3].annotate("", xy=(-0.4, 5.8), xytext=(0, 6.4), arrowprops=dict(arrowstyle="->", lw=1.5))
    axes[3].text(-0.4, 5.4, "1x1 Expansion\n(4-6x filters)", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.2", facecolor="#fef3c7", edgecolor="#d97706"), fontsize=7.5)
    axes[3].text(-0.4, 4.4, "3x3 Depthwise\nConv + Swish", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.2", facecolor="#fef3c7", edgecolor="#d97706"), fontsize=7.5)
    axes[3].text(-0.4, 3.4, "SE Block\n(GAP -> Dense -> Sigmoid)", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.2", facecolor="#fbcfe8", edgecolor="#db2777"), fontsize=7.5)
    axes[3].text(-0.4, 2.4, "1x1 Linear Proj\n(No activation)", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.2", facecolor="#fef3c7", edgecolor="#d97706"), fontsize=7.5)
    axes[3].annotate("", xy=(0.6, 2.0), xytext=(0, 6.4), arrowprops=dict(arrowstyle="->", lw=1.5, color="#d97706", linestyle="--"))
    axes[3].text(0, 1.4, "+", ha="center", va="center", bbox=dict(boxstyle="circle,pad=0.3", facecolor="#c7d2fe", edgecolor="#4338ca"), fontsize=9, fontweight="bold")
    axes[3].annotate("", xy=(0, 0.7), xytext=(0, 1.1), arrowprops=dict(arrowstyle="->", lw=1.5))
    axes[3].text(0, 0.4, "Output y (MBConv)", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.3", facecolor="#dcfce7", edgecolor="#16a34a"), fontsize=8.5, fontweight="bold")

    # 5. ConvNeXt
    axes[4].text(0, 6.8, "Input x", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.3", facecolor="#e0e7ff", edgecolor="#4338ca"), fontsize=9)
    axes[4].annotate("", xy=(-0.4, 5.8), xytext=(0, 6.4), arrowprops=dict(arrowstyle="->", lw=1.5))
    axes[4].text(-0.4, 5.4, "7x7 Depthwise\nConv (Large RF)", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.2", facecolor="#ede9fe", edgecolor="#7c3aed"), fontsize=7.5)
    axes[4].text(-0.4, 4.4, "LayerNorm\n(Channels Last)", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.2", facecolor="#ede9fe", edgecolor="#7c3aed"), fontsize=7.5)
    axes[4].text(-0.4, 3.4, "1x1 Pointwise (4x)\n+ GELU", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.2", facecolor="#ede9fe", edgecolor="#7c3aed"), fontsize=7.5)
    axes[4].text(-0.4, 2.4, "1x1 Pointwise Proj\nback to dim", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.2", facecolor="#ede9fe", edgecolor="#7c3aed"), fontsize=7.5)
    axes[4].annotate("", xy=(0.6, 2.0), xytext=(0, 6.4), arrowprops=dict(arrowstyle="->", lw=1.5, color="#7c3aed", linestyle="--"))
    axes[4].text(0, 1.4, "+", ha="center", va="center", bbox=dict(boxstyle="circle,pad=0.3", facecolor="#c7d2fe", edgecolor="#4338ca"), fontsize=9, fontweight="bold")
    axes[4].annotate("", xy=(0, 0.7), xytext=(0, 1.1), arrowprops=dict(arrowstyle="->", lw=1.5))
    axes[4].text(0, 0.4, "Output y (ConvNeXt)", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.3", facecolor="#dcfce7", edgecolor="#16a34a"), fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved master architecture poster to: {save_path}")
    return fig


def plot_training_dynamics(
    histories: Dict[str, Dict[str, List[float]]],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plots training loss and validation accuracy curves across multiple models.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), dpi=200)
    fig.patch.set_facecolor("#ffffff")

    palette = {
        "ResNet-20": "#2563eb",
        "Plain-20": "#dc2626",
        "EfficientNet-CIFAR": "#d97706",
        "ConvNeXt-CIFAR": "#7c3aed",
        "Inception-CIFAR": "#059669",
    }
    
    # 1. Loss Plot
    ax1.set_title("Training Loss vs Epochs", fontsize=12, fontweight="bold", pad=10)
    for model_name, hist in histories.items():
        color = palette.get(model_name, None)
        epochs = range(1, len(hist["loss"]) + 1)
        ax1.plot(epochs, hist["loss"], label=f"{model_name} (Train)", linestyle="-", color=color, lw=2)
        if "val_loss" in hist and len(hist["val_loss"]) > 0:
            ax1.plot(epochs, hist["val_loss"], label=f"{model_name} (Val)", linestyle="--", color=color, alpha=0.6, lw=1.5)
    ax1.set_xlabel("Epoch", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Categorical Crossentropy Loss", fontsize=10, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(frameon=True, fontsize=8.5)

    # 2. Accuracy Plot
    ax2.set_title("Classification Accuracy vs Epochs", fontsize=12, fontweight="bold", pad=10)
    for model_name, hist in histories.items():
        color = palette.get(model_name, None)
        train_acc = hist.get("accuracy", hist.get("acc", []))
        val_acc = hist.get("val_accuracy", hist.get("val_acc", []))
        epochs = range(1, len(train_acc) + 1)
        ax2.plot(epochs, [a * 100 for a in train_acc], label=f"{model_name} (Train)", linestyle="-", color=color, lw=2)
        if val_acc and len(val_acc) > 0:
            ax2.plot(epochs, [a * 100 for a in val_acc], label=f"{model_name} (Val)", linestyle="--", color=color, alpha=0.6, lw=1.5)
    ax2.set_xlabel("Epoch", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Accuracy (%)", fontsize=10, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(frameon=True, fontsize=8.5)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved training dynamics to: {save_path}")
    return fig


def plot_weight_distributions(
    resnet_model: keras.Model,
    plain_model: keras.Model,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plots the distribution and L2 norms of convolutional weights across network depth.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=200)
    fig.patch.set_facecolor("#ffffff")

    def get_conv_weights(model: keras.Model):
        weights = []
        names = []
        for layer in model.layers:
            if isinstance(layer, (layers.Conv2D, layers.DepthwiseConv2D)) and len(layer.get_weights()) > 0:
                w = layer.get_weights()[0].flatten()
                weights.append(w)
                names.append(layer.name)
        return weights, names

    res_w, res_names = get_conv_weights(resnet_model)
    plain_w, plain_names = get_conv_weights(plain_model)

    res_norms = [np.linalg.norm(w) for w in res_w]
    plain_norms = [np.linalg.norm(w) for w in plain_w]

    ax1.set_title("Conv Weight L2 Norms across Depth", fontsize=12, fontweight="bold", pad=10)
    ax1.plot(range(1, len(res_norms) + 1), res_norms, marker="o", label=f"ResNet ({len(res_norms)} convs)", color="#2563eb", lw=2)
    ax1.plot(range(1, len(plain_norms) + 1), plain_norms, marker="s", label=f"Plain CNN ({len(plain_norms)} convs)", color="#dc2626", lw=2)
    ax1.set_xlabel("Convolutional Layer Index (Input -> Output)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Weight Tensor L2 Norm ||W||_2", fontsize=10, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(frameon=True, fontsize=9)

    all_res_w = np.concatenate(res_w) if res_w else np.array([])
    all_plain_w = np.concatenate(plain_w) if plain_w else np.array([])

    ax2.set_title("Overall Weight Parameter Distribution", fontsize=12, fontweight="bold", pad=10)
    if len(all_res_w) > 0:
        ax2.hist(all_res_w, bins=60, density=True, alpha=0.6, color="#2563eb", label=f"ResNet (std={np.std(all_res_w):.4f})")
    if len(all_plain_w) > 0:
        ax2.hist(all_plain_w, bins=60, density=True, alpha=0.6, color="#dc2626", label=f"Plain CNN (std={np.std(all_plain_w):.4f})")
    ax2.set_xlabel("Weight Value", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Density", fontsize=10, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(frameon=True, fontsize=9)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved weight distributions to: {save_path}")
    return fig
