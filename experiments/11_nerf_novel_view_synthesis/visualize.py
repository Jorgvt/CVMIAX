"""
Visualization routines for NeRF: 3D ray geometries, training dynamics,
qualitative novel-view renderings, depth maps, and spectral bias ablation plots.
"""

from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import tensorflow as tf

try:
    from .rays import get_rays_np, sample_stratified_points
except (ImportError, ValueError):
    from rays import get_rays_np, sample_stratified_points


def plot_camera_rays_3d(
    c2w: np.ndarray,
    focal: float,
    height: int = 100,
    width: int = 100,
    near: float = 2.0,
    far: float = 6.0,
    n_samples: int = 32,
    num_ray_subsample: int = 8,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize camera frustum, ray casting in 3D world space, and stratified point sampling.
    """
    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    rays_o, rays_d = get_rays_np(height, width, focal, c2w)
    
    # Subsample rays for clean 3D visual presentation
    step_y = max(1, height // num_ray_subsample)
    step_x = max(1, width // num_ray_subsample)
    sub_o = rays_o[::step_y, ::step_x].reshape(-1, 3)
    sub_d = rays_d[::step_y, ::step_x].reshape(-1, 3)

    pts, _ = sample_stratified_points(
        tf.constant(sub_o), tf.constant(sub_d), near=near, far=far, n_samples=n_samples, randomize=False
    )
    pts_np = pts.numpy()

    # Camera origin
    cam_pos = c2w[:3, 3]
    ax.scatter(cam_pos[0], cam_pos[1], cam_pos[2], color="black", s=100, marker="^", label="Camera Center (c2w)")

    # Plot sample points and ray lines
    for i in range(len(sub_o)):
        p_start = sub_o[i] + near * sub_d[i]
        p_end = sub_o[i] + far * sub_d[i]
        ax.plot([sub_o[i, 0], p_end[0]], [sub_o[i, 1], p_end[1]], [sub_o[i, 2], p_end[2]], color="gray", alpha=0.3, linewidth=0.8)
        
        # Color samples along the ray
        ax.scatter(
            pts_np[i, :, 0],
            pts_np[i, :, 1],
            pts_np[i, :, 2],
            c=np.linspace(0, 1, n_samples),
            cmap="viridis",
            s=8,
            alpha=0.6,
        )

    # World origin marker
    ax.scatter(0, 0, 0, color="crimson", s=80, marker="o", label="World Origin (0,0,0)")

    ax.set_title("NeRF Pinhole Ray Casting & Stratified 3D Sampling", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("X (World)")
    ax.set_ylabel("Y (World)")
    ax.set_zlabel("Z (World)")
    ax.legend(loc="upper right")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved 3D ray sampling figure to {save_path}")

    return fig


def plot_training_dynamics(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot training photometric MSE loss and PSNR progression over iterations.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

    iters = history["iterations"]
    ax1.plot(iters, history["train_loss"], color="#2c3e50", lw=2, label="Photometric MSE Loss")
    ax1.set_xlabel("Iteration", fontweight="bold")
    ax1.set_ylabel("Mean Squared Error (MSE)", fontweight="bold")
    ax1.set_title("Training Loss Convergence", fontweight="bold", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.set_yscale("log")
    ax1.legend()

    ax2.plot(iters, history["train_psnr"], color="#27ae60", lw=2, label="Train PSNR (dB)")
    if len(history.get("val_psnr", [])) > 0:
        val_iters = np.linspace(iters[0], iters[-1], len(history["val_psnr"]))
        ax2.plot(val_iters, history["val_psnr"], color="#e74c3c", lw=2, marker="o", label="Validation PSNR (dB)")
    ax2.set_xlabel("Iteration", fontweight="bold")
    ax2.set_ylabel("PSNR (dB)", fontweight="bold")
    ax2.set_title("Reconstruction Quality (PSNR)", fontweight="bold", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved training dynamics figure to {save_path}")

    return fig


def plot_qualitative_results(
    gt_rgb: np.ndarray,
    pred_rgb: np.ndarray,
    pred_depth: np.ndarray,
    pred_acc: np.ndarray,
    metrics: Dict[str, float],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Display comprehensive multi-modal qualitative assessment:
    Ground Truth RGB vs. Rendered RGB vs. Absolute Error vs. Rendered Depth Map vs. Ray Opacity.
    """
    fig, axes = plt.subplots(1, 5, figsize=(18, 4), dpi=150)

    # 1. Ground Truth RGB
    axes[0].imshow(gt_rgb)
    axes[0].set_title("Ground Truth RGB", fontweight="bold")
    axes[0].axis("off")

    # 2. Rendered RGB
    axes[1].imshow(pred_rgb)
    axes[1].set_title(f"NeRF Rendered RGB\nPSNR: {metrics['psnr']:.2f} dB", fontweight="bold")
    axes[1].axis("off")

    # 3. Absolute Error Heatmap
    err_map = np.mean(np.abs(gt_rgb - pred_rgb), axis=-1)
    im3 = axes[2].imshow(err_map, cmap="inferno", vmin=0.0, vmax=0.3)
    axes[2].set_title(f"Absolute Error\nL1: {metrics['l1']:.4f}", fontweight="bold")
    axes[2].axis("off")
    fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    # 4. Volumetric Depth Map
    # Invert colormap so closer objects appear brighter
    im4 = axes[3].imshow(pred_depth, cmap="turbo")
    axes[3].set_title("Rendered Depth D(r)", fontweight="bold")
    axes[3].axis("off")
    fig.colorbar(im4, ax=axes[3], fraction=0.046, pad=0.04)

    # 5. Optical Accumulation / Opacity
    im5 = axes[4].imshow(pred_acc, cmap="gray", vmin=0.0, vmax=1.0)
    axes[4].set_title("Transmittance / Opacity", fontweight="bold")
    axes[4].axis("off")
    fig.colorbar(im5, ax=axes[4], fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved qualitative assessment figure to {save_path}")

    return fig


def plot_spectral_bias_ablation(
    gt_rgb: np.ndarray,
    pred_no_pe: np.ndarray,
    psnr_no_pe: float,
    pred_with_pe: np.ndarray,
    psnr_with_pe: float,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Demonstrate the critical role of Positional Encoding in overcoming Spectral Bias.
    Compares raw coordinate inputs (L=0) vs. Fourier feature encoding (L=6).
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=150)

    # Ground Truth
    axes[0].imshow(gt_rgb)
    axes[0].set_title("Ground Truth Target", fontweight="bold", fontsize=12)
    axes[0].axis("off")

    # Without Positional Encoding (L=0)
    axes[1].imshow(pred_no_pe)
    axes[1].set_title(
        f"Without Positional Encoding (L=0)\nPSNR: {psnr_no_pe:.2f} dB (Oversmoothed)",
        fontweight="bold",
        color="#c0392b",
        fontsize=11,
    )
    axes[1].axis("off")

    # With Positional Encoding (L=6)
    axes[2].imshow(pred_with_pe)
    axes[2].set_title(
        f"With Positional Encoding (L=6)\nPSNR: {psnr_with_pe:.2f} dB (Sharp High-Freq)",
        fontweight="bold",
        color="#27ae60",
        fontsize=11,
    )
    axes[2].axis("off")

    plt.suptitle("Spectral Bias Ablation: Why Fourier Positional Encoding is Essential", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved spectral bias ablation figure to {save_path}")

    return fig


def plot_novel_view_strip(
    novel_frames: List[Dict[str, np.ndarray]],
    num_views_to_show: int = 6,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a horizontal filmstrip of novel synthesized 360-degree orbital viewpoints and depths.
    """
    step = max(1, len(novel_frames) // num_views_to_show)
    selected_indices = [i * step for i in range(num_views_to_show)]

    fig, axes = plt.subplots(2, num_views_to_show, figsize=(16, 5.5), dpi=150)

    for col, idx in enumerate(selected_indices):
        frame = novel_frames[idx]
        azimuth_deg = int(360.0 * idx / len(novel_frames))

        # Top row: RGB synthesis
        axes[0, col].imshow(frame["rgb"])
        axes[0, col].set_title(f"Azimuth: {azimuth_deg}°", fontweight="bold", fontsize=11)
        axes[0, col].axis("off")

        # Bottom row: Depth map
        axes[1, col].imshow(frame["depth"], cmap="turbo")
        axes[1, col].axis("off")

    axes[0, 0].text(-0.15, 0.5, "Rendered RGB", transform=axes[0, 0].transAxes, rotation=90, va="center", ha="right", fontweight="bold", fontsize=12)
    axes[1, 0].text(-0.15, 0.5, "Depth D(r)", transform=axes[1, 0].transAxes, rotation=90, va="center", ha="right", fontweight="bold", fontsize=12)

    plt.suptitle("360° Novel View Synthesis & Continuous 3D Depth Consistency", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved novel view strip to {save_path}")

    return fig


def create_orbital_gif(
    novel_frames: List[Dict[str, np.ndarray]],
    gif_path: str,
    fps: int = 15,
) -> None:
    """
    Save novel viewpoint frames as an animated GIF.
    """
    pil_images = []
    for f in novel_frames:
        # Scale to uint8
        img_uint8 = (f["rgb"] * 255.0).clip(0, 255).astype(np.uint8)
        pil_images.append(Image.fromarray(img_uint8))

    if pil_images:
        duration_ms = int(1000 / fps)
        pil_images[0].save(
            gif_path,
            save_all=True,
            append_images=pil_images[1:],
            duration=duration_ms,
            loop=0,
        )
        print(f"Saved 360-degree novel view animation to {gif_path}")
