"""
Publication-quality visualization and animation utilities for 3D Gaussian Splatting.

Provides:
- 3D Gaussian cloud & camera spatial arrangement plots
- Training dynamics curves (Loss, L1, L2, PSNR)
- Multi-view qualitative comparison (GT vs Render vs Residual Error)
- Novel orbital viewpoint strips
- 360-degree animated orbital GIF generation
"""

import os
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from models import GaussianModel


def plot_3d_gaussian_cloud(
    model: GaussianModel,
    camera_poses: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    max_points: int = 500,
) -> plt.Figure:
    """
    Render a 3D scatter plot of Gaussian centers colored by their learned diffuse RGB,
    along with camera positions in world coordinates.
    """
    means = model.means.numpy()
    colors = model.get_colors().numpy()
    opacities = model.get_opacities().numpy()[:, 0]

    # Subsample if large
    if len(means) > max_points:
        idx = np.random.choice(len(means), max_points, replace=False)
        means = means[idx]
        colors = colors[idx]
        opacities = opacities[idx]

    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    rgba = np.concatenate([colors, np.clip(opacities[:, None] * 1.2, 0.2, 1.0)], axis=-1)

    # Scatter 3D Gaussians
    ax.scatter(
        means[:, 0],
        means[:, 2],
        means[:, 1],
        c=rgba,
        s=30,
        edgecolors="none",
        label=f"3D Gaussians (N={len(means)})",
    )

    # Plot camera poses if provided
    if camera_poses is not None:
        cam_centers = camera_poses[:, :3, 3]
        ax.scatter(
            cam_centers[:, 0],
            cam_centers[:, 2],
            cam_centers[:, 1],
            c="red",
            marker="^",
            s=50,
            label=f"Camera Poses (N={len(cam_centers)})",
        )

    ax.set_title("3D Gaussian Explicit Scene Representation & Camera Poses", fontsize=14, pad=12, fontweight="bold")
    ax.set_xlabel("X (World)")
    ax.set_ylabel("Z (World)")
    ax.set_zlabel("Y (World)")
    ax.legend(loc="upper right", frameon=True)
    ax.view_init(elev=25, azim=45)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved 3D Gaussian cloud plot to: {save_path}")

    return fig


def plot_training_dynamics(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot training loss, L1 reconstruction error, and PSNR across optimization iterations.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    # 1. Loss & L1
    axes[0].plot(history["loss"], label="Total Loss", color="#1f77b4", linewidth=2)
    if "loss_l1" in history:
        axes[0].plot(history["loss_l1"], label="L1 Loss", color="#ff7f0e", linestyle="--", linewidth=1.5)
    axes[0].set_title("Training Loss Convergence", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(frameon=True)

    # 2. PSNR
    axes[1].plot(history["psnr"], label="Reconstruction PSNR (dB)", color="#2ca02c", linewidth=2)
    axes[1].set_title("Peak Signal-to-Noise Ratio (PSNR)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("PSNR (dB)")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(frameon=True)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved training dynamics plot to: {save_path}")

    return fig


def plot_qualitative_reconstruction(
    gt_images: List[np.ndarray],
    rendered_images: List[np.ndarray],
    view_indices: Optional[List[int]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot Ground Truth vs 3DGS Render vs Absolute Residual Error for selected viewpoints.
    """
    if view_indices is None:
        num_show = min(4, len(gt_images))
        view_indices = list(range(num_show))

    num_views = len(view_indices)
    fig, axes = plt.subplots(num_views, 3, figsize=(10, 3.2 * num_views), dpi=150)
    if num_views == 1:
        axes = np.expand_dims(axes, 0)

    for i, idx in enumerate(view_indices):
        gt = gt_images[idx]
        pred = rendered_images[idx]
        diff = np.abs(pred - gt)
        error_map = np.mean(diff, axis=-1)

        # Ground Truth
        axes[i, 0].imshow(np.clip(gt, 0.0, 1.0))
        axes[i, 0].set_title(f"View {idx} - Ground Truth", fontsize=10, fontweight="bold")
        axes[i, 0].axis("off")

        # 3DGS Render
        psnr_val = -10.0 * np.log10(np.mean((pred - gt) ** 2) + 1e-10)
        axes[i, 1].imshow(np.clip(pred, 0.0, 1.0))
        axes[i, 1].set_title(f"3DGS Render (PSNR: {psnr_val:.1f} dB)", fontsize=10, fontweight="bold")
        axes[i, 1].axis("off")

        # Error Map
        im = axes[i, 2].imshow(error_map, cmap="inferno", vmin=0.0, vmax=0.5)
        axes[i, 2].set_title(f"Residual Error (|GT - Pred|)", fontsize=10, fontweight="bold")
        axes[i, 2].axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved qualitative reconstruction plot to: {save_path}")

    return fig


def plot_novel_view_strip(
    novel_frames: np.ndarray,
    num_views_to_display: int = 6,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a horizontal strip of novel views along a 360-degree circular orbit.
    """
    total_frames = len(novel_frames)
    indices = np.linspace(0, total_frames - 1, num_views_to_display, dtype=int)

    fig, axes = plt.subplots(1, num_views_to_display, figsize=(3 * num_views_to_display, 3.2), dpi=150)
    for i, idx in enumerate(indices):
        deg = int((idx / total_frames) * 360)
        axes[i].imshow(np.clip(novel_frames[idx], 0.0, 1.0))
        axes[i].set_title(f"Orbit {deg}°", fontsize=11, fontweight="bold")
        axes[i].axis("off")

    fig.suptitle("360-Degree Novel View Orbital Synthesis (3D Gaussian Splatting)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved novel view strip to: {save_path}")

    return fig


def create_orbital_gif(
    novel_frames: np.ndarray,
    save_path: str,
    fps: int = 20,
) -> str:
    """
    Save novel viewpoint frames as an animated GIF.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    uint8_frames = [(np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8) for frame in novel_frames]

    try:
        from PIL import Image
        pil_images = [Image.fromarray(f) for f in uint8_frames]
        pil_images[0].save(
            save_path,
            save_all=True,
            append_images=pil_images[1:],
            duration=int(1000 / fps),
            loop=0,
        )
        print(f"Successfully created orbital GIF at: {save_path}")
    except Exception as e:
        print(f"Failed to create GIF: {e}")

    return save_path
