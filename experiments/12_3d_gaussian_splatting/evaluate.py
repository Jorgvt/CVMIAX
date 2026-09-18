"""
Evaluation and trajectory rendering routines for 3D Gaussian Splatting.

Computes PSNR, SSIM, and L1 reconstruction metrics, and renders novel-view orbital paths.
"""

from typing import Dict, List, Tuple
import numpy as np
import tensorflow as tf

from models import GaussianModel
from splatting import render_gaussians


def compute_metrics(
    pred_rgb: np.ndarray,
    gt_rgb: np.ndarray,
) -> Dict[str, float]:
    """
    Compute quantitative image reconstruction metrics between predicted and ground truth RGB images.

    Args:
        pred_rgb: [H, W, 3] Predicted image in [0, 1].
        gt_rgb: [H, W, 3] Ground truth image in [0, 1].

    Returns:
        Dictionary containing PSNR (dB), SSIM, L1, and MSE metrics.
    """
    pred_tf = tf.convert_to_tensor(pred_rgb, dtype=tf.float32)
    gt_tf = tf.convert_to_tensor(gt_rgb, dtype=tf.float32)

    l1_loss = float(tf.reduce_mean(tf.abs(pred_tf - gt_tf)).numpy())
    mse_loss = float(tf.reduce_mean(tf.square(pred_tf - gt_tf)).numpy())
    psnr_val = float(tf.image.psnr(pred_tf, gt_tf, max_val=1.0).numpy())
    ssim_val = float(tf.image.ssim(pred_tf, gt_tf, max_val=1.0).numpy())

    return {
        "psnr": psnr_val,
        "ssim": ssim_val,
        "l1": l1_loss,
        "mse": mse_loss,
    }


def evaluate_dataset(
    model: GaussianModel,
    images: np.ndarray,
    poses: np.ndarray,
    focal: float,
    height: int,
    width: int,
) -> Tuple[Dict[str, float], List[np.ndarray]]:
    """
    Evaluate 3D Gaussian Splatting model across an entire validation / test split.

    Args:
        model: Trained GaussianModel.
        images: [N_views, H, W, 3] Ground truth images.
        poses: [N_views, 4, 4] Camera poses.
        focal: Focal length in pixels.
        height: Image height.
        width: Image width.

    Returns:
        Tuple of (mean_metrics_dict, list_of_rendered_images).
    """
    scales = model.get_scales()
    quats = model.get_quats()
    opacities = model.get_opacities()
    colors = model.get_colors()

    rendered_images = []
    all_metrics = {"psnr": [], "ssim": [], "l1": [], "mse": []}

    for i in range(len(images)):
        c2w = tf.constant(poses[i], dtype=tf.float32)
        rendered = render_gaussians(
            means_3d=model.means,
            scales=scales,
            quats=quats,
            opacities=opacities,
            colors=colors,
            c2w=c2w,
            focal=focal,
            height=height,
            width=width,
            bg_color=0.0,
        ).numpy()

        rendered_images.append(rendered)
        m = compute_metrics(rendered, images[i])
        for k in all_metrics:
            all_metrics[k].append(m[k])

    mean_metrics = {k: float(np.mean(v)) for k, v in all_metrics.items()}
    return mean_metrics, rendered_images


def render_novel_trajectory(
    model: GaussianModel,
    poses: np.ndarray,
    focal: float,
    height: int,
    width: int,
) -> np.ndarray:
    """
    Render a sequence of novel camera viewpoints (e.g. 360-degree orbital spin).

    Args:
        model: Trained GaussianModel.
        poses: [N_poses, 4, 4] Camera poses along trajectory.
        focal: Focal length in pixels.
        height: Image height.
        width: Image width.

    Returns:
        np.ndarray: [N_poses, H, W, 3] Rendered RGB sequence in [0, 1].
    """
    scales = model.get_scales()
    quats = model.get_quats()
    opacities = model.get_opacities()
    colors = model.get_colors()

    frames = []
    for pose in poses:
        c2w = tf.constant(pose, dtype=tf.float32)
        rendered = render_gaussians(
            means_3d=model.means,
            scales=scales,
            quats=quats,
            opacities=opacities,
            colors=colors,
            c2w=c2w,
            focal=focal,
            height=height,
            width=width,
            bg_color=0.0,
        ).numpy()
        frames.append(rendered)

    return np.stack(frames, axis=0)
