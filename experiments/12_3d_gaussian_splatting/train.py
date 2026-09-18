"""
Optimization and training routines for 3D Gaussian Splatting.

Implements photometric loss backpropagation with custom per-parameter-group
learning rates (positions, scales, quaternions, opacities, and colors).
"""

import time
from typing import Dict, List, Optional
import numpy as np
import tensorflow as tf

from models import GaussianModel
from splatting import render_gaussians


def train_gaussian_splatting(
    model: GaussianModel,
    images: np.ndarray,
    poses: np.ndarray,
    focal: float,
    height: int,
    width: int,
    num_iterations: int = 100,
    lr_means: float = 0.005,
    lr_scales: float = 0.005,
    lr_quats: float = 0.005,
    lr_opacities: float = 0.03,
    lr_colors: float = 0.02,
    l1_weight: float = 0.8,
    scale_reg_weight: float = 0.01,
    log_interval: int = 10,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """
    Train 3D Gaussian Splatting scene model using multi-view photometric supervision.

    Args:
        model: GaussianModel instance.
        images: [N_views, H, W, 3] Ground truth training images.
        poses: [N_views, 4, 4] Camera-to-world extrinsic matrices.
        focal: Camera focal length in pixels.
        height: Image height.
        width: Image width.
        num_iterations: Total number of gradient descent iterations.
        lr_means: Learning rate for 3D Gaussian positions.
        lr_scales: Learning rate for 3D Gaussian log scales.
        lr_quats: Learning rate for rotation quaternions.
        lr_opacities: Learning rate for logit opacities.
        lr_colors: Learning rate for logit RGB colors.
        l1_weight: Weight for L1 loss term (remainder is L2 loss).
        scale_reg_weight: Regularization penalty weight to prevent giant floaters.
        log_interval: Frequency of console logging.
        verbose: Whether to print training progress.

    Returns:
        Dictionary containing loss and PSNR histories.
    """
    opt_means = tf.keras.optimizers.Adam(learning_rate=lr_means)
    opt_scales = tf.keras.optimizers.Adam(learning_rate=lr_scales)
    opt_quats = tf.keras.optimizers.Adam(learning_rate=lr_quats)
    opt_opacities = tf.keras.optimizers.Adam(learning_rate=lr_opacities)
    opt_colors = tf.keras.optimizers.Adam(learning_rate=lr_colors)

    num_views = len(images)
    images_tf = tf.constant(images, dtype=tf.float32)
    poses_tf = tf.constant(poses, dtype=tf.float32)

    history = {"loss": [], "loss_l1": [], "loss_l2": [], "psnr": []}

    @tf.function
    def train_step(target_img: tf.Tensor, c2w: tf.Tensor):
        with tf.GradientTape() as tape:
            scales = model.get_scales()
            quats = model.get_quats()
            opacities = model.get_opacities()
            colors = model.get_colors()

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
            )

            loss_l1 = tf.reduce_mean(tf.abs(rendered - target_img))
            loss_l2 = tf.reduce_mean(tf.square(rendered - target_img))
            scale_penalty = scale_reg_weight * tf.reduce_mean(scales)
            total_loss = l1_weight * loss_l1 + (1.0 - l1_weight) * loss_l2 + scale_penalty

        grads = tape.gradient(
            total_loss,
            [model.means, model.scales_log, model.quats, model.opacities_logit, model.colors_logit],
        )

        opt_means.apply_gradients([(grads[0], model.means)])
        opt_scales.apply_gradients([(grads[1], model.scales_log)])
        opt_quats.apply_gradients([(grads[2], model.quats)])
        opt_opacities.apply_gradients([(grads[3], model.opacities_logit)])
        opt_colors.apply_gradients([(grads[4], model.colors_logit)])

        return total_loss, loss_l1, loss_l2

    if verbose:
        print(f"Starting 3DGS Optimization ({num_iterations} iterations, {num_views} views)...", flush=True)

    t0 = time.time()
    for step in range(1, num_iterations + 1):
        view_idx = (step - 1) % num_views
        target_img = images_tf[view_idx]
        c2w = poses_tf[view_idx]

        loss_val, l1_val, l2_val = train_step(target_img, c2w)

        loss_float = float(loss_val.numpy())
        l1_float = float(l1_val.numpy())
        l2_float = float(l2_val.numpy())
        psnr_float = float(-10.0 * np.log10(l2_float + 1e-10))

        history["loss"].append(loss_float)
        history["loss_l1"].append(l1_float)
        history["loss_l2"].append(l2_float)
        history["psnr"].append(psnr_float)

        if verbose and (step % log_interval == 0 or step == 1 or step == num_iterations):
            elapsed = time.time() - t0
            iter_speed = step / (elapsed + 1e-8)
            print(
                f"Iter {step:03d}/{num_iterations:03d} | "
                f"Loss: {loss_float:.4f} | "
                f"L1: {l1_float:.4f} | "
                f"PSNR: {psnr_float:.2f} dB | "
                f"Speed: {iter_speed:.1f} it/s",
                flush=True,
            )

    return history
