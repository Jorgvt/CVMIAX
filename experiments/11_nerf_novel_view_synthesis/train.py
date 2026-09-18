"""
Training routines and optimization loops for Tiny NeRF in Keras/TensorFlow.

Optimizes photometric reconstruction loss between rendered rays and ground-truth pixel colors.
"""

import time
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from tensorflow import keras

try:
    from .rays import get_rays_np, get_rays_tf
    from .rendering import render_image, render_rays
except (ImportError, ValueError):
    from rays import get_rays_np, get_rays_tf
    from rendering import render_image, render_rays


def compute_psnr(mse: Union[float, tf.Tensor, np.ndarray]) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR) from Mean Squared Error."""
    if isinstance(mse, tf.Tensor):
        mse = float(mse.numpy())
    if mse <= 1e-10:
        return 100.0
    return float(-10.0 * np.log10(mse))


def train_nerf_step(
    model: keras.Model,
    optimizer: keras.optimizers.Optimizer,
    rays_o: tf.Tensor,
    rays_d: tf.Tensor,
    target_rgb: tf.Tensor,
    near: float = 2.0,
    far: float = 6.0,
    n_samples: int = 64,
    white_bkgd: bool = True,
) -> Tuple[float, float]:
    """
    Perform a single forward and backward optimization step on a batch of rays.

    Args:
        model: NeRF MLP model.
        optimizer: Keras optimizer.
        rays_o: [N_batch, 3] Ray origins.
        rays_d: [N_batch, 3] Ray directions.
        target_rgb: [N_batch, 3] Ground-truth pixel RGB values.
        near: Near depth plane.
        far: Far depth plane.
        n_samples: Samples per ray.
        white_bkgd: White background composite flag.

    Returns:
        (loss, psnr): Scalar MSE loss and PSNR value for this step.
    """
    with tf.GradientTape() as tape:
        outputs = render_rays(
            model=model,
            rays_o=rays_o,
            rays_d=rays_d,
            near=near,
            far=far,
            n_samples=n_samples,
            randomize=True,
            white_bkgd=white_bkgd,
        )
        rendered_rgb = outputs["rgb_map"]
        loss = tf.reduce_mean(tf.square(rendered_rgb - target_rgb))

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    psnr = compute_psnr(loss)
    return float(loss.numpy()), psnr


def train_nerf(
    model: keras.Model,
    images: np.ndarray,
    poses: np.ndarray,
    focal: float,
    val_images: Optional[np.ndarray] = None,
    val_poses: Optional[np.ndarray] = None,
    num_iterations: int = 800,
    batch_size: int = 1024,
    n_samples: int = 64,
    near: float = 2.0,
    far: float = 6.0,
    learning_rate: float = 5e-4,
    eval_every: int = 100,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """
    Train a NeRF model using ray-based stochastic gradient descent.

    Args:
        model: TinyNeRF Keras model.
        images: [N_train, H, W, 3] Ground-truth training images.
        poses: [N_train, 4, 4] Ground-truth camera poses.
        focal: Camera focal length.
        val_images: Optional validation images for tracking generalization.
        val_poses: Optional validation camera poses.
        num_iterations: Total training iterations.
        batch_size: Number of random rays sampled per gradient step.
        n_samples: Quadrature samples along each ray.
        near: Near depth bound.
        far: Far depth bound.
        learning_rate: Initial Adam learning rate.
        eval_every: Frequency of evaluation and logging.
        verbose: Print iteration logs.

    Returns:
        Dictionary with training 'losses', 'train_psnr', 'val_psnr', 'iteration_times'.
    """
    n_images, height, width, _ = images.shape
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=500,
        decay_rate=0.8,
        staircase=True,
    )
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

    # Precompute all rays for all training images for blazing-fast random indexing
    print(f"Precomputing rays for {n_images} training images ({height}x{width})...")
    all_rays_o = []
    all_rays_d = []
    all_target_rgb = []

    for i in range(n_images):
        ro, rd = get_rays_np(height, width, focal, poses[i])
        all_rays_o.append(ro.reshape(-1, 3))
        all_rays_d.append(rd.reshape(-1, 3))
        all_target_rgb.append(images[i].reshape(-1, 3))

    all_rays_o = np.concatenate(all_rays_o, axis=0)        # [N_total_rays, 3]
    all_rays_d = np.concatenate(all_rays_d, axis=0)        # [N_total_rays, 3]
    all_target_rgb = np.concatenate(all_target_rgb, axis=0) # [N_total_rays, 3]
    total_rays = all_rays_o.shape[0]

    history = {
        "iterations": [],
        "train_loss": [],
        "train_psnr": [],
        "val_psnr": [],
        "step_times": [],
    }

    start_time = time.time()

    for step in range(1, num_iterations + 1):
        step_start = time.time()

        # Sample random batch of rays from ray pool
        ray_indices = np.random.randint(0, total_rays, size=batch_size)
        batch_ro = tf.constant(all_rays_o[ray_indices], dtype=tf.float32)
        batch_rd = tf.constant(all_rays_d[ray_indices], dtype=tf.float32)
        batch_rgb = tf.constant(all_target_rgb[ray_indices], dtype=tf.float32)

        loss, psnr = train_nerf_step(
            model=model,
            optimizer=optimizer,
            rays_o=batch_ro,
            rays_d=batch_rd,
            target_rgb=batch_rgb,
            near=near,
            far=far,
            n_samples=n_samples,
            white_bkgd=True,
        )

        step_elapsed = time.time() - step_start
        history["iterations"].append(step)
        history["train_loss"].append(loss)
        history["train_psnr"].append(psnr)
        history["step_times"].append(step_elapsed)

        if step == 1 or step % eval_every == 0 or step == num_iterations:
            val_psnr_val = None
            if val_images is not None and val_poses is not None and len(val_images) > 0:
                # Evaluate on first validation view
                val_rendered = render_image(
                    model=model,
                    height=height,
                    width=width,
                    focal=focal,
                    c2w=val_poses[0],
                    near=near,
                    far=far,
                    n_samples=n_samples,
                )
                val_mse = float(np.mean((val_rendered["rgb"] - val_images[0]) ** 2))
                val_psnr_val = compute_psnr(val_mse)
                history["val_psnr"].append(val_psnr_val)

            if verbose:
                val_str = f" | Val PSNR: {val_psnr_val:.2f} dB" if val_psnr_val is not None else ""
                print(
                    f"Step {step:4d}/{num_iterations} | Loss: {loss:.5f} | "
                    f"Train PSNR: {psnr:.2f} dB{val_str} | ({step_elapsed*1000:.1f} ms/step)"
                )

    total_time = time.time() - start_time
    if verbose:
        print(f"Training completed in {total_time:.1f}s ({total_time/num_iterations*1000:.1f} ms/step average).")

    return history
