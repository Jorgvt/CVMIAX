"""
Evaluation metrics, novel-view trajectory generators, and test benchmarks for NeRF.

Computes PSNR, SSIM, L1 photometric errors, and synthesizes 360-degree orbital camera paths.
"""

from typing import Dict, List, Tuple, Union
import numpy as np
import tensorflow as tf
from tensorflow import keras

try:
    from .dataset import pose_spherical
    from .rendering import render_image
    from .train import compute_psnr
except (ImportError, ValueError):
    from dataset import pose_spherical
    from rendering import render_image
    from train import compute_psnr


def compute_metrics(
    pred_rgb: np.ndarray,
    gt_rgb: np.ndarray,
) -> Dict[str, float]:
    """
    Compute standard image quality assessment metrics: MSE, PSNR, SSIM, L1.

    Args:
        pred_rgb: [H, W, 3] Predicted RGB image in [0, 1].
        gt_rgb: [H, W, 3] Ground-truth RGB image in [0, 1].

    Returns:
        Dictionary containing 'mse', 'psnr', 'ssim', and 'l1'.
    """
    mse = float(np.mean((pred_rgb - gt_rgb) ** 2))
    l1 = float(np.mean(np.abs(pred_rgb - gt_rgb)))
    psnr = compute_psnr(mse)

    # Compute Structural Similarity (SSIM) in TensorFlow
    pred_tf = tf.convert_to_tensor(pred_rgb[None, ...], dtype=tf.float32)
    gt_tf = tf.convert_to_tensor(gt_rgb[None, ...], dtype=tf.float32)
    ssim_val = float(tf.image.ssim(pred_tf, gt_tf, max_val=1.0).numpy()[0])

    return {
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim_val,
        "l1": l1,
    }


def evaluate_dataset(
    model: keras.Model,
    images: np.ndarray,
    poses: np.ndarray,
    focal: float,
    near: float = 2.0,
    far: float = 6.0,
    n_samples: int = 64,
) -> Tuple[Dict[str, float], List[Dict[str, np.ndarray]]]:
    """
    Evaluate trained NeRF model across a dataset of test views.

    Args:
        model: Trained NeRF MLP.
        images: [N_test, H, W, 3] Test ground-truth images.
        poses: [N_test, 4, 4] Test camera poses.
        focal: Focal length in pixels.
        near: Near depth bound.
        far: Far depth bound.
        n_samples: Number of quadrature samples along each ray.

    Returns:
        (aggregated_metrics, rendered_results_list)
    """
    n_test, height, width, _ = images.shape
    all_metrics = {"mse": [], "psnr": [], "ssim": [], "l1": []}
    rendered_results = []

    for i in range(n_test):
        rendered = render_image(
            model=model,
            height=height,
            width=width,
            focal=focal,
            c2w=poses[i],
            near=near,
            far=far,
            n_samples=n_samples,
        )
        metrics = compute_metrics(rendered["rgb"], images[i])
        for k, v in metrics.items():
            all_metrics[k].append(v)

        rendered["gt_rgb"] = images[i]
        rendered["metrics"] = metrics
        rendered_results.append(rendered)

    mean_metrics = {k: float(np.mean(v)) for k, v in all_metrics.items()}
    return mean_metrics, rendered_results


def generate_novel_view_poses(
    n_frames: int = 36,
    phi_deg: float = -30.0,
    radius: float = 4.0,
) -> np.ndarray:
    """
    Generate smooth 360-degree orbital camera poses for novel view synthesis animation.

    Args:
        n_frames: Number of continuous camera viewpoints around the azimuth circle.
        phi_deg: Camera elevation angle in degrees.
        radius: Distance from camera to scene origin.

    Returns:
        [n_frames, 4, 4] Array of Camera-to-World (c2w) transformation matrices.
    """
    thetas = np.linspace(0.0, 360.0, n_frames, endpoint=False)
    poses = [pose_spherical(th, phi_deg, radius) for th in thetas]
    return np.stack(poses, axis=0)


def render_novel_trajectory(
    model: keras.Model,
    height: int,
    width: int,
    focal: float,
    novel_poses: np.ndarray,
    near: float = 2.0,
    far: float = 6.0,
    n_samples: int = 64,
) -> List[Dict[str, np.ndarray]]:
    """
    Render a sequence of novel viewpoints along an orbital trajectory.

    Returns:
        List of rendered frames containing 'rgb', 'depth', and 'acc'.
    """
    rendered_frames = []
    for pose in novel_poses:
        frame = render_image(
            model=model,
            height=height,
            width=width,
            focal=focal,
            c2w=pose,
            near=near,
            far=far,
            n_samples=n_samples,
        )
        rendered_frames.append(frame)
    return rendered_frames
