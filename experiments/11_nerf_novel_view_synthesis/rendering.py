"""
Differentiable Volumetric Rendering Engine for Neural Radiance Fields.

Implements the classical discrete numerical approximation of the volume rendering integral
(Max 1995, Mildenhall et al. 2020) mapping 3D density and radiance predictions to 2D pixel colors,
depth maps, and optical accumulation.
"""

from typing import Dict, Tuple, Union
import numpy as np
import tensorflow as tf
from tensorflow import keras

try:
    from .rays import sample_stratified_points, get_rays_tf
except (ImportError, ValueError):
    from rays import sample_stratified_points, get_rays_tf


def raw2outputs(
    raw: tf.Tensor,
    z_vals: tf.Tensor,
    rays_d: tf.Tensor,
    white_bkgd: bool = True,
) -> Dict[str, tf.Tensor]:
    """
    Transform raw model predictions (RGB emission + volume density) into accumulated ray outputs.

    Discrete Volumetric Rendering Formulation:
        delta_i = (t_{i+1} - t_i) * ||d||_2
        alpha_i = 1 - exp(-sigma_i * delta_i)
        T_i = prod_{j=1}^{i-1} (1 - alpha_j)
        w_i = T_i * alpha_i
        C(r) = sum_i w_i * c_i
        D(r) = sum_i w_i * t_i

    Args:
        raw: [N_rays, N_samples, 4] Tensor of [R, G, B, sigma] predictions.
        z_vals: [N_rays, N_samples] Depth partition values along each ray.
        rays_d: [N_rays, 3] Ray direction vectors.
        white_bkgd: If True, composite rendered foreground onto a pure white background.

    Returns:
        Dictionary containing:
            - 'rgb_map': [N_rays, 3] Estimated RGB color of the ray.
            - 'depth_map': [N_rays] Estimated distance / depth along the ray.
            - 'acc_map': [N_rays] Accumulated optical transmittance / foreground opacity.
            - 'weights': [N_rays, N_samples] Ray integration weights w_i.
    """
    rgb = raw[..., :3]      # [N_rays, N_samples, 3]
    sigma = raw[..., 3]    # [N_rays, N_samples]

    # Compute step intervals between adjacent sample points: delta_i = z_{i+1} - z_i
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    # For the last sample point, pad with a large distance (1e10)
    dists = tf.concat([dists, tf.broadcast_to([1e10], tf.shape(dists[..., :1]))], axis=-1)

    # Scale step intervals by ray direction magnitude to obtain true Euclidean distance
    ray_dir_norm = tf.linalg.norm(rays_d[..., None, :], axis=-1)
    dists = dists * ray_dir_norm

    # Discrete opacity alpha_i = 1 - exp(-sigma_i * delta_i)
    alpha = 1.0 - tf.exp(-sigma * dists)

    # Cumulative transmittance T_i = prod_{j < i} (1 - alpha_j)
    # cumprod with exclusive=True ensures T_1 = 1
    transmittance = tf.math.cumprod(1.0 - alpha + 1e-10, axis=-1, exclusive=True)

    # Quadrature ray integration weights w_i = T_i * alpha_i
    weights = alpha * transmittance

    # Integrate ray color: C(r) = sum_i w_i * c_i
    rgb_map = tf.reduce_sum(weights[..., None] * rgb, axis=-2)

    # Integrate expected depth: D(r) = sum_i w_i * z_i
    depth_map = tf.reduce_sum(weights * z_vals, axis=-1)

    # Accumulated foreground opacity
    acc_map = tf.reduce_sum(weights, axis=-1)

    # Composite onto white background if requested (standard for synthetic objects)
    if white_bkgd:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    return {
        "rgb_map": rgb_map,
        "depth_map": depth_map,
        "acc_map": acc_map,
        "weights": weights,
    }


def render_rays(
    model: keras.Model,
    rays_o: tf.Tensor,
    rays_d: tf.Tensor,
    near: float = 2.0,
    far: float = 6.0,
    n_samples: int = 64,
    randomize: bool = False,
    white_bkgd: bool = True,
) -> Dict[str, tf.Tensor]:
    """
    Execute end-to-end ray rendering: stratified 3D sampling -> MLP query -> volume integration.

    Args:
        model: NeRF Keras Model predicting [R, G, B, sigma] from 3D coordinates.
        rays_o: [N_rays, 3] Ray origins.
        rays_d: [N_rays, 3] Ray directions.
        near: Near depth clipping plane.
        far: Far depth clipping plane.
        n_samples: Number of quadrature samples along each ray.
        randomize: Whether to apply stratified jitter (True during training, False for evaluation).
        white_bkgd: Composite onto white background.

    Returns:
        Rendered output dict with 'rgb_map', 'depth_map', 'acc_map', and 'weights'.
    """
    # 1. Sample 3D coordinates along rays
    pts, z_vals = sample_stratified_points(
        rays_o=rays_o,
        rays_d=rays_d,
        near=near,
        far=far,
        n_samples=n_samples,
        randomize=randomize,
    )

    # 2. Query NeRF MLP at 3D spatial points: flatten to [N_rays * N_samples, 3]
    pts_flat = tf.reshape(pts, [-1, 3])
    raw_flat = model(pts_flat, training=randomize)
    raw = tf.reshape(raw_flat, tf.concat([tf.shape(pts)[:2], [4]], axis=0))

    # 3. Integrate volume rendering equation
    outputs = raw2outputs(raw=raw, z_vals=z_vals, rays_d=rays_d, white_bkgd=white_bkgd)
    return outputs


def render_image(
    model: keras.Model,
    height: int,
    width: int,
    focal: float,
    c2w: Union[np.ndarray, tf.Tensor],
    near: float = 2.0,
    far: float = 6.0,
    n_samples: int = 64,
    chunk_size: int = 4096,
    white_bkgd: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Render a complete 2D image frame by chunking rays to avoid GPU/memory OOM.

    Args:
        model: Trained NeRF MLP.
        height: Image height.
        width: Image width.
        focal: Focal length.
        c2w: Camera pose matrix.
        near: Near clipping distance.
        far: Far clipping distance.
        n_samples: Number of samples per ray.
        chunk_size: Number of rays rendered per batch chunk.
        white_bkgd: White background composite flag.

    Returns:
        Dictionary with rendered 'rgb' [H, W, 3], 'depth' [H, W], and 'acc' [H, W] arrays.
    """
    if not isinstance(c2w, tf.Tensor):
        c2w = tf.constant(c2w, dtype=tf.float32)

    rays_o, rays_d = get_rays_tf(height=height, width=width, focal=focal, c2w=c2w)
    rays_o_flat = tf.reshape(rays_o, [-1, 3])
    rays_d_flat = tf.reshape(rays_d, [-1, 3])

    n_rays = tf.shape(rays_o_flat)[0]
    rgb_chunks = []
    depth_chunks = []
    acc_chunks = []

    for i in range(0, n_rays, chunk_size):
        chunk_o = rays_o_flat[i : i + chunk_size]
        chunk_d = rays_d_flat[i : i + chunk_size]

        outputs = render_rays(
            model=model,
            rays_o=chunk_o,
            rays_d=chunk_d,
            near=near,
            far=far,
            n_samples=n_samples,
            randomize=False,
            white_bkgd=white_bkgd,
        )
        rgb_chunks.append(outputs["rgb_map"].numpy())
        depth_chunks.append(outputs["depth_map"].numpy())
        acc_chunks.append(outputs["acc_map"].numpy())

    rgb_img = np.concatenate(rgb_chunks, axis=0).reshape(height, width, 3)
    depth_img = np.concatenate(depth_chunks, axis=0).reshape(height, width)
    acc_img = np.concatenate(acc_chunks, axis=0).reshape(height, width)

    return {
        "rgb": np.clip(rgb_img, 0.0, 1.0),
        "depth": depth_img,
        "acc": acc_img,
    }
