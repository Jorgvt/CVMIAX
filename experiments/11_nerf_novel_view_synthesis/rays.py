"""
Camera ray casting and 3D stratified spatial sampling utilities for NeRF.

Translates 2D image plane pixel coordinates through camera extrinsics into 3D ray origins
and directions, and handles stratified point sampling along rays.
"""

from typing import Tuple, Union
import numpy as np
import tensorflow as tf


def get_rays_np(
    height: int,
    width: int,
    focal: float,
    c2w: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate ray origins and direction vectors for all pixels of a pinhole camera (NumPy).

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        focal: Camera focal length in pixels.
        c2w: Camera-to-world 4x4 or 3x4 extrinsic transformation matrix.

    Returns:
        rays_o: [H, W, 3] Ray origins in world coordinates.
        rays_d: [H, W, 3] Ray unit/view directions in world coordinates.
    """
    cx = width / 2.0
    cy = height / 2.0
    u, v = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
        indexing="xy",
    )

    # OpenGL / NeRF camera convention: x right, y up, z backwards (-z viewing direction)
    dirs = np.stack([(u - cx) / focal, -(v - cy) / focal, -np.ones_like(u)], axis=-1)

    # Transform ray directions to world frame: dirs_world = dirs @ R^T
    rot = c2w[:3, :3]
    rays_d = np.sum(dirs[..., None, :] * rot, axis=-1)

    # All rays for a single perspective pinhole camera originate from the camera center
    rays_o = np.broadcast_to(c2w[:3, 3], rays_d.shape).copy()

    return rays_o.astype(np.float32), rays_d.astype(np.float32)


@tf.function
def get_rays_tf(
    height: int,
    width: int,
    focal: float,
    c2w: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Generate ray origins and directions for all pixels in TensorFlow.

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        focal: Camera focal length in pixels.
        c2w: Camera-to-world transformation matrix [4, 4] or [3, 4].

    Returns:
        rays_o: [H, W, 3] Ray origins.
        rays_d: [H, W, 3] Ray direction vectors.
    """
    cx = tf.cast(width, tf.float32) / 2.0
    cy = tf.cast(height, tf.float32) / 2.0
    u = tf.range(width, dtype=tf.float32)
    v = tf.range(height, dtype=tf.float32)
    uu, vv = tf.meshgrid(u, v, indexing="xy")

    dirs = tf.stack([(uu - cx) / focal, -(vv - cy) / focal, -tf.ones_like(uu)], axis=-1)
    rot = c2w[:3, :3]
    rays_d = tf.reduce_sum(dirs[..., None, :] * rot, axis=-1)
    rays_o = tf.broadcast_to(c2w[:3, 3], tf.shape(rays_d))

    return rays_o, rays_d


def sample_stratified_points(
    rays_o: tf.Tensor,
    rays_d: tf.Tensor,
    near: float,
    far: float,
    n_samples: int,
    randomize: bool = True,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Sample 3D points along rays using stratified sampling between near and far depth planes.

    For continuous 3D representation during training, samples are jittered uniformly
    within their discrete bin intervals:
        t_i ~ U[ t_n + (i-1)/N * (t_f - t_n), t_n + i/N * (t_f - t_n) ]
        x_i = o + t_i * d

    Args:
        rays_o: [N_rays, 3] Ray origins.
        rays_d: [N_rays, 3] Ray directions.
        near: Near clipping depth plane.
        far: Far clipping depth plane.
        n_samples: Number of sample points per ray.
        randomize: Whether to add uniform jitter within bin intervals.

    Returns:
        pts: [N_rays, N_samples, 3] Sampled 3D point coordinates in world space.
        z_vals: [N_rays, N_samples] Sampled depth values along rays.
    """
    # Linear partition between near and far
    t_vals = tf.linspace(0.0, 1.0, n_samples)
    z_vals = near * (1.0 - t_vals) + far * t_vals
    z_vals = tf.broadcast_to(z_vals, [tf.shape(rays_o)[0], n_samples])

    if randomize:
        # Compute bin step sizes and add uniform noise
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = tf.concat([mids, z_vals[..., -1:]], axis=-1)
        lower = tf.concat([z_vals[..., :1], mids], axis=-1)
        t_rand = tf.random.uniform(tf.shape(z_vals))
        z_vals = lower + (upper - lower) * t_rand

    # 3D points: pts = o[:, None, :] + z_vals[:, :, None] * d[:, None, :]
    pts = rays_o[:, None, :] + z_vals[..., None] * rays_d[:, None, :]
    return pts, z_vals
