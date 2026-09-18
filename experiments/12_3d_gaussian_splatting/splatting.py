"""
Differentiable EWA Splatting and Front-to-Back Alpha Compositing Engine.

Implements:
1. Camera Coordinate Transformation and Projective Jacobian Calculation
2. Elliptical Weighted Average (EWA) 3D-to-2D Covariance Projection
3. Depth Sorting & Differentiable Point-Based Volume Rendering ("Over" Operator)
"""

from typing import Dict, Tuple
import tensorflow as tf
from models import compute_cov3d


def project_gaussians_to_screen(
    means_3d: tf.Tensor,
    c2w: tf.Tensor,
    focal: float,
    height: int,
    width: int,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Project 3D Gaussian means into camera coordinates and 2D pixel space,
    computing the first-order projective Jacobian matrix J.

    Args:
        means_3d: [N, 3] 3D Gaussian center coordinates in world space.
        c2w: [4, 4] Camera-to-world extrinsic matrix.
        focal: Camera focal length in pixels.
        height: Image height in pixels.
        width: Image width in pixels.

    Returns:
        means_cam: [N, 3] Gaussian centers in camera space (X_c, Y_c, Z_c).
        depths: [N] Positive depth distance in front of camera (-Z_c).
        means_2d: [N, 2] Projected (u, v) pixel coordinates.
        J: [N, 2, 3] Projective camera transformation Jacobian.
    """
    R_c2w = c2w[:3, :3]
    t_c2w = c2w[:3, 3]

    # Transform 3D world coordinates to camera coordinates: X_cam = (X_world - t) @ R_c2w
    means_cam = tf.matmul(means_3d - t_c2w[None, :], R_c2w)
    X_c = means_cam[:, 0]
    Y_c = means_cam[:, 1]
    Z_c = means_cam[:, 2]

    # In standard OpenGL / NeRF coordinates, camera looks along -Z
    depths = -Z_c

    cx = float(width) / 2.0
    cy = float(height) / 2.0

    # Pinhole projection to pixel coordinates (u, v)
    u = focal * (X_c / (depths + 1e-7)) + cx
    v = -focal * (Y_c / (depths + 1e-7)) + cy  # v points downwards
    means_2d = tf.stack([u, v], axis=-1)

    # First-order affine Jacobian J = d(u, v) / d(X_c, Y_c, Z_c)
    inv_z = 1.0 / (depths + 1e-7)
    inv_z2 = inv_z ** 2

    J00 = focal * inv_z
    J01 = tf.zeros_like(inv_z)
    J02 = focal * X_c * inv_z2

    J10 = tf.zeros_like(inv_z)
    J11 = -focal * inv_z
    J12 = -focal * Y_c * inv_z2

    J0 = tf.stack([J00, J01, J02], axis=-1)
    J1 = tf.stack([J10, J11, J12], axis=-1)
    J = tf.stack([J0, J1], axis=-2)  # [N, 2, 3]

    return means_cam, depths, means_2d, J


def compute_cov2d(
    cov3d: tf.Tensor,
    c2w: tf.Tensor,
    J: tf.Tensor,
    low_pass_filter: float = 0.3,
) -> tf.Tensor:
    """
    Project 3D covariance Sigma into 2D image plane covariance Sigma_2D via EWA splatting:
    Sigma_2D = J * W * Sigma_3D * W^T * J^T + nu * I_2

    Args:
        cov3d: [N, 3, 3] 3D covariance matrices in world space.
        c2w: [4, 4] Camera-to-world extrinsic matrix.
        J: [N, 2, 3] Projective Jacobian matrices.
        low_pass_filter: Anti-aliasing regularization added to diagonal (default: 0.3).

    Returns:
        [N, 2, 2] Projected 2D covariance matrices on the image plane.
    """
    R_c2w = c2w[:3, :3]
    R_w2c = tf.transpose(R_c2w)

    # Covariance in camera coordinate frame
    cov_cam = tf.matmul(tf.matmul(R_w2c[None, ...], cov3d), R_w2c[None, ...], transpose_b=True)

    # Projected 2D covariance: J * cov_cam * J^T
    cov2d = tf.matmul(tf.matmul(J, cov_cam), J, transpose_b=True)

    # Add low-pass filter to guarantee invertibility and mitigate aliasing
    num_points = tf.shape(cov3d)[0]
    anti_aliasing = low_pass_filter * tf.eye(2, batch_shape=[num_points])
    return cov2d + anti_aliasing


@tf.function
def render_gaussians(
    means_3d: tf.Tensor,
    scales: tf.Tensor,
    quats: tf.Tensor,
    opacities: tf.Tensor,
    colors: tf.Tensor,
    c2w: tf.Tensor,
    focal: float,
    height: int,
    width: int,
    bg_color: float = 0.0,
) -> tf.Tensor:
    """
    Complete differentiable 3D Gaussian Splatting rendering pipeline for a single viewpoint.

    Args:
        means_3d: [N, 3] Gaussian spatial means.
        scales: [N, 3] Positive scale standard deviations.
        quats: [N, 4] Unit rotation quaternions.
        opacities: [N, 1] Opacities in [0, 1].
        colors: [N, 3] Diffuse RGB colors in [0, 1].
        c2w: [4, 4] Camera-to-world extrinsic matrix.
        focal: Camera focal length in pixels.
        height: Image height.
        width: Image width.
        bg_color: Background color value (1.0 for white, 0.0 for black).

    Returns:
        [height, width, 3] Rendered RGB image.
    """
    # 1. Project 3D centers and compute Jacobian
    _, depths, means_2d, J = project_gaussians_to_screen(means_3d, c2w, focal, height, width)

    # 2. Compute 3D and 2D Covariances
    cov3d = compute_cov3d(scales, quats)
    cov2d = compute_cov2d(cov3d, c2w, J, low_pass_filter=0.3)

    # 3. Sort Gaussians from front-to-back based on camera depth
    sort_idx = tf.argsort(depths, direction="ASCENDING")
    means_2d_sorted = tf.gather(means_2d, sort_idx)
    cov2d_sorted = tf.gather(cov2d, sort_idx)
    opacities_sorted = tf.gather(opacities, sort_idx)
    colors_sorted = tf.gather(colors, sort_idx)

    # 4. Invert 2x2 2D Covariances analytically
    a = cov2d_sorted[:, 0, 0]
    b = cov2d_sorted[:, 0, 1]
    c = cov2d_sorted[:, 1, 0]
    d = cov2d_sorted[:, 1, 1]
    det = a * d - b * c + 1e-7

    inv_cov2d = tf.stack([
        tf.stack([d / det, -b / det], axis=-1),
        tf.stack([-c / det, a / det], axis=-1)
    ], axis=-2)  # [N, 2, 2]

    # 5. Build 2D Pixel Grid (u, v)
    y_coords = tf.cast(tf.range(height), tf.float32)
    x_coords = tf.cast(tf.range(width), tf.float32)
    grid_y, grid_x = tf.meshgrid(y_coords, x_coords, indexing="ij")
    pixels = tf.stack([grid_x, grid_y], axis=-1)  # [H, W, 2]

    # 6. Evaluate 2D Gaussian Radial Density: exp(-0.5 * (p - u)^T * inv_cov2d * (p - u))
    diff = pixels[:, :, None, :] - means_2d_sorted[None, None, :, :]  # [H, W, N, 2]
    diff_exp = tf.expand_dims(diff, axis=-2)  # [H, W, N, 1, 2]
    inv_cov_exp = inv_cov2d[None, None, ...]  # [1, 1, N, 2, 2]

    prod = tf.matmul(diff_exp, inv_cov_exp)  # [H, W, N, 1, 2]
    power = -0.5 * tf.squeeze(tf.matmul(prod, diff_exp, transpose_b=True), axis=[-2, -1])  # [H, W, N]
    power = tf.clip_by_value(power, -50.0, 0.0)
    gaussian_weights = tf.exp(power)  # [H, W, N]

    # 7. Effective Opacity: alpha_i = opacity_i * gaussian_weight_i
    alphas = opacities_sorted[None, None, :, 0] * gaussian_weights  # [H, W, N]
    alphas = tf.clip_by_value(alphas, 0.0, 0.99)

    # 8. Front-to-Back Cumulative Transmittance: T_i = prod_{j < i} (1 - alpha_j)
    alphas_trans = tf.transpose(alphas, [2, 0, 1])  # [N, H, W]
    one_minus_alpha = 1.0 - alphas_trans + 1e-7
    transmittance = tf.math.cumprod(one_minus_alpha, axis=0, exclusive=True)  # [N, H, W]

    # Point-Based Blending Weights
    weights = transmittance * alphas_trans  # [N, H, W]

    # 9. Accumulated Pixel Color: C = sum_i (w_i * c_i)
    colors_exp = colors_sorted[:, None, None, :]  # [N, 1, 1, 3]
    weights_exp = weights[..., None]  # [N, H, W, 1]
    rendered_rgb = tf.reduce_sum(weights_exp * colors_exp, axis=0)  # [H, W, 3]

    # 10. Blend White/Black Background
    total_alpha = tf.reduce_sum(weights, axis=0, keepdims=True)  # [1, H, W]
    total_alpha = tf.transpose(total_alpha, [1, 2, 0])  # [H, W, 1]
    rendered_rgb = rendered_rgb + (1.0 - total_alpha) * bg_color

    return tf.clip_by_value(rendered_rgb, 0.0, 1.0)
