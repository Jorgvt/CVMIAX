"""
3D Gaussian Splatting model representation and covariance utilities.

Defines the explicit 3D Gaussian scene parameters:
- 3D spatial means (positions) mu in R^3
- 3D covariance Sigma = R * S * S^T * R^T via log-scale s in R^3 and quaternion q in H
- Opacity sigma in [0, 1] via logit
- Diffuse RGB colors c in [0, 1]^3 via logit
"""

import numpy as np
import tensorflow as tf


def quaternion_to_rotation_matrix(quats: tf.Tensor) -> tf.Tensor:
    """
    Convert normalized unit quaternions [w, x, y, z] to 3x3 rotation matrices.

    Args:
        quats: Tensor of shape [N, 4] containing (w, x, y, z).

    Returns:
        Tensor of shape [N, 3, 3] representing orthonormal rotation matrices.
    """
    q = tf.math.l2_normalize(quats, axis=-1)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    r00 = 1.0 - 2.0 * (y**2 + z**2)
    r01 = 2.0 * (x * y - w * z)
    r02 = 2.0 * (x * z + w * y)

    r10 = 2.0 * (x * y + w * z)
    r11 = 1.0 - 2.0 * (x**2 + z**2)
    r12 = 2.0 * (y * z - w * x)

    r20 = 2.0 * (x * z - w * y)
    r21 = 2.0 * (y * z + w * x)
    r22 = 1.0 - 2.0 * (x**2 + y**2)

    r0 = tf.stack([r00, r01, r02], axis=-1)
    r1 = tf.stack([r10, r11, r12], axis=-1)
    r2 = tf.stack([r20, r21, r22], axis=-1)

    return tf.stack([r0, r1, r2], axis=-2)


def compute_cov3d(scales: tf.Tensor, quats: tf.Tensor) -> tf.Tensor:
    """
    Compute 3D covariance matrices Sigma = R * S * S^T * R^T from scale vectors and quaternions.

    Args:
        scales: Tensor of shape [N, 3] representing positive standard deviations along principal axes.
        quats: Tensor of shape [N, 4] representing unit quaternions (w, x, y, z).

    Returns:
        Tensor of shape [N, 3, 3] symmetric positive semi-definite 3D covariance matrices.
    """
    R = quaternion_to_rotation_matrix(quats)  # [N, 3, 3]
    S = tf.linalg.diag(scales)  # [N, 3, 3]
    M = tf.matmul(R, S)  # [N, 3, 3]
    return tf.matmul(M, M, transpose_b=True)  # [N, 3, 3]


class GaussianModel(tf.keras.Model):
    """
    Differentiable 3D Gaussian Splatting scene model.
    """

    def __init__(
        self,
        num_gaussians: int = 500,
        spatial_bound: float = 1.0,
        initial_log_scale: float = -3.0,
        name: str = "gaussian_splatting_model",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.num_gaussians = num_gaussians
        self.spatial_bound = spatial_bound

        # 1. Learnable 3D Means (Positions) [N, 3]
        initial_means = tf.random.uniform(
            [num_gaussians, 3],
            -spatial_bound,
            spatial_bound,
            dtype=tf.float32,
        )
        self.means = tf.Variable(initial_means, trainable=True, name="means")

        # 2. Learnable Log Scales [N, 3] -> s = exp(scales_log)
        initial_scales = tf.fill([num_gaussians, 3], initial_log_scale)
        self.scales_log = tf.Variable(initial_scales, trainable=True, name="scales_log")

        # 3. Learnable Quaternions [N, 4] -> q_norm = l2_normalize(quats)
        initial_quats = tf.random.normal([num_gaussians, 4], stddev=1.0)
        self.quats = tf.Variable(initial_quats, trainable=True, name="quats")

        # 4. Learnable Logit Opacities [N, 1] -> opacity = sigmoid(opacities_logit)
        initial_opacities = tf.zeros([num_gaussians, 1], dtype=tf.float32)
        self.opacities_logit = tf.Variable(initial_opacities, trainable=True, name="opacities_logit")

        # 5. Learnable Logit Colors [N, 3] -> rgb = sigmoid(colors_logit)
        initial_colors = tf.random.normal([num_gaussians, 3], mean=0.0, stddev=0.5, dtype=tf.float32)
        self.colors_logit = tf.Variable(initial_colors, trainable=True, name="colors_logit")

    def initialize_from_points(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        initial_log_scale: float = -3.5,
        initial_logit_opacity: float = 1.0,
    ):
        """
        Reinitialize Gaussian model parameters directly on a surface point cloud.
        """
        n = len(points)
        self.num_gaussians = n
        self.means.assign(points.astype(np.float32))
        self.scales_log.assign(tf.fill([n, 3], initial_log_scale))

        init_quats = np.zeros((n, 4), dtype=np.float32)
        init_quats[:, 0] = 1.0  # identity quaternion
        self.quats.assign(init_quats)

        self.opacities_logit.assign(tf.fill([n, 1], initial_logit_opacity))

        # Initialize colors from sampled RGB
        eps = 1e-3
        c_clipped = np.clip(colors, eps, 1.0 - eps)
        colors_logit_init = np.log(c_clipped / (1.0 - c_clipped))
        self.colors_logit.assign(colors_logit_init.astype(np.float32))

    def get_scales(self) -> tf.Tensor:
        """Return positive scale standard deviations bounded to prevent blurry floaters."""
        return tf.clip_by_value(tf.exp(self.scales_log), 0.005, 0.08)

    def get_quats(self) -> tf.Tensor:
        """Return normalized unit quaternions."""
        return tf.math.l2_normalize(self.quats, axis=-1)

    def get_opacities(self) -> tf.Tensor:
        """Return opacities bounded in [0, 1]."""
        return tf.sigmoid(self.opacities_logit)

    def get_colors(self) -> tf.Tensor:
        """Return RGB colors bounded in [0, 1]."""
        return tf.sigmoid(self.colors_logit)

    def get_cov3d(self) -> tf.Tensor:
        """Return 3D covariance matrices Sigma."""
        return compute_cov3d(self.get_scales(), self.get_quats())

    def get_trainable_param_groups(self):
        """Return dictionary of parameter references for separate learning rates."""
        return {
            "means": self.means,
            "scales_log": self.scales_log,
            "quats": self.quats,
            "opacities_logit": self.opacities_logit,
            "colors_logit": self.colors_logit,
        }

