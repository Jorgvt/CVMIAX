"""
2D Gaussian Image Fitting (Pedagogical Intuition Warm-Up).

Fits an explicit collection of 2D Gaussian primitives (position, scale, rotation,
opacity, and RGB color) directly to a target 2D image without 3D camera projections.
"""

from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class Gaussian2DModel(tf.keras.Model):
    """
    2D Gaussian Image Splatting representation.
    """

    def __init__(
        self,
        num_gaussians: int = 200,
        height: int = 100,
        width: int = 100,
        name: str = "gaussian_2d_model",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.num_gaussians = num_gaussians
        self.height = height
        self.width = width

        # 1. Learnable 2D Pixel Means (x, y)
        init_means = np.random.uniform(
            low=[0.1 * width, 0.1 * height],
            high=[0.9 * width, 0.9 * height],
            size=(num_gaussians, 2),
        ).astype(np.float32)
        self.means = tf.Variable(init_means, trainable=True, name="means_2d")

        # 2. Learnable Log Scales (sx, sy) -> s = exp(scales_log)
        init_scales = tf.fill([num_gaussians, 2], np.log(10.0).astype(np.float32))
        self.scales_log = tf.Variable(init_scales, trainable=True, name="scales_log_2d")

        # 3. Learnable Rotation Angles theta in radians
        init_thetas = tf.random.uniform([num_gaussians, 1], -np.pi, np.pi)
        self.thetas = tf.Variable(init_thetas, trainable=True, name="thetas_2d")

        # 4. Learnable Logit Opacities -> opacity = sigmoid(opacities_logit)
        init_opacities = tf.zeros([num_gaussians, 1], dtype=tf.float32)
        self.opacities_logit = tf.Variable(init_opacities, trainable=True, name="opacities_logit_2d")

        # 5. Learnable Logit Colors -> rgb = sigmoid(colors_logit)
        init_colors = tf.random.normal([num_gaussians, 3], mean=0.0, stddev=0.5, dtype=tf.float32)
        self.colors_logit = tf.Variable(init_colors, trainable=True, name="colors_logit_2d")

    def get_scales(self) -> tf.Tensor:
        """Return positive scale standard deviations bounded to prevent unbounded expansion."""
        return tf.clip_by_value(tf.exp(self.scales_log), 1.0, float(max(self.height, self.width) / 2.0))

    def get_cov2d(self) -> tf.Tensor:
        """
        Build 2D covariance matrices Sigma = R(theta) * S * S^T * R(theta)^T
        """
        scales = self.get_scales()
        sx, sy = scales[:, 0], scales[:, 1]
        theta = self.thetas[:, 0]

        cos_t = tf.cos(theta)
        sin_t = tf.sin(theta)

        # R = [[cos, -sin], [sin, cos]]
        r0 = tf.stack([cos_t, -sin_t], axis=-1)
        r1 = tf.stack([sin_t, cos_t], axis=-1)
        R = tf.stack([r0, r1], axis=-2)  # [N, 2, 2]

        S = tf.linalg.diag(scales)  # [N, 2, 2]
        M = tf.matmul(R, S)
        return tf.matmul(M, M, transpose_b=True) + 0.3 * tf.eye(2, batch_shape=[self.num_gaussians])

    def get_opacities(self) -> tf.Tensor:
        return tf.sigmoid(self.opacities_logit)

    def get_colors(self) -> tf.Tensor:
        return tf.sigmoid(self.colors_logit)

    @tf.function
    def render(self, bg_color: float = 1.0) -> tf.Tensor:
        """
        Rasterize all 2D Gaussians onto the (height, width) pixel canvas using front-to-back compositing.
        """
        cov2d = self.get_cov2d()
        opacities = self.get_opacities()
        colors = self.get_colors()
        means = self.means

        # Analytical 2x2 matrix inversion
        a = cov2d[:, 0, 0]
        b = cov2d[:, 0, 1]
        c = cov2d[:, 1, 0]
        d = cov2d[:, 1, 1]
        det = a * d - b * c + 1e-7

        inv_cov2d = tf.stack([
            tf.stack([d / det, -b / det], axis=-1),
            tf.stack([-c / det, a / det], axis=-1)
        ], axis=-2)  # [N, 2, 2]

        # Canvas pixel grid
        y_coords = tf.cast(tf.range(self.height), tf.float32)
        x_coords = tf.cast(tf.range(self.width), tf.float32)
        grid_y, grid_x = tf.meshgrid(y_coords, x_coords, indexing="ij")
        pixels = tf.stack([grid_x, grid_y], axis=-1)  # [H, W, 2]

        # Evaluate 2D Gaussian density
        diff = pixels[:, :, None, :] - means[None, None, :, :]  # [H, W, N, 2]
        diff_exp = tf.expand_dims(diff, axis=-2)
        inv_cov_exp = inv_cov2d[None, None, ...]

        prod = tf.matmul(diff_exp, inv_cov_exp)
        power = -0.5 * tf.squeeze(tf.matmul(prod, diff_exp, transpose_b=True), axis=[-2, -1])
        power = tf.clip_by_value(power, -50.0, 0.0)
        gaussian_weights = tf.exp(power)  # [H, W, N]

        alphas = opacities[None, None, :, 0] * gaussian_weights
        alphas = tf.clip_by_value(alphas, 0.0, 0.99)

        # Transmittance along Gaussian index dimension
        alphas_trans = tf.transpose(alphas, [2, 0, 1])  # [N, H, W]
        one_minus_alpha = 1.0 - alphas_trans + 1e-7
        transmittance = tf.math.cumprod(one_minus_alpha, axis=0, exclusive=True)  # [N, H, W]

        weights = transmittance * alphas_trans  # [N, H, W]

        colors_exp = colors[:, None, None, :]  # [N, 1, 1, 3]
        weights_exp = weights[..., None]  # [N, H, W, 1]
        rendered_rgb = tf.reduce_sum(weights_exp * colors_exp, axis=0)  # [H, W, 3]

        total_alpha = tf.reduce_sum(weights, axis=0, keepdims=True)
        total_alpha = tf.transpose(total_alpha, [1, 2, 0])
        rendered_rgb = rendered_rgb + (1.0 - total_alpha) * bg_color

        return tf.clip_by_value(rendered_rgb, 0.0, 1.0)


def train_2d_gaussian_fitting(
    target_image: np.ndarray,
    num_gaussians: int = 250,
    num_iterations: int = 150,
    lr_means: float = 1.5,
    lr_scales: float = 0.02,
    lr_thetas: float = 0.05,
    lr_opacities: float = 0.08,
    lr_colors: float = 0.05,
    log_interval: int = 25,
) -> Tuple[Gaussian2DModel, Dict[str, List[float]]]:
    """
    Train 2D Gaussian primitives to reconstruct a single target image.
    """
    height, width = target_image.shape[0], target_image.shape[1]
    target_tf = tf.constant(target_image, dtype=tf.float32)

    model = Gaussian2DModel(num_gaussians=num_gaussians, height=height, width=width)

    # Initialize Gaussian colors by sampling the target image at initial Gaussian means
    init_x = np.clip(model.means[:, 0].numpy().astype(int), 0, width - 1)
    init_y = np.clip(model.means[:, 1].numpy().astype(int), 0, height - 1)
    sampled_colors = target_image[init_y, init_x]
    eps = 1e-3
    sampled_colors_clipped = np.clip(sampled_colors, eps, 1.0 - eps)
    model.colors_logit.assign(np.log(sampled_colors_clipped / (1.0 - sampled_colors_clipped)))

    opt_means = tf.keras.optimizers.Adam(learning_rate=lr_means)
    opt_scales = tf.keras.optimizers.Adam(learning_rate=lr_scales)
    opt_thetas = tf.keras.optimizers.Adam(learning_rate=lr_thetas)
    opt_opacities = tf.keras.optimizers.Adam(learning_rate=lr_opacities)
    opt_colors = tf.keras.optimizers.Adam(learning_rate=lr_colors)

    history = {"loss": [], "psnr": [], "l1": []}

    @tf.function
    def step_fn():
        with tf.GradientTape() as tape:
            rendered = model.render(bg_color=1.0)
            l1 = tf.reduce_mean(tf.abs(rendered - target_tf))
            l2 = tf.reduce_mean(tf.square(rendered - target_tf))
            loss = 0.8 * l1 + 0.2 * l2

        grads = tape.gradient(
            loss,
            [model.means, model.scales_log, model.thetas, model.opacities_logit, model.colors_logit],
        )
        opt_means.apply_gradients([(grads[0], model.means)])
        opt_scales.apply_gradients([(grads[1], model.scales_log)])
        opt_thetas.apply_gradients([(grads[2], model.thetas)])
        opt_opacities.apply_gradients([(grads[3], model.opacities_logit)])
        opt_colors.apply_gradients([(grads[4], model.colors_logit)])

        # Keep means inside the image canvas bounds
        model.means.assign(tf.clip_by_value(model.means, 0.0, float(max(height, width))))
        return loss, l1, l2

    print(f"Fitting 2D Gaussians ({num_gaussians} primitives, {num_iterations} iterations)...", flush=True)
    for step in range(1, num_iterations + 1):
        loss_val, l1_val, l2_val = step_fn()
        l2_float = float(l2_val.numpy())
        psnr_float = float(-10.0 * np.log10(l2_float + 1e-10))

        history["loss"].append(float(loss_val.numpy()))
        history["l1"].append(float(l1_val.numpy()))
        history["psnr"].append(psnr_float)

        if step % log_interval == 0 or step == 1 or step == num_iterations:
            print(f"2DGS Step {step:03d} | Loss: {loss_val.numpy():.4f} | L1: {l1_val.numpy():.4f} | PSNR: {psnr_float:.2f} dB", flush=True)

    return model, history


def plot_2d_gaussian_fitting(
    target_image: np.ndarray,
    model: Gaussian2DModel,
    history: Dict[str, List[float]],
    save_path: str = None,
) -> plt.Figure:
    """
    Visualize 2D Gaussian image fitting results: Target vs Render vs Gaussian Ellipses.
    """
    rendered = model.render().numpy()
    means = model.means.numpy()
    colors = model.get_colors().numpy()
    scales = model.get_scales().numpy()
    thetas = model.thetas.numpy()[:, 0]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=150)

    # 1. Target Image
    axes[0].imshow(np.clip(target_image, 0.0, 1.0))
    axes[0].set_title("Target Image", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # 2. 2DGS Render
    psnr_final = history["psnr"][-1]
    axes[1].imshow(np.clip(rendered, 0.0, 1.0))
    axes[1].set_title(f"2DGS Render (PSNR: {psnr_final:.1f} dB)", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    # 3. Gaussian Ellipses Representation
    from matplotlib.patches import Ellipse
    axes[2].set_xlim(0, target_image.shape[1])
    axes[2].set_ylim(target_image.shape[0], 0)  # Invert y for image coordinates
    axes[2].set_facecolor("white")
    for i in range(len(means)):
        e = Ellipse(
            xy=(means[i, 0], means[i, 1]),
            width=scales[i, 0] * 2.0,
            height=scales[i, 1] * 2.0,
            angle=np.rad2deg(thetas[i]),
            facecolor=colors[i],
            edgecolor="black",
            linewidth=0.5,
            alpha=0.6,
        )
        axes[2].add_patch(e)
    axes[2].set_title(f"2D Gaussian Primitives (N={len(means)})", fontsize=11, fontweight="bold")
    axes[2].set_aspect("equal")

    # 4. Convergence Curve
    axes[3].plot(history["psnr"], color="#2ca02c", linewidth=2)
    axes[3].set_title("PSNR Convergence (dB)", fontsize=11, fontweight="bold")
    axes[3].set_xlabel("Iteration")
    axes[3].set_ylabel("PSNR (dB)")
    axes[3].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    if save_path:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved 2D Gaussian fitting plot to: {save_path}")

    return fig
