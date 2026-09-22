"""
Data Augmentation policies and visualization utilities for Experiment 16.

Implements modular Keras 3 augmentation pipelines:
1. Baseline (Identity / None)
2. Geometric Invariance (Horizontal Flip, Translation, Subtle Zoom)
3. Photometric / Color Invariance (Brightness, Contrast)
4. Combined Standard Augmentation (Geometric + Photometric)
5. Mixup Regularization (Vicinal Risk Minimization / VRM)
"""

from typing import Dict, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import layers, ops


class MixupLayer(layers.Layer):
    """
    Implements Mixup data augmentation (Zhang et al., 2017).
    Constructs virtual training examples:
        x_tilde = lambda * x_i + (1 - lambda) * x_j
        y_tilde = lambda * y_i + (1 - lambda) * y_j
    where lambda ~ Beta(alpha, alpha).
    """

    def __init__(self, alpha: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.alpha = float(alpha)

    def call(self, images, labels, training=True):
        if not training:
            return images, labels

        batch_size = ops.shape(images)[0]
        # Sample lambda from Beta(alpha, alpha) via ratio of Gamma variables
        gamma_1 = keras.random.gamma((batch_size, 1, 1, 1), self.alpha)
        gamma_2 = keras.random.gamma((batch_size, 1, 1, 1), self.alpha)
        lam = gamma_1 / (gamma_1 + gamma_2 + 1e-8)

        # Generate permutation indices
        indices = keras.random.shuffle(ops.arange(batch_size))
        mixed_images = lam * images + (1.0 - lam) * ops.take(images, indices, axis=0)

        lam_labels = ops.reshape(lam, (batch_size, 1))
        mixed_labels = lam_labels * labels + (1.0 - lam_labels) * ops.take(labels, indices, axis=0)

        return mixed_images, mixed_labels

    def get_config(self):
        config = super().get_config()
        config.update({"alpha": self.alpha})
        return config


def get_augmentation_model(policy: str = "none") -> layers.Layer:
    """
    Builds a Keras preprocessing/augmentation block.

    Args:
        policy: One of ['none', 'geometric', 'photometric', 'combined']
    """
    policy = policy.lower()

    if policy == "none":
        return layers.Identity(name="aug_none")

    elif policy == "geometric":
        return keras.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomTranslation(
                    height_factor=0.08, width_factor=0.08, fill_mode="nearest"
                ),
                layers.RandomZoom(height_factor=(-0.05, 0.05), fill_mode="nearest"),
            ],
            name="aug_geometric",
        )

    elif policy == "photometric":
        return keras.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomBrightness(factor=0.1, value_range=(0.0, 1.0)),
                layers.RandomContrast(factor=0.1),
            ],
            name="aug_photometric",
        )

    elif policy == "combined":
        return keras.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomTranslation(
                    height_factor=0.08, width_factor=0.08, fill_mode="nearest"
                ),
                layers.RandomZoom(height_factor=(-0.05, 0.05), fill_mode="nearest"),
                layers.RandomBrightness(factor=0.1, value_range=(0.0, 1.0)),
                layers.RandomContrast(factor=0.1),
            ],
            name="aug_combined",
        )

    else:
        raise ValueError(
            f"Unknown augmentation policy: {policy}. Choose from ['none', 'geometric', 'photometric', 'combined']."
        )


def plot_augmentation_gallery(
    sample_images: np.ndarray,
    class_names: list,
    sample_labels: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    num_samples: int = 4,
):
    """
    Generates a publication-grade comparison gallery showing the visual effect
    of different augmentation policies across multiple sample images.
    """
    policies = ["Original", "Geometric", "Photometric", "Combined", "Mixup (alpha=0.2)"]
    geom_model = get_augmentation_model("geometric")
    photo_model = get_augmentation_model("photometric")
    comb_model = get_augmentation_model("combined")
    mixup_layer = MixupLayer(alpha=0.2)

    fig, axes = plt.subplots(
        num_samples, len(policies), figsize=(14, 2.6 * num_samples), dpi=200
    )

    if num_samples == 1:
        axes = np.expand_dims(axes, 0)

    # Prepare mixup batch
    batch_tensor = ops.convert_to_tensor(sample_images[:num_samples], dtype="float32")
    if sample_labels is not None:
        lbl_tensor = ops.convert_to_tensor(sample_labels[:num_samples], dtype="float32")
    else:
        lbl_tensor = ops.zeros((num_samples, len(class_names)))
    mix_imgs, mix_lbls = mixup_layer(batch_tensor, lbl_tensor, training=True)
    mix_imgs_np = ops.convert_to_numpy(mix_imgs)

    for i in range(num_samples):
        img = sample_images[i : i + 1]

        # Policy transforms
        img_orig = np.clip(img[0], 0.0, 1.0)
        img_geom = np.clip(geom_model(img, training=True).numpy()[0], 0.0, 1.0)
        img_photo = np.clip(photo_model(img, training=True).numpy()[0], 0.0, 1.0)
        img_comb = np.clip(comb_model(img, training=True).numpy()[0], 0.0, 1.0)
        img_mix = np.clip(mix_imgs_np[i], 0.0, 1.0)

        views = [img_orig, img_geom, img_photo, img_comb, img_mix]

        for j, (v, title) in enumerate(zip(views, policies)):
            ax = axes[i, j]
            ax.imshow(v)
            if i == 0:
                ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
            if j == 0 and sample_labels is not None:
                cls_idx = np.argmax(sample_labels[i])
                ax.set_ylabel(
                    f"Sample {i+1}\n({class_names[cls_idx]})",
                    fontsize=10,
                    fontweight="bold",
                )
            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle(
        "Data Augmentation Policies: Inducing Invariances & Vicinal Distributions",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()

    if save_path:
        if os.path.dirname(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
        print(f" Saved augmentation gallery to: {save_path}")
    return fig

