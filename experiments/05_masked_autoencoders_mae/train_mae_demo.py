"""
Enhanced Vision Transformer Masked Autoencoder (ViT-MAE) Training on CIFAR-10.

Key Improvements:
1. Increased training epochs & dataset scale (using 25,000+ CIFAR-10 images).
2. Cosine Decay Learning Rate Schedule with Warmup.
3. Per-Patch Target Normalization (He et al., 2022):
   Normalizing each patch by its local mean and std (x_norm = (x - μ) / σ) forces the model
   to predict sharp contrast and structural textures rather than blurred average colors.
4. Loss convergence curve tracking.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras

from mae_utils import patchify, unpatchify, random_masking
from models import MaskedAutoencoderViT


def normalize_patches(patches, eps=1e-6):
    """
    Applies per-patch normalization (He et al., CVPR 2022):
    Computes mean and variance across pixels within each patch and normalizes.
    
    Returns:
        norm_patches: Normalized patches of shape (B, N, P*P*C).
        mean: (B, N, 1)
        std: (B, N, 1)
    """
    mean = np.mean(patches, axis=-1, keepdims=True)
    var = np.var(patches, axis=-1, keepdims=True)
    std = np.sqrt(var + eps)
    norm_patches = (patches - mean) / std
    return norm_patches, mean, std


def unnormalize_patches(norm_patches, mean, std):
    """Restores pixel scale from normalized patches."""
    return norm_patches * std + mean


def run_mae_training_demo(
    num_samples=25000,
    epochs=15,
    batch_size=128,
    mask_ratio=0.75,
    output_dir="figures",
):
    os.makedirs(output_dir, exist_ok=True)
    print("\n===================================================================")
    print(f" Enhanced ViT-MAE Training: {num_samples} images, {epochs} epochs, {mask_ratio*100:.0f}% mask")
    print("===================================================================")

    # 1. Load data
    (x_train, _), (x_test, _) = keras.datasets.cifar10.load_data()
    x_train = x_train[:num_samples].astype("float32") / 255.0
    x_test = x_test[:1000].astype("float32") / 255.0

    patch_size = 4
    image_shape = (32, 32, 3)

    # 2. Build ViT-MAE
    mae_model = MaskedAutoencoderViT(
        image_shape=image_shape,
        patch_size=patch_size,
        enc_dim=128,
        enc_depth=4,
        enc_heads=4,
        dec_dim=64,
        dec_depth=2,
        dec_heads=4,
        mlp_ratio=4,
        mask_ratio=mask_ratio,
    )

    # Cosine learning rate schedule
    total_steps = (len(x_train) // batch_size) * epochs
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=2e-3,
        decay_steps=total_steps,
        alpha=0.05,
    )
    optimizer = keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=0.02)

    # 3. Training Loop with Per-Patch Target Normalization
    num_batches = len(x_train) // batch_size
    train_losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        indices = np.random.permutation(len(x_train))
        x_train_shuffled = x_train[indices]

        for b_idx in range(num_batches):
            batch_imgs = x_train_shuffled[b_idx * batch_size:(b_idx + 1) * batch_size]
            patches_arr = patchify(batch_imgs, patch_size=patch_size)
            norm_patches, p_mean, p_std = normalize_patches(patches_arr)

            vis_patches, mask, restore_indices, shuffle_indices = random_masking(
                norm_patches, mask_ratio=mask_ratio
            )

            with tf.GradientTape() as tape:
                pred_norm_patches = mae_model(
                    (norm_patches, shuffle_indices, restore_indices),
                    training=True,
                )
                loss = mae_model.compute_mae_loss(norm_patches, pred_norm_patches, mask)

            grads = tape.gradient(loss, mae_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, mae_model.trainable_variables))
            epoch_loss += float(loss.numpy())

        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1:2d}/{epochs} | Masked Patch Normalized MSE: {avg_loss:.5f}")

    # 4. Plot Loss Curve
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(1, epochs + 1), train_losses, "o-", color="#1f77b4", linewidth=2.2, label="MAE Inpainting Loss (Normalized MSE)")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Masked Patch Reconstruction Loss", fontsize=12)
    ax.set_title("ViT-MAE Pre-training Loss on CIFAR-10 (75% Masking)", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=11)
    plt.tight_layout()
    curve_path = os.path.join(output_dir, "mae_training_loss_curve.png")
    fig.savefig(curve_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {curve_path}")

    # 5. Qualitative Test Set Inpainting Evaluation
    print("Generating High-Quality Qualitative Test Inpainting Visualizations...")
    sample_indices = [3, 12, 19, 27, 45, 88]
    test_batch = x_test[sample_indices]
    test_patches = patchify(test_batch, patch_size=patch_size)
    norm_test_patches, t_mean, t_std = normalize_patches(test_patches)

    vis_p, mask_test, restore_idx_test, shuffle_idx_test = random_masking(
        norm_test_patches, mask_ratio=mask_ratio, seed=42
    )

    pred_norm_test = mae_model(
        (norm_test_patches, shuffle_idx_test, restore_idx_test),
        training=False,
    ).numpy()

    # Unnormalize predicted patches back to pixel RGB space
    pred_test_patches = unnormalize_patches(pred_norm_test, t_mean, t_std)

    # Create 75% masked view
    masked_view_patches = test_patches.copy()
    mask_patch_val = np.full(patch_size * patch_size * 3, 0.08, dtype=np.float32)
    for b in range(len(sample_indices)):
        for i in range(mask_test.shape[1]):
            if mask_test[b, i] == 1.0:
                masked_view_patches[b, i] = mask_patch_val
    masked_test_imgs = unpatchify(masked_view_patches, image_shape=image_shape, patch_size=patch_size)

    # Inpainted composite (original visible + model predicted masked)
    composite_patches = test_patches.copy()
    for b in range(len(sample_indices)):
        for i in range(mask_test.shape[1]):
            if mask_test[b, i] == 1.0:
                composite_patches[b, i] = pred_test_patches[b, i]
    composite_imgs = np.clip(unpatchify(composite_patches, image_shape=image_shape, patch_size=patch_size), 0, 1)

    full_pred_imgs = np.clip(unpatchify(pred_test_patches, image_shape=image_shape, patch_size=patch_size), 0, 1)

    # Plot grid
    fig, axes = plt.subplots(len(sample_indices), 4, figsize=(14, 2.7 * len(sample_indices)))
    col_titles = [
        "1. Original Ground Truth",
        "2. Input to Encoder (75% Masked)",
        "3. Full Model Prediction",
        "4. Inpainting Composite\n(Visible + Reconstructed)",
    ]

    for row_idx in range(len(sample_indices)):
        axes[row_idx, 0].imshow(test_batch[row_idx])
        axes[row_idx, 1].imshow(masked_test_imgs[row_idx])
        axes[row_idx, 2].imshow(full_pred_imgs[row_idx])
        axes[row_idx, 3].imshow(composite_imgs[row_idx])

        for col_idx in range(4):
            axes[row_idx, col_idx].axis("off")
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(col_titles[col_idx], fontsize=11, fontweight="bold", pad=8)

    plt.suptitle("Masked Autoencoder (ViT-MAE): Test Reconstructions with Per-Patch Normalization (15 Epochs)", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "04_mae_cifar_reconstructions.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(current_dir, "figures")
    run_mae_training_demo(num_samples=25000, epochs=12, batch_size=128, mask_ratio=0.75, output_dir=figures_dir)
