# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
# ---

# %% [markdown]
# # Vision Transformer Masked Autoencoder (ViT-MAE) Training on CIFAR-10
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jorgvt/CVMIAX/blob/main/experiments/05_masked_autoencoders_mae/train_mae_demo.ipynb)
#
# **Pedagogical Objectives:**
# This notebook demonstrates the complete end-to-end self-supervised pre-training loop of an **Asymmetric Vision Transformer Masked Autoencoder (ViT-MAE)** on CIFAR-10.
#
# Key Architectural & Training Features:
# 1. **75% Random Patch Masking**: Forcing the ViT encoder to infer high-level semantics from sparse (25%) visual cues.
# 2. **Per-Patch Target Normalization (He et al., CVPR 2022)**: Normalizing each patch by its local mean and std ($x_{\text{norm}} = (x - \mu) / \sigma$) to encourage sharp textures and structural features rather than blurry average colors.
# 3. **Cosine Decay Learning Rate Schedule with AdamW**.
# 4. **Qualitative Test Inpainting**: Visualizing ground truth vs. 75% masked input vs. ViT reconstruction.

# %% [markdown]
# ## 1. Setup & Environment

# %%
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
from keras import layers

from mae_utils import patchify, unpatchify, random_masking
from models import MaskedAutoencoderViT

np.random.seed(42)
tf.random.set_seed(42)

print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")
print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")

# %% [markdown]
# ## 2. Per-Patch Target Normalization (He et al., 2022)
#
# In vanilla pixel regression, MSE loss tends to predict the mean color of missing regions (causing blurriness).
# By normalizing each patch $p$ to have zero mean and unit variance:
# $$p_{\text{norm}} = \frac{p - \mu_p}{\sigma_p + \epsilon}$$
# The model is forced to predict high-frequency local contrast and structural patterns.

# %%
def normalize_patches(patches, eps=1e-6):
    """
    Applies per-patch normalization:
    Computes mean and variance across pixels within each patch.
    """
    mean = np.mean(patches, axis=-1, keepdims=True)
    var = np.var(patches, axis=-1, keepdims=True)
    std = np.sqrt(var + eps)
    norm_patches = (patches - mean) / std
    return norm_patches, mean, std


def unnormalize_patches(norm_patches, mean, std):
    """Restores original pixel scale from normalized patches."""
    return norm_patches * std + mean

# %% [markdown]
# ## 3. Data Preparation (CIFAR-10)

# %%
from dataset import load_mae_cifar10

# Use 20,000 training images for thorough demonstration
num_samples = 20000
x_train, x_test = load_mae_cifar10(num_train=num_samples, num_test=1000)

patch_size = 4
image_shape = (32, 32, 3)
mask_ratio = 0.75

print(f"Training images: {x_train.shape[0]} | Test images: {x_test.shape[0]}")
print(f"Image Shape: {image_shape} | Patch Size: {patch_size}x{patch_size} | Mask Ratio: {mask_ratio*100:.0f}%")

# %% [markdown]
# ## 4. Build Asymmetric ViT-MAE Architecture
#
# * **Encoder**: 4 Transformer blocks, embedding dimension 128, 4 attention heads (processes visible 25% patches only).
# * **Decoder**: 2 Transformer blocks, embedding dimension 64, 4 attention heads (processes full sequence with learnable `[MASK]` tokens).

# %%
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

epochs = 12
batch_size = 128
total_steps = (len(x_train) // batch_size) * epochs

lr_schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=2e-3,
    decay_steps=total_steps,
    alpha=0.05,
)
optimizer = keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=0.02)

print(f"Total training steps: {total_steps} across {epochs} epochs")

# %% [markdown]
# ## 5. Training Loop with Per-Patch Normalization

# %%
num_batches = len(x_train) // batch_size
train_losses = []

print("Starting ViT-MAE Self-Supervised Pre-training...")
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
    print(f"Epoch {epoch+1:2d}/{epochs} | Masked Patch Normalized MSE Loss: {avg_loss:.5f}")

# %% [markdown]
# ### 5.1 Training Convergence Curve

# %%
plt.figure(figsize=(8, 4))
plt.plot(range(1, epochs + 1), train_losses, "o-", color="#1f77b4", linewidth=2.2, label="Normalized MSE Loss")
plt.xlabel("Epoch", fontsize=11)
plt.ylabel("Masked Patch Loss", fontsize=11)
plt.title("ViT-MAE Pre-training Loss on CIFAR-10 (75% Masking)", fontsize=13, fontweight="bold")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Qualitative Test Set Inpainting Evaluation
#
# Let's inspect test images never seen during pre-training and visualize the reconstructions.

# %%
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

# Unnormalize back to RGB pixel space
pred_test_patches = unnormalize_patches(pred_norm_test, t_mean, t_std)

# Construct 75% masked input view
masked_view_patches = test_patches.copy()
mask_patch_val = np.full(patch_size * patch_size * 3, 0.08, dtype=np.float32)
for b in range(len(sample_indices)):
    for i in range(mask_test.shape[1]):
        if mask_test[b, i] == 1.0:
            masked_view_patches[b, i] = mask_patch_val
masked_test_imgs = unpatchify(masked_view_patches, image_shape=image_shape, patch_size=patch_size)

# Construct Inpainted composite (original visible + model predicted masked)
composite_patches = test_patches.copy()
for b in range(len(sample_indices)):
    for i in range(mask_test.shape[1]):
        if mask_test[b, i] == 1.0:
            composite_patches[b, i] = pred_test_patches[b, i]
composite_imgs = np.clip(unpatchify(composite_patches, image_shape=image_shape, patch_size=patch_size), 0, 1)
full_pred_imgs = np.clip(unpatchify(pred_test_patches, image_shape=image_shape, patch_size=patch_size), 0, 1)

# Render comparison grid
fig, axes = plt.subplots(len(sample_indices), 4, figsize=(14, 2.7 * len(sample_indices)))
col_titles = [
    "1. Original Ground Truth",
    "2. Input to ViT (75% Masked)",
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

plt.suptitle("Masked Autoencoder (ViT-MAE): Test Reconstructions (75% Masking)", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Summary & Takeaways
#
# 1. **High Masking Enables Holistic Learning**: At 75% masking, adjacent pixel interpolation is impossible—the network must recognize global semantics (wheels, wings, eyes) to reconstruct the image.
# 2. **Per-Patch Normalization Sharpens Output**: Normalizing local patch targets prevents blurriness and forces the ViT to focus on texture and edge details.
# 3. **Asymmetric ViT Scaling**: Because the heavy Encoder processes only 25% of tokens, ViT-MAE scales efficiently to very deep Vision Transformer backbones.
