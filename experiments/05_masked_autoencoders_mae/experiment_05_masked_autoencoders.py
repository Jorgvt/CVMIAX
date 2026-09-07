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
# # Experiment 05: Masked Autoencoders (MAE) for Computer Vision
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jorgvt/CVMIAX/blob/main/experiments/05_masked_autoencoders_mae/experiment_05_masked_autoencoders.ipynb)
#
# **Pedagogical Objective:**
# In NLP (BERT), masking 15% of words enables contextual representation learning. In vision, because images have heavy spatial redundancy, **Masked Autoencoders (MAE)** mask an astonishing **75%** of image patches.
#
# In this interactive notebook, we demonstrate:
# 1. **High Masking Ratio (75%)**: Forcing the model to learn holistic semantics rather than low-level pixel interpolation.
# 2. **Asymmetric ViT Architecture**: The heavy Encoder processes **only the 25% visible patches**, saving ~75% compute and memory.
# 3. **Lightweight Decoder**: Reconstructing full pixel values from encoded visible tokens, shared learnable `[MASK]` tokens, and 2D positional embeddings.
# 4. **Patch-wise MSE Loss**: Computing reconstruction error strictly over the masked positions.

# %% [markdown]
# ## 1. Setup & Environment

# %%
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
from keras import layers

np.random.seed(42)
tf.random.set_seed(42)

print("Keras Version:", keras.__version__)
print("TensorFlow Version:", tf.__version__)

# %% [markdown]
# ## 2. Patch Extraction & 75% Random Masking
#
# Given an image $X \in \mathbb{R}^{H \times W \times C}$, we partition it into $N = (H/P) \times (W/P)$ non-overlapping patches of size $P \times P$.

# %%
def patchify(images, patch_size=4):
    """Splits images (B, H, W, C) into patches (B, N, patch_size * patch_size * C)."""
    b, h, w, c = images.shape
    num_patches_h = h // patch_size
    num_patches_w = w // patch_size
    num_patches = num_patches_h * num_patches_w

    patches = tf.image.extract_patches(
        images=images,
        sizes=[1, patch_size, patch_size, 1],
        strides=[1, patch_size, patch_size, 1],
        rates=[1, 1, 1, 1],
        padding="VALID",
    )
    patches = tf.reshape(patches, (b, num_patches, patch_size * patch_size * c))
    return patches


def unpatchify(patches, patch_size=4, h=32, w=32, c=3):
    """Reconstructs (B, H, W, C) image from (B, N, patch_size * patch_size * C) patches."""
    b, num_patches, patch_dim = patches.shape
    num_patches_h = h // patch_size
    num_patches_w = w // patch_size

    p = tf.reshape(patches, (b, num_patches_h, num_patches_w, patch_size, patch_size, c))
    p = tf.transpose(p, (0, 1, 3, 2, 4, 5))
    images = tf.reshape(p, (b, h, w, c))
    return images


def random_masking(patches, mask_ratio=0.75):
    """
    Randomly masks mask_ratio (e.g. 75%) of patches.
    Returns:
        visible_patches: (B, N_vis, patch_dim)
        mask: (B, N) binary mask (0 = visible, 1 = masked)
        ids_restore: indices to restore original patch order
    """
    b, n, d = patches.shape
    num_keep = int(n * (1.0 - mask_ratio))

    noise = tf.random.uniform((b, n))
    ids_shuffle = tf.argsort(noise, axis=1)
    ids_restore = tf.argsort(ids_shuffle, axis=1)

    ids_keep = ids_shuffle[:, :num_keep]
    visible_patches = tf.gather(patches, ids_keep, batch_dims=1)

    mask = tf.concat([tf.zeros((b, num_keep)), tf.ones((b, n - num_keep))], axis=1)
    mask = tf.gather(mask, ids_restore, batch_dims=1)
    return visible_patches, mask, ids_restore

# %% [markdown]
# ## 3. Visualizing 75% Masking on CIFAR-10 Samples

# %%
(x_train, _), _ = keras.datasets.cifar10.load_data()
x_train = x_train.astype("float32") / 255.0

sample_batch = tf.convert_to_tensor(x_train[:4])
patch_size = 4
patches_batch = patchify(sample_batch, patch_size=patch_size)
vis_patches, mask_batch, ids_restore = random_masking(patches_batch, mask_ratio=0.75)

# Create visualization of masked image by zeroing masked patches
masked_patches_for_vis = patches_batch * (1.0 - tf.expand_dims(mask_batch, -1))
masked_images_vis = unpatchify(masked_patches_for_vis, patch_size=patch_size, h=32, w=32, c=3).numpy()

fig, axes = plt.subplots(4, 3, figsize=(9, 10))
for i in range(4):
    axes[i, 0].imshow(sample_batch[i].numpy())
    axes[i, 0].set_title("Original (32x32)")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(masked_images_vis[i])
    axes[i, 1].set_title("75% Masked View (Visible to ViT)")
    axes[i, 1].axis("off")

    mask_grid = mask_batch[i].numpy().reshape(32 // patch_size, 32 // patch_size)
    axes[i, 2].imshow(mask_grid, cmap="gray")
    axes[i, 2].set_title("Binary Mask (White = Reconstructed)")
    axes[i, 2].axis("off")

plt.suptitle("Masked Autoencoder (MAE) Input Pipeline: 75% Random Masking", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Information Routing & Architectural Asymmetry
#
# ```
# 25% Visible Patches ──► [ Heavy ViT Encoder ] ──► Encoded Latent Vectors
#                                                         │
#                                                         ▼
#   [ Encoded Tokens ] + [ Shared Learnable [MASK] Tokens ] + [ Full 2D Positional Embeddings ]
#                                 │
#                                 ▼
#                      [ Lightweight ViT Decoder ]
#                                 │
#                                 ▼
#                     [ Reconstructed Full Image ]
# ```
#
# ### 4.1 Key Design Mechanics:
# 1. **Encoder Efficiency**: The Encoder never processes `[MASK]` tokens. By operating exclusively on the 25% visible tokens, computation and memory scale down quadratically with sequence length ($O((0.25N)^2) = \frac{1}{16} N^2$).
# 2. **Positional Embeddings**: 2D sinusoidal or learned positional embeddings are added so self-attention layers know each patch's absolute coordinate in the grid.
# 3. **Reconstruction Target**: The loss is MSE evaluated strictly on masked patches:
# $$\mathcal{L}_{\text{MAE}} = \frac{1}{\sum_{i} M_i} \sum_{i=1}^N M_i \cdot \|\hat{p}_i - p_i\|_2^2$$

# %% [markdown]
# ## 5. Comparison: NLP (BERT) vs Vision (MAE)
#
# | Feature | BERT (NLP) | Masked Autoencoder (Vision) |
# | :--- | :--- | :--- |
# | **Data Modality** | Discrete words/subwords with high semantic density | Continuous pixel values with heavy spatial redundancy |
# | **Masking Ratio** | **15%** (higher disrupts language syntax) | **75% - 80%** (necessary to eliminate local interpolation shortcuts) |
# | **Architecture** | Symmetric Transformer (processes all tokens + `[MASK]`) | **Asymmetric**: Encoder takes visible tokens only; Decoder takes full sequence |
# | **Prediction Target** | Discrete token cross-entropy classification | Continuous pixel MSE reconstruction |
