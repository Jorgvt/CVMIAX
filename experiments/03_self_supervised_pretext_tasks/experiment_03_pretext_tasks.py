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
# # Experiment 03: Self-Supervised Learning Pretext Tasks (Rotation, Jigsaw & Colorization)
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jorgvt/CVMIAX/blob/main/experiments/03_self_supervised_pretext_tasks/experiment_03_pretext_tasks.ipynb)
#
# **Pedagogical Objective:**
# Self-Supervised Learning (SSL) eliminates manual annotation costs by formulating **pretext tasks** where pseudo-labels are generated algorithmically from the data itself.
#
# In this interactive notebook, we cover:
# 1. **Rotation Prediction**: 4-class rotation invariance and canonical orientation learning.
# 2. **Jigsaw / Spatial Context**: Solving relative position puzzles to learn part-whole spatial relationships.
# 3. **Image Colorization**: Predicting ab chrominance from luminance $L$ in CIE Lab space.
# 4. **Shortcut Learning Pitfalls**: Chromatic aberration and edge continuity shortcuts.
# 5. **Empirical Demonstration**: Training a Rotation Pretext model in Keras and evaluating feature representations.

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
# ## 2. Rotation Prediction Pretext Task
#
# Given an unlabeled image $X$, we generate 4 rotated variants:
# $$\tilde{X}_k = \text{Rot}(X, 90^\circ \times k), \quad y_k = k \quad \text{for } k \in \{0, 1, 2, 3\}$$

# %%
def generate_rotation_batch(images):
    """Creates a 4x augmented batch with 0, 90, 180, 270 degree rotations."""
    rot0 = images
    rot90 = np.rot90(images, k=1, axes=(1, 2))
    rot180 = np.rot90(images, k=2, axes=(1, 2))
    rot270 = np.rot90(images, k=3, axes=(1, 2))

    X_rot = np.concatenate([rot0, rot90, rot180, rot270], axis=0)
    y_rot = np.concatenate([
        np.zeros(len(images), dtype=np.int32),
        np.ones(len(images), dtype=np.int32),
        np.full(len(images), 2, dtype=np.int32),
        np.full(len(images), 3, dtype=np.int32),
    ], axis=0)
    return X_rot, y_rot

# %%
# Load CIFAR-10 sample
from dataset import load_unlabeled_cifar10

x_train, _ = load_unlabeled_cifar10(num_train=100)
sample_images = x_train[:4]

x_rot_batch, y_rot_batch = generate_rotation_batch(sample_images)

fig, axes = plt.subplots(4, 4, figsize=(10, 10))
classes = ["0° (Original)", "90° (Counter-Clockwise)", "180° (Upside-Down)", "270° (Clockwise)"]

for img_idx in range(4):
    for rot_idx in range(4):
        batch_pos = rot_idx * 4 + img_idx
        axes[img_idx, rot_idx].imshow(x_rot_batch[batch_pos])
        if img_idx == 0:
            axes[img_idx, rot_idx].set_title(classes[rot_idx], fontsize=10, fontweight="bold")
        axes[img_idx, rot_idx].axis("off")

plt.suptitle("Rotation Prediction Pretext Task (4-Class Classification)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Spatial Context & Jigsaw Puzzle Pretext Task
#
# By dividing an image into a $3 \times 3$ grid of patches and shuffling them (or predicting the relative direction of patch $B$ from central patch $A$), the network is forced to learn anatomical compositions and object parts.

# %%
def extract_3x3_jigsaw_patches(image, patch_size=24, jitter=4):
    """Extracts 9 patches from an image with random jitter to prevent edge-continuity shortcuts."""
    h, w, c = image.shape
    grid_h, grid_w = h // 3, w // 3
    patches_list = []

    for i in range(3):
        for j in range(3):
            # Center of the cell with jitter
            cy = i * grid_h + grid_h // 2 + np.random.randint(-jitter, jitter + 1)
            cx = j * grid_w + grid_w // 2 + np.random.randint(-jitter, jitter + 1)
            y1 = max(0, cy - patch_size // 2)
            y2 = min(h, y1 + patch_size)
            x1 = max(0, cx - patch_size // 2)
            x2 = min(w, x1 + patch_size)
            patch = image[y1:y2, x1:x2]
            patches_list.append(patch)
    return patches_list

sample_img = x_train[0]
patches_9 = extract_3x3_jigsaw_patches(sample_img, patch_size=10, jitter=0)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(sample_img)
axes[0].set_title("Original Image (32x32)")
axes[0].axis("off")

# Render 3x3 jigsaw grid
grid_img = np.zeros((32, 32, 3), dtype=np.float32)
for idx, p in enumerate(patches_9):
    r, c = idx // 3, idx % 3
    y_start, x_start = r * 11, c * 11
    ph, pw, _ = p.shape
    grid_img[y_start:y_start+ph, x_start:x_start+pw] = p

axes[1].imshow(grid_img)
axes[1].set_title("Extracted 3x3 Jigsaw Patches")
axes[1].axis("off")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Empirical Demo: Training Rotation Classifier in Keras

# %%
def build_rotation_convnet(input_shape=(32, 32, 3)):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(4, activation="softmax")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="rotation_ssl_model")

# Prepare self-supervised dataset
train_subset = x_train[:4000]
val_subset = x_train[4000:5000]

X_ssl_train, y_ssl_train = generate_rotation_batch(train_subset)
X_ssl_val, y_ssl_val = generate_rotation_batch(val_subset)

# Shuffle
perm = np.random.permutation(len(X_ssl_train))
X_ssl_train, y_ssl_train = X_ssl_train[perm], y_ssl_train[perm]

model = build_rotation_convnet()
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

print("Training Rotation Prediction SSL Backbone...")
history = model.fit(X_ssl_train, y_ssl_train, validation_data=(X_ssl_val, y_ssl_val), epochs=5, batch_size=128, verbose=1)

# %% [markdown]
# ### 4.1 Training Convergence

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history.history["loss"], label="Train Loss")
ax1.plot(history.history["val_loss"], label="Val Loss")
ax1.set_title("Rotation Task: Cross-Entropy Loss")
ax1.legend()
ax1.grid(True, linestyle="--", alpha=0.5)

ax2.plot(history.history["accuracy"], label="Train Acc")
ax2.plot(history.history["val_accuracy"], label="Val Acc")
ax2.set_title("Rotation Task: Accuracy (Random Guess = 25%)")
ax2.legend()
ax2.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Key Takeaways & Modern Evolution
#
# 1. **Zero Annotation Cost**: Pretext tasks create supervisory training signals directly from image structure.
# 2. **Shortcut Pitfalls**: Neural networks are lazy optimizers—they will exploit chromatic aberration or low-level pixel continuity unless patches are jittered and color-perturbed.
# 3. **From Heuristic Puzzles to Contrastive & Masked Learning**: Classic heuristic pretext tasks (Rotation, Jigsaw) laid the groundwork for modern SSL methods like **Contrastive Learning (SimCLR)** and **Masked Autoencoders (MAE)**.
