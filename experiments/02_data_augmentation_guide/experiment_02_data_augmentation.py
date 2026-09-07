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
# # Experiment 02: Visual Guide to Data Augmentation (Good Defaults & Label Preservation)
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jorgvt/CVMIAX/blob/main/experiments/02_data_augmentation_guide/experiment_02_data_augmentation.ipynb)
#
# **Pedagogical Objective:**
# Data augmentation is a domain-dependent regularization technique that introduces synthetic variations while preserving semantic labels.
#
# In this interactive notebook, we explore:
# 1. **Good Defaults for Natural Images**: Random Resized Crops, Horizontal Flips, Mild Color Jitter, Mixup & CutMix, and Object-Aware Geometric Transforms.
# 2. **Label-Altering Failure Modes (Pitfalls)**: When augmentations inadvertently corrupt ground-truth semantic labels (OCR flips, Medical X-ray inversions, Histopathology stain shifts, and Occlusion pitfalls).

# %% [markdown]
# ## 1. Setup & Imports

# %%
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf
import keras
from keras import layers

plt.rcParams.update({
    "font.sans-serif": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "figure.titlesize": 14,
    "figure.titleweight": "bold",
    "figure.dpi": 150,
})

print("Keras Version:", keras.__version__)
print("TensorFlow Version:", tf.__version__)

# %% [markdown]
# ## 2. Procedural Image Generators
#
# We generate synthetic high-contrast pedagogical scenes (e.g. natural scenes with animals, OCR digits, medical chest X-rays, histopathology stains) to cleanly demonstrate augmentation mechanics.

# %%
def create_natural_scene():
    img = Image.new("RGB", (300, 300), color=(135, 206, 235))
    draw = ImageDraw.Draw(img)
    draw.ellipse([220, 30, 270, 80], fill=(255, 220, 50))  # Sun
    draw.ellipse([-50, 180, 350, 450], fill=(34, 139, 34))  # Hill
    # Dog body
    draw.ellipse([80, 140, 210, 230], fill=(160, 82, 45))
    draw.ellipse([170, 100, 240, 170], fill=(160, 82, 45))
    draw.ellipse([210, 130, 255, 165], fill=(210, 140, 90))
    draw.ellipse([245, 140, 255, 150], fill=(20, 20, 20))
    draw.ellipse([200, 120, 210, 130], fill=(20, 20, 20))
    draw.rectangle([100, 210, 120, 270], fill=(140, 70, 35))
    draw.rectangle([170, 210, 190, 270], fill=(140, 70, 35))
    return np.array(img)

def create_second_scene():
    img = Image.new("RGB", (300, 300), color=(255, 228, 196))
    draw = ImageDraw.Draw(img)
    draw.ellipse([50, 50, 250, 250], fill=(220, 20, 60))  # Red circle
    draw.polygon([(150, 60), (90, 220), (210, 220)], fill=(255, 255, 255))
    return np.array(img)

# %% [markdown]
# ## 3. Good Defaults in Action
#
# ### 3.1 Random Resized Crops
# Forces scale and translation invariance by extracting random bounding boxes with varying scale (e.g. 50%-100%) and aspect ratio (3:4 to 4:3).

# %%
base_img = create_natural_scene()
fig, axes = plt.subplots(1, 4, figsize=(14, 4))
axes[0].imshow(base_img)
axes[0].set_title("Original Image (300x300)")
axes[0].axis("off")

crop_layer = layers.RandomCrop(height=200, width=200, seed=42)
for i in range(1, 4):
    cropped = crop_layer(tf.expand_dims(base_img, 0), training=True).numpy()[0]
    axes[i].imshow(cropped)
    axes[i].set_title(f"Random Crop #{i}")
    axes[i].axis("off")

plt.suptitle("Random Crop Augmentation", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 3.2 Horizontal Flips & Color Jitter
# - **Horizontal Flip**: Label-preserving for isotropic natural scenes.
# - **Color Jitter**: Mild brightness and contrast variations ($\pm 15\%$) simulate varying sensor exposures.

# %%
flip_layer = layers.RandomFlip(mode="horizontal")
bright_layer = layers.RandomBrightness(factor=0.25, seed=123)
contrast_layer = layers.RandomContrast(factor=0.25, seed=123)

flipped = flip_layer(tf.expand_dims(base_img, 0)).numpy()[0]
jittered = contrast_layer(bright_layer(tf.expand_dims(base_img, 0), training=True), training=True).numpy()[0]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
ax1.imshow(base_img)
ax1.set_title("Original")
ax1.axis("off")

ax2.imshow(flipped)
ax2.set_title("Horizontal Flip (Label: Dog)")
ax2.axis("off")

ax3.imshow(jittered)
ax3.set_title("Color Jitter (Brightness/Contrast)")
ax3.axis("off")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 3.3 Mixup & CutMix
# - **Mixup**: $\tilde{x} = \lambda x_1 + (1-\lambda) x_2$, with $\tilde{y} = \lambda y_1 + (1-\lambda) y_2$.
# - **CutMix**: Cuts a patch from Image B and pastes it into Image A, weighting labels by area.

# %%
img_a = create_natural_scene().astype(np.float32) / 255.0
img_b = create_second_scene().astype(np.float32) / 255.0

# Mixup with lambda = 0.6
lam = 0.6
mixup_img = lam * img_a + (1 - lam) * img_b

# CutMix
cutmix_img = img_a.copy()
cutmix_img[80:220, 80:220] = img_b[80:220, 80:220]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
ax1.imshow(img_a)
ax1.set_title("Image A (Label: Dog [1.0])")
ax1.axis("off")

ax2.imshow(mixup_img)
ax2.set_title(f"Mixup ($\lambda={lam}$)\nLabel: 0.6 Dog + 0.4 Geometry")
ax2.axis("off")

ax3.imshow(cutmix_img)
ax3.set_title("CutMix (Patch Insertion)\nLabel: 0.78 Dog + 0.22 Geometry")
ax3.axis("off")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Label-Altering Failure Modes (Pitfalls)
#
# Not all transformations preserve ground-truth semantics. Applying the wrong augmentation can corrupt annotations and degrade performance.

# %% [markdown]
# ### 4.1 Pitfall 1: Flipping in OCR / Digit Recognition (6 vs 9)

# %%
def create_digit_6():
    img = Image.new("L", (150, 150), color=0)
    draw = ImageDraw.Draw(img)
    draw.ellipse([35, 60, 115, 130], outline=255, width=16)
    draw.arc([35, 20, 115, 110], start=100, end=270, fill=255, width=16)
    return np.array(img)

digit_6 = create_digit_6()
digit_flip_v = np.flipud(digit_6)
digit_rot180 = np.rot90(digit_6, 2)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 4))
ax1.imshow(digit_6, cmap="gray")
ax1.set_title("Original: '6' (Label: 6)", color="green")
ax1.axis("off")

ax2.imshow(digit_flip_v, cmap="gray")
ax2.set_title("Vertical Flip -> '9' (Label: 6)", color="red")
ax2.axis("off")

ax3.imshow(digit_rot180, cmap="gray")
ax3.set_title("180° Rotation -> '9' (Label: 6)", color="red")
ax3.axis("off")

plt.suptitle("CRITICAL PITFALL: Digit Inversion Corrupts Ground Truth", fontsize=13, fontweight="bold", color="darkred")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 4.2 Pitfall 2: Horizontal Flips on Medical Chest Radiographs
# In medical imaging, organ laterality is critical (heart on anatomical left, liver on right). Flipping introduces artificial dextrocardia or situs inversus.

# %%
def create_chest_xray():
    img = Image.new("L", (200, 200), color=30)
    draw = ImageDraw.Draw(img)
    # Lungs (darker)
    draw.ellipse([30, 40, 85, 160], fill=15)
    draw.ellipse([115, 40, 170, 160], fill=15)
    # Spine (bright)
    draw.rectangle([95, 20, 105, 180], fill=180)
    # Heart silhouette (located on anatomical left, viewer right)
    draw.ellipse([100, 100, 150, 160], fill=140)
    return np.array(img)

xray = create_chest_xray()
xray_flipped = np.fliplr(xray)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
ax1.imshow(xray, cmap="bone")
ax1.set_title("Normal Radiograph\n(Heart correctly on Left)", color="green")
ax1.axis("off")

ax2.imshow(xray_flipped, cmap="bone")
ax2.set_title("Horizontal Flip\n(Artificial Dextrocardia Pitfall)", color="red")
ax2.axis("off")

plt.suptitle("Medical Imaging: Laterality Inversion Pitfall", fontsize=13, fontweight="bold", color="darkred")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Summary Checklist for Pipeline Design
#
# | Augmentation | Safe For | Unsafe / Pitfall For |
# | :--- | :--- | :--- |
# | **Horizontal Flip** | Natural objects (dogs, vehicles, general scenes) | OCR, text, symbols, medical scans (X-ray laterality) |
# | **Vertical Flip** | Satellite images, microscopy, histopathology | Natural outdoor scenes, autonomous driving |
# | **Random Resized Crop** | General classification | Small object detection (can crop out tiny defects/tumors) |
# | **Aggressive Color Jitter** | General RGB photographs | Histopathology (H&E stain ratios), blood oxygenation scans |
# | **Mixup / CutMix** | High-capacity classifiers fighting severe overfit | Dense pixel-level semantic segmentation boundaries |
