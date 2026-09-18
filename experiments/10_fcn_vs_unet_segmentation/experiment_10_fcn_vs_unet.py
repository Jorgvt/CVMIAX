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
# # Experiment 10: Plain FCN vs U-Net Segmentation (The Power of Skip Connections)
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jorgvt/CVMIAX/blob/main/experiments/10_fcn_vs_unet_segmentation/experiment_10_fcn_vs_unet.ipynb)
#
# **Pedagogical Objective:**
# In semantic image segmentation, the model must output a discrete class label for every individual pixel in the input canvas. This requires balancing two conflicting objectives:
# 1. **Semantic Abstraction ("What")**: Deep receptive fields to recognize categories regardless of size or style.
# 2. **Spatial Localization ("Where")**: Fine-grained pixel coordinates to delineate sharp object boundaries and corners.
#
# In this experiment, we explore:
# - **The Spatial Bottleneck in Plain Encoder-Decoders (FCN without skips)**: Why downsampling discards high-frequency spatial edge information that cannot be recovered purely by upsampling from the bottleneck.
# - **The U-Net Skip Connection Paradigm**: How concatenating encoder feature maps at matching resolutions restores high-resolution spatial details directly into the decoder.
# - **Empirical Proof of Boundary Sharpness**: Demonstrating the dramatic improvement in boundary precision, sharp corner delineation, and Boundary IoU on a controlled multi-shape segmentation task.

# %% [markdown]
# ## 1. Mathematical Formulation
#
# ### 1.1 The Semantic Segmentation Formulation
# Given an input image $X \in \mathbb{R}^{H \times W \times 3}$, semantic segmentation aims to learn a dense mapping $f_\theta: X \to \hat{Y}$, where $\hat{Y} \in \{0, 1, \dots, C-1\}^{H \times W}$.
#
# The model computes pixel-wise class posterior probabilities via the softmax function:
# $$P(Y_{i, j} = c \mid X) = \frac{\exp(z_{i, j, c})}{\sum_{k=0}^{C-1} \exp(z_{i, j, k})}$$
#
# The model parameters $\theta$ are optimized using pixel-wise Categorical Cross-Entropy loss:
# $$\mathcal{L}_{\text{CE}}(\theta) = - \frac{1}{H \cdot W} \sum_{i=1}^H \sum_{j=1}^W \sum_{c=0}^{C-1} Y_{i, j, c} \log P(Y_{i, j} = c \mid X)$$
#
# ---
#
# ### 1.2 The "What" vs. "Where" Dilemma and Information Loss
# In deep convolutional encoders, successive pooling / striding operations downsample feature maps by a factor of $2^D$ (where $D$ is the depth):
# $$z_{\text{bottleneck}} \in \mathbb{R}^{\frac{H}{2^D} \times \frac{W}{2^D} \times C_{\text{deep}}}$$
#
# - **Semantic Context ("What")**: Downsampling aggregates receptive fields across large spatial contexts, allowing deep kernels to recognize complex object semantics.
# - **Spatial Degradation ("Where")**: Downsampling acts as a spatial low-pass filter. High-frequency spatial phase information (exact boundary locations, corners, thin structures) is permanently discarded.
#
# In a **Plain Encoder-Decoder (No Skips)**, the decoder must reconstruct full-resolution masks $\mathbb{R}^{H \times W}$ exclusively from the low-resolution bottleneck $z_{\text{bottleneck}}$:
# $$\hat{Y} = \mathcal{D}_{\text{plain}}(z_{\text{bottleneck}})$$
# Because exact spatial coordinate information is absent in $z_{\text{bottleneck}}$, upsampling operations (bilinear interpolation or transposed convolutions) produce spatially smoothed, blurry predictions with rounded corners.
#
# ---
#
# ### 1.3 U-Net: High-Resolution Skip Connections
# U-Net (Ronneberger et al., 2015) solves this fundamental trade-off by introducing lateral **skip connections** between corresponding encoder and decoder stages:
# $$z_l^{\text{dec}} = \text{ConvBlock}\left( \left[ \text{UpSample}(z_{l+1}^{\text{dec}}), \, z_l^{\text{enc}} \right] \right)$$
#
# where $[\cdot, \cdot]$ denotes feature concatenation along the channel dimension.
#
# - $z_{l+1}^{\text{dec}}$ provides high-level semantic guidance ("this region contains a rectangle").
# - $z_l^{\text{enc}}$ injects pristine, uncompressed high-resolution spatial coordinates ("the edge is precisely at pixel column 42").
#
# This enables the decoder to synthesize semantic decisions with razor-sharp boundary delineation.

# %% [markdown]
# ## 2. Environment Setup & Imports

# %%
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# Set reproducibility seed
np.random.seed(42)
tf.random.set_seed(42)

# Verify environment
print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")
print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")

# %% [markdown]
# ## 3. Synthetic Dataset Generation & Inspection
#
# To cleanly isolate boundary sharpness, we use a procedural geometric shapes dataset:
# - **Background (0)**: Soft canvas
# - **Rectangle (1)**: Straight edges with sharp 90-degree corners
# - **Circle (2)**: Smooth continuous curvature
# - **Triangle (3)**: Sharp acute and obtuse vertices

# %%
from dataset import get_dataset_splits, CLASS_NAMES, CLASS_COLORS, NUM_CLASSES
from visualize import get_segmentation_cmap

# Generate dataset splits (Train=400, Val=100, Test=100)
(x_train, y_train), (x_val, y_val), (x_test, y_test) = get_dataset_splits(
    num_train=400, num_val=100, num_test=100, img_size=128, seed=42
)

print(f"Train set: {x_train.shape}, Masks: {y_train.shape}")
print(f"Val set:   {x_val.shape}, Masks: {y_val.shape}")
print(f"Test set:  {x_test.shape}, Masks: {y_test.shape}")

# %% [markdown]
# ### Visualizing Ground-Truth Dataset Samples

# %%
cmap, norm = get_segmentation_cmap()

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for i in range(4):
    axes[0, i].imshow(x_train[i])
    axes[0, i].set_title(f"Train Image #{i+1}", fontsize=11, fontweight="bold")
    axes[0, i].axis("off")

    axes[1, i].imshow(y_train[i], cmap=cmap, norm=norm)
    axes[1, i].set_title(f"Ground Truth Mask #{i+1}", fontsize=11, fontweight="bold")
    axes[1, i].axis("off")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Model Architectures: Plain FCN vs U-Net
#
# Both models share the EXACT SAME encoder backbone (3 downsampling stages) and decoder depth.
# The only structural difference is the presence of lateral skip connections (`layers.Concatenate`).

# %%
from models import build_plain_fcn, build_unet

plain_fcn = build_plain_fcn(input_shape=(128, 128, 3), num_classes=4, base_filters=32)
unet = build_unet(input_shape=(128, 128, 3), num_classes=4, base_filters=32)

print("=" * 60)
print(f"Plain FCN Total Parameters : {plain_fcn.count_params():,}")
print(f"U-Net Total Parameters     : {unet.count_params():,}")
print("=" * 60)

# %% [markdown]
# ## 5. Model Training & Checkpoint Persistence
#
# We train both models under identical conditions (Adam optimizer, lr=$10^{-3}$, 25 epochs).
# Checkpoints and histories are automatically cached in `checkpoints/` to avoid redundant retraining.

# %%
from train import train_models

fcn_model, unet_model, fcn_history, unet_history, splits = train_models(
    epochs=25,
    batch_size=16,
    learning_rate=1e-3,
    num_train=400,
    num_val=100,
    num_test=100,
    base_filters=32,
    seed=42,
    force_retrain=False,
    verbose=1,
)

# %% [markdown]
# ### Training Dynamics Comparison (Loss & Accuracy)

# %%
from visualize import plot_training_dynamics

fig = plot_training_dynamics(fcn_history, unet_history)
plt.show()

# %% [markdown]
# ## 6. Quantitative Evaluation: IoU & Boundary IoU
#
# We evaluate both models on the independent 100-sample test set.
# In addition to standard **Mean IoU (mIoU)**, we compute **Boundary IoU** within a narrow 2-pixel band around shape contours to measure edge precision.

# %%
from evaluate import evaluate_model

fcn_metrics = evaluate_model(fcn_model, x_test, y_test)
unet_metrics = evaluate_model(unet_model, x_test, y_test)

# Display results table
print("=" * 80)
print(f"{'Evaluation Metric':<28} | {'Plain FCN (No Skips)':<20} | {'U-Net (With Skips)':<18} | {'Delta':<10}")
print("-" * 80)
for metric_name, key in [
    ("Mean IoU (mIoU)", "mean_iou"),
    ("Boundary IoU (Sharpness)", "boundary_iou"),
    ("Pixel Accuracy", "pixel_accuracy"),
    ("Rectangle IoU (Corners)", "Rectangle"),
    ("Circle IoU (Curves)", "Circle"),
    ("Triangle IoU (Angles)", "Triangle"),
    ("Background IoU", "Background"),
]:
    f_val = fcn_metrics[key]
    u_val = unet_metrics[key]
    delta = u_val - f_val
    print(f"{metric_name:<28} | {f_val*100:6.2f}%              | {u_val*100:6.2f}%            | +{delta*100:5.2f}%")
print("=" * 80)

# %% [markdown]
# ## 7. Qualitative Visual Comparison & Error Analysis
#
# Notice how:
# - **Plain FCN**: Shows rounded, blurry boundaries and produces a thick halo of misclassification errors along shape edges.
# - **U-Net**: Delineates crisp, sharp boundaries and corners with almost zero edge error.

# %%
from visualize import plot_qualitative_comparison

fcn_probs = fcn_model.predict(x_test, batch_size=16, verbose=0)
unet_probs = unet_model.predict(x_test, batch_size=16, verbose=0)
fcn_preds = np.argmax(fcn_probs, axis=-1)
unet_preds = np.argmax(unet_probs, axis=-1)

fig = plot_qualitative_comparison(x_test, y_test, fcn_preds, unet_preds, num_samples=5)
plt.show()

# %% [markdown]
# ## 8. Diagnostic Analysis: Edge Profiles & Corner Sharpness

# %%
from visualize import plot_boundary_and_sharpness_analysis

fig = plot_boundary_and_sharpness_analysis(
    fcn_metrics=fcn_metrics,
    unet_metrics=unet_metrics,
    test_images=x_test,
    test_masks=y_test,
    fcn_probs=fcn_probs,
    unet_probs=unet_probs,
)
plt.show()

# %% [markdown]
# ## 9. Key Pedagogical Takeaways
#
# 1. **The Role of Downsampling**:
#    Downsampling via pooling is crucial for expanding the receptive field and learning invariant semantic representations ("what is in the scene").
#
# 2. **The Bottleneck Loss**:
#    Without skip connections, the decoder has to hallucinate high-resolution edge details from a downsampled bottleneck, leading to blurred boundaries and rounded corners.
#
# 3. **The Skip Connection Advantage**:
#    U-Net bypasses the bottleneck, supplying high-frequency spatial coordinate details directly to the corresponding decoder stages. This enables pixel-accurate segmentation with razor-sharp edges.
