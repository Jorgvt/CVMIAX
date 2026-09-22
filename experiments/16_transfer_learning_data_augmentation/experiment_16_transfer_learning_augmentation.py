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
# # Experiment 16: Mitigating Overfitting in Transfer Learning via Data Augmentation
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jorgvt/CVMIAX/blob/main/experiments/16_transfer_learning_data_augmentation/experiment_16_transfer_learning_augmentation.ipynb)
#
# **Pedagogical Objective:**
# In computer vision, **Transfer Learning** is the predominant paradigm for training deep models on downstream tasks where labeled data is scarce. By leveraging representations learned from large-scale datasets (such as ImageNet-1k), transfer learning drastically reduces data requirements and training time.
#
# However, fine-tuning high-capacity pre-trained neural networks on small target datasets poses a severe challenge: **catastrophic overfitting**. Because the network possesses millions of trainable parameters and strong representational capacity, it can rapidly memorize training samples, driving training loss to near zero while failing to generalize to unseen test data.
#
# In this interactive experiment, we:
# 1. **Empirically demonstrate the Transfer Learning Overfitting Trap**: Fine-tuning an ImageNet pre-trained MobileNetV2 on a limited dataset without data augmentation.
# 2. **Formulate Data Augmentation as Vicinal Risk Minimization (VRM)**: Moving beyond standard Empirical Risk Minimization (ERM) by constructing continuous semantic neighborhoods around training samples.
# 3. **Benchmark Multiple Augmentation Strategies**:
#    - Geometric Invariances (Random Flips, Rotations, Translations, Zooms).
#    - Photometric Invariances (Brightness & Contrast jitter).
#    - Combined Transformations.
#    - Advanced Interpolation Regularization (Mixup).
# 4. **Quantify the Generalization Gap**: Tracking $\Delta_{acc} = \text{Acc}_{train} - \text{Acc}_{val}$ and crossentropy divergence.

# %% [markdown]
# ---
# ## 1. Mathematical Formulation
#
# ### 1.1 Transfer Learning & Fine-Tuning Mechanics
# Let the source domain be $\mathcal{D}_S = \{\mathcal{X}_S, P(X_S)\}$ with task $\mathcal{T}_S = \{\mathcal{Y}_S, f_S(\cdot)\}$, pre-trained on ImageNet ($N_S \approx 1.28 \times 10^6$).
#
# We adapt the learned representation to a target domain $\mathcal{D}_T = \{\mathcal{X}_T, P(X_T)\}$ with task $\mathcal{T}_T = \{\mathcal{Y}_T, f_T(\cdot)\}$ where target sample size is small ($N_T \ll N_S$).
#
# The model decomposes into a convolutional feature extractor $f_\theta: \mathcal{X}_T \to \mathbb{R}^d$ and a task-specific classification head $g_\phi: \mathbb{R}^d \to \mathcal{P}(\mathcal{Y}_T)$:
# $$\hat{y} = g_\phi(f_\theta(x)) = \text{softmax}(W \cdot f_\theta(x) + b)$$
#
# During fine-tuning, the parameter updates are governed by:
# $$\min_{\theta, \phi} \mathcal{L}_{CE}(g_\phi(f_\theta(x)), y) = - \sum_{k=1}^K y_k \log \hat{y}_k$$
#
# ---
#
# ### 1.2 Empirical Risk Minimization (ERM) vs. The Overfitting Gap
# Under classic Empirical Risk Minimization (ERM), we approximate the true expected risk $R(f) = \mathbb{E}_{(x,y)\sim P}[\mathcal{L}(f(x), y)]$ using the empirical distribution $P_\delta(x, y) = \frac{1}{N} \sum_{i=1}^N \delta_{(x_i, y_i)}(x, y)$:
#
# $$R_{emp}(f) = \frac{1}{N} \sum_{i=1}^N \mathcal{L}(f(x_i), y_i)$$
#
# When sample size $N$ is small relative to model capacity $\text{VC}(f)$, ERM allows the network to memorize idiosyncratic sample artifacts without learning generalizable manifolds:
# $$\text{Generalization Gap} = R(f) - R_{emp}(f) \gg 0$$
#
# ---
#
# ### 1.3 Vicinal Risk Minimization (VRM) & Data Augmentation
# To regularize learning, **Vicinal Risk Minimization (VRM)** replaces point-mass delta distributions with a continuous vicinal distribution $\nu(\tilde{x}, \tilde{y} \mid x_i, y_i)$:
#
# $$R_{vrm}(f) = \frac{1}{N} \sum_{i=1}^N \mathbb{E}_{(\tilde{x}, \tilde{y}) \sim \nu(x_i, y_i)} [\mathcal{L}(f(\tilde{x}), \tilde{y})]$$
#
# 1. **Geometric Vicinity ($\nu_{geom}$)**:
#    Transforms image coordinates using spatial transformations $T_\omega \in \text{SE}(2)$:
#    $$\tilde{x} = T_\omega(x), \quad \tilde{y} = y, \quad \omega \sim \Omega_{spatial}$$
#    Enforces affine and viewpoint invariance.
#
# 2. **Photometric Vicinity ($\nu_{photo}$)**:
#    Perturbs pixel intensity distributions:
#    $$\tilde{x} = \alpha x + \beta, \quad \tilde{y} = y, \quad \alpha \in [1-\epsilon, 1+\epsilon], \beta \in [-\delta, \delta]$$
#    Enforces illumination and contrast invariance.
#
# 3. **Mixup Regularization ($\nu_{mix}$)**:
#    Constructs linear convex interpolations between pairs of distinct training examples (Zhang et al., 2017):
#    $$\tilde{x} = \lambda x_i + (1 - \lambda) x_j, \quad \tilde{y} = \lambda y_i + (1 - \lambda) y_j, \quad \lambda \sim \text{Beta}(\alpha, \alpha)$$
#    Mixup enforces linear behavior between training manifolds, prevents overconfident predictions, and acts as effective label smoothing.

# %% [markdown]
# ## 2. Environment & Dependency Setup

# %%
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import tensorflow as tf
import keras
from keras import layers, ops

# Ensure reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version:      {keras.__version__}")
print(f"GPU Available:      {len(tf.config.list_physical_devices('GPU')) > 0}")

# %% [markdown]
# ## 3. Dataset Loading: Creating the Small-Sample Overfitting Regime
#
# To clearly isolate the overfitting phenomenon, we subsample CIFAR-10 into a small target dataset of **40 training images per class** (200 training images total across 5 distinct classes) and **200 validation images per class** (1,000 validation images).

# %%
from dataset import load_cifar_subsample, create_tf_dataset

# Load subsampled dataset (5 classes: airplane, automobile, bird, dog, ship)
x_train, y_train, x_val, y_val, class_names = load_cifar_subsample(
    selected_classes=[0, 1, 2, 5, 8],
    samples_per_class_train=40,
    samples_per_class_val=200,
    target_size=(96, 96),
    random_seed=42,
)

num_classes = len(class_names)
print(f"Dataset summary: {x_train.shape[0]} train images, {x_val.shape[0]} validation images.")

# %% [markdown]
# ## 4. Visualizing Data Augmentation Policies
#
# We inspect how different transformations perturb the input manifold:
# - **Original**: Unaugmented inputs.
# - **Geometric**: Random horizontal flips, small rotations ($\pm 15^\circ$), translation, and zoom.
# - **Photometric**: Contrast and brightness variations.
# - **Combined**: Geometric + Photometric transformations.
# - **Mixup**: Convex linear combination of sample pairs and soft target labels.

# %%
from augmentations import plot_augmentation_gallery, get_augmentation_model, MixupLayer

# Generate and display gallery
plot_augmentation_gallery(
    sample_images=x_train,
    class_names=class_names,
    sample_labels=y_train,
    save_path=None,
    num_samples=4,
)
plt.show()

# %% [markdown]
# ## 5. Transfer Learning Model Architecture
#
# We instantiate an ImageNet pre-trained **MobileNetV2** backbone. We unfreeze the top 25 convolutional layers to enable fine-tuning of high-level semantic filters while keeping early low-level texture/edge extractors stable.

# %%
from models import build_transfer_learning_model

# Build a sample baseline model to inspect parameter distribution
sample_model = build_transfer_learning_model(
    input_shape=(96, 96, 3),
    num_classes=num_classes,
    aug_policy="none",
    unfreeze_top_layers=25,
)
sample_model.summary()

# %% [markdown]
# ## 6. Comparative Training Suite
#
# We train identical transfer learning models across 5 distinct data augmentation policies:
# 1. `none`: Baseline without data augmentation (ERM).
# 2. `geometric`: Geometric invariances only.
# 3. `photometric`: Photometric invariances only.
# 4. `combined`: Geometric + Photometric combined.
# 5. `mixup`: Combined spatial augmentation + batch-level Mixup regularization.

# %%
from train_comparison import run_training_experiment

epochs = 35
policies = ["none", "geometric", "photometric", "combined", "mixup"]

exp_results = run_training_experiment(
    epochs=epochs,
    batch_size=32,
    samples_per_class_train=40,
    samples_per_class_val=200,
    unfreeze_top_layers=25,
    learning_rate=2e-4,
    random_seed=42,
    policies=policies,
    save_dir="figures",
)
results = exp_results["results"]

# %% [markdown]
# ## 7. Empirical Results & Visualizations
#
# ### 7.1 Training Loss & Accuracy Dynamics (The Overfitting Proof)
# We plot the loss trajectories and validation accuracy progression.
#
# In the **Baseline (No Augmentation)** run:
# - Training accuracy quickly surges to $100\%$, and training loss drops near $0.0$.
# - Validation loss steadily diverges upwards, and validation accuracy stagnates.
# - A massive **Generalization Gap** ($\Delta_{acc} > 30\%$) develops.
#
# In the **Augmented & Regularized** runs:
# - Training loss decreases gracefully without memorizing sample noise.
# - Validation loss remains controlled and low.
# - Validation accuracy significantly improves.

# %%
from visualize_results import plot_training_curves_comparison, plot_generalization_summary_barchart

# Plot dynamic training curves
plot_training_curves_comparison(results, save_path=None)
plt.show()

# %% [markdown]
# ### 7.2 Quantitative Generalization Gap Analysis
#
# We compare the final performance metrics and generalization gaps ($\Delta_{acc} = \text{Acc}_{train} - \text{Acc}_{val}$) across all five policies.

# %%
# Plot quantitative summary barcharts
plot_generalization_summary_barchart(results, save_path=None)
plt.show()

# %% [markdown]
# ### 7.3 Quantitative Summary Table

# %%
from visualize_results import LABELS

print("=" * 80)
print(f"{'Policy':<25} | {'Train Acc':<10} | {'Val Acc (Best)':<16} | {'Gen Gap':<10} | {'Val Loss':<10}")
print("=" * 80)

for policy in policies:
    d = results[policy]
    name = LABELS.get(policy, policy)
    print(
        f"{name:<25} | {d['final_train_acc']*100:>8.2f}% | "
        f"{d['final_val_acc']*100:>6.2f}% ({d['best_val_acc']*100:>5.2f}%) | "
        f"{d['final_acc_gap']*100:>8.2f}% | {d['final_val_loss']:>10.3f}"
    )
print("=" * 80)

# %% [markdown]
# ## 8. Key Takeaways & Pedagogical Insights
#
# 1. **High Capacity vs. Small Datasets**: Even with pre-trained feature extractors, high-capacity models easily memorize small target sets via Empirical Risk Minimization.
# 2. **Implicit Regularization via Vicinal Sampling**: Data augmentation does not merely "create more data"—it changes the optimization geometry by enforcing smoothness and transformation invariance over local neighborhoods ($\nu_{geom}$, $\nu_{photo}$, $\nu_{mix}$).
# 3. **Additive Invariance Benefits**: Combining spatial (geometric) and photometric augmentations provides complementary regularization, shrinking the generalization gap and boosting top-line accuracy.
# 4. **Mixup & Convex Interpolation**: By training on convex combinations of inputs and targets, Mixup smooths decision boundaries between classes, drastically curbing overconfident memorization.
