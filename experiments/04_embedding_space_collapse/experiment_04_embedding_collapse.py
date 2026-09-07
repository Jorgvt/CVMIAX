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
# # Experiment 04: Embedding Space Collapse in Self-Supervised Learning
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jorgvt/CVMIAX/blob/main/experiments/04_embedding_space_collapse/experiment_04_embedding_collapse.ipynb)
#
# **Pedagogical Objective:**
# In self-supervised Siamese architectures, two augmented views $v_1, v_2 \sim \mathcal{T}(x)$ are mapped to latent representations $z_1, z_2$.
# If the objective only minimizes distance between positive pairs without counteracting forces:
# $$\min_\theta \mathbb{E}_{x} \left[ \|f_\theta(v_1) - f_\theta(v_2)\|^2 \right]$$
#
# The model finds a trivial global minimum: **it maps every input image to a single constant vector $\mathbf{z}_0$**, discarding all mutual information.
#
# In this interactive notebook, we analyze:
# 1. **Complete (Point) Collapse vs Dimensional (Subspace) Collapse**.
# 2. **Diagnostic Signals**: Per-dimension variance, Covariance heatmaps, and SVD Singular Value Spectra.
# 3. **Architectural Countermeasures**: Negative pairs (SimCLR/InfoNCE), Stop-Gradient Asymmetry (SimSiam/BYOL), and Variance Regularization (VICReg).

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
# ## 2. Visualizing Latent Spaces: Healthy vs Collapsed
#
# We generate synthetic 2D representations across 5 semantic classes to observe how collapse manifests geometrically.

# %%
n_samples_per_class = 60
n_classes = 5

# 1. Healthy Latent Space (Clusters distributed on unit circle)
angles = np.linspace(0, 2 * np.pi, n_classes, endpoint=False)
healthy_embeddings = []
labels = []
for c, ang in enumerate(angles):
    pts = np.random.randn(n_samples_per_class, 2) * 0.15 + [np.cos(ang), np.sin(ang)]
    # Normalize to unit circle
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    healthy_embeddings.append(pts)
    labels.extend([c] * n_samples_per_class)
healthy_embeddings = np.vstack(healthy_embeddings)

# 2. Complete (Point) Collapse: All vectors map to (1.0, 0.0) + tiny noise
point_collapse = np.tile([1.0, 0.0], (n_samples_per_class * n_classes, 1)) + np.random.randn(n_samples_per_class * n_classes, 2) * 0.005
point_collapse = point_collapse / np.linalg.norm(point_collapse, axis=1, keepdims=True)

# 3. Dimensional (Subspace) Collapse: Vectors span only a 1D line
line_pts = np.random.uniform(-1, 1, size=(n_samples_per_class * n_classes, 1))
subspace_collapse = np.hstack([line_pts, np.random.randn(n_samples_per_class * n_classes, 1) * 0.01])
subspace_collapse = subspace_collapse / np.linalg.norm(subspace_collapse, axis=1, keepdims=True)

colors = plt.cm.tab10(np.array(labels))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
circle = plt.Circle((0, 0), 1.0, color="gray", fill=False, linestyle="--", alpha=0.5)

# Healthy
ax1.add_patch(plt.Circle((0, 0), 1.0, color="gray", fill=False, linestyle="--", alpha=0.5))
ax1.scatter(healthy_embeddings[:, 0], healthy_embeddings[:, 1], c=colors, alpha=0.7, edgecolors="k")
ax1.set_title("Healthy Embedding Space\n(High Variance, Well-Separated)")
ax1.set_xlim(-1.3, 1.3); ax1.set_ylim(-1.3, 1.3)
ax1.set_aspect("equal"); ax1.grid(True, linestyle="--", alpha=0.3)

# Point Collapse
ax2.add_patch(plt.Circle((0, 0), 1.0, color="gray", fill=False, linestyle="--", alpha=0.5))
ax2.scatter(point_collapse[:, 0], point_collapse[:, 1], c=colors, alpha=0.7, edgecolors="k")
ax2.set_title("Complete (Point) Collapse\n(Var ≈ 0, 0 Bits Information)")
ax2.set_xlim(-1.3, 1.3); ax2.set_ylim(-1.3, 1.3)
ax2.set_aspect("equal"); ax2.grid(True, linestyle="--", alpha=0.3)

# Subspace Collapse
ax3.add_patch(plt.Circle((0, 0), 1.0, color="gray", fill=False, linestyle="--", alpha=0.5))
ax3.scatter(subspace_collapse[:, 0], subspace_collapse[:, 1], c=colors, alpha=0.7, edgecolors="k")
ax3.set_title("Dimensional (Subspace) Collapse\n(Loss of Orthogonal Dimensions)")
ax3.set_xlim(-1.3, 1.3); ax3.set_ylim(-1.3, 1.3)
ax3.set_aspect("equal"); ax3.grid(True, linestyle="--", alpha=0.3)

plt.suptitle("Latent Space Geometry Under Representation Collapse", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Spectral Diagnostics: SVD Singular Values & Covariance Heatmaps
#
# A healthy $d$-dimensional representation has isotropic singular values and a diagonally dominant covariance matrix.

# %%
d = 16
N = 500

# Healthy: isotropic Gaussian normalized
Z_healthy = np.random.randn(N, d)
Z_healthy = Z_healthy / np.linalg.norm(Z_healthy, axis=1, keepdims=True)

# Collapsed: Rank-1 projection
v = np.random.randn(d)
v /= np.linalg.norm(v)
Z_collapsed = np.outer(np.random.randn(N), v) + np.random.randn(N, d) * 0.01
Z_collapsed = Z_collapsed / np.linalg.norm(Z_collapsed, axis=1, keepdims=True)

# Covariance matrices
cov_healthy = np.cov(Z_healthy, rowvar=False)
cov_collapsed = np.cov(Z_collapsed, rowvar=False)

# SVD Singular values
_, s_healthy, _ = np.linalg.svd(Z_healthy - np.mean(Z_healthy, axis=0))
_, s_collapsed, _ = np.linalg.svd(Z_collapsed - np.mean(Z_collapsed, axis=0))

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Covariance heatmaps
im1 = ax1.imshow(cov_healthy, cmap="coolwarm", vmin=-0.05, vmax=0.1)
ax1.set_title("Healthy Covariance (Decorrelated Dimensions)")
plt.colorbar(im1, ax=ax1, fraction=0.046)

im2 = ax2.imshow(cov_collapsed, cmap="coolwarm", vmin=-0.05, vmax=0.1)
ax2.set_title("Collapsed Covariance (Extreme Collinearity)")
plt.colorbar(im2, ax=ax2, fraction=0.046)

# Scree plot (Singular values)
dim_indices = range(1, d + 1)
ax3.plot(dim_indices, s_healthy / s_healthy[0], "o-", color="blue", label="Healthy Space")
ax3.plot(dim_indices, s_collapsed / s_collapsed[0], "s--", color="red", label="Collapsed Space (Rank ~ 1)")
ax3.set_xlabel("Singular Value Rank")
ax3.set_ylabel("Normalized Singular Value $\sigma_i / \sigma_1$")
ax3.set_title("SVD Singular Value Spectrum (Scree Plot)")
ax3.grid(True, linestyle="--", alpha=0.5)
ax3.legend()

# Per-dimension variance bar plot
ax4.bar(np.arange(d) - 0.2, np.var(Z_healthy, axis=0), width=0.4, label="Healthy Variance", color="blue")
ax4.bar(np.arange(d) + 0.2, np.var(Z_collapsed, axis=0), width=0.4, label="Collapsed Variance", color="red")
ax4.set_xlabel("Embedding Dimension Index")
ax4.set_ylabel("Variance $\mathrm{Var}(z_j)$")
ax4.set_title("Per-Dimension Embedding Variance")
ax4.grid(True, linestyle="--", alpha=0.5)
ax4.legend()

plt.suptitle("Spectral Diagnostics for Detecting Representation Collapse", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Summary of Collapse Prevention Families
#
# | Mechanism / Method | Representatives | How it Prevents Collapse |
# | :--- | :--- | :--- |
# | **Contrastive Pairs** | SimCLR, MoCo | Negative pairs push dissimilar instances apart on the hypersphere via InfoNCE loss. |
# | **Architectural Asymmetry** | SimSiam, BYOL | Stop-gradient operator (`tf.stop_gradient`) breaks gradient symmetry; predictor and EMA teacher prevent static equilibria. |
# | **Information Regularization** | VICReg, Barlow Twins | Explicit loss penalties on cross-correlation matrix off-diagonals and hinge loss on per-dimension variance ($\mathrm{Var}(z) \ge 1$). |
