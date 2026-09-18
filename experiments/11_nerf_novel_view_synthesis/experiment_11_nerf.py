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
# # Experiment 11: Neural Radiance Fields (NeRF) for 3D Novel View Synthesis
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jorgvt/CVMIAX/blob/main/experiments/11_nerf_novel_view_synthesis/experiment_11_nerf.ipynb)
#
# **Pedagogical Objectives:**
# 1. **Implicit Coordinate Representations**: Understand how a 3D physical scene can be encoded directly inside the weights of an MLP mapping continuous coordinates $(x, y, z) \to (\text{RGB}, \sigma)$ rather than discrete voxel lattices or polygonal meshes.
# 2. **Pinhole Camera Ray Casting**: Derive the transformation mapping 2D pixel coordinates and camera extrinsics into 3D ray origins $\mathbf{o}$ and directions $\mathbf{d}$.
# 3. **Spectral Bias & Positional Encoding**: Discover why MLPs are biased toward low frequencies (Neural Tangent Kernel behavior) and how Fourier features enable high-frequency texture and sharp edge recovery.
# 4. **Differentiable Volumetric Rendering**: Implement discrete optical numerical integration, accumulating transmittance and opacity to render photorealistic RGB images and continuous depth maps.
# 5. **360° Novel View Synthesis**: Render smooth orbital camera trajectories around synthesized 3D objects.

# %% [markdown]
# ## 1. Mathematical Formulation & Foundations
#
# ### 1.1 The Pinhole Camera Model & Ray Generation
# Given an image plane of resolution $H \times W$, focal length $f$, optical center $(c_x, c_y) = (W/2, H/2)$, and a Camera-to-World extrinsic matrix $[R \mid \mathbf{t}] \in \mathbb{R}^{3 \times 4}$:
#
# For every pixel $(u, v)$ on the sensor grid:
# 1. Normalized ray direction in camera frame:
#    $$\mathbf{d}_{\text{cam}} = \begin{pmatrix} \frac{u - c_x}{f} \\ -\frac{v - c_y}{f} \\ -1 \end{pmatrix}$$
# 2. Transformed direction in world frame:
#    $$\mathbf{d} = R \cdot \mathbf{d}_{\text{cam}}$$
# 3. Ray origin in world frame:
#    $$\mathbf{o} = \mathbf{t}$$
# 4. Parametric ray equation:
#    $$\mathbf{r}(t) = \mathbf{o} + t \mathbf{d}, \quad t \ge 0$$
#
# ---
#
# ### 1.2 Stratified 3D Point Sampling
# To evaluate the continuous radiance field along ray $\mathbf{r}(t)$ between near plane $t_n$ and far plane $t_f$, we partition the depth range into $N$ bins and sample uniformly within each bin during training:
# $$t_i \sim \mathcal{U}\left[ t_n + \frac{i-1}{N}(t_f - t_n), \; t_n + \frac{i}{N}(t_f - t_n) \right]$$
# $$\mathbf{x}_i = \mathbf{o} + t_i \mathbf{d}$$
#
# Stratified stochastic sampling enables continuous spatial learning, avoiding discrete depth discretization artifacts.
#
# ---
#
# ### 1.3 Fourier Positional Encoding & Spectral Bias
# Standard coordinate-based MLPs acting on raw coordinates $\mathbf{x} \in \mathbb{R}^3$ fail to fit high frequencies due to spectral bias (Rahaman et al., 2019). We map inputs through Fourier feature embeddings $\gamma: \mathbb{R} \to \mathbb{R}^{2L}$:
# $$\gamma(p) = \Big( \sin(2^0 \pi p), \; \cos(2^0 \pi p), \; \dots, \; \sin(2^{L-1} \pi p), \; \cos(2^{L-1} \pi p) \Big)$$
#
# For $L=6$, this projects 3D spatial points into a 39-dimensional rich frequency hyperspace.
#
# ---
#
# ### 1.4 Differentiable Volumetric Rendering
# The continuous light arriving at pixel $\mathbf{r}$ is governed by the volume rendering equation:
# $$\mathbf{C}(\mathbf{r}) = \int_{t_n}^{t_f} T(t) \, \sigma(\mathbf{r}(t)) \, \mathbf{c}(\mathbf{r}(t)) \, dt, \quad T(t) = \exp\left(-\int_{t_n}^t \sigma(\mathbf{r}(s)) \, ds\right)$$
#
# Numerically discretized over $N$ quadrature sample intervals $\delta_i = (t_{i+1} - t_i) \|\mathbf{d}\|_2$:
# - **Alpha Opacity**: $\alpha_i = 1 - \exp(-\sigma_i \delta_i)$
# - **Cumulative Transmittance**: $T_i = \prod_{j=1}^{i-1} (1 - \alpha_j) = \exp\left(-\sum_{j=1}^{i-1} \sigma_j \delta_j\right)$ with $T_1 = 1$
# - **Sample Weight**: $w_i = T_i \alpha_i$
# - **Integrated Color**: $\hat{\mathbf{C}}(\mathbf{r}) = \sum_{i=1}^N w_i \mathbf{c}_i$
# - **Integrated Depth**: $\hat{D}(\mathbf{r}) = \sum_{i=1}^N w_i t_i$
# - **Total Foreground Opacity**: $A(\mathbf{r}) = \sum_{i=1}^N w_i$
#
# ---
#
# ### 1.5 Photometric Loss & PSNR
# Given ground-truth pixel colors $\mathbf{C}_{\text{gt}}$, the network parameters $\Theta$ are optimized end-to-end via MSE loss:
# $$\mathcal{L}_{\text{MSE}} = \frac{1}{|\mathcal{R}|} \sum_{\mathbf{r} \in \mathcal{R}} \|\hat{\mathbf{C}}(\mathbf{r}) - \mathbf{C}_{\text{gt}}(\mathbf{r})\|_2^2, \quad \text{PSNR} = -10 \log_{10}(\mathcal{L}_{\text{MSE}})$$

# %% [markdown]
# ## 2. Environment Setup & Imports

# %%
import os
import sys
from pathlib import Path

# Add experiment directory to path for clean modular imports
EXPERIMENT_DIR = Path.cwd() if "experiments" in str(Path.cwd()) else Path.cwd() / "experiments" / "11_nerf_novel_view_synthesis"
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from dataset import generate_synthetic_3d_dataset, load_tiny_nerf_dataset, pose_spherical
from evaluate import compute_metrics, evaluate_dataset, generate_novel_view_poses, render_novel_trajectory
from models import PositionalEncoding, build_tiny_nerf
from rays import get_rays_np, get_rays_tf, sample_stratified_points
from rendering import render_image, render_rays
from train import compute_psnr, train_nerf
from visualize import (
    create_orbital_gif,
    plot_camera_rays_3d,
    plot_novel_view_strip,
    plot_qualitative_results,
    plot_spectral_bias_ablation,
    plot_training_dynamics,
)

print(f"TensorFlow Version: {tf.__version__}")
print(f"Available GPUs: {tf.config.list_physical_devices('GPU')}")

# %% [markdown]
# ## 3. Loading the Dataset & Visualizing Camera Poses

# %%
# Load Tiny NeRF synthetic Lego dataset (auto-downloads and caches locally)
data = load_tiny_nerf_dataset(data_dir=str(EXPERIMENT_DIR / "data"), val_split_index=100)

images_train = data["images_train"]
poses_train = data["poses_train"]
images_val = data["images_val"]
poses_val = data["poses_val"]
focal = data["focal"]
height = data["height"]
width = data["width"]

print(f"Train views: {images_train.shape} | Poses: {poses_train.shape}")
print(f"Val views:   {images_val.shape} | Poses: {poses_val.shape}")
print(f"Resolution:  {height}x{width} | Focal length: {focal:.2f}px")

# Display a grid of multi-view training observations
fig, axes = plt.subplots(1, 6, figsize=(16, 3), dpi=120)
for i in range(6):
    axes[i].imshow(images_train[i * 15])
    axes[i].set_title(f"Train View #{i*15}", fontsize=10, fontweight="bold")
    axes[i].axis("off")
plt.suptitle("Multi-View Training Images (Synthetic Lego Benchmark)", fontsize=13, fontweight="bold", y=1.03)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Ray Casting & 3D Stratified Sampling Geometry

# %%
# Visualize camera frustum, ray casting in 3D world space, and stratified point sampling
fig_3d = plot_camera_rays_3d(
    c2w=poses_train[0],
    focal=focal,
    height=height,
    width=width,
    near=2.0,
    far=6.0,
    n_samples=32,
    num_ray_subsample=8,
)
plt.show()

# %% [markdown]
# ## 5. Building the NeRF Architecture in Keras

# %%
# Build Tiny NeRF MLP with Positional Encoding (L=6)
nerf_model = build_tiny_nerf(num_freqs=6, hidden_dim=128, num_layers=4)
nerf_model.summary()

# %% [markdown]
# ## 6. Training the Model with Photometric Loss

# %%
# Train Tiny NeRF model
history = train_nerf(
    model=nerf_model,
    images=images_train,
    poses=poses_train,
    focal=focal,
    val_images=images_val,
    val_poses=poses_val,
    num_iterations=600,
    batch_size=1024,
    n_samples=64,
    near=2.0,
    far=6.0,
    learning_rate=5e-4,
    eval_every=100,
    verbose=True,
)

# Plot loss and PSNR training progression
plot_training_dynamics(history)
plt.show()

# %% [markdown]
# ## 7. Spectral Bias Ablation: With vs. Without Positional Encoding

# %%
# Train baseline model without Positional Encoding (L=0)
print("Training ablation baseline without Positional Encoding (L=0)...")
nerf_no_pe = build_tiny_nerf(num_freqs=0, hidden_dim=128, num_layers=4)
train_nerf(
    model=nerf_no_pe,
    images=images_train,
    poses=poses_train,
    focal=focal,
    num_iterations=300,
    batch_size=1024,
    n_samples=64,
    near=2.0,
    far=6.0,
    learning_rate=5e-4,
    eval_every=150,
    verbose=False,
)

test_pose = poses_val[0]
test_gt = images_val[0]

render_with_pe = render_image(nerf_model, height, width, focal, test_pose, near=2.0, far=6.0, n_samples=64)
render_no_pe = render_image(nerf_no_pe, height, width, focal, test_pose, near=2.0, far=6.0, n_samples=64)

metrics_with_pe = compute_metrics(render_with_pe["rgb"], test_gt)
metrics_no_pe = compute_metrics(render_no_pe["rgb"], test_gt)

# Visual comparison showcasing spectral bias
plot_spectral_bias_ablation(
    gt_rgb=test_gt,
    pred_no_pe=render_no_pe["rgb"],
    psnr_no_pe=metrics_no_pe["psnr"],
    pred_with_pe=render_with_pe["rgb"],
    psnr_with_pe=metrics_with_pe["psnr"],
)
plt.show()

# %% [markdown]
# ## 8. Multi-Modal Quantitative & Qualitative Evaluation

# %%
# Quantitative evaluation on validation views
val_metrics, val_renders = evaluate_dataset(
    model=nerf_model,
    images=images_val,
    poses=poses_val,
    focal=focal,
    near=2.0,
    far=6.0,
    n_samples=64,
)

print("=" * 50)
print("Validation View Synthesis Performance:")
print(f"  • PSNR: {val_metrics['psnr']:.2f} dB")
print(f"  • SSIM: {val_metrics['ssim']:.4f}")
print(f"  • MSE:  {val_metrics['mse']:.6f}")
print(f"  • L1:   {val_metrics['l1']:.4f}")
print("=" * 50)

# Multi-modal qualitative visualization: RGB vs Error vs Depth vs Opacity
plot_qualitative_results(
    gt_rgb=test_gt,
    pred_rgb=render_with_pe["rgb"],
    pred_depth=render_with_pe["depth"],
    pred_acc=render_with_pe["acc"],
    metrics=metrics_with_pe,
)
plt.show()

# %% [markdown]
# ## 9. 360° Novel View Synthesis & Orbital Trajectory Rendering

# %%
# Generate smooth 360-degree orbital camera trajectory
novel_poses = generate_novel_view_poses(n_frames=36, phi_deg=-30.0, radius=4.0)
novel_frames = render_novel_trajectory(
    model=nerf_model,
    height=height,
    width=width,
    focal=focal,
    novel_poses=novel_poses,
    near=2.0,
    far=6.0,
    n_samples=64,
)

# Display filmstrip of synthesized novel views
plot_novel_view_strip(novel_frames, num_views_to_show=6)
plt.show()

# Save animated 360-degree GIF
gif_path = str(EXPERIMENT_DIR / "nerf_novel_views_360.gif")
create_orbital_gif(novel_frames, gif_path=gif_path, fps=15)
