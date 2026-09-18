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
# # Experiment 12: 3D Gaussian Splatting for Real-Time Novel View Synthesis
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jorgvt/CVMIAX/blob/main/experiments/12_3d_gaussian_splatting/experiment_12_3d_gaussian_splatting.ipynb)
#
# ---
#
# ## Pedagogical Overview
#
# **3D Gaussian Splatting (3DGS)** (Kerbl et al., SIGGRAPH 2023) revolutionized 3D computer vision and neural rendering. While Neural Radiance Fields (NeRF) represent continuous 3D scenes implicitly inside the weights of a Multi-Layer Perceptron (MLP), 3DGS represents scenes **explicitly** using a collection of 3D anisotropic Gaussian ellipsoids.
#
# ### Key Learning Objectives:
# 1. **2D Gaussian Image Fitting (Intuition Warm-Up)**: Fit 2D Gaussian primitives directly to a 2D image to understand how position $(x,y)$, scale $(s_x, s_y)$, rotation angle $\theta$, opacity $\sigma$, and RGB color $\mathbf{c}$ fit geometric boundaries.
# 2. **3D Explicit Representation**: Parameterize 3D covariances $\boldsymbol{\Sigma} = R S S^T R^T$ with log-scales $\mathbf{s} \in \mathbb{R}^3$ and unit quaternions $\mathbf{q} \in \mathbb{H}$.
# 3. **Surface Anchoring via Visual Hull**: Sample initial 3D positions directly along multi-view silhouette rays so Gaussians start on the object rather than in empty air.
# 4. **Differentiable EWA Splatting**: Derive the projective camera Jacobian $J$ and project 3D covariances to the 2D image plane:
#    $$\boldsymbol{\Sigma}_{2D} = J W \boldsymbol{\Sigma}_{3D} W^T J^T + \nu I_{2\times2}$$
# 5. **Differentiable Front-to-Back Alpha Compositing**: Implement point-based volumetric rendering using depth sorting and the "Over" operator:
#    $$C(\mathbf{p}) = \sum_{i=1}^N \mathbf{c}_i \alpha_i(\mathbf{p}) \prod_{j=1}^{i-1} (1 - \alpha_j(\mathbf{p}))$$
# 6. **Scale Regularization & Novel View Synthesis**: Prevent giant floaters via scale bounding and synthesize 360-degree orbital camera trajectories.

# %%
import os
import sys
from pathlib import Path

EXPERIMENT_DIR = Path.cwd()
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from dataset import generate_orbital_camera_poses, load_tiny_nerf_dataset, sample_visual_hull_point_cloud
from evaluate import compute_metrics, evaluate_dataset, render_novel_trajectory
from gaussian_2d import Gaussian2DModel, plot_2d_gaussian_fitting, train_2d_gaussian_fitting
from models import GaussianModel, compute_cov3d, quaternion_to_rotation_matrix
from splatting import compute_cov2d, project_gaussians_to_screen, render_gaussians
from train import train_gaussian_splatting
from visualize import (
    create_orbital_gif,
    plot_3d_gaussian_cloud,
    plot_novel_view_strip,
    plot_qualitative_reconstruction,
    plot_training_dynamics,
)

print(f"TensorFlow Version: {tf.__version__}")
print(f"GPU / Accelerators: {tf.config.list_physical_devices()}")

# %% [markdown]
# ---
# ## Part 1: 2D Gaussian Image Splatting (Intuition Warm-Up)
#
# Before dealing with 3D camera matrices, depth sorting, and multi-view occlusion, we fit a collection of 200 2D Gaussian ellipses directly to a target 2D image.
#
# Each 2D Gaussian has:
# - Center $\boldsymbol{\mu}_i = (x_i, y_i) \in \mathbb{R}^2$
# - Scale $\mathbf{s}_i = (s_{x, i}, s_{y, i}) \in \mathbb{R}_{>0}^2$
# - Rotation angle $\theta_i \in [-\pi, \pi]$
# - Opacity $\sigma_i \in [0, 1]$
# - Color $\mathbf{c}_i \in [0, 1]^3$

# %%
DATA_DIR = EXPERIMENT_DIR / "data"
data = load_tiny_nerf_dataset(data_dir=str(DATA_DIR), val_split_index=100)
target_2d = data["images_train"][0]

# Train 2D Gaussian Image Fitting
model_2d, history_2d = train_2d_gaussian_fitting(
    target_image=target_2d,
    num_gaussians=200,
    num_iterations=100,
    log_interval=25,
)

# Visualize 2D image fitting results
fig_2d = plot_2d_gaussian_fitting(target_2d, model_2d, history_2d)
plt.show()

# %% [markdown]
# ---
# ## Part 2: 3D Multi-View Scene Representation
#
# We load the multi-view synthetic dataset consisting of calibrated RGB images and $4 \times 4$ Camera-to-World extrinsic matrices $[R_{\text{c2w}} \mid \mathbf{t}_{\text{c2w}}]$.

# %%
images_train = data["images_train"]
poses_train = data["poses_train"]
images_val = data["images_val"]
poses_val = data["poses_val"]
focal = float(data["focal"])
height = int(data["height"])
width = int(data["width"])

print(f"Loaded {len(images_train)} training views and {len(images_val)} test views ({height}x{width}, focal={focal:.2f})")

# Preview sample training views
fig, axes = plt.subplots(1, 4, figsize=(12, 3), dpi=150)
for i, idx in enumerate([0, 10, 20, 30]):
    axes[i].imshow(images_train[idx])
    axes[i].set_title(f"Train View {idx}", fontsize=10, fontweight="bold")
    axes[i].axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## Part 3: Visual Hull Point Cloud & Surface Anchoring
#
# Random initialization in empty space leads to floating colored blobs. By back-projecting foreground pixels from calibrated views, we sample 3D points directly on the physical object surface.

# %%
num_gaussians = 600
pts, colors_init = sample_visual_hull_point_cloud(
    images=images_train[:30],
    poses=poses_train[:30],
    focal=focal,
    height=height,
    width=width,
    num_points=num_gaussians,
    spatial_bound=0.8,
)

model = GaussianModel(num_gaussians=len(pts))
model.initialize_from_points(pts, colors_init, initial_log_scale=-3.5, initial_logit_opacity=1.0)

print(f"Model initialized with {len(pts)} 3D surface-anchored Gaussians.")

# Visualize initial 3D Gaussian cloud on the object surface
fig_cloud = plot_3d_gaussian_cloud(model, camera_poses=poses_train[:20])
plt.show()

# %% [markdown]
# ---
# ## Part 4: Differentiable EWA Splatting & Forward Rendering
#
# For a camera with rotation $W = R_{\text{w2c}}$ and intrinsics $(f, c_x, c_y)$, the 3D Gaussian is projected to 2D image coordinates:
#
# 1. **Projective Jacobian Matrix $J$**:
#    $$J = \begin{pmatrix} \frac{f}{z} & 0 & \frac{f X_c}{z^2} \\ 0 & -\frac{f}{z} & -\frac{f Y_c}{z^2} \end{pmatrix}$$
# 2. **Projected 2D Covariance Matrix $\boldsymbol{\Sigma}_{2D}$**:
#    $$\boldsymbol{\Sigma}_{2D} = J W \boldsymbol{\Sigma}_{3D} W^T J^T + \nu I_{2\times2}$$
# 3. **Alpha Compositing ("Over" Operator)**:
#    Sort Gaussians by depth $z_1 \le z_2 \le \dots \le z_N$.
#    $$T_i(\mathbf{p}) = \prod_{j=1}^{i-1} \big(1 - \alpha_j(\mathbf{p})\big)$$
#    $$C(\mathbf{p}) = \sum_{i=1}^N T_i(\mathbf{p}) \alpha_i(\mathbf{p}) \mathbf{c}_i + \Big(1 - \sum_{i=1}^N T_i(\mathbf{p}) \alpha_i(\mathbf{p})\Big) \mathbf{C}_{\text{bg}}$$

# %%
test_c2w = tf.constant(poses_train[0], dtype=tf.float32)
initial_render = render_gaussians(
    means_3d=model.means,
    scales=model.get_scales(),
    quats=model.get_quats(),
    opacities=model.get_opacities(),
    colors=model.get_colors(),
    c2w=test_c2w,
    focal=focal,
    height=height,
    width=width,
    bg_color=0.0,
).numpy()

fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=150)
axes[0].imshow(images_train[0])
axes[0].set_title("Ground Truth View 0", fontsize=11, fontweight="bold")
axes[0].axis("off")

axes[1].imshow(initial_render)
axes[1].set_title("Surface-Anchored Initial Splatting", fontsize=11, fontweight="bold")
axes[1].axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## Part 5: End-to-End Multi-View Photometric Optimization
#
# We optimize Gaussian positions, scales, rotations, opacities, and colors with scale regularization to prevent floater expansion.

# %%
num_iterations = 100
history = train_gaussian_splatting(
    model=model,
    images=images_train[:30],
    poses=poses_train[:30],
    focal=focal,
    height=height,
    width=width,
    num_iterations=num_iterations,
    lr_means=0.005,
    lr_scales=0.005,
    lr_quats=0.005,
    lr_opacities=0.03,
    lr_colors=0.02,
    l1_weight=0.8,
    scale_reg_weight=0.01,
    log_interval=10,
    verbose=True,
)

# Plot Loss & PSNR convergence curves
fig_dyn = plot_training_dynamics(history)
plt.show()

# %% [markdown]
# ---
# ## Part 6: Quantitative & Qualitative Evaluation
#
# We evaluate the trained 3D Gaussian representation on unseen test views.

# %%
metrics, rendered_val = evaluate_dataset(
    model=model,
    images=images_val,
    poses=poses_val,
    focal=focal,
    height=height,
    width=width,
)

print(f"Validation PSNR: {metrics['psnr']:.2f} dB")
print(f"Validation SSIM: {metrics['ssim']:.4f}")
print(f"Validation L1:   {metrics['l1']:.4f}")

# Qualitative multi-view comparison
fig_qual = plot_qualitative_reconstruction(
    gt_images=images_val,
    rendered_images=rendered_val,
    view_indices=[0, 1, 2, 3],
)
plt.show()

# Visualize optimized 3D Gaussian cloud
fig_opt_cloud = plot_3d_gaussian_cloud(model, camera_poses=poses_val)
plt.show()

# %% [markdown]
# ---
# ## Part 7: 360-Degree Orbital Novel View Synthesis
#
# We render novel views along a continuous circular orbit around the scene.

# %%
orbital_poses = generate_orbital_camera_poses(num_poses=36, radius=4.03, elevation_deg=-30.0)
novel_frames = render_novel_trajectory(
    model=model,
    poses=orbital_poses,
    focal=focal,
    height=height,
    width=width,
)

# Display horizontal strip of novel views
fig_strip = plot_novel_view_strip(novel_frames, num_views_to_display=6)
plt.show()

# Save animated GIF
gif_path = str(EXPERIMENT_DIR / "3dgs_novel_views_360.gif")
create_orbital_gif(novel_frames, save_path=gif_path, fps=15)
print(f"Saved 360-degree novel view animation to: {gif_path}")
