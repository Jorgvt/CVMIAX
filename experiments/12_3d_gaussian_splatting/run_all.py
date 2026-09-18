"""
Master Runner for Experiment 12:
3D Gaussian Splatting for Real-Time Novel View Synthesis.

Executes the complete pedagogical progression:
[Part 1] 2D Gaussian Image Fitting (Intuition Warm-Up)
[Part 2] Multi-View 3D Gaussian Splatting with Visual Hull Anchoring & Scale Regularization:
  1. Multi-view dataset loading & camera geometry inspection
  2. Visual-hull point cloud extraction and 3D Gaussian surface anchoring
  3. Multi-view training & photometric loss optimization
  4. Training dynamics & convergence diagnostics
  5. Quantitative test set evaluation (PSNR, SSIM, L1)
  6. Qualitative multi-view reconstruction comparison
  7. 360-degree orbital novel view synthesis & animated GIF generation
"""

import os
import sys
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import numpy as np
import tensorflow as tf

from dataset import generate_orbital_camera_poses, load_tiny_nerf_dataset, sample_visual_hull_point_cloud
from evaluate import evaluate_dataset, render_novel_trajectory
from gaussian_2d import plot_2d_gaussian_fitting, train_2d_gaussian_fitting
from models import GaussianModel
from train import train_gaussian_splatting
from visualize import (
    create_orbital_gif,
    plot_3d_gaussian_cloud,
    plot_novel_view_strip,
    plot_qualitative_reconstruction,
    plot_training_dynamics,
)

DATA_DIR = EXPERIMENT_DIR / "data"
OUTPUTS_DIR = EXPERIMENT_DIR


def main():
    print("=" * 80)
    print("Experiment 12: 3D Gaussian Splatting (3DGS) - Real-Time Novel View Synthesis")
    print("=" * 80)

    # 1. Load Dataset
    print("\n[Step 1/7] Loading Multi-View Dataset...")
    data = load_tiny_nerf_dataset(data_dir=str(DATA_DIR), val_split_index=100)
    images_train = data["images_train"]
    poses_train = data["poses_train"]
    images_val = data["images_val"]
    poses_val = data["poses_val"]
    focal = float(data["focal"])
    height = int(data["height"])
    width = int(data["width"])
    print(
        f"Dataset loaded: {len(images_train)} train views, {len(images_val)} test views "
        f"({height}x{width}, focal={focal:.2f})"
    )

    # [Part 1] 2D Gaussian Image Splatting Warm-Up
    print("\n[Step 2/7] Running Part 1: 2D Gaussian Image Splatting Intuition Warm-Up...")
    target_2d = images_train[0]
    model_2d, history_2d = train_2d_gaussian_fitting(
        target_image=target_2d,
        num_gaussians=200,
        num_iterations=100,
        log_interval=25,
    )
    fit_2d_path = str(OUTPUTS_DIR / "2dgs_image_fitting.png")
    plot_2d_gaussian_fitting(target_2d, model_2d, history_2d, save_path=fit_2d_path)

    # [Part 2] 3D Visual Hull Surface Point Cloud Extraction
    print("\n[Step 3/7] Extracting 3D Surface Point Cloud via Visual Hull Ray Sampling...")
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
    print(f"Extracted {len(pts)} 3D surface points anchoring Gaussians directly to the object.")

    # 3. Initialize 3D Gaussian Model directly on surface points
    model = GaussianModel(num_gaussians=len(pts))
    model.initialize_from_points(pts, colors_init, initial_log_scale=-3.5, initial_logit_opacity=1.0)

    # Save initial 3D Gaussian cloud plot
    cloud_path = str(OUTPUTS_DIR / "3dgs_initial_cloud.png")
    plot_3d_gaussian_cloud(model, camera_poses=poses_train[:20], save_path=cloud_path)

    # 4. Train 3D Gaussian Model
    print("\n[Step 4/7] Optimizing 3D Gaussians with Scale Regularization...")
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

    # 5. Plot Training Dynamics
    print("\n[Step 5/7] Generating Training Dynamics Visualizations...")
    dyn_path = str(OUTPUTS_DIR / "3dgs_training_dynamics.png")
    plot_training_dynamics(history, save_path=dyn_path)

    # 6. Evaluate on Test Views
    print("\n[Step 6/7] Evaluating on Unseen Validation / Test Views...")
    metrics, rendered_val = evaluate_dataset(
        model=model,
        images=images_val,
        poses=poses_val,
        focal=focal,
        height=height,
        width=width,
    )
    print("-" * 50)
    print(f"Validation PSNR: {metrics['psnr']:.2f} dB")
    print(f"Validation SSIM: {metrics['ssim']:.4f}")
    print(f"Validation L1:   {metrics['l1']:.4f}")
    print(f"Validation MSE:  {metrics['mse']:.4f}")
    print("-" * 50)

    # Save qualitative reconstruction comparison
    qual_path = str(OUTPUTS_DIR / "3dgs_qualitative_reconstruction.png")
    plot_qualitative_reconstruction(
        gt_images=images_val,
        rendered_images=rendered_val,
        view_indices=[0, 1, 2, 3],
        save_path=qual_path,
    )

    # Save optimized 3D Gaussian cloud plot
    opt_cloud_path = str(OUTPUTS_DIR / "3dgs_optimized_cloud.png")
    plot_3d_gaussian_cloud(model, camera_poses=poses_val, save_path=opt_cloud_path)

    # 7. Novel 360-Degree Orbital View Synthesis
    print("\n[Step 7/7] Synthesizing 360-Degree Orbital Trajectory...")
    orbital_poses = generate_orbital_camera_poses(num_poses=36, radius=4.03, elevation_deg=-30.0)
    novel_frames = render_novel_trajectory(
        model=model,
        poses=orbital_poses,
        focal=focal,
        height=height,
        width=width,
    )

    strip_path = str(OUTPUTS_DIR / "3dgs_novel_view_orbital_strip.png")
    plot_novel_view_strip(novel_frames, num_views_to_display=6, save_path=strip_path)

    gif_path = str(OUTPUTS_DIR / "3dgs_novel_views_360.gif")
    create_orbital_gif(novel_frames, save_path=gif_path, fps=15)

    print("\n" + "=" * 80)
    print("Experiment 12 Execution Complete!")
    print(f"Artifacts saved in: {OUTPUTS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
