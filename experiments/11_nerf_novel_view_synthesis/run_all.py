"""
Master Runner for Experiment 11:
Neural Radiance Fields (NeRF) for Novel View Synthesis.

Executes the complete end-to-end pipeline:
1. Ray generation & 3D stratified sampling visualization
2. Tiny NeRF model training with positional encoding (L=6)
3. Spectral bias ablation model training without positional encoding (L=0)
4. Quantitative test set evaluation (PSNR, SSIM, L1)
5. Multi-view qualitative diagnostics & 360-degree orbital camera GIF generation
"""

import os
import sys
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import numpy as np
import tensorflow as tf

from dataset import load_tiny_nerf_dataset
from evaluate import compute_metrics, evaluate_dataset, generate_novel_view_poses, render_novel_trajectory
from models import build_tiny_nerf
from rendering import render_image
from train import train_nerf
from visualize import (
    create_orbital_gif,
    plot_camera_rays_3d,
    plot_novel_view_strip,
    plot_qualitative_results,
    plot_spectral_bias_ablation,
    plot_training_dynamics,
)

EXPERIMENT_DIR = Path(__file__).resolve().parent
DATA_DIR = EXPERIMENT_DIR / "data"
OUTPUTS_DIR = EXPERIMENT_DIR


def main():
    print("=" * 80)
    print("Experiment 11: Neural Radiance Fields (NeRF) - 3D Novel View Synthesis")
    print("=" * 80)

    # 1. Load Dataset
    print("\n[Step 1/6] Loading Dataset...")
    data = load_tiny_nerf_dataset(data_dir=str(DATA_DIR), val_split_index=100)
    images_train = data["images_train"]
    poses_train = data["poses_train"]
    images_val = data["images_val"]
    poses_val = data["poses_val"]
    focal = data["focal"]
    height = data["height"]
    width = data["width"]
    print(f"Dataset loaded: {len(images_train)} train views, {len(images_val)} test views ({height}x{width}, focal={focal:.2f})")

    # 2. 3D Ray Casting & Sampling Geometry
    print("\n[Step 2/6] Visualizing 3D Pinhole Ray Casting & Stratified Point Sampling...")
    plot_camera_rays_3d(
        c2w=poses_train[0],
        focal=focal,
        height=height,
        width=width,
        near=2.0,
        far=6.0,
        n_samples=32,
        num_ray_subsample=8,
        save_path=str(OUTPUTS_DIR / "nerf_3d_ray_sampling.png"),
    )

    # 3. Train Full NeRF Model (L=6 Positional Encoding)
    print("\n[Step 3/6] Training Tiny NeRF with Positional Encoding (L=6)...")
    nerf_model = build_tiny_nerf(num_freqs=6, hidden_dim=128, num_layers=4)
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
    plot_training_dynamics(history, save_path=str(OUTPUTS_DIR / "nerf_training_dynamics.png"))

    # 4. Train Ablation Model without Positional Encoding (L=0)
    print("\n[Step 4/6] Training Spectral Bias Ablation Model without Positional Encoding (L=0)...")
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

    # Render comparison on test pose
    test_pose = poses_val[0]
    test_gt = images_val[0]

    render_with_pe = render_image(nerf_model, height, width, focal, test_pose, near=2.0, far=6.0, n_samples=64)
    render_no_pe = render_image(nerf_no_pe, height, width, focal, test_pose, near=2.0, far=6.0, n_samples=64)

    metrics_with_pe = compute_metrics(render_with_pe["rgb"], test_gt)
    metrics_no_pe = compute_metrics(render_no_pe["rgb"], test_gt)

    plot_spectral_bias_ablation(
        gt_rgb=test_gt,
        pred_no_pe=render_no_pe["rgb"],
        psnr_no_pe=metrics_no_pe["psnr"],
        pred_with_pe=render_with_pe["rgb"],
        psnr_with_pe=metrics_with_pe["psnr"],
        save_path=str(OUTPUTS_DIR / "nerf_spectral_bias_ablation.png"),
    )

    # 5. Full Quantitative & Qualitative Evaluation
    print("\n[Step 5/6] Evaluating Test Views and Rendering Diagnostics...")
    val_metrics, val_renders = evaluate_dataset(
        model=nerf_model,
        images=images_val,
        poses=poses_val,
        focal=focal,
        near=2.0,
        far=6.0,
        n_samples=64,
    )
    print("\n--- Test Set Quantitative Performance ---")
    print(f"  • PSNR: {val_metrics['psnr']:.2f} dB")
    print(f"  • SSIM: {val_metrics['ssim']:.4f}")
    print(f"  • MSE:  {val_metrics['mse']:.6f}")
    print(f"  • L1:   {val_metrics['l1']:.4f}")

    plot_qualitative_results(
        gt_rgb=test_gt,
        pred_rgb=render_with_pe["rgb"],
        pred_depth=render_with_pe["depth"],
        pred_acc=render_with_pe["acc"],
        metrics=metrics_with_pe,
        save_path=str(OUTPUTS_DIR / "nerf_qualitative_reconstruction.png"),
    )

    # 6. Novel View Synthesis: 360-degree Orbit & Animation
    print("\n[Step 6/6] Rendering 360-Degree Novel View Orbital Synthesis...")
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

    plot_novel_view_strip(novel_frames, num_views_to_show=6, save_path=str(OUTPUTS_DIR / "nerf_novel_view_orbital_strip.png"))
    create_orbital_gif(novel_frames, gif_path=str(OUTPUTS_DIR / "nerf_novel_views_360.gif"), fps=15)

    print("\n" + "=" * 80)
    print("Experiment 11 Pipeline Finished Successfully!")
    print("Generated Artifacts:")
    print(f"  1. {OUTPUTS_DIR / 'nerf_3d_ray_sampling.png'}")
    print(f"  2. {OUTPUTS_DIR / 'nerf_training_dynamics.png'}")
    print(f"  3. {OUTPUTS_DIR / 'nerf_spectral_bias_ablation.png'}")
    print(f"  4. {OUTPUTS_DIR / 'nerf_qualitative_reconstruction.png'}")
    print(f"  5. {OUTPUTS_DIR / 'nerf_novel_view_orbital_strip.png'}")
    print(f"  6. {OUTPUTS_DIR / 'nerf_novel_views_360.gif'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
