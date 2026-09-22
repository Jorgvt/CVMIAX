"""
Master execution script for Experiment 16: Transfer Learning & Data Augmentation Regularization.

Executes the full pipeline:
1. Data loading & sample subsampling (inducing small-sample overfitting regime).
2. Generates visual gallery of data augmentation policies.
3. Trains transfer learning models (MobileNetV2) under all augmentation regimes.
4. Generates publication-ready comparative training and generalization gap figures.
5. Prints comprehensive summary table of empirical results.
"""

import os
import sys
import numpy as np

# Ensure experiment directory is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from dataset import load_cifar_subsample
from augmentations import plot_augmentation_gallery
from train_comparison import run_training_experiment
from visualize_results import (
    plot_training_curves_comparison,
    plot_generalization_summary_barchart,
    LABELS,
)


def main():
    print("=" * 70)
    print("🎓 EXPERIMENT 16: TRANSFER LEARNING & DATA AUGMENTATION REGULARIZATION")
    print("=" * 70)

    # Output directory
    figures_dir = os.path.join(SCRIPT_DIR, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Generate Augmentation Visual Gallery
    print("\n[Step 1/3] Generating Data Augmentation Gallery...")
    x_train, y_train, x_val, y_val, class_names = load_cifar_subsample(
        samples_per_class_train=25,
        samples_per_class_val=200,
        target_size=(96, 96),
        random_seed=42,
    )

    gallery_path = os.path.join(figures_dir, "data_augmentations_comparison.png")
    plot_augmentation_gallery(
        sample_images=x_train,
        class_names=class_names,
        sample_labels=y_train,
        save_path=gallery_path,
        num_samples=4,
    )

    # 2. Run Comparative Training Suite
    print("\n[Step 2/3] Training Transfer Learning Models across Augmentation Policies...")
    epochs = 30
    policies = ["none", "geometric", "photometric", "combined", "mixup"]

    exp_data = run_training_experiment(
        epochs=epochs,
        batch_size=16,
        samples_per_class_train=25,
        samples_per_class_val=200,
        unfreeze_top_layers=20,
        learning_rate=1.5e-4,
        random_seed=42,
        policies=policies,
        save_dir=figures_dir,
    )
    results = exp_data["results"]

    # 3. Generate Evaluation Figures
    print("\n[Step 3/3] Generating Analysis Visualizations...")
    curves_path = os.path.join(figures_dir, "training_curves_overfitting_comparison.png")
    plot_training_curves_comparison(results, save_path=curves_path)

    summary_path = os.path.join(figures_dir, "generalization_gap_analysis.png")
    plot_generalization_summary_barchart(results, save_path=summary_path)

    # 4. Print Structured Summary Table
    print("\n" + "=" * 75)
    print("📊 EMPIRICAL RESULTS SUMMARY")
    print("=" * 75)
    header = f"{'Policy':<25} | {'Train Acc':<10} | {'Val Acc (Best)':<16} | {'Gen Gap':<10} | {'Val Loss':<10}"
    print(header)
    print("-" * len(header))

    for policy in policies:
        d = results[policy]
        name = LABELS.get(policy, policy)
        print(
            f"{name:<25} | {d['final_train_acc']*100:>8.2f}% | "
            f"{d['final_val_acc']*100:>6.2f}% ({d['best_val_acc']*100:>5.2f}%) | "
            f"{d['final_acc_gap']*100:>8.2f}% | {d['final_val_loss']:>10.3f}"
        )
    print("=" * 75)
    print(f"✨ All outputs saved to: {figures_dir}/\n")


if __name__ == "__main__":
    main()
