"""
Master Runner for Experiment 10:
Plain FCN vs U-Net Semantic Segmentation.

Executes the full pipeline:
1. Architecture comparison diagram generation
2. Dataset generation and training with checkpointing
3. Rigorous quantitative evaluation (mIoU, Per-Class IoU, Boundary IoU)
4. High-resolution diagnostic visualizations
"""

from pathlib import Path
import numpy as np

from train import train_models
from evaluate import evaluate_model
from visualize import (
    plot_architecture_diagram,
    plot_qualitative_comparison,
    plot_boundary_and_sharpness_analysis,
    plot_training_dynamics,
)

EXPERIMENT_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = EXPERIMENT_DIR / "checkpoints"


def main():
    print("=" * 75)
    print("Experiment 10: Plain FCN vs U-Net Segmentation (The Power of Skip Connections)")
    print("=" * 75)

    # 1. Plot Architecture Comparison Diagram
    arch_diagram_path = EXPERIMENT_DIR / "fcn_vs_unet_architecture_comparison.png"
    print("\n[Step 1/4] Generating Architectural Comparison Schematic...")
    plot_architecture_diagram(save_path=arch_diagram_path)

    # 2. Train or Load Models
    print("\n[Step 2/4] Training / Loading Plain FCN and U-Net Models...")
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
        checkpoint_dir=CHECKPOINT_DIR,
        verbose=1,
    )

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = splits

    # 3. Evaluate Models on Test Set
    print("\n[Step 3/4] Evaluating Models on Unseen Test Set...")
    fcn_metrics = evaluate_model(fcn_model, x_test, y_test)
    unet_metrics = evaluate_model(unet_model, x_test, y_test)

    # Compute probability outputs for diagnostics
    fcn_probs = fcn_model.predict(x_test, batch_size=16, verbose=0)
    unet_probs = unet_model.predict(x_test, batch_size=16, verbose=0)
    fcn_preds = np.argmax(fcn_probs, axis=-1)
    unet_preds = np.argmax(unet_probs, axis=-1)

    # 4. Generate Visualizations
    print("\n[Step 4/4] Generating Publication-Quality Diagnostic Visualizations...")
    
    # 4a. Training Dynamics
    train_dynamics_path = EXPERIMENT_DIR / "fcn_vs_unet_training_dynamics.png"
    plot_training_dynamics(fcn_history, unet_history, save_path=train_dynamics_path)

    # 4b. Qualitative Predictions & Error Maps
    qual_path = EXPERIMENT_DIR / "fcn_vs_unet_qualitative_comparison.png"
    plot_qualitative_comparison(x_test, y_test, fcn_preds, unet_preds, num_samples=5, save_path=qual_path)

    # 4c. Boundary Sharpness & Profile Analysis
    sharpness_path = EXPERIMENT_DIR / "fcn_vs_unet_boundary_and_corner_sharpness.png"
    plot_boundary_and_sharpness_analysis(
        fcn_metrics=fcn_metrics,
        unet_metrics=unet_metrics,
        test_images=x_test,
        test_masks=y_test,
        fcn_probs=fcn_probs,
        unet_probs=unet_probs,
        save_path=sharpness_path,
    )

    # Print Summary Table
    print("\n" + "=" * 75)
    print("EXPERIMENT 10: FINAL QUANTITATIVE COMPARISON")
    print("=" * 75)
    print(f"{'Metric':<25} | {'Plain FCN (No Skips)':<22} | {'U-Net (With Skips)':<20} | {'Improvement':<12}")
    print("-" * 88)
    
    metrics_to_show = [
        ("Mean IoU (mIoU)", "mean_iou"),
        ("Boundary IoU (Sharpness)", "boundary_iou"),
        ("Pixel Accuracy", "pixel_accuracy"),
        ("Rectangle IoU (Corners)", "Rectangle"),
        ("Circle IoU (Curves)", "Circle"),
        ("Triangle IoU (Angles)", "Triangle"),
        ("Background IoU", "Background"),
    ]

    for label, key in metrics_to_show:
        f_val = fcn_metrics[key]
        u_val = unet_metrics[key]
        diff = u_val - f_val
        diff_str = f"+{diff*100:.2f}%" if diff >= 0 else f"{diff*100:.2f}%"
        print(f"{label:<25} | {f_val*100:6.2f}%                 | {u_val*100:6.2f}%               | {diff_str:<12}")

    print("=" * 88)
    print("All figures successfully saved to:", EXPERIMENT_DIR)


if __name__ == "__main__":
    main()
