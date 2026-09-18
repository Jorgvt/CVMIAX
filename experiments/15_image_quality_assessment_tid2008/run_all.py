"""Master runner script for Experiment 15: Image Quality Assessment (IQA) on TID2008.

Executes the full pipeline:
1. Loads and splits TID2008 (reference-independent content split).
2. Trains Full-Reference Siamese Difference CNN and No-Reference Blind CNN.
3. Evaluates Classical Baselines (PSNR, SSIM) vs Deep IQA models.
4. Generates all 5 publication-quality visualization figures.
"""

import os
import sys

# Support direct script execution
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from dataset import (
    load_tid2008_raw,
    prepare_numpy_arrays,
    split_tid2008_by_reference,
)
from evaluate import (
    evaluate_all_methods_on_test,
)
from train import train_models
from visualize import (
    plot_classical_vs_perceptual_correlation,
    plot_distortion_taxonomy_gallery,
    plot_distortion_wise_breakdown,
    plot_perceptual_error_heatmaps,
    plot_training_dynamics,
)


def run_experiment(epochs: int = 15, batch_size: int = 32):
    figures_dir = os.path.join(current_dir, "figures")
    weights_dir = os.path.join(current_dir, "weights")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)

    print("=" * 70)
    print("EXPERIMENT 15: IMAGE QUALITY ASSESSMENT (IQA) ON TID2008")
    print("=" * 70)

    # 1. Train models
    fr_model, nr_model, fr_history, nr_history, dataset_bundle = train_models(
        target_size=(224, 224),
        epochs=epochs,
        batch_size=batch_size,
        weights_dir=weights_dir,
    )

    te_refs, te_dists, te_mos, te_rids, te_dids, te_ints = dataset_bundle["test"]

    # 2. Evaluate all methods on test set (unseen reference images)
    print("\n" + "=" * 70)
    print("QUANTITATIVE BENCHMARK EVALUATION (UNSEEN REFERENCE SCENES)")
    print("=" * 70)
    overall_df, distortion_df, predictions = evaluate_all_methods_on_test(
        te_refs,
        te_dists,
        te_mos,
        te_dids,
        fr_model=fr_model,
        nr_model=nr_model,
    )

    print("\n--- Overall Benchmark Performance ---")
    print(overall_df.to_string(index=False))

    print("\n--- SROCC Across Distortion Categories ---")
    srocc_pivot = distortion_df.pivot(
        index="Distortion Group", columns="Model", values="SROCC"
    )
    print(srocc_pivot.to_string())

    # 3. Generate Visualizations
    print("\n" + "=" * 70)
    print("GENERATING PUBLICATION-QUALITY FIGURES")
    print("=" * 70)

    # Load raw data for gallery
    raw_data = load_tid2008_raw()

    # Fig 1: Distortion taxonomy gallery
    plot_distortion_taxonomy_gallery(
        raw_data,
        ref_id=1,
        save_path=os.path.join(figures_dir, "01_distortion_taxonomy_gallery.png"),
    )

    # Fig 2: Correlation scatter plots
    plot_classical_vs_perceptual_correlation(
        te_mos,
        predictions,
        save_path=os.path.join(figures_dir, "02_classical_vs_perceptual_correlation.png"),
    )

    # Fig 3: Training dynamics
    plot_training_dynamics(
        fr_history,
        nr_history,
        save_path=os.path.join(figures_dir, "03_training_dynamics.png"),
    )

    # Fig 4: Distortion breakdown bar chart
    plot_distortion_wise_breakdown(
        distortion_df,
        save_path=os.path.join(figures_dir, "04_distortion_wise_performance_breakdown.png"),
    )

    # Fig 5: Perceptual error heatmaps
    plot_perceptual_error_heatmaps(
        fr_model,
        te_refs,
        te_dists,
        te_mos,
        sample_indices=[0, 15, 30, 45],
        save_path=os.path.join(figures_dir, "05_perceptual_error_heatmaps.png"),
    )

    print("\n" + "=" * 70)
    print("EXPERIMENT 15 COMPLETED SUCCESSFULLY!")
    print(f"Figures saved to: {figures_dir}")
    print(f"Weights saved to: {weights_dir}")
    print("=" * 70)


if __name__ == "__main__":
    run_experiment(epochs=15)
