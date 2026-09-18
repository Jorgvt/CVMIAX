"""
Master Execution Script for Experiment 13: Traditional Computer Vision Pipeline.

Runs the complete 4-stage pipeline:
1. Preprocessing (Deskewing, Denoising, Normalization)
2. Feature Extraction (HOG, LBP, Moments, Raw Baseline)
3. Feature Processing (Standardization, PCA Dimensionality Reduction)
4. Classification & Evaluation (k-NN, Random Forest, Linear SVM, RBF SVM)
5. Generates all 5 publication-ready diagnostic figures.
"""

import os
import sys
import numpy as np

# Ensure local experiment directory is on Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from train_and_evaluate import run_benchmark_suite
from visualize import (
    plot_pipeline_stages_overview,
    plot_feature_extraction_deep_dive,
    plot_pca_feature_space_and_variance,
    plot_classifier_benchmark_comparison,
    plot_sample_predictions_and_failure_analysis
)


def main():
    print("=" * 80)
    print("EXPERIMENT 13: TRADITIONAL COMPUTER VISION PIPELINE")
    print("Preprocessing -> Feature Extraction -> Feature Processing -> Classifier")
    print("=" * 80)

    # 1. Run full benchmarking suite
    suite_data = run_benchmark_suite(n_train=10000, n_test=2000)
    results = suite_data['results']

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY RESULTS:")
    print("=" * 80)
    print(f"{'Pipeline Configuration':<35} | {'Test Accuracy':<15} | {'Train Time (s)':<15} | {'Inference Time (s)':<15}")
    print("-" * 86)
    for name, res in results.items():
        print(f"{name:<35} | {res['accuracy']*100:>13.2f}% | {res['train_time_sec']:>14.2f}s | {res['test_time_sec']:>18.3f}s")
    print("=" * 86)

    # 2. Generate Visualizations
    print("\n[Visualizations] Generating Publication-Quality Diagnostic Figures...")
    
    # Figure 1: 4-Stage Pipeline Schematic
    sample_img = suite_data['x_te_raw'][0]
    sample_label = suite_data['y_te'][0]
    fig1_path = os.path.join(CURRENT_DIR, "01_pipeline_stages_overview.png")
    plot_pipeline_stages_overview(sample_img, sample_label, fig1_path)

    # Figure 2: Hand-Crafted Feature Representations
    fig2_path = os.path.join(CURRENT_DIR, "02_feature_extraction_deep_dive.png")
    plot_feature_extraction_deep_dive(suite_data['x_te_raw'], suite_data['y_te'], fig2_path)

    # Figure 3: PCA Feature Disentanglement & Scree Plot
    fig3_path = os.path.join(CURRENT_DIR, "03_pca_feature_space_and_variance.png")
    plot_pca_feature_space_and_variance(
        suite_data['X_tr_raw'][:2000],
        suite_data['X_tr_hog'][:2000],
        suite_data['y_tr'][:2000],
        suite_data['cum_var_raw'],
        suite_data['cum_var_hog'],
        fig3_path
    )

    # Figure 4: Model Comparison & Confusion Matrix
    fig4_path = os.path.join(CURRENT_DIR, "04_classifier_benchmark_comparison.png")
    plot_classifier_benchmark_comparison(results, fig4_path)

    # Figure 5: Qualitative Successes & Failure Modes
    fig5_path = os.path.join(CURRENT_DIR, "05_sample_predictions_and_failure_analysis.png")
    plot_sample_predictions_and_failure_analysis(
        suite_data['x_te_raw'],
        suite_data['y_te'],
        suite_data['best_predictions'],
        fig5_path
    )

    print("\n[SUCCESS] Experiment 13 pipeline execution and visual asset generation complete!")


if __name__ == "__main__":
    main()
