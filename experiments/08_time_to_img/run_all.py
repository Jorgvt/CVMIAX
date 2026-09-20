"""Master Orchestrator for Experiment 08: Time-Series to Image Encodings.

Executes the complete experimental pipeline:
1. Multi-asset historical market data ingestion & date-based causal regime labeling.
2. Generating 2D visual transformation gallery (1D Series -> GAF -> CWT -> STFT).
3. Cone of Influence (COI) diagnostic plot.
4. Transform Tournament: Training or loading cached weights for all 8 models
   (GAF, CWT, STFT, Fusion) x (ResNet-18, MobileNetV2).
5. Comprehensive 2x4 Multi-Class Confusion Matrix Grid across all models & encodings.
6. Grad-CAM visual interpretability analysis.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure experiment folder is in Python path for local module imports
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from dataset import (
    compute_regime_labels,
    extract_asset_windows,
    load_market_data,
    REGIME_NAMES,
)
from evaluate import evaluate_model
from train import train_or_load_regime_classifier
from visualize import (
    plot_coi_demonstration,
    plot_confusion_matrices,
    plot_gradcam_explanations,
    plot_tournament_summary,
    plot_transformation_gallery,
)


BASE_DIR = Path(__file__).resolve().parent


def run_experiment_pipeline(
    epochs: int = 12,
    batch_size: int = 32,
    image_size: int = 128,
    assets_dir: Optional[str | Path] = None,
    weights_dir: Optional[str | Path] = None,
    force_train: bool = False,
) -> pd.DataFrame:
    """Execute the full end-to-end experiment pipeline."""
    assets_path = Path(assets_dir) if assets_dir is not None else (BASE_DIR / "assets")
    assets_path.mkdir(parents=True, exist_ok=True)
    weights_path = Path(weights_dir) if weights_dir is not None else (BASE_DIR / "weights")
    weights_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STEP 1: Ingesting Market Data & Extracting Temporal Windows")
    print("=" * 70)
    df = load_market_data()
    w_ret, w_cum, vols, rets, assets = extract_asset_windows(df, window_size=128, step_size=8)
    labels, _ = compute_regime_labels(vols, rets)

    print(f"Total extracted multi-asset windows: {len(w_ret):,}")
    for k, name in REGIME_NAMES.items():
        count = int(np.sum(labels == k))
        print(f"  - Class {k} ({name:18s}): {count:4d} ({count / len(labels):.1%})")

    print("\n" + "=" * 70)
    print("STEP 2: Generating Transformation Gallery & COI Diagnostics")
    print("=" * 70)
    gallery_file = assets_path / "transform_gallery.png"
    plot_transformation_gallery(w_ret, labels, image_size=image_size, save_path=gallery_file)
    print(f"Saved: {gallery_file}")

    coi_file = assets_path / "coi_demonstration.png"
    plot_coi_demonstration(w_ret[0], image_size=image_size, save_path=coi_file)
    print(f"Saved: {coi_file}")

    print("\n" + "=" * 70)
    print("STEP 3: Running Transform Tournament across Encodings & Architectures")
    print("=" * 70)
    methods = ["gaf", "cwt", "stft", "fusion"]
    models = ["resnet18", "mobilenet_v2"]

    tournament_records = []
    eval_results_map = {}
    best_models = {}

    for method in methods:
        for model_type in models:
            exp_name = f"{method.upper()} + {model_type}"
            print(f"\n>>> Running: {exp_name} (Epochs: {epochs}, Batch Size: {batch_size})")

            model, history, data_dict = train_or_load_regime_classifier(
                method=method,
                model_type=model_type,
                epochs=epochs,
                batch_size=batch_size,
                image_size=image_size,
                weights_dir=weights_path,
                force_train=force_train,
                verbose=1,
            )

            res = evaluate_model(model, data_dict["x_test"], data_dict["y_test"])
            print(f"  --> Test Accuracy: {res['accuracy']:.2%}, Macro F1: {res['macro_f1']:.4f}")

            tournament_records.append({
                "Experiment": exp_name,
                "Encoding": method.upper(),
                "Architecture": model_type,
                "Accuracy": res["accuracy"],
                "Macro_F1": res["macro_f1"],
            })

            eval_results_map[f"{method.upper()} ({model_type})"] = res
            best_models[f"{method}_{model_type}"] = (model, data_dict)

    tournament_df = pd.DataFrame(tournament_records)
    print("\n" + "=" * 70)
    print("STEP 4: Tournament Results Summary")
    print("=" * 70)
    print(tournament_df.to_string(index=False))

    # Save summary chart
    summary_file = assets_path / "tournament_benchmark.png"
    plot_tournament_summary(tournament_df, save_path=summary_file)
    print(f"\nSaved Tournament Chart: {summary_file}")

    # Plot Confusion Matrices for ALL 8 experimental conditions (2 rows x 4 cols)
    cm_file = assets_path / "confusion_matrices.png"
    plot_confusion_matrices(eval_results_map, ncols=4, save_path=cm_file)
    print(f"Saved Complete Confusion Matrix Grid (8 models): {cm_file}")

    print("\n" + "=" * 70)
    print("STEP 5: Generating Grad-CAM Attention Heatmaps")
    print("=" * 70)
    fusion_model, fusion_data = best_models["fusion_resnet18"]
    gradcam_file = assets_path / "gradcam_explanations.png"
    plot_gradcam_explanations(
        fusion_model,
        fusion_data["x_test"][:4],
        fusion_data["y_test"][:4],
        method_name="Fusion (ResNet-18)",
        num_samples=4,
        save_path=gradcam_file,
    )
    print(f"Saved Grad-CAM Explanations: {gradcam_file}")

    print("\n" + "=" * 70)
    print("EXPERIMENT 08 PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)

    return tournament_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Experiment 08 Full Pipeline")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--image_size", type=int, default=128, help="Image resolution")
    parser.add_argument("--force_train", action="store_true", help="Force retraining even if weights exist")
    args = parser.parse_args()

    run_experiment_pipeline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        force_train=args.force_train,
    )
