"""End-to-end headless execution script for Candlestick Pattern Detection (Experiment 07).

Performs:
1. Fast synthetic and market dataset generation
2. YOLOv8n fine-tuning
3. Evaluation & visualization saving
4. Sim-to-Real Domain shift analysis
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add experiments/07_candles to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from data_generator import build_candlestick_dataset, generate_synthetic_ohlc
from evaluate import evaluate_model, plot_ground_truth_vs_prediction, run_domain_shift_benchmark
from patterns import PATTERN_NAMES
from renderer import render_candlestick_chart
from train import train_yolo_detector


def run_experiment_pipeline(
    dataset_dir: str = "data/candlestick_dataset",
    num_images: int = 250,
    epochs: int = 12,
    imgsz: int = 640,
    batch_size: int = 16,
):
    print("=" * 60)
    print("EXPERIMENT 07: CANDLESTICK PATTERN DETECTION PIPELINE")
    print("=" * 60)

    base_dir = Path(__file__).parent.resolve()
    data_path = base_dir / dataset_dir

    # Step 1: Generate Dataset
    print(f"\n[1/4] Generating Candlestick Dataset ({num_images} images)...")
    stats = build_candlestick_dataset(
        output_dir=data_path,
        num_images=num_images,
        window_size=30,
        image_size=imgsz,
        include_yfinance=True,
        seed=42,
    )
    print(f"Dataset generated at: {data_path}")
    print(f"Split distribution: Train={stats['train_images']}, Val={stats['val_images']}, Test={stats['test_images']}")

    # Step 2: Train Model
    print(f"\n[2/4] Fine-tuning YOLOv8n for {epochs} epochs...")
    yaml_path = data_path / "data.yaml"
    model, train_res = train_yolo_detector(
        data_yaml_path=yaml_path,
        model_variant="yolov8n.pt",
        epochs=epochs,
        imgsz=imgsz,
        batch_size=batch_size,
        project_dir=base_dir / "runs",
        experiment_name="exp_candles_run",
        verbose=False,
    )
    print("Model fine-tuning complete.")

    # Step 3: Evaluate on Test Set
    print("\n[3/4] Evaluating Model on Test Set...")
    metrics = evaluate_model(model, data_yaml_path=yaml_path, split="test", imgsz=imgsz, conf=0.15)
    print(f"Test Metrics: mAP50 = {metrics['mAP50']:.4f}, mAP50-95 = {metrics['mAP50_95']:.4f}")

    # Plot sample prediction
    test_img_dir = data_path / "images" / "test"
    test_lbl_dir = data_path / "labels" / "test"
    test_imgs = sorted(list(test_img_dir.glob("*.png")))

    if test_imgs:
        sample_img_path = test_imgs[0]
        sample_lbl_path = test_lbl_dir / (sample_img_path.stem + ".txt")

        # Load ground truth
        gt_boxes = []
        if sample_lbl_path.exists():
            with open(sample_lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id, xc, yc, bw, bh = map(float, parts)
                        gt_boxes.append((int(cls_id), xc, yc, bw, bh))

        # Predict with calibrated threshold
        pred_res = model.predict(str(sample_img_path), conf=0.15, verbose=False)[0]
        predictions = []
        if pred_res.boxes is not None:
            for b in pred_res.boxes:
                cls_id = int(b.cls.item())
                conf = float(b.conf.item())
                xyxy = b.xyxy.cpu().numpy()[0].tolist()
                predictions.append({"class_id": cls_id, "conf": conf, "xyxy": xyxy})

        fig = plot_ground_truth_vs_prediction(
            image=sample_img_path,
            ground_truth_boxes=gt_boxes,
            predictions=predictions,
            title=f"Sample Test Prediction ({sample_img_path.name})",
            save_path=base_dir / "sample_prediction_result.png",
        )
        plt.close(fig)
        print(f"Saved sample visualization to {base_dir / 'sample_prediction_result.png'}")

    # Step 4: Sim-to-Real Domain Shift Benchmark
    print("\n[4/4] Running Sim-to-Real Domain Shift Benchmark...")
    test_df_samples = [generate_synthetic_ohlc(n_candles=30, seed=500 + i) for i in range(10)]
    shift_df = run_domain_shift_benchmark(model, test_df_samples)
    print("\nDomain Shift Results:")
    print(shift_df.to_string(index=False))

    # Save benchmark table
    fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
    ax.axis("off")
    ax.axis("tight")
    tbl = ax.table(cellText=shift_df.values, colLabels=shift_df.columns, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.5)
    plt.title("Sim-to-Real Domain Shift Robustness", fontsize=12, fontweight="bold", pad=20)
    plt.tight_layout()
    fig.savefig(base_dir / "domain_shift_benchmark.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved domain shift plot to {base_dir / 'domain_shift_benchmark.png'}")

    print("\n" + "=" * 60)
    print("ALL STAGES COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_experiment_pipeline()
