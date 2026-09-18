"""YOLO training and fine-tuning pipeline for Candlestick Pattern Detection.

Uses Ultralytics YOLO (v8n/v11n) with custom augmentations suitable for financial charts:
- Horizontal flip (market reversal symmetry)
- Mosaic and mixup
- Color jitter (minor illumination/hue invariance)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from ultralytics import YOLO


def train_yolo_detector(
    data_yaml_path: str | Path,
    model_variant: str = "yolov8n.pt",
    epochs: int = 15,
    imgsz: int = 640,
    batch_size: int = 16,
    project_dir: str | Path = "runs/detect",
    experiment_name: str = "candles_exp",
    seed: int = 42,
    device: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[YOLO, Any]:
    """Fine-tune a YOLO model on the candlestick dataset.

    Args:
        data_yaml_path: Path to dataset data.yaml file.
        model_variant: Pretrained model weights (e.g. 'yolov8n.pt', 'yolo11n.pt').
        epochs: Number of training epochs.
        imgsz: Image resolution (e.g. 640).
        batch_size: Mini-batch size.
        project_dir: Directory where runs/checkpoints are saved.
        experiment_name: Run subfolder name.
        seed: Random seed for reproducibility.
        device: 'cpu', '0', 'mps', or None (auto-detect).
        verbose: Print progress.

    Returns:
        model: Trained YOLO instance.
        results: Ultralytics training results summary.
    """
    model = YOLO(model_variant)

    results = model.train(
        data=str(Path(data_yaml_path).resolve()),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        project=str(project_dir),
        name=experiment_name,
        seed=seed,
        device=device,
        verbose=verbose,
        # Financial chart augmentation settings
        mosaic=0.0,      # Disable multi-image stitching to preserve crisp candle pixel wicks
        scale=0.1,       # Gentle scale variation
        fliplr=0.5,      # Horizontal mirror simulates inverted time/trend structure
        flipud=0.0,      # Keep vertical price orientation intact (bull vs bear distinction)
        hsv_h=0.015,
        hsv_s=0.2,
        hsv_v=0.2,
        lr0=0.005,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=2.0,
        plots=True,
        save=True,
    )

    return model, results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train YOLO Candlestick Detector")
    parser.add_argument("--data", type=str, default="data/candlestick_dataset/data.yaml")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    train_yolo_detector(
        data_yaml_path=args.data,
        model_variant=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
    )
