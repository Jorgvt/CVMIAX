"""Evaluation, Visualization, and Sim-to-Real Domain Shift Analysis.

Provides:
- Test-set evaluation (mAP@50, mAP@50:95, per-class breakdown)
- Visual comparison plots (Ground Truth vs. YOLO Predictions)
- Domain Shift benchmark (evaluating robustness against dark mode, gridlines, volume, noise)
- Real broker chart screenshot inference
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add experiments/07_candles to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO

from patterns import ID_TO_PATTERN, PATTERN_NAMES, PATTERN_TO_ID
from renderer import render_candlestick_chart


def evaluate_model(
    model: YOLO,
    data_yaml_path: str | Path,
    split: str = "test",
    imgsz: int = 640,
    conf: float = 0.15,
    iou: float = 0.45,
) -> Dict[str, float]:
    """Compute formal YOLO validation/test metrics.

    Returns dict with mAP50, mAP50-95, precision, and recall.
    """
    metrics = model.val(
        data=str(Path(data_yaml_path).resolve()),
        split=split,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        verbose=False,
    )

    results = {
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }
    return results


def plot_ground_truth_vs_prediction(
    image: Union[np.ndarray, Image.Image, str, Path],
    ground_truth_boxes: Optional[List[Tuple[int, float, float, float, float]]] = None,
    predictions: Optional[List[Dict[str, Any]]] = None,
    title: str = "Ground Truth vs Predicted Candlestick Patterns",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot an image with ground truth (green) and predicted (cyan/orange) bounding boxes."""
    if isinstance(image, (str, Path)):
        img_np = np.array(Image.open(image).convert("RGB"))
    elif isinstance(image, Image.Image):
        img_np = np.array(image.convert("RGB"))
    else:
        img_np = image.copy()

    h, w, _ = img_np.shape
    fig, (ax_gt, ax_pred) = plt.subplots(1, 2, figsize=(14, 7), dpi=120)

    # Plot 1: Ground Truth
    ax_gt.imshow(img_np)
    ax_gt.set_title("Ground Truth (TA-Lib Rules)", fontsize=13, fontweight="bold", color="#2E7D32")
    ax_gt.axis("off")

    if ground_truth_boxes:
        for cls_id, xc, yc, bw, bh in ground_truth_boxes:
            x1 = (xc - bw / 2.0) * w
            y1 = (yc - bh / 2.0) * h
            rect_w = bw * w
            rect_h = bh * h
            cls_name = ID_TO_PATTERN.get(int(cls_id), f"Class {cls_id}")

            rect = patches.Rectangle(
                (x1, y1), rect_w, rect_h,
                linewidth=2.2, edgecolor="#2E7D32", facecolor="none", linestyle="-"
            )
            ax_gt.add_patch(rect)
            ax_gt.text(
                x1, max(y1 - 6, 12), cls_name,
                color="white", fontsize=9, fontweight="bold",
                bbox=dict(facecolor="#2E7D32", edgecolor="none", alpha=0.85, pad=2)
            )

    # Plot 2: Predictions
    ax_pred.imshow(img_np)
    ax_pred.set_title("YOLO Model Predictions", fontsize=13, fontweight="bold", color="#1565C0")
    ax_pred.axis("off")

    if predictions:
        for pred in predictions:
            cls_id = pred["class_id"]
            conf = pred["conf"]
            x1, y1, x2, y2 = pred["xyxy"]
            cls_name = ID_TO_PATTERN.get(int(cls_id), f"Class {cls_id}")

            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2.2, edgecolor="#1565C0", facecolor="none", linestyle="-"
            )
            ax_pred.add_patch(rect)
            ax_pred.text(
                x1, max(y1 - 6, 12), f"{cls_name} ({conf:.2f})",
                color="white", fontsize=9, fontweight="bold",
                bbox=dict(facecolor="#1565C0", edgecolor="none", alpha=0.85, pad=2)
            )

    plt.suptitle(title, fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig


def run_domain_shift_benchmark(
    model: YOLO,
    test_df_list: List[pd.DataFrame],
    styles: List[str] = ["clean", "dark", "grid", "volume", "noisy"],
    conf: float = 0.15,
) -> pd.DataFrame:
    """Benchmark model detection performance across varied visual styles / domain shifts."""
    records = []

    for style in styles:
        detected_count = 0
        total_gt = 0
        total_conf = []

        for df_sample in test_df_list:
            from patterns import detect_all_patterns
            pts = detect_all_patterns(df_sample)
            if not pts:
                continue

            vol = df_sample["Volume"].values if "Volume" in df_sample.columns else None
            img_rgb, _ = render_candlestick_chart(df_sample, patterns=pts, style=style, volume=vol)

            # Inference
            res = model.predict(img_rgb, conf=conf, verbose=False)[0]
            boxes = res.boxes
            pred_count = len(boxes) if boxes is not None else 0
            total_gt += len(pts)
            detected_count += pred_count

            if boxes is not None and len(boxes) > 0:
                total_conf.extend(boxes.conf.cpu().numpy().tolist())

        avg_conf = float(np.mean(total_conf)) if total_conf else 0.0
        records.append({
            "Style / Domain Shift": style.capitalize(),
            "Ground Truth Count": total_gt,
            "Predicted Count": detected_count,
            "Recall Ratio (Pred/GT)": round(detected_count / max(total_gt, 1), 3),
            "Mean Confidence": round(avg_conf, 3),
        })

    return pd.DataFrame(records)
