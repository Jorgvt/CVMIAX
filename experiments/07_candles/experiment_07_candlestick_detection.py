# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
# ---

# %% [markdown]
# # Experiment 07: Candlestick Pattern Detection with Modern Object Detectors
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jorgvt/CVMIAX/blob/main/experiments/07_candles/experiment_07_candlestick_detection.ipynb)
#
# **Course:** Deep Learning for Economics & Trading (Computer Vision module)  
# **Author:** Jorge Vila  
# **Topic:** Single-stage Object Detection (YOLO) applied to financial charts with deterministic rule-based auto-labeling and Sim-to-Real domain transfer analysis.
#
# ---
#
# ## 1. Pedagogical Motivation & Framing
#
# In algorithmic trading and quantitative finance, technical analysts frequently visually scan candlestick (OHLC) charts for localized geometric formations known as **candlestick patterns** (e.g., *Hammers*, *Dojis*, *Engulfing patterns*, *Morning/Evening Stars*).
#
# While practitioners traditionally formulate these rules algorithmically or through discrete feature engineering on 1D series, framing this as a **2D Computer Vision Object Detection problem** provides profound pedagogical insights:
#
# 1. **Visual Pattern Grounding:** How spatial convolution and multi-scale feature pyramids capture multi-candle spatial-temporal relationships directly from pixels without manual feature engineering.
# 2. **Programmatic Ground Truth (Zero Manual Annotation):** How deterministic domain rules (e.g., TA-Lib definitions) can generate thousands of labeled bounding box images automatically.
# 3. **The "Sim-to-Real" & Label Imitation Gap:** A trained model learns to *imitate the rule recognizer*, highlighting the vital epistemological distinction between visual pattern replication and genuine predictive market alpha.

# %%
from __future__ import annotations
import os
import sys
from pathlib import Path

# Add current experiment folder to path
sys.path.insert(0, str(Path.cwd()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

# Import experiment modules
from patterns import (
    ID_TO_PATTERN,
    PATTERN_NAMES,
    PATTERN_TO_ID,
    detect_all_patterns,
)
from renderer import render_candlestick_chart
from data_generator import build_candlestick_dataset, generate_synthetic_ohlc, fetch_real_ohlc
from train import train_yolo_detector
from evaluate import evaluate_model, plot_ground_truth_vs_prediction, run_domain_shift_benchmark

# Set plotting style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
print("Dependencies loaded successfully!")
print(f"Taxonomy ({len(PATTERN_NAMES)} classes): {PATTERN_NAMES}")

# %% [markdown]
# ---
# ## 2. Candlestick Anatomy & Mathematical Formulation of Patterns
#
# A single candle $t$ represents the price evolution over a fixed interval $[t, t+\Delta t]$ with 4 scalar values:
# - $\text{Open}_t$ ($O_t$)
# - $\text{High}_t$ ($H_t$)
# - $\text{Low}_t$ ($L_t$)
# - $\text{Close}_t$ ($C_t$)
#
# ### Geometric Definitions:
# $$\text{Body}_t = |C_t - O_t|$$
# $$\text{Range}_t = H_t - L_t$$
# $$\text{UpperShadow}_t = H_t - \max(O_t, C_t)$$
# $$\text{LowerShadow}_t = \min(O_t, C_t) - L_t$$
#
# ### The 8 Target Pattern Formulations:
# 1. **Doji (Indecision):** $\text{Body}_t \le 0.10 \times \text{Range}_t$
# 2. **Hammer (Bullish Reversal):** $\text{Body}_t \le 0.35 \times \text{Range}_t$, $\text{LowerShadow}_t \ge 2 \times \text{Body}_t$, $\text{UpperShadow}_t \le 0.25 \times \text{Body}_t$ following a downtrend.
# 3. **Inverted Hammer (Bullish Reversal):** Small body near low, $\text{UpperShadow}_t \ge 2 \times \text{Body}_t$, minimal lower shadow in a downtrend.
# 4. **Shooting Star (Bearish Reversal):** Small body near low, $\text{UpperShadow}_t \ge 2 \times \text{Body}_t$ following an uptrend.
# 5. **Bullish Engulfing (2-candle):** Candle 1 is bearish; Candle 2 is bullish with $O_2 \le C_1$ and $C_2 \ge O_1$.
# 6. **Bearish Engulfing (2-candle):** Candle 1 is bullish; Candle 2 is bearish with $O_2 \ge C_1$ and $C_2 \le O_1$.
# 7. **Morning Star (3-candle):** Long bearish candle $\to$ gapping small-body star $\to$ long bullish candle penetrating the upper half of Candle 1.
# 8. **Evening Star (3-candle):** Long bullish candle $\to$ gapping small-body star $\to$ long bearish candle penetrating the lower half of Candle 1.

# %% [markdown]
# ---
# ## 3. Synthetic Data Synthesis & Ground-Truth Bounding Box Mapping
#
# To train our detector without expensive manual annotation, we generate synthetic Geometric Brownian Motion (GBM) price series:
# $$S_{t+\Delta t} = S_t \exp\left( (\mu - \frac{1}{2}\sigma^2)\Delta t + \sigma \sqrt{\Delta t} Z_t \right), \quad Z_t \sim \mathcal{N}(0, 1)$$
#
# ### Bounding Box Coordinate Mapping
# For a chart window spanning $N$ candles (indices $0, \dots, N-1$) and price axis $[P_{\min}, P_{\max}]$, a detected pattern spanning candles $[t_{\text{start}}, t_{\text{end}}]$ with extreme prices $[L^*, H^*]$ maps to normalized YOLO coordinates $(x_c, y_c, w, h) \in [0, 1]^4$:
#
# $$x_c = \frac{t_{\text{start}} + t_{\text{end}} + 1}{2N}, \quad w = \frac{t_{\text{end}} - t_{\text{start}} + 1}{N}$$
# $$y_c = \frac{P_{\max} - \frac{L^* + H^*}{2}}{P_{\max} - P_{\min}}, \quad h = \frac{H^* - L^*}{P_{\max} - P_{\min}}$$

# %%
# Generate a sample synthetic price series and inspect detected ground truth
df_sample = generate_synthetic_ohlc(n_candles=30, seed=42)
sample_patterns = detect_all_patterns(df_sample)

print(f"Sample window generated with {len(df_sample)} candles.")
print(f"Detected {len(sample_patterns)} patterns:")
for p in sample_patterns:
    print(f" - [{p.class_name}] at candles {p.start_idx}..{p.end_idx}, Price range: [{p.price_min:.2f}, {p.price_max:.2f}]")

# Render chart with normalized bounding boxes
img_rgb, bboxes = render_candlestick_chart(
    df=df_sample,
    patterns=sample_patterns,
    image_size=640,
    style="clean",
)

# Visualize ground truth rendering
gt_boxes = [(b.class_id, b.x_center, b.y_center, b.width, b.height) for b in bboxes]
fig = plot_ground_truth_vs_prediction(
    image=img_rgb,
    ground_truth_boxes=gt_boxes,
    predictions=None,
    title="Clean Rendered Chart with Ground Truth Annotations",
)
plt.show()

# %% [markdown]
# ---
# ## 4. Fast Dynamic Dataset Generation
#
# We generate a curated, reproducible training and validation dataset with train/val/test splits (70% / 20% / 10%) and output a standard `data.yaml` file for the YOLO engine.

# %%
dataset_dir = Path("data/candlestick_dataset")
print("Building dataset...")
stats = build_candlestick_dataset(
    output_dir=dataset_dir,
    num_images=300,
    window_size=30,
    image_size=640,
    train_ratio=0.70,
    val_ratio=0.20,
    test_ratio=0.10,
    include_yfinance=True,
    seed=42,
)

print(f"\nDataset construction complete!")
print(f"Total Images: {stats['total_images']}")
print(f"Train: {stats['train_images']} | Val: {stats['val_images']} | Test: {stats['test_images']}")
print("\nClass distribution:")
for name in PATTERN_NAMES:
    print(f"  - {name:20s}: {stats[name]} instances")

# %% [markdown]
# ---
# ## 5. Detector Architecture: Single-Stage YOLO
#
# Modern single-stage detectors (such as YOLOv8 / YOLOv11) formulate object detection as a dense spatial regression and classification task:
#
# ```text
# Image (640x640x3)
#        │
# ┌──────▼─────────────────────────────────────────────────┐
# │ Backbone: CSPDarknet (Multi-scale feature extraction)   │
# │ P3 (80x80), P4 (40x40), P5 (20x20)                     │
# └──────┬─────────────────────────────────────────────────┘
#        │
# ┌──────▼─────────────────────────────────────────────────┐
# │ Neck: PANet / BiFPN (Top-down & Bottom-up aggregation) │
# └──────┬─────────────────────────────────────────────────┘
#        │
# ┌──────▼─────────────────────────────────────────────────┐
# │ Head: Decoupled Anchor-Free Detection Head             │
# │ ├── Classification Branch: BCE Loss                    │
# │ └── Box Regression Branch: CIoU + Distribution Focal    │
# └────────────────────────────────────────────────────────┘
# ```
#
# ### Key Mathematical Loss Components:
# 1. **Complete IoU Loss ($\mathcal{L}_{\text{CIoU}}$):**
#    $$\mathcal{L}_{\text{CIoU}} = 1 - \text{IoU} + \frac{\rho^2(b, b^{gt})}{c^2} + \alpha v$$
#    where $\rho$ is the Euclidean distance between center points, $c$ is the diagonal length of the smallest enclosing box, and $v$ measures aspect ratio consistency.
# 2. **Distribution Focal Loss ($\mathcal{L}_{\text{DFL}}$):** Learns continuous box boundary distributions around anchor points to handle uncertain edge boundaries.
# 3. **Binary Cross Entropy ($\mathcal{L}_{\text{cls}}$):** Multi-class probability scoring assigned via the **Task-Aligned Assigner**.

# %% [markdown]
# ---
# ## 6. Model Training & Fine-Tuning
#
# We fine-tune a lightweight `yolov8n.pt` model pretrained on COCO. Pretrained weights provide general edge, gradient, and corner detectors that rapidly adapt to financial line geometry.

# %%
data_yaml_path = dataset_dir / "data.yaml"

print("Starting YOLO fine-tuning on candlestick patterns...")
model, train_results = train_yolo_detector(
    data_yaml_path=data_yaml_path,
    model_variant="yolov8n.pt",
    epochs=12,
    imgsz=640,
    batch_size=16,
    project_dir="runs/detect",
    experiment_name="yolo_candles_run",
    seed=42,
    verbose=True,
)
print("Fine-tuning completed successfully!")

# %% [markdown]
# ---
# ## 7. Quantitative & Qualitative Evaluation
#
# We evaluate the fine-tuned detector on the held-out test split using standard COCO metrics ($mAP@0.50$, $mAP@0.50:0.95$, Precision, Recall).

# %%
# Quantitative evaluation on Test split
metrics = evaluate_model(model, data_yaml_path=data_yaml_path, split="test", imgsz=640, conf=0.15)

print("=" * 45)
print("TEST SET EVALUATION METRICS:")
print("=" * 45)
print(f"  mAP @ IoU 0.50       : {metrics['mAP50']:.4f}")
print(f"  mAP @ IoU 0.50..0.95 : {metrics['mAP50_95']:.4f}")
print(f"  Mean Precision       : {metrics['precision']:.4f}")
print(f"  Mean Recall          : {metrics['recall']:.4f}")
print("=" * 45)

# %% [markdown]
# ### Qualitative Inferences: Ground Truth vs. YOLO Predictions

# %%
test_img_dir = dataset_dir / "images" / "test"
test_lbl_dir = dataset_dir / "labels" / "test"
test_img_paths = sorted(list(test_img_dir.glob("*.png")))

# Display first 3 test samples
for sample_path in test_img_paths[:3]:
    lbl_path = test_lbl_dir / (sample_path.stem + ".txt")
    gt_boxes = []
    if lbl_path.exists():
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id, xc, yc, bw, bh = map(float, parts)
                    gt_boxes.append((int(cls_id), xc, yc, bw, bh))

    # Inference
    pred_res = model.predict(str(sample_path), conf=0.15, verbose=False)[0]
    preds = []
    if pred_res.boxes is not None:
        for b in pred_res.boxes:
            cls_id = int(b.cls.item())
            conf = float(b.conf.item())
            xyxy = b.xyxy.cpu().numpy()[0].tolist()
            preds.append({"class_id": cls_id, "conf": conf, "xyxy": xyxy})

    fig = plot_ground_truth_vs_prediction(
        image=sample_path,
        ground_truth_boxes=gt_boxes,
        predictions=preds,
        title=f"Evaluation: {sample_path.name}",
    )
    plt.show()

# %% [markdown]
# ---
# ## 8. Sim-to-Real Gap & Domain Shift Robustness Analysis
#
# Real-world trading platforms (TradingView, Bloomberg, Interactive Brokers) present charts with diverse visual features not present in clean synthetic training charts:
# - Dark themes vs Light themes
# - Gridlines and watermark logos
# - Subplot volume histograms
# - Text annotations and cursor crosshairs
#
# Below, we test our clean-trained model against controlled domain shifts and real broker screenshot mockups.

# %%
# Generate synthetic test windows with various domain shifts
test_shifts_df = [generate_synthetic_ohlc(n_candles=30, seed=1000 + i) for i in range(15)]
shift_results = run_domain_shift_benchmark(
    model=model,
    test_df_list=test_shifts_df,
    styles=["clean", "dark", "grid", "volume", "noisy"],
)

print("\n--- DOMAIN SHIFT BENCHMARK RESULTS ---")
print(shift_results.to_string(index=False))

# Plot performance degradation across visual styles
fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
bars = ax.bar(shift_results["Style / Domain Shift"], shift_results["Recall Ratio (Pred/GT)"], color="#1976D2", alpha=0.85)
ax.set_ylabel("Detection Recall Ratio", fontsize=11, fontweight="bold")
ax.set_title("Detector Robustness Degradation under Visual Domain Shifts", fontsize=12, fontweight="bold")
ax.set_ylim(0, 1.2)
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.03, f"{yval:.2f}", ha="center", va="bottom", fontweight="bold")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Inference on Curated Real-World Broker Mockups

# %%
broker_assets_dir = Path("assets/real_charts")
broker_mockups = sorted(list(broker_assets_dir.glob("*.png")))

for mockup_path in broker_mockups:
    pred_res = model.predict(str(mockup_path), conf=0.20, verbose=False)[0]
    preds = []
    if pred_res.boxes is not None:
        for b in pred_res.boxes:
            cls_id = int(b.cls.item())
            conf = float(b.conf.item())
            xyxy = b.xyxy.cpu().numpy()[0].tolist()
            preds.append({"class_id": cls_id, "conf": conf, "xyxy": xyxy})

    fig = plot_ground_truth_vs_prediction(
        image=mockup_path,
        ground_truth_boxes=None,
        predictions=preds,
        title=f"Sim-to-Real Inference: {mockup_path.name}",
    )
    plt.show()

# %% [markdown]
# ---
# ## 9. Critical Epistemological Discussion: Pattern Recognition vs. Alpha
#
# > **Key Takeaway for Students:**
# >
# > 1. **Rule Imitator vs Signal Discovery:** Our deep learning model was trained with bounding boxes generated by deterministic rule algorithms (TA-Lib). Therefore, a model achieving 98% mAP has not discovered an edge in financial markets; it has simply learned a high-fidelity visual surrogate for TA-Lib rules.
# > 2. **Domain Invariance:** Synthetic charts with clean backgrounds yield high accuracy, but real trading interfaces suffer severe precision loss unless domain randomization (dark mode, grids, fonts) is incorporated into data augmentation.
# > 3. **The Multi-Scale Advantage:** Unlike rigid scalar rules that struggle when aspect ratios or volatility scale shifts, spatial convolutions learn scale-invariant spatial proportions (e.g. wick-to-body ratios).
