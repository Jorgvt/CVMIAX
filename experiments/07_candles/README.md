# Experiment 07: Candlestick Pattern Detection with Single-Stage Object Detectors

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jorgvt/CVMIAX/blob/main/experiments/07_candles/experiment_07_candlestick_detection.ipynb)

## 1. Overview & Pedagogical Objective
In quantitative finance and technical analysis, traders visually identify local geometric price formations (candlestick patterns) on financial charts. While classical practitioners rely on rigid scalar rules over 1D series, framing pattern recognition as a **2D Computer Vision Object Detection task** provides profound pedagogical insights into representation learning:

1. **Spatial Representation of Market Dynamics**: Multi-scale feature pyramids (FPN/PANet) and convolutions capture local candle geometry (wick-to-body ratios, relative candle height, gaps) directly from raw image pixels without manual feature engineering.
2. **Deterministic Auto-Labeling Pipeline**: Ground-truth bounding boxes are generated programmatically using domain rule algorithms (TA-Lib definitions), creating large-scale labeled datasets with zero human annotation effort.
3. **The "Sim-to-Real" & Label Imitation Gap**: Deep learning models trained on rule-derived bounding boxes learn to *imitate the rule recognizer*, highlighting the vital epistemological boundary between visual pattern replication and true predictive market alpha.

---

## 2. Mathematical Formulation & Architecture

### 2.1 Candlestick Anatomy
Each candle $t$ comprises $[\text{Open}_t, \text{High}_t, \text{Low}_t, \text{Close}_t]$:
$$\text{Body}_t = |C_t - O_t|$$
$$\text{Range}_t = H_t - L_t$$
$$\text{UpperShadow}_t = H_t - \max(O_t, C_t)$$
$$\text{LowerShadow}_t = \min(O_t, C_t) - L_t$$

### 2.2 Taxonomy (8 Pattern Classes)
1. **Doji**: $\text{Body}_t \le 0.10 \times \text{Range}_t$
2. **Hammer**: $\text{Body}_t \le 0.35 \times \text{Range}_t$, $\text{LowerShadow}_t \ge 2 \times \text{Body}_t$, $\text{UpperShadow}_t \le 0.25 \times \text{Body}_t$ in downtrend.
3. **Inverted Hammer**: $\text{UpperShadow}_t \ge 2 \times \text{Body}_t$, $\text{LowerShadow}_t \le 0.25 \times \text{Body}_t$ in downtrend.
4. **Shooting Star**: $\text{UpperShadow}_t \ge 2 \times \text{Body}_t$, $\text{LowerShadow}_t \le 0.25 \times \text{Body}_t$ in uptrend.
5. **Bullish Engulfing**: $O_2 \le C_1$ and $C_2 \ge O_1$ with $C_1 < O_1$ and $C_2 > O_2$.
6. **Bearish Engulfing**: $O_2 \ge C_1$ and $C_2 \le O_1$ with $C_1 > O_1$ and $C_2 < O_2$.
7. **Morning Star**: 3-candle bullish reversal (Long bear $\to$ small-body star gap $\to$ long bull).
8. **Evening Star**: 3-candle bearish reversal (Long bull $\to$ small-body star gap $\to$ long bear).

### 2.3 Normalized Bounding Box Mapping
For a window of $N$ candles spanning prices $[P_{\min}, P_{\max}]$, a pattern spanning $[t_{\text{start}}, t_{\text{end}}]$ and prices $[L^*, H^*]$ maps to YOLO coordinates $(x_c, y_c, w, h) \in [0, 1]^4$:
$$x_c = \frac{t_{\text{start}} + t_{\text{end}} + 1}{2N}, \quad w = \frac{t_{\text{end}} - t_{\text{start}} + 1}{N}$$
$$y_c = \frac{P_{\max} - \frac{L^* + H^*}{2}}{P_{\max} - P_{\min}}, \quad h = \frac{H^* - L^*}{P_{\max} - P_{\min}}$$

### 2.4 Single-Stage Detector Loss Formulation
$$\mathcal{L}_{\text{total}} = \lambda_{\text{box}} \mathcal{L}_{\text{CIoU}} + \lambda_{\text{dfl}} \mathcal{L}_{\text{DFL}} + \lambda_{\text{cls}} \mathcal{L}_{\text{BCE}}$$
- **CIoU**: Enforces bounding box overlap, center distance, and aspect ratio consistency:
  $$\mathcal{L}_{\text{CIoU}} = 1 - \text{IoU} + \frac{\rho^2(b, b^{gt})}{c^2} + \alpha v$$
- **DFL (Distribution Focal Loss)**: Optimizes probability distributions around continuous bounding box boundaries.
- **BCE**: Assigns class label probabilities to positive anchors determined by the Task-Aligned Assigner.

---

## 3. Experiment Structure

```
experiments/07_candles/
├── project_specification.md                  # Detailed course specification
├── README.md                                 # Theory, mathematical formulation, and guide
├── patterns.py                               # Pure-Python deterministic pattern detection rules
├── renderer.py                               # mplfinance chart renderer & bbox mapping
├── data_generator.py                         # GBM synthesis, yfinance fetching, dataset splits
├── train.py                                  # YOLOv8 fine-tuning routine
├── evaluate.py                               # mAP metrics, qualitative overlays, domain shift tests
├── run_all.py                                # Headless master execution script
├── experiment_07_candlestick_detection.py    # Master interactive lesson (Jupytext py:percent)
├── experiment_07_candlestick_detection.ipynb # Paired Jupyter notebook
└── assets/
    ├── generate_sample_assets.py             # Broker screenshot generator
    └── real_charts/                          # Sample broker chart screenshots (TradingView, Bloomberg)
```

---

## 4. Running the Experiment

### Prerequisites
Run with `uv`:
```bash
# Add dependencies
uv add ultralytics mplfinance yfinance pandas

# Run end-to-end headless pipeline
uv run python experiments/07_candles/run_all.py

# Launch interactive notebook
uv run jupyter lab experiments/07_candles/experiment_07_candlestick_detection.ipynb
```

---

## 5. Sim-to-Real Findings & Pedagogical Summary

1. **Rule Imitation**: The detector converges rapidly to high mAP ($>0.90$) on clean synthetic charts because it fits the exact deterministic pattern definitions provided by the auto-labeler.
2. **Domain Shift Vulnerability**: Evaluating clean-trained models on charts with dark backgrounds, gridlines, or broker watermarks causes notable drop in precision/recall, emphasizing the necessity of domain randomization in visual pipelines.
3. **Educational Takeaway**: Object detection offers a powerful visual mechanism to parse complex multi-scale formations, but practitioners must separate pattern detection precision from actual statistical market predictability.
