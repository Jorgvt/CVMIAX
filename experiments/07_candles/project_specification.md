# Project Specification — Candlestick Pattern Detection

**Course:** Deep Learning for Economics & Trading (Computer Vision module)
**Deliverable type:** Hands-on example for the Object Detection lecture
**Format:** Self-contained, reproducible Python pipeline
**Author:** Jorge Vila
**Date:** September 2026

---

## 1. Motivation & Learning Objectives

This project demonstrates how an **object detection model** can locate and classify
classical **technical-analysis patterns** directly on candlestick (OHLC) charts. It is
designed as a worked example within the CV module of a deep-learning course aimed at
students with an economics/trading background.

By the end of the example, students should be able to:

1. Frame a financial-chart reading task as a computer-vision **object detection** problem
   (bounding boxes + class labels over rendered chart images).
2. Build a **labeled image dataset programmatically**, with ground-truth annotations
   obtained for free from a rule-based pattern recognizer.
3. Train and evaluate a modern detector (YOLO family) end-to-end.
4. Critically discuss the **limitations** of rule-based labels and the gap between
   pattern recognition and genuine market predictability.

> **Pedagogical note.** Because labels come from a deterministic recognizer (TA-Lib),
> the trained detector is effectively learning to *imitate* TA-Lib, not discovering
> "true" market signals. This is intentionally surfaced as a discussion point, and is
> contrasted with a held-out set of real broker screenshots (as in the Stanford
> pipeline referenced below).

---

## 2. Approach: Build a Synthetic, Auto-Labeled Dataset

After surveying available public datasets, the project builds its **own** dataset
rather than consuming a pre-built one. Rationale:

| Option | Verdict |
|---|---|
| Roboflow "Candlestick Pattern Recognition" (5,873 images, YOLO/COCO) | Rejected as primary: community-uploaded, unclear class taxonomy, unverified label quality, unclear license. Unsuitable to hand to students as a black box. |
| ChartScanAI (GitHub, MIT) | Rejected: only 2 classes (Buy/Sell) — a trading signal, not a pattern taxonomy. |
| Stanford CS231n "Fast Candlestick Patterns Detection" pipeline | Adopted as **methodological reference**: synthetic generation + auto-labeling + held-out real test images. Not a downloadable dataset. |

**Chosen pipeline** (fully automated, ground-truth labels, no manual annotation):

1. **Acquire OHLC data** — `yfinance` (real equities/forex) and/or synthetic
   GBM/ARIMA price series for controllable difficulty.
2. **Detect patterns programmatically with [TA-Lib](https://ta-lib.org/)** —
   TA-Lib exposes 60+ deterministic candlestick pattern recognizers that return the
   index of the candle(s) where each pattern occurs. These indices become
   **exact, ground-truth bounding-box coordinates**.
3. **Render candlestick charts with `mplfinance`** — strip axes, grids, and text so
   the model sees only the pattern (following the rendering protocol used in the
   [Stanford CS231n work](https://cs231n.stanford.edu/2025/papers/text_file_840597081-LaTeXAuthor_Guidelines_for_CVPR_Proceedings__1_-2.pdf)).
4. **Auto-generate bounding boxes** — draw a box around the candle window returned by
   TA-Lib; export in YOLO and COCO formats.
5. **Train a detector** — YOLOv8/v11 (or a small Faster R-CNN), tying directly into
   the existing Object Detection lecture (Faster R-CNN vs. YOLO/RetinaNet, FPN, NMS).

### Why this is better for teaching

- **Ground-truth labels with zero annotation effort** — deterministic recognizer ⇒
  exact, reproducible bounding boxes.
- **Controlled taxonomy** — instructor selects the N patterns to teach.
- **Tunable difficulty** — noise, trend, window length, candle count, chart styling
  can be varied to demonstrate generalization, augmentation, and failure modes.
- **No licensing/provenance issues** — fully owned, distributable with course materials.
- **End-to-end CV pipeline** — data → rendering → annotation → detection → evaluation.

---

## 3. Pattern Taxonomy (8 classes)

| ID | Pattern | TA-Lib function | Type |
|---:|---|---|---|
| 1 | Doji | `CDLDOJI` | Indecision |
| 2 | Hammer | `CDLHAMMER` | Bullish reversal |
| 3 | Inverted Hammer | `CDLINVERTEDHAMMER` | Bullish reversal |
| 4 | Morning Star | `CDLMORNINGSTAR` | Bullish reversal |
| 5 | Evening Star | `CDLEVENINGSTAR` | Bearish reversal |
| 6 | Bullish Engulfing | `CDLENGULFING` | Bullish reversal |
| 7 | Bearish Engulfing | `CDLENGULFING` | Bearish reversal |
| 8 | Shooting Star | `CDLSHOOTINGSTAR` | Bearish reversal |

All eight are well-covered by TA-Lib, visually distinct, and standard in the
candlestick literature. A background class (no pattern) is implicitly handled by
YOLO's objectness mechanism.

---

## 4. Dataset Specification

| Field | Value |
|---|---|
| Image source | Rendered programmatically via `mplfinance` from OHLC data |
| Label source | TA-Lib recognizer outputs (deterministic, ground-truth) |
| Annotation format | YOLO TXT + `data.yaml`; COCO JSON (secondary) |
| Image size | 640×640 (YOLO native) |
| Channel format | RGB (axes/grids/text stripped) |
| Target volume | ~5,000–8,000 labeled images |
| Splits | Train 70% / Val 20% / Test 10% |
| Augmentations | Applied at training time (mosaic, flip, color jitter) — not baked into the dataset |
| Held-out test set | ~200 real broker/chart screenshots, manually labeled, to measure real-world generalization (cf. Stanford pipeline) |

### Dataset directory layout

```
candlestick_dataset/
├── data.yaml                 # YOLO class names + paths
├── images/
│   ├── train/  *.png
│   ├── val/    *.png
│   └── test/   *.png
├── labels/
│   ├── train/  *.txt         # YOLO: class x_center y_center w h (normalized)
│   ├── val/    *.txt
│   └── test/   *.txt
└── coco_annotations.json     # COCO-format mirror for Detectron2/TF
```

---

## 5. Technical Stack

| Component | Library / Tool |
|---|---|
| OHLC data | `yfinance` |
| Synthetic price series | `numpy` + `statsmodels` (ARIMA) for augmentation |
| Pattern recognition | [TA-Lib](https://ta-lib.org/) (Python bindings) |
| Chart rendering | `mplfinance` |
| Annotation / dataset I/O | `pylabel` or custom YOLO/COCO writer |
| Detection model | `ultralytics` (YOLOv8/v11) |
| Evaluation | `pycocotools` + `ultralytics` metrics |
| Environment | Python 3.11, conda/pip, Jupyter (for the lecture notebook) |

---

## 6. Pipeline Stages (Build → Train → Evaluate)

**Stage 1 — Data acquisition.** Download daily/hourly OHLC for a basket of liquid
symbols (e.g. AAPL, MSFT, EURUSD, SPY) over multiple years via `yfinance`. Optionally
generate synthetic ARIMA series for additional coverage and controllable noise.

**Stage 2 — Pattern labeling.** Run TA-Lib recognizers over each OHLC series. For
each detected pattern, record the candle index range and map it to one of the 8
classes. Discard overlaps and ambiguous windows.

**Stage 3 — Rendering.** For each labeled window, render a fixed-size candlestick
chart (±N candles around the pattern) with `mplfinance`, stripping axes/grid/text.
Compute the bounding box in normalized image coordinates from the pattern's candle
window.

**Stage 4 — Dataset assembly.** Write YOLO TXT labels + `data.yaml` and a COCO JSON
mirror. Split 70/20/10, stratified by class to handle imbalance.

**Stage 5 — Training.** Fine-tune a YOLOv8n/v11n pretrained on COCO. Use mosaic +
flip + color-jitter augmentation. Report mAP@0.5 and mAP@0.5:0.95 per class.

**Stage 6 — Evaluation & discussion.**
- Quantitative: mAP per class, confusion matrix, precision/recall.
- Qualitative: failure cases (overlapping patterns, heavy noise, long trends).
- Generalization test: run the trained detector on the held-out real screenshots and
  compare against rule-based labels — a direct measure of the sim-to-real gap.

---

## 7. Deliverables

1. **Dataset generation script** — `generate_dataset.py`: yfinance → TA-Lib →
   mplfinance → YOLO/COCO export. Configurable symbol list, date range, pattern
   classes, image size, and split ratios.
2. **Rendered sample charts** — a small visual sample (e.g. one per class) so the
   output can be inspected before committing to a full run.
3. **Training notebook** — `train_detector.ipynb`: loads the dataset, fine-tunes
   YOLOv8, logs metrics, and saves weights.
4. **Evaluation notebook** — `evaluate.ipynb`: mAP, confusion matrix, qualitative
   plots, and the held-out real-image generalization test.
5. **This specification handout** — `project_specification.md`.

---

## 8. Risks & Honest Limitations

- **Rule-based label ceiling.** The detector learns TA-Lib's definitions; it cannot
  exceed TA-Lib's own (imperfect) agreement with human chartists. State this openly.
- **Class imbalance.** Doji/Hammer are common; Stars and Abandoned Babies are rare.
  Mitigate with stratified sampling and class-weighted augmentation.
- **Sim-to-real gap.** Rendered charts differ from real broker screenshots (fonts,
  colors, gridlines). The held-out real test set quantifies this gap explicitly.
- **No financial claim.** Pattern recognition ≠ predictive alpha. The example is a
  CV/detection exercise, not a trading strategy.

---

## 9. References

- TA-Lib — Technical analysis library with 60+ candlestick pattern recognizers. [ta-lib.org](https://ta-lib.org/)
- Stanford CS231n (2022) — *Fast Candlestick Patterns Detection with Limited Training Samples*. Synthetic-generation + YOLO-LITE pipeline, 8 pattern classes, 200 real test images. [report PDF](https://cs231n.stanford.edu/reports/2022/pdfs/157.pdf)
- Stanford CS231n (2025) — *Learning Predictive Candlestick Patterns* (Vision Transformers on mplfinance-rendered charts). [paper](https://cs231n.stanford.edu/2025/papers/text_file_840597081-LaTeXAuthor_Guidelines_for_CVPR_Proceedings__1_-2.pdf)
- Roboflow — *Candlestick Pattern Recognition* dataset (5,873 images, YOLO/COCO/VOC). [dataset](https://universe.roboflow.com/ranyas-workspace/candlestick-pattern-recognition/dataset/1)
- ChartScanAI — YOLOv8 pattern-detection app (MIT), `yfinance` + `mplfinance` + Roboflow. [GitHub](https://github.com/Omar-Karimov/ChartScanAI)
- UCL Discovery — *Deep Candlestick Mining* (DCM): LSTM + RDT + k-means++ for asset-specific pattern discovery. [PDF](https://discovery.ucl.ac.uk/10062933/1/ICONIP_Deep_Candlestick_Mining_update_final.pdf)
