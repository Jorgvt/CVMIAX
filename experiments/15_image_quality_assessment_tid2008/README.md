# Experiment 15: Image Quality Assessment (IQA) on TID2008

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jorgvt/CVMIAX/blob/main/experiments/15_image_quality_assessment_tid2008/experiment_15_image_quality_assessment_tid2008.ipynb)

---

## Pedagogical Objectives & Motivation

Visual quality perceived by human observers is subjective yet exhibits strong statistical regularities across populations. In computer vision, **Image Quality Assessment (IQA)** algorithms predict subjective quality scores (Mean Opinion Score, **MOS**) from digital images.

This experiment provides an end-to-end pedagogical implementation of modern IQA using the **TID2008** benchmark dataset (hosted on Hugging Face at [`Jorgvt/TID2008`](https://huggingface.co/datasets/Jorgvt/TID2008)).

Key educational goals:
1. **Perceptual vs. Signal-Level Disconnect**: Understand why classical pixel-error metrics (MSE, PSNR) fail fundamentally on blur, contrast shifts, and localized artifacts, while deep neural features naturally correlate with the human visual system (HVS).
2. **Full-Reference (FR-IQA) vs. No-Reference (NR-IQA / Blind IQA)**: Compare dual-stream Siamese difference feature extractors against single-stream blind regressors.
3. **Content-Independent Splitting**: Learn why reference-wise splitting is mandatory in IQA to prevent models from memorizing scene semantics rather than distortion characteristics.
4. **Hierarchical Perceptual Error Maps**: Visualize spatial activations $|\phi(I_{\text{ref}}) - \phi(I_{\text{dist}})|$ across shallow, mid, and deep convolutional layers.
5. **Standard IQA Evaluation Metrics**: Formulate and compute Spearman Rank Order Correlation (**SROCC**), Kendall Tau (**KROCC**), and Pearson Linear Correlation (**PLCC**) with 4-parameter logistic calibration.

---

## Mathematical Formulation

```
                                    +-----------------------+
                                    | Reference Image (I_r) |
                                    +-----------+-----------+
                                                |
                                                v
+-----------------------+           +-----------+-----------+
| Distorted Image (I_d) |           |  Shared Conv Backbone |
+-----------+-----------+           +-----------+-----------+
            |                                   |
            v                                   v
+-----------+-----------+                 phi_k(I_r)
|  Shared Conv Backbone |                       |
+-----------+-----------+                       |
            |                                   |
            v                                   |
        phi_k(I_d)                              |
            |                                   |
            +-----------------+-----------------+
                              |
                              v
                Delta_k = |phi_k(I_r) - phi_k(I_d)|  <-- Multi-Scale Spatial Residuals
                              |
                              v
                GAP + Multi-Stage Concatenation
                              |
                              v
                  MLP Quality Regression Head
                              |
                              v
                 Predicted MOS (Quality Score)
```

### 1. Classical Metrics vs. Human Perception

- **Mean Squared Error (MSE)** and **Peak Signal-to-Noise Ratio (PSNR)**:
  $$\text{MSE} = \frac{1}{HWC}\sum_{i,j,c} \left(I_{\text{ref}}(i, j, c) - I_{\text{dist}}(i, j, c)\right)^2$$
  $$\text{PSNR} = 10 \cdot \log_{10}\left(\frac{\text{MAX}^2}{\text{MSE}}\right)$$
  *Limitation*: Assigns identical penalty to a uniform 1-pixel brightness shift as to severe high-frequency noise, despite humans perceiving them completely differently.

- **Structural Similarity Index (SSIM)**:
  $$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}$$
  *Advantage*: Separates luminance, contrast, and structure, improving correlation over PSNR.

### 2. Deep Full-Reference Siamese Multi-Scale Model

Let $\phi_k(\cdot)$ denote feature activation maps from convolutional stage $k \in \{1, 2, 3\}$. The multi-scale perceptual residual is defined as:
$$\Delta \phi_k = \left| \phi_k(I_{\text{ref}}) - \phi_k(I_{\text{dist}}) \right| \in \mathbb{R}^{H_k \times W_k \times C_k}$$

Spatial Global Average Pooling ($\text{GAP}$) collapses each stage into a representation vector:
$$v_k = \text{GAP}(\Delta \phi_k) \in \mathbb{R}^{C_k}$$

The multi-stage representation $v = [v_1 \parallel v_2 \parallel v_3]$ is fed to a multi-layer perceptron $g_\theta(v)$ to regress the scalar quality score:
$$\widehat{\text{MOS}} = g_\theta(v)$$

### 3. Evaluation Metrics for IQA Benchmarks

- **Spearman Rank-Order Correlation Coefficient (SROCC)**:
  Quantifies monotonic ranking agreement between predicted scores $\hat{y}$ and subjective ground truth $y$:
  $$\text{SROCC} = 1 - \frac{6 \sum_{i=1}^N d_i^2}{N(N^2 - 1)}$$
  where $d_i = \text{rank}(y_i) - \text{rank}(\hat{y}_i)$.

- **4-Parameter Logistic Non-Linear Mapping (ITU-R / VQEG Standard)**:
  Before computing linear metrics (PLCC, RMSE), predicted scores are mapped through a monotonic logistic function to account for saturation in subjective scoring:
  $$f(\hat{y}) = \frac{\beta_1 - \beta_2}{1 + \exp\left(\frac{\hat{y} - \beta_3}{|\beta_4|}\right)} + \beta_2$$

- **Pearson Linear Correlation Coefficient (PLCC)**:
  $$\text{PLCC} = \frac{\sum_{i=1}^N (y_i - \bar{y})(f(\hat{y}_i) - \bar{f})}{\sqrt{\sum_{i=1}^N (y_i - \bar{y})^2 \sum_{i=1}^N (f(\hat{y}_i) - \bar{f})^2}}$$

---

## Dataset & Content-Independent Splitting

The **TID2008** dataset contains:
- **25 Pristine Reference Scenes** (`reference_id` 1 to 25).
- **17 Distortion Types** (Gaussian Noise, Spatially Correlated Noise, Blur, JPEG, JPEG2000, Transmission Errors, Contrast Changes, etc.).
- **4 Distortion Intensity Levels** per type.
- **1,700 total evaluated images** with psychophysical MOS ground truth.

### The Content-Leakage Pitfall
If images sharing reference scene $R$ appear in both training and test sets, deep models can memorize specific high-contrast textures (e.g. sharp building edges vs smooth sky) to artificially boost performance.

To ensure strict zero-leakage evaluation:
- **Train Set**: Reference IDs 1–18 (18 scenes, 1,224 images, 72%)
- **Validation Set**: Reference IDs 19–20 (2 scenes, 136 images, 8%)
- **Test Set**: Reference IDs 21–25 (5 unseen scenes, 340 images, 20%)

---

## Codebase Organization

```
experiments/15_image_quality_assessment_tid2008/
├── README.md                                          # Complete pedagogical guide & theory
├── dataset.py                                         # HuggingFace TID2008 loader & content-split
├── metrics.py                                         # PSNR, SSIM, SROCC, PLCC, KROCC, logistic fit
├── models.py                                          # FR-IQA Siamese & NR-IQA Blind architectures
├── train.py                                           # Training loop with custom correlation callbacks
├── evaluate.py                                        # Quantitative benchmark suite
├── visualize.py                                       # Publication-quality figure generator
├── run_all.py                                         # Master CLI runner
├── experiment_15_image_quality_assessment_tid2008.py  # Jupytext master lesson script
├── experiment_15_image_quality_assessment_tid2008.ipynb # Synchronized interactive notebook
├── figures/                                           # Saved visualization figures
│   ├── 01_distortion_taxonomy_gallery.png
│   ├── 02_classical_vs_perceptual_correlation.png
│   ├── 03_training_dynamics.png
│   ├── 04_distortion_wise_performance_breakdown.png
│   └── 05_perceptual_error_heatmaps.png
└── weights/                                           # Saved model checkpoints
    ├── fr_iqa_model.weights.h5
    └── nr_iqa_model.weights.h5
```

---

## Key Visualizations

| Figure | Description |
| :--- | :--- |
| **01. Distortion Taxonomy Gallery** | Visual catalog of 1 reference scene across 4 representative distortions and 4 degradation intensities with MOS labels. |
| **02. Classical vs Perceptual Correlation** | 4-panel scatter plots comparing (PSNR vs MOS), (SSIM vs MOS), (Deep FR-IQA vs MOS), and (Deep NR-IQA vs MOS) on unseen test scenes. |
| **03. Training Dynamics** | Huber loss convergence and validation SROCC progression across training epochs. |
| **04. Distortion-Wise Breakdown** | Grouped bar chart illustrating SROCC rankings across Noise, Blur, Compression, and Tone shift categories. |
| **05. Perceptual Error Heatmaps** | Hierarchical spatial feature residual maps ($\Delta \phi_1, \Delta \phi_2, \Delta \phi_3$) highlighting artifact localization. |

---

## How to Run

### 1. Run Master Pipeline (CLI)
```bash
uv run python -m experiments.15_image_quality_assessment_tid2008.run_all
```

### 2. Launch Interactive Jupytext Notebook
```bash
# Sync paired notebook
uv run jupytext --sync experiments/15_image_quality_assessment_tid2008/experiment_15_image_quality_assessment_tid2008.py

# Open in VS Code or JupyterLab
```
