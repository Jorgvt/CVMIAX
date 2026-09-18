# Experiment 08: Time-Series Image Encoding & Regime Classification

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jorgvt/CVMIAX/blob/main/experiments/08_time_to_img/experiment_08_time_to_img.ipynb)

## 1. Motivation & Pedagogical Objectives

Traditional financial machine learning methods often flatten temporal sequences into hand-crafted tabular features (moving averages, RSI, Bollinger Bands) or feed them into 1D recurrent/autoregressive models (LSTMs, GRUs, Transformers). 

This experiment explores an alternative paradigm: **transforming 1D non-stationary financial time series into 2D spatial-spectral image representations** to leverage state-of-the-art Computer Vision architectures (CNNs, ResNets, Transfer Learning backbones).

### Core Learning Objectives:
1. **Understand 1D-to-2D Transformations**:
   - **Gramian Angular Fields (GAF)**: Preserves temporal geometry and correlations in polar coordinates ($[-1, 1] \to [0, \pi]$).
   - **Continuous Wavelet Transform (CWT Scalograms)**: Multi-resolution time-frequency decomposition with complex Morlet wavelets.
   - **Short-Time Fourier Transform (STFT Spectrograms)**: Fixed-window harmonic energy distributions.
   - **Multi-Channel Vision Fusion**: Constructing composite 3-channel RGB tensors combining temporal trends (GASF), rate-of-change (GADF), and spectral volatility energy (CWT).
2. **Causal Market Regime Formulation**:
   - Classify 128-bar market windows into 4 statistically grounded regimes (*Low-Vol Bull*, *Low-Vol Bear*, *High-Vol Choppy*, *Volatility Shock*) using strictly backward-looking metrics and calendar-date temporal splits (Train $< 2019$, Val $2019-2021$, Test $\ge 2021$) to prevent look-ahead bias and asset contamination.
3. **The "Transform Tournament"**:
   - Benchmark a **Custom Lightweight ResNet-18** against a **Pretrained MobileNetV2** across all visual representations.
4. **Diagnose Visual Artefacts & Mathematical Trade-Offs**:
   - Mathematically analyze the **Normalization Scale Trap** (how per-sample min-max scaling destroys volatility amplitude).
   - Analyze the **Cone of Influence (COI)**.
   - Use **Grad-CAM saliency heatmaps** to interpret what spatial/spectral features the CNN focuses on.

---

## 2. Mathematical Formulations

### A. Gramian Angular Fields (GAF)
Given a 1D time series $X = \{x_1, x_2, \dots, x_N\}$, the values are rescaled to $[-1, 1]$:
$$\tilde{x}_i = \frac{(x_i - \max(X)) + (x_i - \min(X))}{\max(X) - \min(X)}$$

Each point is mapped to polar coordinates where the timestamp $t_i$ defines the radius $r_i$ and the scaled amplitude defines the angle $\phi_i$:
$$\phi_i = \arccos(\tilde{x}_i), \quad r_i = \frac{t_i}{N}$$

The **Gramian Angular Summation Field (GASF)** and **Gramian Angular Difference Field (GADF)** represent pairwise temporal inner products:
$$\text{GASF}_{i,j} = \cos(\phi_i + \phi_j) = \tilde{x}_i \tilde{x}_j - \sqrt{1 - \tilde{x}_i^2}\sqrt{1 - \tilde{x}_j^2}$$
$$\text{GADF}_{i,j} = \sin(\phi_i - \phi_j) = \sqrt{1 - \tilde{x}_i^2}\tilde{x}_j - \tilde{x}_i\sqrt{1 - \tilde{x}_j^2}$$

* **Property**: Diagonal elements $\text{GASF}_{i,i} = \cos(2\phi_i) = 2\tilde{x}_i^2 - 1$ capture the normalized series trajectory, while off-diagonals preserve relative temporal correlations.

---

### B. Continuous Wavelet Transform (CWT Scalogram)
The Continuous Wavelet Transform projects the signal onto a family of dilated and translated wavelets:
$$W(a, b) = \frac{1}{\sqrt{|a|}} \int_{-\infty}^{\infty} x(t) \, \psi^*\left(\frac{t - b}{a}\right) dt$$
where $a > 0$ is the scale parameter (inversely related to frequency), $b$ is the time shift, and $\psi(t)$ is the mother wavelet (e.g., complex Morlet $\psi(t) = \pi^{-1/4} e^{i \omega_0 t} e^{-t^2/2}$).

The **Scalogram** computes the localized energy density:
$$P(a, b) = |W(a, b)|^2$$

---

### C. Short-Time Fourier Transform (STFT Spectrogram)
The STFT applies a windowing function $w[n]$ (e.g., Hann window) to segment the series into overlapping blocks before calculating the Discrete Fourier Transform:
$$\text{STFT}(m, \omega) = \sum_{k=-\infty}^{\infty} x[k] \, w[k - m] \, e^{-j \omega k}$$

The **Log-Power Spectrogram** is:
$$S(m, \omega) = \log\left(1 + 1000 \cdot |\text{STFT}(m, \omega)|^2\right)$$

---

## 3. Market Regime Definition & Causal Design

To eliminate lookahead bias and data leakage:
- Input features are computed **strictly within the 128-bar window**.
- Chronological train/validation/test splits are partitioned by **Calendar Date** across all assets:
  - **Train**: Dates $< 2019-01-01$
  - **Validation**: Dates $2019-01-01 \le t < 2021-06-01$
  - **Out-of-Time Test**: Dates $\ge 2021-06-01$

| Class Index | Regime Name | Statistical Condition |
|:---:|:---|:---|
| **0** | **Low-Vol Bull** | Realized Volatility $< \text{Med}(\sigma)$ AND Cumulative Return $> +\delta$ |
| **1** | **Low-Vol Bear** | Realized Volatility $< \text{Med}(\sigma)$ AND Cumulative Return $< -\delta$ |
| **2** | **High-Vol Choppy** | Realized Volatility $\ge \text{Med}(\sigma)$ OR Return within $[-\delta, +\delta]$ |
| **3** | **Volatility Shock** | Realized Volatility $\ge \text{P}_{75}(\sigma)$ (Tail-risk jump) |

---

## 4. Key Methodological Lessons & Diagnostic Findings

### 1. The Normalization Trap (Per-Sample vs Global Scaling)
* **The Pitfall**: In naive colormapping, each image is min-max scaled independently to $[0, 1]$. Because a quiet day has variance $\sim 10^{-5}$ and a crash has variance $\sim 10^{-3}$, per-sample min-max normalization scales both to the exact same $[0, 1]$ intensity, completely destroying the $40\times\text{--}100\times$ energy difference.
* **The Solution**: Maintain a dataset-level global scaling reference $V_{\max}$ (e.g., 99th percentile across training power) so that quiet regimes appear dark/subdued and shocks glow with high energy.

### 2. Symmetry vs Directionality
* **CWT and STFT on returns** ($|W(a,b)|^2$) are sign-symmetric: $|+r|^2 = |-r|^2$. They identify volatility energy accurately, but cannot distinguish an upward drift (Bull) from a downward drift (Bear).
* **GAF on cumulative prices** provides the missing polar trajectory angles $\phi = \arccos(\tilde{x})$.
* **Multi-Channel Fusion** (Channel 0: GASF Trend, Channel 1: GADF Dynamics, Channel 2: CWT Energy) combines both advantages, achieving the highest performance (**~80% Accuracy, ~0.79 Macro F1**).

### 3. The Cone of Influence (COI)
Wavelet convolutions near the edges of finite windows suffer from boundary truncation. The **Cone of Influence** marks the region where boundary distortion exceeds $e^{-2}$, warning practitioners against treating edge filter flare as genuine economic signals.

---

## 5. Tournament Benchmark Results

| Representation | Architecture | Test Accuracy | Macro F1-Score | Key Strength |
|:---|:---|:---:|:---:|:---|
| **FUSION (GASF+GADF+CWT)** | **Custom ResNet-18** | **79.7%** | **0.787** | **Best overall (unites trend geometry + volatility energy)** |
| **CWT Scalogram** | Pretrained MobileNetV2 | 75.2% | 0.426 | High precision on Volatility Shocks & Choppy regimes |
| **STFT Spectrogram** | Pretrained MobileNetV2 | 66.4% | 0.494 | Robust frequency energy detection |
| **GAF (GASF)** | Pretrained MobileNetV2 | 43.0% | 0.238 | Directional trend orientation |

---

## 6. How to Run

### Execute the Full Automated Pipeline
```bash
uv run python experiments/08_time_to_img/run_all.py --epochs 12 --batch_size 32
```

### Run an Individual Training Run
```bash
# Train Custom ResNet-18 on Multi-Channel Fusion
uv run python experiments/08_time_to_img/train.py --method fusion --model resnet18 --epochs 15

# Train Pretrained MobileNetV2 on CWT Scalograms
uv run python experiments/08_time_to_img/train.py --method cwt --model mobilenet_v2 --epochs 15
```

### Launch the Interactive Lesson Notebook
```bash
uv run jupyter lab experiments/08_time_to_img/experiment_08_time_to_img.ipynb
```
