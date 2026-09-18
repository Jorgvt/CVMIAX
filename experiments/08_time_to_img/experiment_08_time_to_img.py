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
# # Experiment 08: Time-Series to Image Encoding & Market Regime Classification
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jorgvt/CVMIAX/blob/main/experiments/08_time_to_img/experiment_08_time_to_img.ipynb)
#
# ## 1. Pedagogical Motivation & Objectives
# Transforming 1D financial time series into 2D spatial and spectral image representations enables the application of powerful Computer Vision architectures (CNNs, Vision Transformers) to economic and financial data.
#
# In this hands-on laboratory, we:
# 1. **Encode windowed market dynamics** into four distinct mathematical 2D representations:
#    - **Gramian Angular Fields (GAF)**: Preserves temporal geometry and correlations in polar coordinates.
#    - **Continuous Wavelet Transform (CWT Scalograms)**: Multi-resolution time-frequency decomposition capturing localized volatility clustering.
#    - **Short-Time Fourier Transform (STFT Spectrograms)**: Fixed-window harmonic energy distributions.
#    - **Multi-Channel Fusion (GASF + GADF + CWT)**: Composite RGB tensor fusing temporal geometry, rate-of-change dynamics, and localized energy.
# 2. **Formulate strictly causal market regime classification** (Low-Vol Bull, Low-Vol Bear, High-Vol Choppy, Volatility Shock) free of lookahead bias using chronological date splits.
# 3. **Run a Transform Tournament**: Benchmark a **Custom Lightweight ResNet-18** against a **Pretrained MobileNetV2** backbone across all visual representations.
# 4. **Persist learned weights** in `weights/` for instant loading and offline evaluation without costly retraining.
# 5. **Diagnose mathematical pitfalls**: Analyze the **Cone of Influence (COI)**, investigate normalization scale preservation, and inspect **Grad-CAM** saliency maps.

# %%
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# Add experiment root to path for local imports
module_path = str(Path(".").resolve())
if module_path not in sys.path:
    sys.path.insert(0, module_path)

from encoders import compute_gaf, compute_cwt, compute_stft, apply_colormap, batch_encode_windows
from dataset import load_market_data, extract_asset_windows, compute_regime_labels, get_encoded_datasets, REGIME_NAMES
from models import build_custom_resnet18, build_pretrained_backbone, get_model, save_model_weights, load_model_weights
from evaluate import evaluate_model, compute_gradcam
from train import train_or_load_regime_classifier
from visualize import (
    plot_transformation_gallery,
    plot_coi_demonstration,
    plot_confusion_matrices,
    plot_tournament_summary,
    plot_gradcam_explanations,
)

print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version:      {keras.__version__}")
print(f"GPU Available:      {len(tf.config.list_physical_devices('GPU')) > 0}")

# %% [markdown]
# ---
# ## 2. Theoretical Foundations: 1D to 2D Encodings
#
# ### A. Gramian Angular Fields (GAF)
# Given a 1D sequence $X = \{x_1, x_2, \dots, x_N\}$, GAF scales the series into $[-1, 1]$:
# $$\tilde{x}_i = \frac{(x_i - \max(X)) + (x_i - \min(X))}{\max(X) - \min(X)}$$
# And encodes each point as a polar angle $\phi_i = \arccos(\tilde{x}_i)$.
#
# The **Gramian Angular Summation Field (GASF)** and **Gramian Angular Difference Field (GADF)** are defined as:
# $$\text{GASF}_{i,j} = \cos(\phi_i + \phi_j) = \tilde{x}_i \tilde{x}_j - \sqrt{1 - \tilde{x}_i^2}\sqrt{1 - \tilde{x}_j^2}$$
# $$\text{GADF}_{i,j} = \sin(\phi_i - \phi_j) = \sqrt{1 - \tilde{x}_i^2}\tilde{x}_j - \tilde{x}_i\sqrt{1 - \tilde{x}_j^2}$$
#
# ### B. Continuous Wavelet Transform (CWT Scalogram)
# The CWT convolves the signal with scaled and translated mother wavelets $\psi(t)$:
# $$W(a, b) = \frac{1}{\sqrt{|a|}} \int_{-\infty}^{\infty} x(t) \, \psi^*\left(\frac{t - b}{a}\right) dt$$
# The scalogram $P(a, b) = |W(a, b)|^2$ captures instantaneous frequency and volatility bursts at varying time scales $a$.
#
# ### C. Short-Time Fourier Transform (STFT Spectrogram)
# The STFT slices the signal with a sliding window $w[n]$:
# $$\text{STFT}(m, \omega) = \sum_{k} x[k] \, w[k - m] \, e^{-j \omega k}$$
# The log-spectrogram is given by $S(m, \omega) = \log(1 + |\text{STFT}(m, \omega)|^2)$.
#
# ### D. Multi-Channel Vision Fusion
# Real financial signals possess both directional trend geometry and multi-scale volatility. We construct a 3-channel visual tensor:
# - **Channel 0**: GASF (Cumulative price trend)
# - **Channel 1**: GADF (Price trajectory rate-of-change)
# - **Channel 2**: CWT Scalogram (Log-power volatility energy)

# %%
# Load multi-asset dataset
df = load_market_data("data/market_data.csv" if os.path.exists("data/market_data.csv") else "experiments/08_time_to_img/data/market_data.csv")
w_ret, w_cum, vols, rets, assets = extract_asset_windows(df, window_size=128, step_size=8)
labels, thresholds = compute_regime_labels(vols, rets)

print(f"Extracted {len(w_ret):,} windows of length 128.")
for k, name in REGIME_NAMES.items():
    count = int(np.sum(labels == k))
    print(f"  Class {k} ({name:18s}): {count:4d} ({count / len(labels):.1%})")

# %% [markdown]
# ---
# ## 3. Visual Gallery Across Market Regimes
#
# Let's inspect how the exact same financial window looks across 1D Price Drift, GASF, GADF, CWT Scalogram, and STFT Spectrogram:

# %%
fig_gallery = plot_transformation_gallery(w_ret, labels, image_size=128, save_path=None)
plt.show()

# %% [markdown]
# ---
# ## 4. The Cone of Influence (COI): Wavelet Boundary Artifacts
#
# In finite-length time series, wavelet filters extend beyond the edges of the window. At low frequencies (large scales), the wavelet is physically wider than the edge distance, introducing **boundary distortion**.
#
# The **Cone of Influence (COI)** defines the region where edge effects become significant (typically where wavelet power drops by a factor of $e^{-2}$). 

# %%
fig_coi = plot_coi_demonstration(w_ret[0], image_size=128, save_path=None)
plt.show()

# %% [markdown]
# ---
# ## 5. The Transform Tournament: Benchmarking Representations & Architectures
#
# We train or load cached weights for both **Custom ResNet-18** and **Pretrained MobileNetV2** across:
# 1. **GAF** (Gramian Angular Field)
# 2. **CWT** (Continuous Wavelet Scalogram)
# 3. **STFT** (Short-Time Fourier Spectrogram)
# 4. **FUSION** (Multi-Channel GASF + GADF + CWT)

# %%
# Tournament settings
EPOCHS = 12
BATCH_SIZE = 32
IMG_SIZE = 128

results = {}
methods = ["gaf", "cwt", "stft", "fusion"]
models_to_test = ["resnet18", "mobilenet_v2"]
tournament_records = []
best_models = {}

for method in methods:
    for model_type in models_to_test:
        exp_name = f"{method.upper()} ({model_type})"
        print(f"Running: {exp_name}")
        
        model, _, data_dict = train_or_load_regime_classifier(
            method=method,
            model_type=model_type,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            image_size=IMG_SIZE,
            force_train=False,
            verbose=0,
        )
        
        res = evaluate_model(model, data_dict["x_test"], data_dict["y_test"])
        print(f"  --> Test Accuracy: {res['accuracy']:.2%}, Macro F1: {res['macro_f1']:.4f}")
        
        results[exp_name] = res
        best_models[f"{method}_{model_type}"] = (model, data_dict)
        
        tournament_records.append({
            "Experiment": f"{method.upper()} + {model_type}",
            "Encoding": method.upper(),
            "Architecture": model_type,
            "Accuracy": res["accuracy"],
            "Macro_F1": res["macro_f1"],
        })

tournament_df = pd.DataFrame(tournament_records)

# %% [markdown]
# ### Tournament Benchmark Summary

# %%
tournament_df.sort_values(by="Macro_F1", ascending=False).reset_index(drop=True)

# %%
fig_summary = plot_tournament_summary(tournament_df, save_path=None)
plt.show()

# %% [markdown]
# ---
# ## 6. Comprehensive Confusion Matrix Grid (All 8 Models)
#
# Inspect how every combination of 2D visual representation and deep neural architecture behaves across all 4 market regimes:

# %%
fig_cm = plot_confusion_matrices(results, ncols=4, save_path=None)
plt.show()

# %% [markdown]
# ---
# ## 7. Model Explainability: Grad-CAM Attention Heatmaps
#
# Grad-CAM highlights the exact visual regions driving the model's regime decision:

# %%
fusion_model, fusion_data = best_models["fusion_resnet18"]
fig_gradcam = plot_gradcam_explanations(
    fusion_model,
    fusion_data["x_test"][:4],
    fusion_data["y_test"][:4],
    method_name="Fusion (ResNet-18)",
    num_samples=4,
    save_path=None,
)
plt.show()

# %% [markdown]
# ---
# ## 8. Key Pedagogical Insights & Diagnostic Findings
#
# 1. **The Normalization Trap**:
#    - Per-sample min-max normalization erases the 40–100× amplitude gap between quiet low-volatility markets and explosive volatility shocks.
#    - Preserving a global scale reference allows spectral encoders (CWT/STFT) to easily identify Volatility Shocks.
# 2. **Symmetry vs Directionality**:
#    - CWT scalograms on returns ($|W|^2$) are sign-symmetric ($|+r|^2 = |-r|^2$) and detect volatility but cannot separate Bull from Bear trends.
#    - GAF on cumulative price curves provides the missing directional polar geometry.
# 3. **The Power of Multi-Channel Fusion**:
#    - Combining GASF (trend), GADF (rate-of-change), and CWT (energy) into an RGB tensor achieves the highest performance (~80% accuracy, ~0.79 Macro F1 on out-of-time test data).
