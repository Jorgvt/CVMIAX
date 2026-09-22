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
# # Experiment 15: Image Quality Assessment (IQA) on TID2008
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jorgvt/CVMIAX/blob/main/experiments/15_image_quality_assessment_tid2008/experiment_15_image_quality_assessment_tid2008.ipynb)
#
# Image Quality Assessment (IQA) seeks to computationally quantify the visual quality of images in alignment with human perceptual judgment. While classical signal-level metrics (e.g. Mean Squared Error, Peak Signal-to-Noise Ratio) quantify point-wise pixel differences, they fail notoriously to reflect biological perceptual degradation—such as blurring, structural distortion, or localized texture corruption.
#
# In this experiment, we explore the **TID2008** benchmark dataset (hosted on Hugging Face at [`Jorgvt/TID2008`](https://huggingface.co/datasets/Jorgvt/TID2008)), contrast classical metrics with deep learning approaches, and investigate:
# 1. **Full-Reference (FR-IQA)** vs. **No-Reference (NR-IQA / Blind IQA)** formulations.
# 2. **Content-Independent Splitting**: Why reference-wise train/test splits are strictly required to prevent semantic leakage.
# 3. **Hierarchical Feature Residuals**: How multi-scale convolutional difference maps $\Delta \phi = |\phi(I_{\text{ref}}) - \phi(I_{\text{dist}})|$ capture perceptual artifacts.
# 4. **Standard IQA Evaluation Metrics**: Spearman Rank Correlation ($\text{SROCC}$), Pearson Linear Correlation ($\text{PLCC}$ with 4-parameter logistic mapping), and Kendall's Tau ($\text{KROCC}$).

# %%
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

# Configure visualization style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["figure.dpi"] = 120
print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")

# %% [markdown]
# ## 1. Loading the TID2008 Dataset
#
# TID2008 contains 25 reference pristine images corrupted by 17 distortion types at 4 intensity levels (1,700 images total). Ground truth **MOS** (Mean Opinion Score) values represent human psychophysical subjective ratings.

# %%
from datasets import load_dataset
from PIL import Image

print("Loading dataset Jorgvt/TID2008 from Hugging Face...")
raw_ds = load_dataset("Jorgvt/TID2008", split="train")
print(f"Dataset successfully loaded: {len(raw_ds)} total samples.")

# Inspect sample record
sample = raw_ds[0]
print(f"Sample keys: {list(sample.keys())}")
print(f"Reference ID: {sample['reference_id']}, Distortion ID: {sample['distortion_id']}, Intensity: {sample['distortion_intensity']}, MOS: {sample['mos']:.3f}")

# %% [markdown]
# ### TID2008 Distortion Taxonomy
#
# Let's inspect the 17 distortion types and visualize how subjective quality degrades across intensity levels.

# %%
DISTORTION_NAMES = {
    1: "Additive Gaussian noise",
    2: "Additive noise in color components",
    3: "Spatially correlated noise",
    4: "Masked noise",
    5: "High frequency noise",
    6: "Impulse noise",
    7: "Quantization noise",
    8: "Gaussian blur",
    9: "Image denoising artifacts",
    10: "JPEG compression",
    11: "JPEG2000 compression",
    12: "JPEG transmission errors",
    13: "JPEG2000 transmission errors",
    14: "Non eccentricity pattern noise",
    15: "Local block-wise distortions",
    16: "Mean shift (intensity shift)",
    17: "Contrast change",
}

# Visualize Reference #1 across 4 key distortion types
selected_dists = [1, 8, 10, 17]  # Gaussian Noise, Gaussian Blur, JPEG, Contrast Change
fig, axes = plt.subplots(len(selected_dists), 5, figsize=(14, 2.8 * len(selected_dists)))

ref_sample = next(item for item in raw_ds if item["reference_id"] == 1 and item["distortion_id"] == 1)
ref_img = ref_sample["reference"]

for r_idx, did in enumerate(selected_dists):
    axes[r_idx, 0].imshow(ref_img)
    axes[r_idx, 0].axis("off")
    if r_idx == 0:
        axes[r_idx, 0].set_title("Reference\n(Pristine)", fontweight="bold")
    axes[r_idx, 0].set_ylabel(DISTORTION_NAMES[did], fontweight="bold", fontsize=9)

    for intensity in range(1, 5):
        s = next(item for item in raw_ds if item["reference_id"] == 1 and item["distortion_id"] == did and item["distortion_intensity"] == intensity)
        ax = axes[r_idx, intensity]
        ax.imshow(s["distorted"])
        ax.axis("off")
        ax.set_title(f"Level {intensity} | MOS: {s['mos']:.2f}", fontsize=9)

plt.suptitle("TID2008 Distortion Taxonomy & Perceptual MOS Degradation", fontsize=14, fontweight="bold", y=0.99)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. Methodology: Content-Independent Reference Splitting
#
# > **Crucial IQA Principle**:
# > If images derived from the same reference scene exist in both training and testing sets, the network can learn semantic shortcuts (e.g., memorizing specific high-frequency grass textures or sky regions) rather than learning the perceptual mapping from distortion artifacts to quality degradation.
# >
# > We perform a strict **Reference-Independent Split**:
# > - **Train Set**: Reference IDs 1 to 18 (18 references, 72% / 1,224 images)
# > - **Val Set**: Reference IDs 19 to 20 (2 references, 8% / 136 images)
# > - **Test Set**: Reference IDs 21 to 25 (5 references, 20% / 340 images)

# %%
train_ref_ids = list(range(1, 19))
val_ref_ids = [19, 20]
test_ref_ids = list(range(21, 26))

train_items = [item for item in raw_ds if item["reference_id"] in train_ref_ids]
val_items = [item for item in raw_ds if item["reference_id"] in val_ref_ids]
test_items = [item for item in raw_ds if item["reference_id"] in test_ref_ids]

print(f"Content Split: Train={len(train_items)} | Val={len(val_items)} | Test={len(test_items)}")

# Helper to convert to preprocessed numpy batches (224x224, float32 [0, 1])
def extract_arrays(items, target_size=(224, 224)):
    n = len(items)
    refs = np.zeros((n, target_size[0], target_size[1], 3), dtype=np.float32)
    dists = np.zeros((n, target_size[0], target_size[1], 3), dtype=np.float32)
    mos = np.zeros((n,), dtype=np.float32)
    dids = np.zeros((n,), dtype=np.int32)
    for i, it in enumerate(items):
        r_arr = np.array(it["reference"].resize(target_size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        d_arr = np.array(it["distorted"].resize(target_size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        refs[i] = r_arr
        dists[i] = d_arr
        mos[i] = float(it["mos"])
        dids[i] = int(it["distortion_id"])
    return refs, dists, mos, dids

tr_refs, tr_dists, tr_mos, tr_dids = extract_arrays(train_items)
val_refs, val_dists, val_mos, val_dids = extract_arrays(val_items)
te_refs, te_dists, te_mos, te_dids = extract_arrays(test_items)
print("Arrays extracted successfully.")

# %% [markdown]
# ## 3. Classical Baselines: PSNR & SSIM
#
# Let's compute Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM) on the test set.
#
# - **PSNR**:
#   $$\text{PSNR} = 10 \cdot \log_{10}\left(\frac{\text{MAX}^2}{\text{MSE}}\right)$$
# - **SSIM**:
#   $$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}$$

# %%
from scipy import stats, optimize

def compute_psnr_batch(refs, dists):
    mse = np.mean((refs.astype(np.float64) - dists.astype(np.float64)) ** 2, axis=(1, 2, 3))
    mse = np.clip(mse, 1e-10, None)
    return 10.0 * np.log10(1.0 / mse)

def compute_ssim_batch(refs, dists):
    ssim_vals = tf.image.ssim(
        tf.convert_to_tensor(refs, dtype=tf.float32),
        tf.convert_to_tensor(dists, dtype=tf.float32),
        max_val=1.0,
    )
    return ssim_vals.numpy().astype(np.float64)

# 4-parameter logistic curve for standard IQA mapping
def logistic_4param(x, beta1, beta2, beta3, beta4):
    return (beta1 - beta2) / (1.0 + np.exp((x - beta3) / (np.abs(beta4) + 1e-7))) + beta2

def fit_logistic(y_pred, y_true):
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    min_t, max_t = float(np.min(y_true)), float(np.max(y_true))
    mean_p, std_p = float(np.mean(y_pred)), float(np.std(y_pred)) + 1e-4
    beta0 = [max_t, min_t, mean_p, std_p]
    try:
        popt, _ = optimize.curve_fit(
            logistic_4param,
            y_pred,
            y_true,
            p0=beta0,
            bounds=([-2.0, -2.0, -np.inf, 1e-4], [12.0, 12.0, np.inf, np.inf]),
            maxfev=5000,
        )
        mapped = logistic_4param(y_pred, *popt)
        if np.all(np.isfinite(mapped)) and np.std(mapped) > 1e-3:
            orig_srocc, _ = stats.spearmanr(y_true, y_pred)
            mapped_plcc, _ = stats.pearsonr(y_true, mapped)
            if np.sign(mapped_plcc) == np.sign(orig_srocc):
                return mapped
    except Exception:
        pass
    try:
        slope, intercept, _, _, _ = stats.linregress(y_pred, y_true)
        return slope * y_pred + intercept
    except Exception:
        return y_pred

psnr_test = compute_psnr_batch(te_refs, te_dists)
ssim_test = compute_ssim_batch(te_refs, te_dists)

srocc_psnr, _ = stats.spearmanr(te_mos, psnr_test)
srocc_ssim, _ = stats.spearmanr(te_mos, ssim_test)
plcc_psnr, _ = stats.pearsonr(te_mos, fit_logistic(psnr_test, te_mos))
plcc_ssim, _ = stats.pearsonr(te_mos, fit_logistic(ssim_test, te_mos))

print(f"PSNR Baseline -> SROCC: {srocc_psnr:.4f} | PLCC: {plcc_psnr:.4f}")
print(f"SSIM Baseline -> SROCC: {srocc_ssim:.4f} | PLCC: {plcc_ssim:.4f}")

# %% [markdown]
# ## 4. Deep Learning IQA Models
#
# We construct two complementary architectures:
# 1. **Full-Reference Siamese Multi-Scale Difference CNN**:
#    Extracts hierarchical feature maps $\phi_1, \phi_2, \phi_3$, computes feature residuals $\Delta \phi_k = |\phi_k(I_{\text{ref}}) - \phi_k(I_{\text{dist}})|$, pools them spatially, and regresses MOS.
# 2. **No-Reference (Blind) CNN**:
#    Directly maps the distorted image $I_{\text{dist}}$ to MOS.

# %%
from keras import layers, Model, ops

def build_conv_block(filters, name_prefix):
    return keras.Sequential([
        layers.Conv2D(filters, kernel_size=3, padding="same", use_bias=False, name=f"{name_prefix}_conv"),
        layers.BatchNormalization(name=f"{name_prefix}_bn"),
        layers.Activation("relu", name=f"{name_prefix}_relu"),
        layers.MaxPooling2D(pool_size=2, strides=2, name=f"{name_prefix}_pool"),
    ], name=name_prefix)

def build_fr_iqa(input_shape=(224, 224, 3)):
    ref_in = layers.Input(shape=input_shape, name="reference")
    dist_in = layers.Input(shape=input_shape, name="distorted")

    s1 = build_conv_block(32, "stage1")
    s2 = build_conv_block(64, "stage2")
    s3 = build_conv_block(128, "stage3")

    # Ref hierarchical passes
    rf1 = s1(ref_in)
    rf2 = s2(rf1)
    rf3 = s3(rf2)

    df1 = s1(dist_in)
    df2 = s2(df1)
    df3 = s3(df2)

    diff1 = layers.Lambda(lambda t: ops.abs(t[0] - t[1]), name="diff_stage1")([rf1, df1])
    diff2 = layers.Lambda(lambda t: ops.abs(t[0] - t[1]), name="diff_stage2")([rf2, df2])
    diff3 = layers.Lambda(lambda t: ops.abs(t[0] - t[1]), name="diff_stage3")([rf3, df3])

    p1 = layers.GlobalAveragePooling2D()(diff1)
    p2 = layers.GlobalAveragePooling2D()(diff2)
    p3 = layers.GlobalAveragePooling2D()(diff3)

    merged = layers.Concatenate()([p1, p2, p3])
    x = layers.Dense(128, activation="relu")(merged)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, name="mos_pred")(x)

    return Model(inputs={"reference": ref_in, "distorted": dist_in}, outputs=out, name="FR_IQA_CNN")

def build_nr_iqa(input_shape=(224, 224, 3)):
    dist_in = layers.Input(shape=input_shape, name="distorted_input")
    x = build_conv_block(32, "nr_s1")(dist_in)
    x = build_conv_block(64, "nr_s2")(x)
    x = build_conv_block(128, "nr_s3")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, name="nr_mos_pred")(x)
    return Model(inputs=dist_in, outputs=out, name="NR_IQA_CNN")

fr_model = build_fr_iqa()
nr_model = build_nr_iqa()
fr_model.summary()

# %% [markdown]
# ## 5. Training the IQA Models

# %%
fr_model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="huber", metrics=["mae"])
nr_model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="huber", metrics=["mae"])

# Prepare tf.data pipelines
fr_train_ds = tf.data.Dataset.from_tensor_slices(({"reference": tr_refs, "distorted": tr_dists}, tr_mos)).shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
fr_val_ds = tf.data.Dataset.from_tensor_slices(({"reference": val_refs, "distorted": val_dists}, val_mos)).batch(32).prefetch(tf.data.AUTOTUNE)

nr_train_ds = tf.data.Dataset.from_tensor_slices((tr_dists, tr_mos)).shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
nr_val_ds = tf.data.Dataset.from_tensor_slices((val_dists, val_mos)).batch(32).prefetch(tf.data.AUTOTUNE)

print("Training Full-Reference IQA Model...")
fr_hist = fr_model.fit(fr_train_ds, validation_data=fr_val_ds, epochs=15, verbose=1)

print("\nTraining No-Reference IQA Model...")
nr_hist = nr_model.fit(nr_train_ds, validation_data=nr_val_ds, epochs=15, verbose=1)

# %% [markdown]
# ## 6. Comprehensive Test Set Benchmark & Correlation Analysis

# %%
# Predict on test set (unseen references 21..25)
fr_preds = fr_model.predict({"reference": te_refs, "distorted": te_dists}, verbose=0).flatten()
nr_preds = nr_model.predict(te_dists, verbose=0).flatten()

predictions_dict = {
    "PSNR": psnr_test,
    "SSIM": ssim_test,
    "Deep FR-IQA": fr_preds,
    "Deep NR-IQA": nr_preds,
}

benchmark_rows = []
for name, preds in predictions_dict.items():
    srocc, _ = stats.spearmanr(te_mos, preds)
    krocc, _ = stats.kendalltau(te_mos, preds)
    mapped = fit_logistic(preds, te_mos)
    plcc, _ = stats.pearsonr(te_mos, mapped)
    rmse = np.sqrt(np.mean((te_mos - mapped) ** 2))
    mae = np.mean(np.abs(te_mos - mapped))
    benchmark_rows.append({
        "Model": name,
        "SROCC": srocc,
        "KROCC": krocc,
        "PLCC": plcc,
        "RMSE": rmse,
        "MAE": mae,
    })

benchmark_df = pd.DataFrame(benchmark_rows)
display(benchmark_df)

# %% [markdown]
# ## 7. Visualizing Correlation & Failure Modes

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
colors = ["#2b5c8f", "#2e7d32", "#d32f2f", "#7b1fa2"]

for idx, (m_name, preds) in enumerate(predictions_dict.items()):
    ax = axes[idx]
    mapped = fit_logistic(preds, te_mos)
    srocc, _ = stats.spearmanr(te_mos, preds)
    plcc, _ = stats.pearsonr(te_mos, mapped)

    ax.scatter(te_mos, mapped, c=colors[idx], alpha=0.6, edgecolors="none", s=30)
    min_v = min(np.min(te_mos), np.min(mapped))
    max_v = max(np.max(te_mos), np.max(mapped))
    ax.plot([min_v, max_v], [min_v, max_v], "k--", alpha=0.7, label="y = x")

    ax.set_title(f"{m_name}\nSROCC: {srocc:.4f} | PLCC: {plcc:.4f}", fontweight="bold")
    ax.set_xlabel("Subjective MOS (Ground Truth)")
    ax.set_ylabel("Predicted Quality Score")
    ax.legend()

plt.suptitle("Quantitative IQA Alignment Across Unseen Reference Scenes", fontsize=14, fontweight="bold", y=0.99)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Multi-Scale Perceptual Feature Difference Maps
#
# By tapping into intermediate activations $\Delta \phi_1, \Delta \phi_2, \Delta \phi_3$, we can inspect where the deep network perceives quality degradation in spatial coordinate space.

# %%
diff_extractor = Model(
    inputs=fr_model.inputs,
    outputs=[
        fr_model.get_layer("diff_stage1").output,
        fr_model.get_layer("diff_stage2").output,
        fr_model.get_layer("diff_stage3").output,
    ]
)

sample_idx = [0, 15, 30]
sub_r = te_refs[sample_idx]
sub_d = te_dists[sample_idx]
d1, d2, d3 = diff_extractor.predict({"reference": sub_r, "distorted": sub_d}, verbose=0)

fig, axes = plt.subplots(len(sample_idx), 5, figsize=(14, 3 * len(sample_idx)))

for i in range(len(sample_idx)):
    axes[i, 0].imshow(sub_r[i])
    axes[i, 0].axis("off")
    if i == 0: axes[i, 0].set_title("Reference Image", fontweight="bold")

    axes[i, 1].imshow(sub_d[i])
    axes[i, 1].axis("off")
    if i == 0: axes[i, 1].set_title("Distorted Image", fontweight="bold")
    axes[i, 1].set_ylabel(f"Sample #{i+1}\nMOS: {te_mos[sample_idx[i]]:.2f}", fontweight="bold")

    axes[i, 2].imshow(np.mean(d1[i], axis=-1), cmap="magma")
    axes[i, 2].axis("off")
    if i == 0: axes[i, 2].set_title("Stage 1 Residual ($\Delta \phi_1$)\n[Texture / Edges]", fontweight="bold")

    axes[i, 3].imshow(np.mean(d2[i], axis=-1), cmap="magma")
    axes[i, 3].axis("off")
    if i == 0: axes[i, 3].set_title("Stage 2 Residual ($\Delta \phi_2$)\n[Structure]", fontweight="bold")

    axes[i, 4].imshow(np.mean(d3[i], axis=-1), cmap="magma")
    axes[i, 4].axis("off")
    if i == 0: axes[i, 4].set_title("Stage 3 Residual ($\Delta \phi_3$)\n[High-Level]", fontweight="bold")

plt.suptitle("Hierarchical Perceptual Error Maps in Full-Reference CNN", fontsize=14, fontweight="bold", y=0.99)
plt.tight_layout()
plt.show()
