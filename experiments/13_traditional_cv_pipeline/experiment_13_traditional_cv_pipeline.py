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
# # Experiment 13: Traditional Computer Vision Pipeline
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jorgvt/CVMIAX/blob/main/experiments/13_traditional_cv_pipeline/experiment_13_traditional_cv_pipeline.ipynb)
#
# ## 1. Pedagogical Overview & Problem Statement
#
# Before the deep learning revolution popularized end-to-end representation learning ($f_\theta(x) \to y$), computer vision systems relied on a modular, hand-engineered **4-stage sequential pipeline**:
#
# $$\text{Raw Input Image } x \xrightarrow{\text{Stage 1}} \text{Preprocessing } \tilde{x} \xrightarrow{\text{Stage 2}} \text{Feature Extraction } \phi(\tilde{x}) \xrightarrow{\text{Stage 3}} \text{Feature Processing } z \xrightarrow{\text{Stage 4}} \text{Classifier } \hat{y}$$
#
# This experiment explores each stage in depth by solving a classic vision benchmark (handwritten digit classification) without any neural networks or backpropagation:
# 1. **Stage 1 (Preprocessing)**: Moments-based deskewing ($\alpha = \mu_{11}/\mu_{02}$), Gaussian noise filtering, and intensity normalization.
# 2. **Stage 2 (Feature Extraction)**: Hand-crafted gradient descriptors (Histogram of Oriented Gradients - HOG), texture patterns (Local Binary Patterns - LBP), and Hu Moment invariants.
# 3. **Stage 3 (Feature Processing)**: Standardization and Principal Component Analysis (PCA) eigen-decomposition.
# 4. **Stage 4 (Classification)**: Maximum-margin hyperplane separation via Support Vector Machines (Linear and RBF kernel SVMs) and Random Forests.

# %% [markdown]
# ## 2. Theoretical Formulation & Mathematics
#
# ### 2.1 Stage 1: Preprocessing & Moments Deskewing
#
# Hand-written digits often suffer from varying writing angles (shear). Given an image intensity function $I(x, y)$, the $(p+q)$-th order spatial moment is:
# $$m_{pq} = \sum_{x} \sum_{y} x^p y^q I(x, y)$$
#
# The centroid (center of mass) is:
# $$\bar{x} = \frac{m_{10}}{m_{00}}, \quad \bar{y} = \frac{m_{01}}{m_{00}}$$
#
# The central moments $\mu_{pq}$, invariant to translation, are:
# $$\mu_{pq} = \sum_{x} \sum_{y} (x - \bar{x})^p (y - \bar{y})^q I(x, y)$$
#
# The shear slope $\alpha$ is derived from the spatial covariance ratio:
# $$\alpha = \frac{\mu_{11}}{\mu_{02}}$$
#
# The image is then upright-aligned using the inverse affine shear transform:
# $$\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} 1 & \alpha \\ 0 & 1 \end{bmatrix} \begin{bmatrix} x - \bar{x} \\ y - \bar{y} \end{bmatrix} + \begin{bmatrix} \bar{x} \\ \bar{y} \end{bmatrix}$$

# %% [markdown]
# ### 2.2 Stage 2: Histogram of Oriented Gradients (HOG)
#
# HOG (Dalal & Triggs, 2005) captures object shape and boundary directions:
# 1. **Gradient Computation**:
#    $$G_x(x, y) = I(x+1, y) - I(x-1, y), \quad G_y(x, y) = I(x, y+1) - I(x, y-1)$$
#    $$M(x, y) = \sqrt{G_x^2 + G_y^2}, \quad \theta(x, y) = \arctan\left(\frac{G_y}{G_x}\right) \pmod{180^\circ}$$
#
# 2. **Spatial Cell Histograms**: The image is divided into small cells (e.g., $7 \times 7$ pixels). Each pixel's gradient magnitude $M(x, y)$ votes into 9 orientation bins.
#
# 3. **Block Normalization ($L_2\text{-Hys}$)**: To achieve illumination invariance, histograms are concatenated across overlapping blocks (e.g., $2 \times 2$ cells) and normalized:
#    $$v \leftarrow \frac{v}{\sqrt{\|v\|_2^2 + \epsilon^2}}, \quad v_i \leftarrow \min(v_i, 0.2), \quad v \leftarrow \frac{v}{\sqrt{\|v\|_2^2 + \epsilon^2}}$$

# %% [markdown]
# ### 2.3 Stage 3: Feature Processing & PCA Eigen-decomposition
#
# Let $\mathbf{X} \in \mathbb{R}^{N \times D}$ be the standardized feature matrix ($z = \frac{x - \mu}{\sigma}$). The sample covariance matrix is:
# $$\mathbf{\Sigma} = \frac{1}{N-1} \mathbf{X}^T \mathbf{X}$$
#
# Computing the eigen-decomposition $\mathbf{\Sigma} \mathbf{v}_i = \lambda_i \mathbf{v}_i$ sorted by eigenvalues $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_D$:
# - **Projection onto $k$-dimensional subspace**: $\mathbf{Z} = \mathbf{X} \mathbf{W}_k$, where $\mathbf{W}_k = [\mathbf{v}_1, \dots, \mathbf{v}_k]$.
# - **Cumulative Explained Variance**: $\eta(k) = \frac{\sum_{i=1}^k \lambda_i}{\sum_{j=1}^D \lambda_j}$.

# %% [markdown]
# ### 2.4 Stage 4: Support Vector Classification (SVM)
#
# Given training pairs $(\mathbf{z}_i, y_i)$, the dual formulation of soft-margin Support Vector Machine solves:
# $$\max_{\alpha} \sum_{i=1}^N \alpha_i - \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j K(\mathbf{z}_i, \mathbf{z}_j)$$
# $$\text{subject to } 0 \le \alpha_i \le C \quad \text{and} \quad \sum_{i=1}^N \alpha_i y_i = 0$$
#
# Where the Radial Basis Function (RBF) kernel is:
# $$K(\mathbf{z}_i, \mathbf{z}_j) = \exp\left( -\gamma \|\mathbf{z}_i - \mathbf{z}_j\|^2 \right)$$

# %%
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add current experiment folder to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing import preprocess_single_image, deskew_image, gaussian_filter, normalize_intensity
from features import compute_gradients, extract_hog_features, extract_lbp_features, extract_hu_moments
from feature_processing import FeatureProcessor, compute_scree_analysis
from classifiers import get_classifier, train_and_evaluate_classifier
from train_and_evaluate import load_and_preprocess_dataset

# %% [markdown]
# ## 3. Interactive Walkthrough: Step-by-Step Pipeline Execution

# %%
# 1. Load sample data
x_tr_raw, x_tr_proc, y_tr, x_te_raw, x_te_proc, y_te = load_and_preprocess_dataset(
    n_train=5000, n_test=1000, deskew=True
)

sample_idx = 0
sample_img = x_te_raw[sample_idx]
sample_label = y_te[sample_idx]

# Stage 1: Preprocessing steps
raw_norm = normalize_intensity(sample_img)
deskewed = deskew_image(raw_norm)
smoothed = gaussian_filter(deskewed, kernel_size=3, sigma=0.5)

# Stage 2: Feature Extraction
gx, gy, mag, ori = compute_gradients(smoothed)
hog_feats, hog_vis = extract_hog_features(smoothed, orientations=9, pixels_per_cell=(7, 7), cells_per_block=(2, 2), visualize=True)
lbp_feats, lbp_vis = extract_lbp_features(smoothed, num_points=8, radius=1, visualize=True)
hu_feats = extract_hu_moments(smoothed)

print(f"Sample Digit Class: {sample_label}")
print(f"HOG Feature Vector Dimension: {hog_feats.shape[0]}")
print(f"LBP Feature Vector Dimension: {lbp_feats.shape[0]}")
print(f"Hu Moments: {hu_feats}")

# %% [markdown]
# ### 3.1 Visualizing the 4 Stages for a Single Sample

# %%
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

axes[0, 0].imshow(raw_norm, cmap='gray')
axes[0, 0].set_title(f"1. Raw Input (Class {sample_label})", fontweight='bold')
axes[0, 0].axis('off')

axes[0, 1].imshow(deskewed, cmap='gray')
axes[0, 1].set_title("2. Moments Deskewed", fontweight='bold')
axes[0, 1].axis('off')

axes[0, 2].imshow(gx, cmap='coolwarm')
axes[0, 2].set_title("3. Horizontal Gradient $G_x$", fontweight='bold')
axes[0, 2].axis('off')

axes[0, 3].imshow(mag, cmap='magma')
axes[0, 3].set_title("4. Gradient Magnitude", fontweight='bold')
axes[0, 3].axis('off')

axes[1, 0].imshow(hog_vis, cmap='inferno')
axes[1, 0].set_title("5. HOG Cell Orientations", fontweight='bold')
axes[1, 0].axis('off')

axes[1, 1].imshow(lbp_vis, cmap='viridis')
axes[1, 1].set_title("6. LBP Micro-textures", fontweight='bold')
axes[1, 1].axis('off')

axes[1, 2].bar(range(1, 8), hu_feats, color='#2ca02c')
axes[1, 2].set_title("7. Hu Invariant Moments", fontweight='bold')
axes[1, 2].set_xlabel("Moment $\phi_i$")

# Dummy 2D feature projection
axes[1, 3].scatter(hog_feats[:72], hog_feats[72:], alpha=0.7, c='orange')
axes[1, 3].set_title("8. Feature Vector Distribution", fontweight='bold')
axes[1, 3].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Feature Extraction & Dataset-wide Benchmarking
#
# Now let's extract features across the entire training and test sets and train classic classifiers.

# %%
from features import extract_features_dataset

# Extract HOG descriptors across training & testing subsets
print("Extracting HOG descriptors...")
X_tr_hog = extract_features_dataset(x_tr_proc, feature_type='hog', pixels_per_cell=(7, 7), cells_per_block=(2, 2))
X_te_hog = extract_features_dataset(x_te_proc, feature_type='hog', pixels_per_cell=(7, 7), cells_per_block=(2, 2))

# Extract raw pixel baseline
X_tr_raw = extract_features_dataset(x_tr_raw, feature_type='raw')
X_te_raw = extract_features_dataset(x_te_raw, feature_type='raw')

print(f"Training HOG Matrix: {X_tr_hog.shape}")
print(f"Training Raw Matrix: {X_tr_raw.shape}")

# %% [markdown]
# ## 5. Stage 3: Feature Processing (Standardization & PCA)

# %%
# Standardize features
proc = FeatureProcessor(standardize=True, n_components=50)
X_tr_hog_pca = proc.fit_transform(X_tr_hog)
X_te_hog_pca = proc.transform(X_te_hog)

scaler_raw = FeatureProcessor(standardize=True, n_components=None)
X_tr_raw_scaled = scaler_raw.fit_transform(X_tr_raw)
X_te_raw_scaled = scaler_raw.transform(X_te_raw)

scaler_hog = FeatureProcessor(standardize=True, n_components=None)
X_tr_hog_scaled = scaler_hog.fit_transform(X_tr_hog)
X_te_hog_scaled = scaler_hog.transform(X_te_hog)

print(f"PCA reduced HOG dimension from {X_tr_hog.shape[1]} to {X_tr_hog_pca.shape[1]}")
print(f"Cumulative Explained Variance: {proc.cumulative_explained_variance[-1]*100:.2f}%")

# %% [markdown]
# ## 6. Stage 4: Training Classifiers & Benchmark Comparison

# %%
models = {
    "Raw Pixels + k-NN (k=5)": (get_classifier('knn', n_neighbors=5), X_tr_raw_scaled, X_te_raw_scaled),
    "Raw Pixels + Linear SVM": (get_classifier('svm_linear', C=0.1), X_tr_raw_scaled, X_te_raw_scaled),
    "HOG + Random Forest": (get_classifier('random_forest', n_estimators=100), X_tr_hog_scaled, X_te_hog_scaled),
    "HOG + Linear SVM": (get_classifier('svm_linear', C=1.0), X_tr_hog_scaled, X_te_hog_scaled),
    "HOG + PCA (50D) + SVM (RBF)": (get_classifier('svm_rbf', C=5.0), X_tr_hog_pca, X_te_hog_pca),
    "HOG + SVM (RBF) [Gold Standard]": (get_classifier('svm_rbf', C=5.0), X_tr_hog_scaled, X_te_hog_scaled),
}

benchmark_results = {}
for name, (clf, X_tr, X_te) in models.items():
    res = train_and_evaluate_classifier(clf, X_tr, y_tr, X_te, y_te)
    benchmark_results[name] = res
    print(f"--> {name:<35}: Accuracy = {res['accuracy']*100:.2f}% | Train Time = {res['train_time_sec']:.2f}s | Test Time = {res['test_time_sec']:.2f}s")

# %% [markdown]
# ## 7. Qualitative Failure Mode Analysis
#
# Inspecting where hand-crafted features succeed and where they fail due to ambiguous strokes or overlapping morphology.

# %%
best_res = benchmark_results["HOG + SVM (RBF) [Gold Standard]"]
y_pred = best_res['y_pred']
mistakes = np.where(y_te != y_pred)[0]

print(f"Total test errors: {len(mistakes)} out of {len(y_te)} ({100 - best_res['accuracy']*100:.2f}% error rate)")

fig, axes = plt.subplots(1, 5, figsize=(15, 3.5))
for col in range(min(5, len(mistakes))):
    idx = mistakes[col]
    img = x_te_raw[idx]
    axes[col].imshow(img, cmap='gray')
    axes[col].set_title(f"True: {y_te[idx]} | Pred: {y_pred[idx]}", color='red', fontweight='bold')
    axes[col].axis('off')
plt.suptitle("Misclassified Digits (Traditional Pipeline Failure Modes)", fontsize=14, fontweight='bold')
plt.show()

# %% [markdown]
# ## 8. Summary & Key Takeaways
#
# 1. **Hand-Crafted Invariances**: Preprocessing (moments deskewing) and feature descriptors (HOG orientation bins) hand-engineer translation, rotation, and illumination invariance.
# 2. **Linear Separability**: HOG drastically increases class separability over raw pixels, boosting Linear SVM accuracy from ~91% to >97.5%, and RBF SVM to >98.5%.
# 3. **Traditional vs Deep Learning**:
#    - *Traditional CV*: Explicit, interpretable, ultra-fast to train on CPUs, but requires domain-expert feature engineering that does not scale well to complex open-world semantics.
#    - *Deep Learning*: End-to-end differentiable optimization ($f_\theta$), automatically discovering optimal hierarchical filters directly from raw pixel gradients.
