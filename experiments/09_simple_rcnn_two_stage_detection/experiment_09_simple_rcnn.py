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
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jorgvt/CVMIAX/blob/main/experiments/09_simple_rcnn_two_stage_detection/experiment_09_simple_rcnn.ipynb)
#
# # Experiment 09: Two-Stage Object Detection (Pedagogical R-CNN from Scratch)
#
# Welcome to this deep dive into **Two-Stage Object Detection**! In image classification, a convolutional network predicts a single categorical label for the entire image. In object detection, however, our objective is twofold:
# 1. **Classification**: *What* categories of objects are present in the scene?
# 2. **Localization**: *Where* is each object located in pixel space (bounding boxes)?
#
# In this interactive lesson, we implement the classic **Regions with CNN features (R-CNN)** paradigm from first principles using Keras and TensorFlow.
#
# ---
#
# ## The Two-Stage Architecture
#
# ```
#   +-------------------------------------------------------------------------------+
#   |                                  Input Image                                  |
#   +-------------------------------------------------------------------------------+
#                                          |
#                                          v
#                       +--------------------------------------+
#                       |     Stage 1: Region Proposer         |
#                       |  (Candidate Bounding Box Generation) |
#                       +--------------------------------------+
#                                          |
#                                          v
#                       +--------------------------------------+
#                       |          Crop & Canonical Warp       |
#                       |   (Resize all ROIs to 32x32x3)       |
#                       +--------------------------------------+
#                                          |
#                                          v
#                       +--------------------------------------+
#                       |      Stage 2: Multi-Task CNN         |
#                       +--------------------------------------+
#                                    /            \
#                                   /              \
#                                  v                v
#                    +-------------------+   +--------------------+
#                    |  Classification   |   |  BBox Regression   |
#                    |  (C+1 categories) |   | (Δx, Δy, Δw, Δh)   |
#                    +-------------------+   +--------------------+
#                                   \              /
#                                    \            /
#                                     v          v
#                       +--------------------------------------+
#                       |   Non-Maximum Suppression (NMS)      |
#                       |   (Eliminate Redundant Overlaps)     |
#                       +--------------------------------------+
#                                          |
#                                          v
#                       +--------------------------------------+
#                       |        Final Object Detections       |
#                       +--------------------------------------+
# ```

# %%
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Add current directory to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from dataset import (
    CLASS_NAMES,
    ID_TO_CLASS,
    generate_dataset,
    generate_single_sample,
)
from evaluate import (
    detect_objects_rcnn,
    evaluate_dataset_map,
    non_max_suppression,
)
from proposals import (
    clip_boxes,
    compute_iou_matrix,
    decode_bbox_deltas,
    encode_bbox_deltas,
    extract_and_warp_crops,
    generate_sliding_window_proposals,
    match_proposals_to_ground_truth,
)
from train import (
    load_or_train_rcnn_detector,
    prepare_rcnn_training_data,
)
from visualize import (
    visualize_complete_pipeline,
    visualize_crops_grid,
    visualize_dataset_samples,
    visualize_proposals_and_iou,
)

print(f"TensorFlow Version: {tf.__version__}")
print(f"Classes: {CLASS_NAMES}")

# %% [markdown]
# ---
# ## 1. Pedagogical Dataset: Synthetic Multi-Shape Canvas
#
# To keep the learning loop fast, deterministic, and visually intuitive, we generate synthetic $128 \times 128$ RGB canvases with 1–3 colored geometric shapes:
# - **Circles** (Class 1)
# - **Rectangles / Squares** (Class 2)
# - **Triangles** (Class 3)
# - **Background** (Class 0)
#
# Let's generate a batch of samples and inspect their ground-truth annotations.

# %%
train_images, train_annos = generate_dataset(num_samples=250, img_size=128, min_objects=1, max_objects=3, seed=42)
val_images, val_annos = generate_dataset(num_samples=60, img_size=128, min_objects=1, max_objects=3, seed=101)
test_images, test_annos = generate_dataset(num_samples=40, img_size=128, min_objects=2, max_objects=3, seed=202)

print(f"Dataset generated:")
print(f"  - Training samples:   {len(train_images)}")
print(f"  - Validation samples: {len(val_images)}")
print(f"  - Testing samples:    {len(test_images)}")

fig = visualize_dataset_samples(train_images, train_annos, num_samples=6)
plt.show()

# %% [markdown]
# ---
# ## 2. Stage 1: Region Proposal Generation & IoU Matching
#
# ### The Intuition Behind Region Proposals
# An exhaustive sliding-window search at every pixel, scale, and aspect ratio requires evaluating tens of thousands of windows per image. In classical R-CNN (Girshick et al., 2014), **Selective Search** is used as an external algorithm to propose a few hundred high-probability object candidates.
#
# In our implementation, we generate systematic candidate proposals across multiple scales ($24, 36, 48, 64$ pixels) and aspect ratios ($0.75, 1.0, 1.33$).
#
# ### Intersection-over-Union (IoU) Matching
# Given candidate proposal $P$ and ground-truth box $G$, their overlap is measured via the Jaccard Index:
#
# $$\text{IoU}(P, G) = \frac{\text{Area}(P \cap G)}{\text{Area}(P \cup G)}$$
#
# We categorize each proposal based on its maximum IoU with any ground truth box:
# 1. **Positive / Foreground ($\text{IoU} \ge 0.5$)**: Assigned the object's class label and target bounding box offsets.
# 2. **Negative / Background ($\text{IoU} < 0.2$)**: Assigned class 0 (Background).
# 3. **Ambiguous / Neutral ($0.2 \le \text{IoU} < 0.5$)**: Ignored during training to prevent noisy gradients.

# %%
# Select a sample containing multiple shapes
sample_idx = 0
for i, anno in enumerate(train_annos):
    if len(anno["boxes"]) >= 3:
        sample_idx = i
        break

sample_img = train_images[sample_idx]
sample_boxes = train_annos[sample_idx]["boxes"]
sample_classes = train_annos[sample_idx]["classes"]

proposals = generate_sliding_window_proposals(img_size=128)
matched_labels, target_deltas, max_ious = match_proposals_to_ground_truth(
    proposals, sample_boxes, sample_classes, pos_thresh=0.5, neg_thresh=0.2
)

fig = visualize_proposals_and_iou(
    sample_img, sample_boxes, sample_classes, proposals, matched_labels, max_ious
)
plt.show()

# %% [markdown]
# ---
# ## 3. Bounding Box Parameterization (Offsets / Deltas)
#
# Instead of directly predicting raw absolute coordinates $[x_1, y_1, x_2, y_2]$, R-CNN predicts scale-invariant coordinate offsets relative to the proposal bounding box $P = (x_p, y_p, w_p, h_p)$:
#
# $$t_x = \frac{x_{gt} - x_p}{w_p}, \quad t_y = \frac{y_{gt} - y_p}{h_p}$$
#
# $$t_w = \log\left(\frac{w_{gt}}{w_p}\right), \quad t_h = \log\left(\frac{h_{gt}}{h_p}\right)$$
#
# During inference, given the network's predicted deltas $(\hat{t}_x, \hat{t}_y, \hat{t}_w, \hat{t}_h)$, the refined bounding box is reconstructed as:
#
# $$\hat{x} = x_p + w_p \cdot \hat{t}_x, \quad \hat{y} = y_p + h_p \cdot \hat{t}_y$$
#
# $$\hat{w} = w_p \cdot \exp(\hat{t}_w), \quad \hat{h} = h_p \cdot \exp(\hat{t}_h)$$

# %% [markdown]
# ---
# ## 4. Crop & Canonical Warping
#
# Every candidate proposal represents an image crop of arbitrary size and aspect ratio. To feed these into a standard CNN, each region is extracted and warped (bilinearly resized) to a fixed canonical shape of $32 \times 32 \times 3$.

# %%
sample_crops = extract_and_warp_crops(sample_img, proposals[matched_labels >= 0])
sample_crop_labels = matched_labels[matched_labels >= 0]

fig = visualize_crops_grid(sample_crops, sample_crop_labels, num_crops=16)
plt.show()

# %% [markdown]
# ---
# ## 5. Stage 2: Multi-Task CNN & Training with Checkpoint
#
# The Stage 2 CNN consists of:
# - **Shared Feature Extractor**: Conv2D + BatchNorm + MaxPool layers with Global Average Pooling.
# - **Classification Head**: Softmax distribution over $C+1$ classes (Background, Circle, Rectangle, Triangle).
# - **Bounding Box Regression Head**: 4 linear outputs $(t_x, t_y, t_w, t_h)$.
#
# ### Multi-Task Loss Formulation
# $$\mathcal{L} = \mathcal{L}_{\text{cls}}(p, u) + \lambda \cdot [u \ge 1] \cdot \mathcal{L}_{\text{reg}}(t, v)$$
#
# where:
# - $\mathcal{L}_{\text{cls}}$ is Sparse Categorical Cross-Entropy across all proposals.
# - $[u \ge 1]$ is an Iverson indicator: regression loss is **only computed for foreground/positive proposals** (background has no target box).
# - $\mathcal{L}_{\text{reg}}$ is Huber / Smooth $L_1$ loss.
# - $\lambda = 1.0$.

# %%
checkpoint_dir = os.path.join(CURRENT_DIR, "checkpoints")
checkpoint_path = os.path.join(checkpoint_dir, "rcnn_detector.weights.h5")

# Extract training ROI crops with hard negative sampling
X_train, y_cls_train, y_bbox_train = prepare_rcnn_training_data(
    train_images, train_annos, neg_to_pos_ratio=4.0, seed=42
)
X_val, y_cls_val, y_bbox_val = prepare_rcnn_training_data(
    val_images, val_annos, neg_to_pos_ratio=4.0, seed=101
)

print(f"Prepared Training Crops: {len(X_train)} (Positives: {np.sum(y_cls_train > 0)}, Background: {np.sum(y_cls_train == 0)})")
print(f"Prepared Validation Crops: {len(X_val)}")

# Load pre-trained checkpoint if available, or train smoothly
model, history = load_or_train_rcnn_detector(
    checkpoint_path=checkpoint_path,
    X_train=X_train,
    y_cls_train=y_cls_train,
    y_bbox_train=y_bbox_train,
    X_val=X_val,
    y_cls_val=y_cls_val,
    y_bbox_val=y_bbox_val,
    epochs=15,
)

model.summary()

# %% [markdown]
# ---
# ## 6. Non-Maximum Suppression (NMS)
#
# Because many overlapping proposals can cover the same object, the raw CNN output produces multiple redundant bounding box detections around each shape.
#
# ### The Greedy NMS Algorithm:
# 1. Sort all candidate bounding boxes by confidence score in descending order.
# 2. Select the candidate box with the highest score and add it to the final detection list.
# 3. Compute $\text{IoU}$ between this selected box and all remaining candidate boxes.
# 4. Discard any candidate box whose $\text{IoU} > \text{threshold}$ (suppression).
# 5. Repeat until no candidate boxes remain.

# %% [markdown]
# ---
# ## 7. Full End-to-End Inference Pipeline on a Multi-Shape Scene
#
# Let's run the complete two-stage detector on a multi-shape test image and visualize all four stages side-by-side!

# %%
# Select a test image containing at least 3 distinct shapes
multi_test_idx = 0
for i, anno in enumerate(test_annos):
    if len(anno["boxes"]) >= 3 and len(np.unique(anno["classes"])) >= 2:
        multi_test_idx = i
        break

test_img = test_images[multi_test_idx]
test_gt_boxes = test_annos[multi_test_idx]["boxes"]
test_gt_classes = test_annos[multi_test_idx]["classes"]

# Run full end-to-end detection
results = detect_objects_rcnn(
    model, test_img, conf_thresh=0.75, nms_iou_thresh=0.2
)

fig = visualize_complete_pipeline(
    test_img,
    test_gt_boxes,
    test_gt_classes,
    results["proposals"],
    results["raw_boxes"],
    results["raw_scores"],
    results["raw_classes"],
    results["nms_boxes"],
    results["nms_scores"],
    results["nms_classes"],
)
plt.show()

# %% [markdown]
# ---
# ## 8. Quantitative Evaluation: Mean Average Precision (mAP@0.5)
#
# Let's evaluate the model across the entire unseen test set to compute overall Precision, Recall, and per-class Average Precision.

# %%
eval_metrics = evaluate_dataset_map(
    model, test_images, test_annos, conf_thresh=0.75, nms_iou_thresh=0.2
)

print("=" * 45)
print("     TEST SET EVALUATION SUMMARY (mAP@0.5)")
print("=" * 45)
for k, v in eval_metrics.items():
    print(f"  {k:<20}: {v:.4f}")
print("=" * 45)

# %% [markdown]
# ---
# ## 9. Summary: From R-CNN to Fast and Faster R-CNN
#
# | Model | Stage 1 (Proposals) | Feature Extraction | Speed / Bottleneck |
# | :--- | :--- | :--- | :--- |
# | **R-CNN** (Girshick et al., 2014) | External Selective Search | 2,000 forward passes per image (slow) | ~47s / image |
# | **Fast R-CNN** (Girshick, 2015) | External Selective Search | Single full-image feature map + **RoI Pooling** | ~2s / image |
# | **Faster R-CNN** (Ren et al., 2015) | **Region Proposal Network (RPN)** | Fully end-to-end differentiable CNN | ~0.2s / image |
#
# Congratulations! You have built, trained, and evaluated a complete two-stage object detector from scratch.
