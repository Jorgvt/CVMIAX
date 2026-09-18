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
# # Experiment 14: Contrastive Language-Image Pre-Training (CLIP)
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jorgvt/CVMIAX/blob/main/experiments/14_clip_multimodal_contrastive/experiment_14_clip_multimodal_contrastive.ipynb)
#
# **Pedagogical Objective:**
# Explore the foundational paradigm of multimodal contrastive representation learning (Radford et al., 2021).
# In this experiment, we investigate:
# 1. **Dual-Encoder Architecture & Joint Embedding Space**: How independent convolutional vision and Transformer text encoders map disparate modalities into a shared unit-hypersphere metric space.
# 2. **Symmetric InfoNCE Loss**: Why bidirectional contrastive alignment (Image $\to$ Text and Text $\to$ Image) prevents representation collapse and establishes sharp semantic correspondences.
# 3. **Compositional Zero-Shot Generalization**: How the model correctly binds unseen combinations of seen concepts (e.g., recognizing *"yellow square"* when yellow squares were strictly excluded from training).
# 4. **Novel Shape Topology Showcase**: How CLIP handles completely novel geometric primitives (e.g., 5-pointed *stars* and *diamonds*), demonstrating how color is preserved with high confidence while shape uncertainty is cleanly distributed across nearest geometric features.
# 5. **Cross-Modal Bidirectional Retrieval**: Querying images via free-form text prompts and ranking descriptive captions for images.

# %% [markdown]
# ## 1. Mathematical Formulation
#
# ### 1.1 The Dual-Encoder Metric Formulation
# Given a batch of $N$ paired image-text instances $\{(x_i, t_i)\}_{i=1}^N$:
# 1. **Vision Encoder $f_I$**: Maps image $x_i \in \mathbb{R}^{H \times W \times C}$ to embedding $u_i \in \mathbb{R}^d$, normalized onto the unit hypersphere:
#    $$\hat{I}_i = \frac{f_I(x_i)}{\|f_I(x_i)\|_2}$$
# 2. **Text Encoder $f_T$**: Maps token sequence $t_j$ to embedding $v_j \in \mathbb{R}^d$, normalized onto the unit hypersphere:
#    $$\hat{T}_j = \frac{f_T(t_j)}{\|f_T(t_j)\|_2}$$
#
# ### 1.2 The Scaled Cosine Similarity Matrix
# The pairwise similarity between all images and texts in the minibatch forms an $N \times N$ matrix:
# $$S_{i, j} = \tau \cdot \left( \hat{I}_i \cdot \hat{T}_j \right)$$
# where $\tau = \exp(\text{logit\_scale})$ is a learnable positive temperature parameter that modulates the sharpness of the posterior distribution.
#
# ### 1.3 Symmetric InfoNCE Loss
# In a batch of $N$ pairs, the diagonal entries $(i, i)$ represent the true positive pairs, while off-diagonal entries $(i, j)$ ($i \neq j$) act as negative pairs.
#
# The loss is computed symmetrically across both modalities:
#
# **Image-to-Text Cross-Entropy Loss:**
# $$\mathcal{L}_{I \to T} = - \frac{1}{N} \sum_{i=1}^N \log \frac{\exp(S_{i, i})}{\sum_{j=1}^N \exp(S_{i, j})}$$
#
# **Text-to-Image Cross-Entropy Loss:**
# $$\mathcal{L}_{T \to I} = - \frac{1}{N} \sum_{j=1}^N \log \frac{\exp(S_{j, j})}{\sum_{i=1}^N \exp(S_{i, j})}$$
#
# **Total Symmetric InfoNCE Objective:**
# $$\mathcal{L}_{\text{CLIP}} = \frac{1}{2} \left( \mathcal{L}_{I \to T} + \mathcal{L}_{T \to I} \right)$$
#
# ### 1.4 Zero-Shot Classification via Cosine Similarity
# At test time, zero-shot classification is performed by computing dot products between the image embedding $\hat{I}_{\text{query}}$ and the text embeddings of $K$ candidate class prompts $\{\hat{T}_k\}_{k=1}^K$:
# $$P(c_k \mid x) = \frac{\exp(\tau \cdot \hat{I}_{\text{query}} \cdot \hat{T}_k)}{\sum_{m=1}^K \exp(\tau \cdot \hat{I}_{\text{query}} \cdot \hat{T}_m)}$$

# %%
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# %% [markdown]
# ## 2. Data Generation: Multi-Attribute Shapes & Hold-Out Splits
#
# We generate procedural images of geometric shapes (*circle, square, triangle, hexagon*) in distinct colors (*red, blue, green, yellow, purple, cyan, orange*) paired with descriptive captions.
#
# We create three distinct evaluation splits:
# 1. **In-Distribution Validation Set**: Seen shapes and colors.
# 2. **Compositional Zero-Shot Set**: Seen shapes and colors in combinations *never paired during training* (e.g., `(yellow, square)`, `(cyan, triangle)`).
# 3. **Novel Shape Zero-Shot Set**: Completely novel geometric primitives (*star*, *diamond*) that never appeared during training.

# %%
from experiments.14_clip_multimodal_contrastive.dataset import (
    ALL_SHAPES,
    COLORS,
    COMPOSITIONAL_HOLDOUT,
    KNOWN_SHAPES,
    NOVEL_SHAPES,
    build_text_vectorizer,
    create_multimodal_dataset,
    create_tf_dataset,
    draw_geometric_shape,
)

train_data, val_data, comp_data, novel_data = create_multimodal_dataset(
    num_samples_per_combo=120,
    seed=42,
)

print(f"Dataset summary:")
print(f"  • Training samples:                 {len(train_data['images'])}")
print(f"  • In-distribution validation:       {len(val_data['images'])}")
print(f"  • Compositional zero-shot holdout:  {len(comp_data['images'])}")
print(f"  • Novel shape zero-shot holdout:    {len(novel_data['images'])}")

# Build Text Vectorizer
vectorizer = build_text_vectorizer(train_data["texts"])
vocab_size = vectorizer.vocabulary_size()
print(f"Vocabulary size: {vocab_size}")

# Build tf.data datasets
train_ds = create_tf_dataset(train_data, vectorizer, batch_size=64, shuffle=True)
val_ds = create_tf_dataset(val_data, vectorizer, batch_size=64, shuffle=False)

# %% [markdown]
# ### Visualizing Training and Zero-Shot Evaluation Samples

# %%
fig, axes = plt.subplots(2, 4, figsize=(14, 7), dpi=150)

# Training samples
for idx in range(4):
    ax = axes[0, idx]
    ax.imshow(train_data["images"][idx])
    ax.axis("off")
    ax.set_title(f"Train Pair\n\"{train_data['texts'][idx]}\"", fontsize=9)

# Zero-Shot evaluation samples (Compositional + Novel)
zs_samples = [
    (comp_data["images"][0], comp_data["texts"][0], "Compositional (Unseen Combo)"),
    (comp_data["images"][10], comp_data["texts"][10], "Compositional (Unseen Combo)"),
    (novel_data["images"][0], novel_data["texts"][0], "Novel Shape (Star)"),
    (novel_data["images"][100], novel_data["texts"][100], "Novel Shape (Diamond)"),
]

for idx, (img, caption, tag) in enumerate(zs_samples):
    ax = axes[1, idx]
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(f"[{tag}]\n\"{caption}\"", fontsize=9)

plt.suptitle("Sample Dataset Pairs: Training vs Zero-Shot Holdouts", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Model Architecture: Dual Encoders & CLIP Dual-Encoder
#
# - **Vision Encoder**: ConvNet with Batch Normalization, Global Average Pooling, and Dense projection layer with UnitNormalization ($L_2$ norm).
# - **Text Encoder**: Token & Positional Embeddings, Mini Multi-Head Attention Transformer block, Global Pooling, and Dense projection with UnitNormalization.
# - **Learnable Temperature ($\tau$)**: Initialized to $\tau = 1/0.07 \approx 14.28$ ($\text{logit\_scale} \approx 2.66$).

# %%
from experiments.14_clip_multimodal_contrastive.models import (
    CLIPDualEncoder,
    build_text_encoder,
    build_vision_encoder,
)

vision_enc = build_vision_encoder(input_shape=(64, 64, 3), embedding_dim=64)
text_enc = build_text_encoder(vocab_size=vocab_size, seq_length=8, embedding_dim=64)

clip_model = CLIPDualEncoder(
    vision_encoder=vision_enc,
    text_encoder=text_enc,
    init_temperature=0.07,
)

optimizer = keras.optimizers.Adam(learning_rate=1e-3)
clip_model.compile(optimizer=optimizer)

# %% [markdown]
# ## 4. Training with Symmetric InfoNCE Loss

# %%
from experiments.14_clip_multimodal_contrastive.train import train_clip_model

clip_model, vectorizer, history, eval_results = train_clip_model(
    epochs=16,
    batch_size=64,
    learning_rate=1e-3,
    embedding_dim=64,
    samples_per_combo=120,
    seed=42,
)

# %% [markdown]
# ### Training Dynamics & Temperature Scaling

# %%
from experiments.14_clip_multimodal_contrastive.visualize import (
    plot_similarity_matrix_heatmaps,
    plot_training_dynamics,
    plot_zero_shot_showcase,
    plot_cross_modal_retrieval,
    plot_joint_embedding_space,
)

fig_dyn = plot_training_dynamics(history)
plt.show()

# %% [markdown]
# ## 5. Multimodal Alignment: Similarity Matrix Heatmaps
#
# Before training, cosine similarities between random pairs are close to zero or uniformly distributed. After InfoNCE pre-training, the diagonal elements (matched image-text pairs) dominate with high similarity.

# %%
fig_heat = plot_similarity_matrix_heatmaps(
    eval_results["sim_before"],
    eval_results["sim_after"],
    eval_results["sample_txts"],
)
plt.show()

# %% [markdown]
# ## 6. Zero-Shot Classification: Seen, Compositional, and Novel Shapes
#
# We evaluate zero-shot classification across:
# 1. **Seen Pairs** (Baseline in-distribution)
# 2. **Compositional Zero-Shot** (`yellow square`, `cyan triangle` - concepts seen separately but never together)
# 3. **Novel Shape Primitives** (`star`, `diamond` - shapes never seen during training)

# %%
# Build zero-shot test cases with softmax probability distributions
all_candidate_classes = [f"{c}_{s}" for s in ALL_SHAPES for c in COLORS.keys()]
candidate_prompts = [f"a photo of a {c.replace('_', ' ')}" for c in all_candidate_classes]
tokenized_all_prompts = vectorizer(tf.constant(candidate_prompts))
all_text_embeddings = clip_model.encode_texts(tokenized_all_prompts).numpy()

def get_zs_probabilities(img_arr):
    img_emb = clip_model.encode_images(tf.constant(np.expand_dims(img_arr, 0))).numpy()
    cos_sim = np.matmul(img_emb, all_text_embeddings.T)[0]
    temp = float(tf.math.exp(clip_model.logit_scale).numpy())
    exp_sim = np.exp(cos_sim * temp)
    probs = exp_sim / np.sum(exp_sim)
    return probs

test_cases = []

# Seen
seen_img = draw_geometric_shape("circle", COLORS["red"], jitter=False)
seen_probs = get_zs_probabilities(seen_img)
top_idx = np.argsort(-seen_probs)[:5]
test_cases.append({
    "image": seen_img,
    "category": "Seen In-Distribution",
    "true_label": "red circle",
    "candidate_prompts": [candidate_prompts[i] for i in top_idx],
    "probabilities": seen_probs[top_idx],
})

# Compositional 1
comp_img1 = draw_geometric_shape("square", COLORS["yellow"], jitter=False)
comp_probs1 = get_zs_probabilities(comp_img1)
top_idx = np.argsort(-comp_probs1)[:5]
test_cases.append({
    "image": comp_img1,
    "category": "Compositional Zero-Shot (Unseen Pair)",
    "true_label": "yellow square",
    "candidate_prompts": [candidate_prompts[i] for i in top_idx],
    "probabilities": comp_probs1[top_idx],
})

# Novel Shape: Star
star_img = draw_geometric_shape("star", COLORS["yellow"], jitter=False)
star_probs = get_zs_probabilities(star_img)
top_idx = np.argsort(-star_probs)[:5]
test_cases.append({
    "image": star_img,
    "category": "Novel Geometric Figure (Star)",
    "true_label": "yellow star",
    "candidate_prompts": [candidate_prompts[i] for i in top_idx],
    "probabilities": star_probs[top_idx],
})

# Novel Shape: Diamond
diamond_img = draw_geometric_shape("diamond", COLORS["purple"], jitter=False)
diamond_probs = get_zs_probabilities(diamond_img)
top_idx = np.argsort(-diamond_probs)[:5]
test_cases.append({
    "image": diamond_img,
    "category": "Novel Geometric Figure (Diamond)",
    "true_label": "purple diamond",
    "candidate_prompts": [candidate_prompts[i] for i in top_idx],
    "probabilities": diamond_probs[top_idx],
})

fig_zs = plot_zero_shot_showcase(test_cases)
plt.show()

# %% [markdown]
# ## 7. Cross-Modal Retrieval (Text $\to$ Image)

# %%
query_prompts = [
    "a photo of a blue square",
    "a green triangle",
    "a photo of a yellow circle",
    "a purple hexagon",
]
val_images = eval_results["val_data"]["images"]
tokenized_queries = vectorizer(tf.constant(query_prompts))
query_txt_emb = clip_model.encode_texts(tokenized_queries).numpy()
val_img_emb = clip_model.encode_images(tf.constant(val_images)).numpy()

retrieval_images = []
retrieval_scores = []
for q_emb in query_txt_emb:
    sims = np.matmul(val_img_emb, q_emb)
    top5_idx = np.argsort(-sims)[:5]
    retrieval_images.append([val_images[i] for i in top5_idx])
    retrieval_scores.append([sims[i] for i in top5_idx])

fig_ret = plot_cross_modal_retrieval(query_prompts, retrieval_images, retrieval_scores)
plt.show()

# %% [markdown]
# ## 8. Joint Multimodal Metric Space (2D PCA)

# %%
sample_vis_classes = ["red_circle", "blue_square", "green_triangle", "yellow_square"]
vis_imgs, vis_txts, vis_lbls = [], [], []

for c_lbl in sample_vis_classes:
    src = eval_results["comp_data"] if "yellow_square" in c_lbl else eval_results["val_data"]
    mask = src["labels"] == c_lbl
    for img, txt in zip(src["images"][mask][:12], src["texts"][mask][:12]):
        vis_imgs.append(img)
        vis_txts.append(txt)
        vis_lbls.append(c_lbl.replace("_", " "))

vis_img_emb = clip_model.encode_images(tf.constant(np.array(vis_imgs))).numpy()
vis_txt_emb = clip_model.encode_texts(vectorizer(tf.constant(vis_txts))).numpy()

fig_pca = plot_joint_embedding_space(vis_img_emb, vis_txt_emb, vis_lbls)
plt.show()
