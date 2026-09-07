# Experiment 03: Self-Supervised Learning Pretext Tasks (Rotation, Jigsaw, and Colorization)

## 1. Introduction: What is Self-Supervised Learning?

In traditional supervised learning, deep convolutional networks require millions of human-annotated labels $(x_i, y_i)$. **Self-Supervised Learning (SSL)** eliminates the need for manual annotations by formulating a **pretext task** where:
1. The **input data $X$** is transformed or degraded via a known algorithmic transformation.
2. The **pseudo-ground-truth label $y$** (or target signal $Y$) is automatically derived from the transformation metadata itself, at zero annotation cost.

By forcing a neural network to solve these pretext puzzles, the network is compelled to learn rich, high-level semantic representations (such as object boundaries, anatomical composition, orientation, and context) that transfer effectively to downstream target tasks (e.g., classification, object detection, segmentation).

```
Unlabeled Images ──► [ Algorithmic Transformation ] ──► (Transformed Input X, Pseudo-Label y)
                                                                    │
                                                                    ▼
                                                            [ ConvNet Backbone ] ──► Feature Representation
```

---

## 2. The Three Classic Pretext Tasks

### 2.1 Rotation Prediction (Gidaris et al., ICLR 2018)

#### How Inputs and Labels are Created
Given an unlabeled image $X \in \mathbb{R}^{H \times W \times C}$:
1. We apply four discrete 2D rotations: $\theta \in \{0^\circ, 90^\circ, 180^\circ, 270^\circ\}$.
2. We assign a discrete 4-class categorical label $y \in \{0, 1, 2, 3\}$.

$$\tilde{X}_k = \text{Rot}(X, 90^\circ \times k), \quad y_k = k \quad \text{for } k \in \{0, 1, 2, 3\}$$

- **Inputs**: Batch of 4 rotated copies of each image.
- **Targets**: Categorical integer $y \in \{0, 1, 2, 3\}$ trained with Categorical Cross-Entropy Loss:
$$\mathcal{L}_{\text{rot}} = - \sum_{k=0}^3 \log P(Y = k \mid \tilde{X}_k)$$

#### What the Model Learns (Inductive Bias)
To recognize whether a dog, tree, or car is upside down ($180^\circ$) or sideways ($90^\circ / 270^\circ$), the model **cannot rely on low-level edge statistics**. It must learn high-level semantic features:
- Where the ground (gravity) vs. sky is located.
- Anatomy: heads are usually on top, legs on the bottom.
- Canonical viewpoints of common physical objects.

#### Visual Walkthrough
![Rotation Prediction Task](file:///Users/jorgvt/Developer/CVMIAX/experiments/03_self_supervised_pretext_tasks/figures/01_rotation_prediction_task.png)

```python
# Keras / NumPy Input & Label Generation
def create_rotation_batch(images):
    rot0 = images
    rot90 = np.rot90(images, k=1, axes=(1, 2))
    rot180 = np.rot90(images, k=2, axes=(1, 2))
    rot270 = np.rot90(images, k=3, axes=(1, 2))
    
    X_ssl = np.concatenate([rot0, rot90, rot180, rot270], axis=0)
    y_ssl = np.concatenate([
        np.zeros(len(images)),
        np.ones(len(images)),
        np.full(len(images), 2),
        np.full(len(images), 3),
    ], axis=0)
    return X_ssl, y_ssl
```

---

### 2.2 Jigsaw Puzzle Solving (Noroozi & Favaro, ECCV 2016)

#### How Inputs and Labels are Created
Given an unlabeled image $X$:
1. We partition the image into a $3 \times 3$ grid of 9 spatial patches $\{t_0, t_1, \dots, t_8\}$.
2. We define a predefined vocabulary of $K$ distinct permutations $\mathcal{P} = \{\pi_0, \pi_1, \dots, \pi_{K-1}\}$ (selected to have high pairwise Hamming distance).
3. We select permutation $\pi_k$, shuffle the patches accordingly: $\tilde{T} = [t_{\pi_k(0)}, t_{\pi_k(1)}, \dots, t_{\pi_k(8)}]$.
4. The target label is the permutation class index $y = k$.

- **Inputs**: 9 shuffled image patches fed to a **Siamese / Shared ConvNet backbone**.
- **Targets**: Permutation index $k \in \{0, \dots, K-1\}$ optimized via Multi-class Cross-Entropy.

#### What the Model Learns (Inductive Bias)
To solve the spatial puzzle, the network must identify:
- **Object part compositionality**: e.g., how a dog's head connects to its torso, and how torso connects to paws.
- **Spatial continuity & semantic layout**: understanding horizon lines, texture continuities, and geometric alignment.
- *Note on Shortcut Learning*: In practice, a small spatial gap or color jitter between tiles is applied to prevent the model from cheating using pixel continuity at tile borders.

#### Visual Walkthrough
![Jigsaw Puzzle Task](file:///Users/jorgvt/Developer/CVMIAX/experiments/03_self_supervised_pretext_tasks/figures/02_jigsaw_puzzle_task.png)

---

### 2.3 Image Colorization (Zhang et al., ECCV 2016)

#### How Inputs and Labels are Created
Given a full-color RGB image $X$:
1. We extract the **Grayscale / Luminance channel** ($L$ channel in CIE-Lab space, or luminance $Y = 0.299R + 0.587G + 0.114B$).
2. The grayscale image acts as the **Input**: $X_{\text{gray}} \in \mathbb{R}^{H \times W \times 1}$.
3. The original color channels (RGB or chrominance channels $a, b$) serve as the **Supervisory Target**: $Y_{\text{color}} \in \mathbb{R}^{H \times W \times 3}$.

$$\mathcal{L}_{\text{color}} = \frac{1}{HW} \sum_{h=1}^H \sum_{w=1}^W \| \hat{Y}_{\text{color}}(h, w) - Y_{\text{color}}(h, w) \|_2^2$$

#### What the Model Learns (Inductive Bias)
Grayscale images contain structural texture and edges but no color information. To colorize realistically:
- The network must recognize that a spherical textured region in the sky is the **Sun** $\implies$ color Yellow.
- The canopy of a tree is **Foliage** $\implies$ color Green.
- Water / Sky regions $\implies$ color Blue.
- Thus, the model is forced to perform **implicit object recognition and semantic segmentation** without a single human label!

#### Visual Walkthrough
![Colorization Task](file:///Users/jorgvt/Developer/CVMIAX/experiments/03_self_supervised_pretext_tasks/figures/03_colorization_task.png)

---

## 3. Master Slide Overview Poster

A comprehensive 3-row pedagogical poster comparing all three pretext tasks, their inputs, automated labels, and learned inductive biases:

![Master SSL Pretext Tasks Overview](file:///Users/jorgvt/Developer/CVMIAX/experiments/03_self_supervised_pretext_tasks/figures/slide_ssl_pretext_tasks_overview.png)

---

## 4. Experiment Structure

```
experiments/03_self_supervised_pretext_tasks/
├── data_generators.py                  # Creation of inputs and pseudo-labels for Rotation, Jigsaw, and Colorization
├── models.py                           # Keras models: RotationClassifier, JigsawSolver, ColorizationUNet
├── visualize_tasks.py                  # Generates publication-ready figures for slides & course notes
├── train_pretext_demo.py               # Empirical training demo on CIFAR-10 (Rotation SSL)
├── run_all.py                          # Master execution script
├── figures/
│   ├── 01_rotation_prediction_task.png
│   ├── 02_jigsaw_puzzle_task.png
│   ├── 03_colorization_task.png
│   ├── slide_ssl_pretext_tasks_overview.png
│   ├── ssl_rotation_training_curve.png
│   └── ssl_rotation_qualitative_predictions.png
└── README.md                           # This guide
```

---

## 5. How to Run

```bash
# Run all visualizations and training demonstration
uv run python experiments/03_self_supervised_pretext_tasks/run_all.py

# Or run visualization generation individually
uv run python experiments/03_self_supervised_pretext_tasks/visualize_tasks.py
```
