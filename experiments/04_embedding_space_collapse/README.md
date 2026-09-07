# Experiment 04: Embedding Space Collapse in Self-Supervised Learning

## 1. Pedagogical Overview: What is Embedding Space Collapse?

In self-supervised and representation learning (especially joint-embedding / Siamese architectures), a network $f_\theta$ is trained to map two augmented views $v_1, v_2 \sim \mathcal{T}(x)$ of the same image to similar latent vectors $z_1, z_2 \in \mathbb{R}^d$.

If the optimization objective simply minimizes the distance between positive pairs without counteracting mechanisms:
$$\min_\theta \mathbb{E}_{x \sim \mathcal{D}} \left[ \mathcal{D}(f_\theta(v_1), f_\theta(v_2)) \right]$$

The optimization finds a trivial, catastrophic global minimum: **the network outputs the exact same constant vector $\mathbf{z}_0$ for every input in the universe**, regardless of the image content:
$$\forall x, \quad f_\theta(x) = \mathbf{z}_0$$

This failure mode is known as **Embedding Space Collapse** (or Representation Collapse). When collapsed, the representation carries **0 bits of mutual information** and is completely useless for downstream classification or detection.

---

## 2. Taxonomy of Collapse

### 2.1 Complete (Constant / Point) Collapse
- **Behavior**: All samples collapse to a single point $\mathbf{c}$ on the unit hypersphere.
- **Diagnostic Signal**:
  - Embedding variance per dimension $\text{Var}(z_j) \to 0$ for all $j=1, \dots, d$.
  - Cosine similarity between any two random images equals $1.0$.
  - Rank of embedding matrix $\text{Rank}(Z) = 0$ (or 1 if centered).

### 2.2 Dimensional (Subspace / Partial) Collapse
- **Behavior**: Embeddings do not collapse to a single point, but span only a lower-dimensional subspace or narrow 1D line/manifold, ignoring all orthogonal dimensions.
- **Diagnostic Signal**:
  - The first 1 or 2 singular values in the SVD spectrum account for $100\%$ of total variance ($\sigma_1 \gg \sigma_2 \approx \dots \approx \sigma_d \to 0$).
  - Off-diagonal covariance entries are strongly correlated.

---

## 3. Visual Comparisons

### 3.1 2D Latent Space Comparison (Healthy vs Point vs Subspace Collapse)
![2D Embedding Space Comparison](file:///Users/jorgvt/Developer/CVMIAX/experiments/04_embedding_space_collapse/figures/01_2d_embedding_space_comparison.png)

- **Healthy Space (Left)**: Distinct semantic classes are uniformly distributed around the unit circle with high variance and clear cluster separation.
- **Complete Collapse (Middle)**: All 5 classes collapse onto a single point $\mathbf{c} = (1.0, 0.0)$.
- **Subspace Collapse (Right)**: All samples are squashed onto a 1D line, losing one full degree of freedom.

---

### 3.2 Spectral Analysis & Covariance Matrix Diagnostics
![Spectral and Covariance Analysis](file:///Users/jorgvt/Developer/CVMIAX/experiments/04_embedding_space_collapse/figures/02_spectral_analysis_and_covariance.png)

- **Covariance Heatmaps**:
  - *Healthy*: Diagonal dominance with near-zero off-diagonals (dimensions are decorrelated and encode independent features).
  - *Collapsed*: Off-diagonal entries are either zero (constant collapse) or completely collinear (subspace collapse).
- **SVD Singular Value Spectrum (Scree Plot)**:
  - *Healthy*: Smooth, isotropic distribution of singular values across all $d=16$ latent dimensions.
  - *Collapsed*: Rank drops precipitously to $\le 1$.

---

### 3.3 Empirical Siamese Training Dynamics (CIFAR-10)
![Empirical Training Dynamics](file:///Users/jorgvt/Developer/CVMIAX/experiments/04_embedding_space_collapse/figures/04_empirical_collapse_curves.png)

When training Siamese networks without negative pairs:
1. **Naive Siamese + Weight Decay**: The output variance $\text{Std}(z)$ plunges to near-zero ($< 0.0003$) while weight decay actively shrinks all weights to zero ($W \to \mathbf{0}$).
2. **Naive Siamese (No Weight Decay)**: The output variance collapses just as rapidly, but because the task gradient vanishes ($\nabla_W \mathcal{L} \approx \mathbf{0}$ once $z_1 \approx z_2$), the weights freeze near their random initialization scale.
3. **Variance-Regularized (VICReg)**: The explicit variance hinge penalty $\max(0, 1 - \text{Std}(z))$ provides active gradients throughout training, maintaining healthy variance ($\text{Std}(z) \approx 1.0$) and preventing collapse.

---

### 3.4 Post-Training Weight Inspection: Collapse Regimes
![Weight Distributions and Norms](file:///Users/jorgvt/Developer/CVMIAX/experiments/04_embedding_space_collapse/figures/05_weight_distributions_and_norms.png)

Comparing the network weights post-training:
- **Weight Histograms (Top Row)**:
  - In **Naive + Weight Decay**, weights contract tightly to a sharp spike at **0.0**.
  - In **Naive without Weight Decay**, weights remain in their initial random distribution because task gradients vanish the moment output collapse occurs.
  - In **VICReg Regularized**, weights adapt into an active, wide distribution driven by representation variance.
- **Layer-wise Frobenius Norms & Heatmaps (Bottom Row)**:
  - Weight Decay drives the idle collapsed weights to near-zero Frobenius norm across all layers.
  - VICReg maintains full-rank, non-degenerate projection matrices.

---

## 4. How Modern SSL Architectures Prevent Collapse

| Method | Key Mechanism | Representative Papers |
|---|---|---|
| **Contrastive Repulsion** | Adds negative pairs via InfoNCE loss to push distinct images apart | SimCLR (Chen et al., 2020), MoCo (He et al., 2020) |
| **Asymmetric Dynamics** | Stop-gradient operator $\text{stop\_gradient}$ + Predictor MLP | BYOL (Grill et al., 2020), SimSiam (Chen & He, 2021) |
| **Covariance Regularization** | Explicit loss terms penalizing low variance and cross-correlation | Barlow Twins (Zbontar et al., 2021), VICReg (Bardes et al., 2022) |
| **Online Clustering** | Enforces equal partition over learnable cluster centroids | SwAV (Caron et al., 2020) |

---

## 5. Master Lecture Slide Poster

![Master Collapse Overview Slide](file:///Users/jorgvt/Developer/CVMIAX/experiments/04_embedding_space_collapse/figures/slide_embedding_collapse_overview.png)

---

## 6. How to Run

```bash
# Run all visualizations and empirical CIFAR-10 collapse experiment
uv run python experiments/04_embedding_space_collapse/run_all.py

# Run visualization generator individually
uv run python experiments/04_embedding_space_collapse/visualize_collapse.py
```
