# Experiment 11: Neural Radiance Fields (NeRF) for 3D Novel View Synthesis

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jorgvt/CVMIAX/blob/main/experiments/11_nerf_novel_view_synthesis/experiment_11_nerf.ipynb)

---

## Pedagogical Objectives
1. **Implicit Neural Representations**: Understand how a continuous 3D physical scene can be parameterized inside the weights of a Multi-Layer Perceptron (MLP) mapping 3D coordinates $(x, y, z) \to (\text{RGB}, \sigma)$ rather than discrete voxel grids, meshes, or point clouds.
2. **Pinhole Camera Ray Casting**: Master the geometry connecting 2D pixel coordinates, camera intrinsics (focal length $f$, optical center), and camera extrinsics (Camera-to-World matrix $[R \mid \mathbf{t}]$) to generate 3D ray origins $\mathbf{o}$ and directions $\mathbf{d}$.
3. **Spectral Bias & Positional Encoding**: Unpack why standard MLPs struggle to learn high-frequency details (Neural Tangent Kernel behavior) and how Fourier positional embeddings $\gamma(p)$ enable razor-sharp geometric boundaries and texture recovery.
4. **Differentiable Volumetric Rendering**: Derive and implement the discrete numerical approximation of the volume rendering integral (classical optical absorption formulation) with accumulated transmittance, rendering both photorealistic RGB images and continuous depth maps.
5. **360-Degree Novel View Synthesis**: Synthesize smooth orbital camera trajectories around an object from sparse multi-view observations.

---

## 1. Background & The Implicit Paradigm Shift

Traditional 3D computer vision models represent geometry discretely:
- **Voxel Grids**: Discretize space into a 3D lattice. Memory scales cubically $\mathcal{O}(N^3)$, limiting resolution.
- **Triangle Meshes**: Require fixed topology and cannot easily represent fuzzy, semi-transparent, or complex volumetric structures (smoke, hair, fine foliage).
- **Point Clouds**: Lack connectivity and continuous surface normals.

**NeRF (Mildenhall et al., ECCV 2020)** replaces explicit 3D structures with a continuous implicit function:
$$F_\Theta: (\mathbf{x}, \mathbf{d}) \mapsto (\mathbf{c}, \sigma)$$
where:
- $\mathbf{x} = (x, y, z) \in \mathbb{R}^3$ is a 3D spatial coordinate.
- $\mathbf{d} = (\theta, \phi) \in \mathbb{S}^2$ is a 2D viewing direction vector.
- $\mathbf{c} = (r, g, b) \in [0, 1]^3$ is the directional emitted radiance (color).
- $\sigma \in [0, \infty)$ is the volume density (differential probability of ray termination).

---

## 2. Mathematical Formulation

### 2.1 Camera Pinhole Ray Casting
Given an image plane of dimensions $H \times W$, focal length $f$, optical center $(c_x, c_y) = (W/2, H/2)$, and a Camera-to-World extrinsic matrix $[R \mid \mathbf{t}] \in \mathbb{R}^{3 \times 4}$:

For each pixel coordinate $(u, v)$ where $u \in [0, W-1], v \in [0, H-1]$:
1. Direction in camera frame:
   $$\mathbf{d}_{\text{cam}} = \begin{pmatrix} \frac{u - c_x}{f} \\ -\frac{v - c_y}{f} \\ -1 \end{pmatrix}$$
2. Direction in world frame:
   $$\mathbf{d} = R \cdot \mathbf{d}_{\text{cam}}$$
3. Ray origin in world frame:
   $$\mathbf{o} = \mathbf{t}$$
4. Parametric ray equation:
   $$\mathbf{r}(t) = \mathbf{o} + t \mathbf{d}, \quad t \ge 0$$

```
Camera Origin [o]
      │
      ├─── Ray r(t) = o + t * d ────────►
     ┌┴──────┐                     • x_1 (Near t_n)
     │ Image │                     • x_2
     │ Plane │                     • x_i = o + t_i * d
     └───────┘                     • x_N (Far t_f)
```

---

### 2.2 Stratified 3D Point Sampling
To evaluate the continuous radiance field along ray $\mathbf{r}(t)$ between near plane $t_n$ and far plane $t_f$, we partition $[t_n, t_f]$ into $N$ uniformly spaced bins and draw random samples (stratified sampling):
$$t_i \sim \mathcal{U}\left[ t_n + \frac{i-1}{N}(t_f - t_n), \; t_n + \frac{i}{N}(t_f - t_n) \right], \quad i \in \{1, \dots, N\}$$
$$\mathbf{x}_i = \mathbf{o} + t_i \mathbf{d}$$

*Pedagogical Takeaway*: Random jitter during training allows the MLP to learn a truly continuous 3D representation across space, preventing the network from overfitting to fixed discrete depth intervals.

---

### 2.3 Positional Encoding & Mitigating Spectral Bias
Standard fully connected networks acting on raw low-dimensional coordinates $\mathbf{x} \in \mathbb{R}^3$ suffer from **spectral bias** (Rahaman et al., 2019): lower frequency components are learned exponentially faster, resulting in overly smooth, blurry outputs without sharp edges.

NeRF overcomes this by mapping inputs through a high-frequency Fourier embedding $\gamma: \mathbb{R} \to \mathbb{R}^{2L}$:
$$\gamma(p) = \Big( \sin(2^0 \pi p), \; \cos(2^0 \pi p), \; \dots, \; \sin(2^{L-1} \pi p), \; \cos(2^{L-1} \pi p) \Big)$$

For a 3D coordinate $\mathbf{x} = (x, y, z)$, applying $\gamma(\cdot)$ with $L=6$ octaves expands the input from $3$ dimensions to $3 \times (2 \times 6) = 36$ (or $39$ if including raw coordinates).

---

### 2.4 Differentiable Volumetric Rendering
The expected light $\mathbf{C}(\mathbf{r})$ arriving at camera pixel $\mathbf{r}(t)$ is governed by the volume rendering integral:
$$\mathbf{C}(\mathbf{r}) = \int_{t_n}^{t_f} T(t) \, \sigma(\mathbf{r}(t)) \, \mathbf{c}(\mathbf{r}(t), \mathbf{d}) \, dt$$
where $T(t) = \exp\left(-\int_{t_n}^t \sigma(\mathbf{r}(s)) \, ds\right)$ is the **cumulative transmittance** (the probability that the ray travels from $t_n$ to $t$ without hitting any occluding particle).

#### Discrete Numerical Quadrature:
Using the $N$ stratified sample points with step intervals $\delta_i = (t_{i+1} - t_i) \|\mathbf{d}\|_2$:
1. **Discrete Opacity**:
   $$\alpha_i = 1 - \exp(-\sigma_i \delta_i)$$
2. **Cumulative Transmittance**:
   $$T_i = \prod_{j=1}^{i-1} (1 - \alpha_j) = \exp\left(-\sum_{j=1}^{i-1} \sigma_j \delta_j\right), \quad T_1 = 1$$
3. **Integration Weight**:
   $$w_i = T_i \alpha_i$$
4. **Accumulated Ray RGB Color**:
   $$\hat{\mathbf{C}}(\mathbf{r}) = \sum_{i=1}^N w_i \mathbf{c}_i$$
5. **Accumulated Ray Depth**:
   $$\hat{D}(\mathbf{r}) = \sum_{i=1}^N w_i t_i$$

For synthetic objects on a white background with total foreground opacity $A(\mathbf{r}) = \sum_{i=1}^N w_i$:
$$\hat{\mathbf{C}}_{\text{comp}}(\mathbf{r}) = \hat{\mathbf{C}}(\mathbf{r}) + \big(1 - A(\mathbf{r})\big)$$

---

### 2.5 Photometric Loss & PSNR
NeRF requires **zero 3D supervision**. It is trained purely on multi-view 2D images by minimizing the mean squared photometric reconstruction error across a batch of sampled rays $\mathcal{R}$:
$$\mathcal{L}_{\text{MSE}} = \frac{1}{|\mathcal{R}|} \sum_{\mathbf{r} \in \mathcal{R}} \big\| \hat{\mathbf{C}}(\mathbf{r}) - \mathbf{C}_{\text{gt}}(\mathbf{r}) \big\|_2^2$$

Reconstruction quality is evaluated using Peak Signal-to-Noise Ratio (PSNR):
$$\text{PSNR} = -10 \log_{10}(\mathcal{L}_{\text{MSE}})$$

---

## 3. Architecture Breakdown

```
Input 3D Coordinate (x, y, z)
          │
          ▼
   PositionalEncoding (L=6)  ──► [39 dims]
          │
          ▼
   Dense(128, ReLU)
          │
   Dense(128, ReLU)
          │
   Dense(128, ReLU)
          │
   Dense(128, ReLU)
          ├───► Dense(3, Sigmoid) ──► Emitted Color c in [0, 1]
          │
          └───► Dense(1, ReLU)    ──► Volume Density σ >= 0
```

---

## 4. Key Experimental Results & Visualizations

| Metric | NeRF without PE ($L=0$) | NeRF with Positional Encoding ($L=6$) |
| :--- | :---: | :---: |
| **Test PSNR (dB)** | ~18.5 dB | **~24.5 - 27.0 dB** |
| **Visual Sharpness** | Oversmoothed, blurry boundaries | Razor-sharp edges, high-frequency textures |
| **3D Depth Estimation** | Ambiguous | Crisp, accurate metric geometry |

---

## 5. How to Run the Experiment

```bash
# Run the automated end-to-end training and visualization suite
uv run python experiments/11_nerf_novel_view_synthesis/run_all.py

# Launch interactive Jupytext notebook
uv run jupyter notebook experiments/11_nerf_novel_view_synthesis/experiment_11_nerf.ipynb
```
