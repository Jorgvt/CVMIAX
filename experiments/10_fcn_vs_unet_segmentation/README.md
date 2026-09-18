# Experiment 10: Plain FCN vs U-Net (The Power of Skip Connections in Semantic Segmentation)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jorgvt/CVMIAX/blob/main/experiments/10_fcn_vs_unet_segmentation/experiment_10_fcn_vs_unet.ipynb)

## 1. Overview & Pedagogical Objective

In dense computer vision tasks like semantic segmentation, the neural network must assign a semantic class label to every individual pixel in an image. This task embodies a fundamental architectural conflict known as the **"What" vs. "Where" dilemma**:
- **Semantic Abstraction ("What")**: To recognize an object class reliably (regardless of scale, deformation, or background clutter), the network needs a large receptive field achieved by progressive spatial downsampling (e.g. max pooling, strided convolutions).
- **Spatial Localization ("Where")**: To delineate precise object boundaries, crisp corners, and fine structural edges, the network requires high-frequency, uncompressed spatial coordinate information.

This experiment provides students with a clean, intuitive, and mathematically rigorous comparison between:
1. **Plain Fully Convolutional Network (Encoder-Decoder without skip connections)**: Reconstructs high-resolution masks solely from the compressed bottleneck representations, resulting in blurry contours and rounded corners.
2. **U-Net Architecture (with lateral skip connections)**: Direct concatenation of high-resolution encoder feature maps to matching decoder stages, enabling razor-sharp boundary delineation.

---

## 2. Mathematical Formulation

### 2.1 The Semantic Segmentation Objective
Let $X \in \mathbb{R}^{H \times W \times 3}$ be an input RGB image and $Y \in \{0, 1, \dots, C-1\}^{H \times W}$ be the ground-truth categorical mask for $C$ semantic classes.

The network computes dense class posteriors through a $1 \times 1$ convolution followed by spatial softmax:
$$P(Y_{i, j} = c \mid X) = \frac{\exp\left(z_{i, j, c}\right)}{\sum_{k=0}^{C-1} \exp\left(z_{i, j, k}\right)}$$

The objective is optimized via pixel-wise Categorical Cross-Entropy:
$$\mathcal{L}_{\text{CE}}(\theta) = - \frac{1}{H W} \sum_{i=1}^H \sum_{j=1}^W \sum_{c=0}^{C-1} \mathbb{I}(Y_{i, j} = c) \log P(Y_{i, j} = c \mid X)$$

---

### 2.2 The Information Bottleneck in Plain Encoder-Decoders
In a plain encoder-decoder network, spatial resolution is progressively reduced across $D$ downsampling stages:
$$z_{\text{bottleneck}} = \mathcal{E}(X) \in \mathbb{R}^{\frac{H}{2^D} \times \frac{W}{2^D} \times C_{\text{deep}}}$$

While $z_{\text{bottleneck}}$ captures rich global semantics, the spatial downsampling acts as a spatial low-pass filter: high-frequency spatial phase details (exact edge coordinates, sharp corners, thin lines) are permanently discarded.

The plain decoder must invert this downsampling purely through learned upsampling:
$$\hat{Y} = \mathcal{D}_{\text{plain}}(z_{\text{bottleneck}})$$

Because the precise high-frequency spatial coordinates are absent in $z_{\text{bottleneck}}$, the decoder is forced to interpolate spatial boundaries, leading to **spatially blurred, rounded contours**.

---

### 2.3 U-Net: Lateral Skip Connections
U-Net (Ronneberger et al., 2015) bridges the contracting (encoder) and expanding (decoder) paths with lateral **skip connections** at each spatial scale $l \in \{1, \dots, D\}$:
$$z_l^{\text{dec}} = \text{ConvBlock}\left( \left[ \text{UpSample}(z_{l+1}^{\text{dec}}), \, z_l^{\text{enc}} \right] \right)$$

where $[\cdot, \cdot]$ denotes concatenation along the channel dimension.

- **$\text{UpSample}(z_{l+1}^{\text{dec}})$** provides top-down abstract semantic context ("what").
- **$z_l^{\text{enc}}$** injects bottom-up, uncompressed high-resolution spatial coordinates ("where").

The decoder effectively fuses abstract semantic reasoning with sub-pixel spatial precision, producing **razor-sharp, pixel-accurate segmentation boundaries**.

---

## 3. Experiment Structure

```
experiments/10_fcn_vs_unet_segmentation/
├── dataset.py                                    # Procedural geometric shape dataset generator
├── models.py                                     # Plain FCN and U-Net Keras implementations
├── train.py                                      # Training pipeline with automatic checkpoint saving/loading
├── evaluate.py                                   # Quantitative metrics (mIoU, Boundary IoU, Per-Class IoU)
├── visualize.py                                  # Publication-grade plotting and diagnostic suite
├── run_all.py                                    # End-to-end master execution runner
├── checkpoints/                                  # Saved weights and training history logs
│   ├── plain_fcn_best.weights.h5
│   ├── unet_best.weights.h5
│   ├── plain_fcn_history.json
│   └── unet_history.json
├── fcn_vs_unet_architecture_comparison.png      # Architectural schematic diagram
├── fcn_vs_unet_qualitative_comparison.png         # Multi-sample predictions and error maps
├── fcn_vs_unet_boundary_and_corner_sharpness.png  # Quantitative metrics and edge profile diagnostics
├── fcn_vs_unet_training_dynamics.png             # Loss and accuracy curves over epochs
├── experiment_10_fcn_vs_unet.py                  # Jupytext py:percent interactive notebook script
├── experiment_10_fcn_vs_unet.ipynb               # Synchronized Jupyter notebook
└── README.md                                     # Pedagogical documentation and lesson guide
```

---

## 4. How to Run the Experiment

All code is managed with `uv`.

### Run the Complete Pipeline
```bash
uv run python experiments/10_fcn_vs_unet_segmentation/run_all.py
```

### Sync Interactive Jupyter Notebook
```bash
uv run jupytext --set-formats py:percent,ipynb experiments/10_fcn_vs_unet_segmentation/experiment_10_fcn_vs_unet.py
uv run jupytext --sync experiments/10_fcn_vs_unet_segmentation/experiment_10_fcn_vs_unet.py
```

---

## 5. Visualizations & Key Results

### 1. Architectural Difference
- **Plain FCN**: All information must squeeze through the bottleneck ($16 \times 16$).
- **U-Net**: Skip connections bypass the bottleneck, feeding high-resolution feature maps directly into the decoder.

### 2. Qualitative Visual Comparison & Error Maps
- **Plain FCN (No Skips)**: Outputs blurry edges and rounded vertices. Its misclassification error maps reveal a thick red border (halo) around every shape boundary.
- **U-Net (With Skips)**: Accurately traces straight edges, sharp right-angle corners (rectangles), and acute vertices (triangles) with virtually clean error maps.

### 3. Quantitative Evaluation (Boundary IoU)
- Standard **Mean IoU (mIoU)** measures overall area overlap.
- **Boundary IoU** isolates a narrow 2-pixel band around shape contours, rigorously quantifying the sharpness of boundaries. U-Net shows a substantial improvement in Boundary IoU compared to Plain FCN.

---

## 6. Key Takeaways for Students

1. **Downsampling is necessary for semantics**: We cannot avoid pooling/striding because large receptive fields are required to recognize complex visual patterns.
2. **Bottlenecks discard spatial localization**: Reconstructing high-resolution details from a low-resolution bottleneck is an ill-posed inverse problem.
3. **Skip connections provide the best of both worlds**: By transmitting early encoder feature maps across the network, U-Net retains semantic understanding while achieving pixel-perfect spatial precision.
