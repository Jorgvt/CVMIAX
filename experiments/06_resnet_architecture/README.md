# Experiment 06: Deep Computer Vision Architectures (ResNet, Plain CNN, EfficientNet, ConvNeXt, Inception)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jorgvt/CVMIAX/blob/main/experiments/06_resnet_architecture/experiment_06_resnet.ipynb)

## 1. Overview & Pedagogical Objectives
In this experiment series, we implement, train, and benchmark five foundational deep learning computer vision architectures in **Keras**, isolating the unique design principles and mathematical formulation of each family:

1. **Plain Sequential CNN** ([`plain_cnn.py`](plain_cnn.py)): Standard sequential feedforward convolutional network (VGG-style) where gradients propagate through multiplicative non-linear compositions.
2. **Deep Residual Network (ResNet)** ([`resnet.py`](resnet.py)): Introduces additive identity shortcut connections ($y = \mathcal{F}(x) + x$) that construct an uninterrupted gradient highway resolving the degradation problem.
3. **Inception (GoogLeNet)** ([`inception.py`](inception.py)): Multi-scale parallel receptive fields ($1\times1, 3\times3, 5\times5$) with $1\times1$ bottleneck dimensionality reductions concatenated across channels.
4. **EfficientNet** ([`efficientnet.py`](efficientnet.py)): Mobile Inverted Bottleneck Convolutions (MBConv) leveraging depthwise separable convolutions, inverted channel expansions, and Squeeze-and-Excitation (SE) channel-wise attention.
5. **ConvNeXt** ([`convnext.py`](convnext.py)): Modernized pure convolutional architecture adopting Vision Transformer design principles ($7\times7$ depthwise convolutions, inverted bottleneck ratio, Layer Normalization, and GELU activations).

---

## 2. Mathematical Formulations & Architecture Breakdown

### 2.1 ResNet: Additive Residual Learning & Gradient Highway
- **Residual Formulation**: $\mathcal{H}(x) = \mathcal{F}(x) + x \implies \mathcal{F}(x) = \mathcal{H}(x) - x$.
- **Gradient Highway**:
  $$\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L} \left( \mathbf{I} + \frac{\partial}{\partial x_l} \sum_{i=l}^{L-1} \mathcal{F}(x_i, W_i) \right)$$
  The identity matrix term $\frac{\partial \mathcal{L}}{\partial x_L} \mathbf{I}$ allows lossless backpropagation directly to initial layers regardless of depth.

---

### 2.2 Inception: Multi-Scale Feature Extraction with Bottlenecks
- Concatenates 4 parallel branches:
  $$\text{Inception}(x) = \text{Concat}\Big[ \text{Conv}_{1\times1}(x),\; \text{Conv}_{3\times3}(\text{Red}_{1\times1}(x)),\; \text{Conv}_{5\times5}(\text{Red}_{1\times1}(x)),\; \text{Proj}_{1\times1}(\text{MaxPool}_{3\times3}(x)) \Big]$$
- Factorizes expensive $5\times5$ filters into two stacked $3\times3$ convolutions ($2 \times 3^2 = 18$ parameters vs $5^2 = 25$ parameters, $28\%$ parameter reduction with an additional non-linearity).

---

### 2.3 EfficientNet: MBConv & Squeeze-and-Excitation (SE) Attention
- **Inverted Bottleneck**: Low-channel input is expanded by factor $e \in [4, 6]$ via $1\times1$ conv, transformed by $3\times3$ Depthwise Conv, and projected linearly back to output dimension.
- **Squeeze-and-Excitation Attention**:
  $$s = \sigma\Big( W_2 \cdot \text{Swish}(W_1 \cdot \text{GAP}(x)) \Big), \quad \tilde{x} = x \odot s$$
  Computes global channel-wise recalibration weights with minimal parameter overhead.

---

### 2.4 ConvNeXt: Modernizing Pure Convolutions
- **Large Kernel Receptive Field**: $7\times7$ depthwise convolution at the beginning of each block.
- **Modern Normalization & Activation**: Replaces BatchNorm with LayerNorm (channels-last) and ReLU with GELU.
- **Inverted Bottleneck**: Pointwise $1\times1$ convolution expands channels $4\times$, followed by a single GELU activation and linear projection back to base dimension.

---

## 3. Architecture Comparison Matrix

| Model | Module | Key Building Block | Receptive Field | Normalization | Activation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Plain CNN** | [`plain_cnn.py`](plain_cnn.py) | Sequential Conv-BN-ReLU | $3\times3$ sequential | BatchNorm | ReLU |
| **ResNet-20** | [`resnet.py`](resnet.py) | BasicBlock ($F(x) + x$) | $3\times3$ + Identity | BatchNorm | ReLU |
| **Inception-CIFAR** | [`inception.py`](inception.py) | 4-branch multi-scale | $1\times1, 3\times3, 5\times5$ | BatchNorm | ReLU |
| **EfficientNet-CIFAR** | [`efficientnet.py`](efficientnet.py) | MBConv + SE Attention | $3\times3$ Depthwise | BatchNorm | Swish / SiLU |
| **ConvNeXt-CIFAR** | [`convnext.py`](convnext.py) | 7x7 DW + Inverted Bottleneck | $7\times7$ Depthwise | LayerNorm | GELU |

---

## 4. Experiment Structure

```
experiments/06_resnet_architecture/
├── resnet.py                          # Standalone ResNet architecture & weight save/load
├── plain_cnn.py                       # Standalone Plain CNN architecture & weight save/load
├── efficientnet.py                    # Standalone EfficientNet architecture & weight save/load
├── convnext.py                        # Standalone ConvNeXt architecture & weight save/load
├── inception.py                       # Standalone Inception architecture & weight save/load
├── train_resnet.py                    # Dedicated training script for ResNet-20
├── train_plain_cnn.py                 # Dedicated training script for Plain-20 CNN
├── train_efficientnet.py              # Dedicated training script for EfficientNet
├── train_convnext.py                  # Dedicated training script for ConvNeXt
├── train_inception.py                 # Dedicated training script for Inception
├── compare.py                         # Unified comparative evaluation across all models
├── dataset.py                         # CIFAR-10 data loading, standardization, & augmentation
├── visualize.py                       # High-resolution architectural posters & training curves
├── run_all.py                         # Master CLI orchestrator
├── experiment_06_resnet.py            # Master interactive lesson script (Jupytext py:percent)
├── experiment_06_resnet.ipynb         # Interactive Jupyter Notebook (Open in Colab)
├── weights/                           # Stored pre-trained model weights (.weights.h5)
│   ├── resnet20_cifar10.weights.h5
│   ├── plain20_cifar10.weights.h5
│   ├── efficientnet_cifar10.weights.h5
│   ├── convnext_cifar10.weights.h5
│   ├── inception_cifar10.weights.h5
│   └── *_history.json
├── figures/                           # Publication-ready visualizations
│   ├── 00_all_architectures_overview.png
│   ├── 01_architecture_plain_vs_resnet.png
│   ├── 02_training_dynamics_comparison.png
│   └── 03_weight_distributions.png
└── README.md                          # Theory and experiment guide
```

---

## 5. How to Run the Experiments

Ensure dependencies are installed and execute via `uv`:

```bash
# 1. Run the entire multi-model benchmark suite (all models)
uv run python experiments/06_resnet_architecture/run_all.py --quick-demo

# 2. Or train individual architectures independently:
uv run python experiments/06_resnet_architecture/train_resnet.py --epochs 10 --batch-size 128
uv run python experiments/06_resnet_architecture/train_plain_cnn.py --epochs 10 --batch-size 128
uv run python experiments/06_resnet_architecture/train_efficientnet.py --epochs 10 --batch-size 128
uv run python experiments/06_resnet_architecture/train_convnext.py --epochs 10 --batch-size 128
uv run python experiments/06_resnet_architecture/train_inception.py --epochs 10 --batch-size 128

# 3. Compare all stored models and generate visual figures:
uv run python experiments/06_resnet_architecture/compare.py
```

---

## 6. Storing & Reusing Model Weights

Each architecture module provides standardized `save_<model>_weights` and `load_<model>_weights` utilities:

```python
# Reusing pre-trained ResNet weights
from experiments.06_resnet_architecture.resnet import build_resnet20, load_resnet_weights

model = build_resnet20(input_shape=(32, 32, 3), num_classes=10)
load_resnet_weights(model, "experiments/06_resnet_architecture/weights/resnet20_cifar10.weights.h5")

# Reusing pre-trained EfficientNet weights
from experiments.06_resnet_architecture.efficientnet import build_efficientnet_cifar, load_efficientnet_weights

eff_model = build_efficientnet_cifar(input_shape=(32, 32, 3), num_classes=10)
load_efficientnet_weights(eff_model, "experiments/06_resnet_architecture/weights/efficientnet_cifar10.weights.h5")

# Ready for fine-tuning, linear probing, or Grad-CAM feature analysis!
```
