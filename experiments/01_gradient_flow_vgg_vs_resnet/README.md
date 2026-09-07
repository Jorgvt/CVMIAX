# Experiment 01: Gradient Propagation & Vanishing Gradients (Plain VGG vs ResNet)

## 1. Overview & Pedagogical Objective
In deep learning computer vision architectures, increasing depth theoretically allows models to learn representations at multiple levels of abstraction. However, historically, simply stacking more convolutional layers caused **the degradation problem**: as network depth increased, accuracy saturated and then rapidly degraded, caused by severe optimization failure rather than overfitting.

This experiment empirically proves and visualizes the mathematical mechanics behind this phenomenon:
1. **Vanishing Gradient in Plain Networks**: In sequential architectures (e.g., VGG-style feedforward stacks), backpropagated gradients decay exponentially as they travel from the output layer to the input layers.
2. **Gradient Highways in Residual Networks (ResNet)**: The introduction of additive identity shortcuts ($x_{l+1} = \mathcal{F}(x_l) + x_l$) creates an uninterrupted gradient highway, allowing gradients to propagate directly to early layers regardless of depth.

---

## 2. Mathematical Formulation

### 2.1 Plain Network Gradient Propagation (Chain Rule Decay)
In a plain network, the output of layer $l$ is a composition of non-linear transformations:
$$x_{l} = \sigma(W_l x_{l-1} + b_l)$$

By the chain rule, the gradient of the loss $\mathcal{L}$ with respect to the activations of an earlier layer $x_l$ given downstream layer $x_L$ ($L > l$) is:
$$\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L} \prod_{k=l}^{L-1} \frac{\partial x_{k+1}}{\partial x_k} = \frac{\partial \mathcal{L}}{\partial x_L} \prod_{k=l}^{L-1} \left( W_{k+1}^T \cdot \text{diag}(\sigma'(z_k)) \right)$$

As $L - l$ (the depth distance) increases:
- If the spectral norm of the Jacobian $\left\| \frac{\partial x_{k+1}}{\partial x_k} \right\| < 1$, the gradient magnitude decays exponentially:
$$\lim_{L - l \to \infty} \left\| \frac{\partial \mathcal{L}}{\partial x_l} \right\| \to 0$$
Consequently, the earliest layers receive almost zero gradient update ($\nabla_{W_1} \mathcal{L} \approx 0$), leaving their initial feature extractors effectively untrained.

---

### 2.2 Residual Network Gradient Propagation (Additive Gradient Highway)
In a ResNet building block with identity mapping:
$$x_{l+1} = x_l + \mathcal{F}(x_l, W_l)$$

Recursively applying this relation from layer $l$ to a deeper layer $L$:
$$x_L = x_l + \sum_{i=l}^{L-1} \mathcal{F}(x_i, W_i)$$

Taking the derivative of the loss $\mathcal{L}$ with respect to $x_l$:
$$\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L} \frac{\partial x_L}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L} \left( \mathbf{I} + \frac{\partial}{\partial x_l} \sum_{i=l}^{L-1} \mathcal{F}(x_i, W_i) \right)$$

Notice the crucial term:
$$\frac{\partial \mathcal{L}}{\partial x_l} = \underbrace{\frac{\partial \mathcal{L}}{\partial x_L}}_{\text{Direct gradient flow}} + \underbrace{\frac{\partial \mathcal{L}}{\partial x_L} \left( \sum_{i=l}^{L-1} \frac{\partial \mathcal{F}(x_i, W_i)}{\partial x_l} \right)}_{\text{Residual branch gradient}}$$

Even if the residual term $\sum_{i=l}^{L-1} \frac{\partial \mathcal{F}}{\partial x_l}$ vanishes or decays, the identity matrix $\mathbf{I}$ ensures that **the gradient $\frac{\partial \mathcal{L}}{\partial x_L}$ is transmitted directly to layer $l$ without multiplicative decay**.

---

## 3. Experiment Structure

```
experiments/01_gradient_flow_vgg_vs_resnet/
├── models.py                          # Plain VGG and ResNet architectures in Keras
├── gradient_analysis.py               # Layerwise gradient norms & depth scaling
├── train_comparison.py                # Training convergence & degradation test
├── run_all.py                         # Master execution script
├── gradient_norm_by_layer.png         # Layerwise gradient comparison plot
├── gradient_at_first_layer_vs_depth.png # Scaling depth vs input layer gradient
├── training_curves_comparison.png     # Training loss and validation accuracy curves
└── README.md                          # Theory and experiment guide
```

---

## 4. How to Run the Experiment

Ensure dependencies are installed and run via `uv`:

```bash
# Run the complete experiment suite
uv run python experiments/01_gradient_flow_vgg_vs_resnet/run_all.py
```

Or run individual components:

```bash
# 1. Compute and plot layerwise gradient propagation
uv run python experiments/01_gradient_flow_vgg_vs_resnet/gradient_analysis.py

# 2. Train Plain vs ResNet on CIFAR-10 across depths
uv run python experiments/01_gradient_flow_vgg_vs_resnet/train_comparison.py
```

---

## 5. Visualizations & Empirical Results

### 1. Layer-wise Gradient Magnitude
- **Plain VGG (36 layers)**: Gradient norms drop exponentially from $\approx 10^{-2}$ at output layers down to $< 10^{-8}$ at the earliest convolutional layers.
- **ResNet (36 layers)**: Gradient norms remain bounded and well-behaved across all 36 layers ($10^{-2}$ to $10^{-3}$ throughout).

### 2. Gradient Magnitude at Input Layer vs Network Depth
- As total depth increases from 6 to 48 layers:
  - In Plain VGG, the input layer gradient collapses towards zero.
  - In ResNet, the input layer gradient remains stable, allowing deep architectures to train effectively.

### 3. Training Dynamics & The Degradation Problem
- **Plain-12 vs Plain-30**: The 30-layer Plain network achieves **higher** training loss than the 12-layer model due to gradient starvation in the early representation layers.
- **ResNet-12 vs ResNet-30**: ResNet-30 trains effortlessly, converges faster, and achieves significantly higher validation accuracy.
