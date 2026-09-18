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
# # Deep Computer Vision Model Architectures: ResNet, Plain CNN, Inception, EfficientNet & ConvNeXt
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jorgvt/CVMIAX/blob/main/experiments/06_resnet_architecture/experiment_06_resnet.ipynb)
#
# ## 1. Overview & Pedagogical Objectives
# In this experiment series, we implement, train, and benchmark foundational deep learning computer vision architectures in **Keras**:
#
# 1. **Plain Sequential CNN** ([`plain_cnn.py`](plain_cnn.py)): Standard sequential feedforward convolutional network (VGG-style) with multiplicative gradient flow.
# 2. **Deep Residual Network (ResNet)** ([`resnet.py`](resnet.py)): Introduces additive identity shortcut connections ($y = \mathcal{F}(x) + x$) creating an uninterrupted gradient highway.
# 3. **Inception (GoogLeNet)** ([`inception.py`](inception.py)): Multi-branch parallel receptive fields ($1\times1, 3\times3, 5\times5$) with $1\times1$ dimensionality reduction bottlenecks.
# 4. **EfficientNet** ([`efficientnet.py`](efficientnet.py)): Mobile Inverted Bottleneck Convolutions (MBConv) with Depthwise Separable Convolutions and Squeeze-and-Excitation (SE) channel attention.
# 5. **ConvNeXt** ([`convnext.py`](convnext.py)): Modernized pure convolutional architecture incorporating Vision Transformer design principles ($7\times7$ depthwise conv, inverted bottleneck ratio, LayerNorm, GELU).
#
# Crucially:
# - Each architecture is implemented in its **own isolated module**.
# - Each model has a dedicated training script that trains the network on CIFAR-10 and **persists its learned weights (`.weights.h5`)** for downstream transfer learning and analysis.

# %%
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import layers, optimizers, losses, metrics

# Add local directory to module path for clean imports
module_path = os.path.abspath(".")
if module_path not in sys.path:
    sys.path.insert(0, module_path)

# Independent imports of each model architecture
from plain_cnn import build_plain20, save_plain_weights, load_plain_weights
from resnet import build_resnet20, save_resnet_weights, load_resnet_weights
from inception import build_inception_cifar, save_inception_weights, load_inception_weights
from efficientnet import build_efficientnet_cifar, save_efficientnet_weights, load_efficientnet_weights
from convnext import build_convnext_cifar, save_convnext_weights, load_convnext_weights

from dataset import load_cifar10_data, create_tf_dataset
from visualize import (
    plot_architecture_comparison,
    plot_all_architectures_poster,
    plot_training_dynamics,
    plot_weight_distributions,
)

print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version:      {keras.__version__}")

# %% [markdown]
# ---
# ## 2. Visualizing the Architectural Paradigms
#
# Below is a side-by-side comparison of the core building blocks defining each of the 5 architectures:

# %%
# Visualizing all 5 architectural paradigms
fig_poster = plot_all_architectures_poster()
plt.show()

# Detailed ResNet vs Plain CNN Contrast
fig_arch = plot_architecture_comparison()
plt.show()

# %% [markdown]
# ---
# ## 3. Instantiating the Independent Models & Inspecting Parameter Counts
#
# Let's instantiate all 5 models and inspect their parameter footprint on $32\times32\times3$ images:

# %%
models = {
    "Plain-20": build_plain20(input_shape=(32, 32, 3), num_classes=10),
    "ResNet-20": build_resnet20(input_shape=(32, 32, 3), num_classes=10),
    "Inception-CIFAR": build_inception_cifar(input_shape=(32, 32, 3), num_classes=10),
    "EfficientNet-CIFAR": build_efficientnet_cifar(input_shape=(32, 32, 3), num_classes=10),
    "ConvNeXt-CIFAR": build_convnext_cifar(input_shape=(32, 32, 3), num_classes=10),
}

print("=" * 60)
print(f"{'Architecture':<22} | {'Parameters':<15}")
print("-" * 60)
for name, model in models.items():
    print(f"{name:<22} | {model.count_params():<15,}")
print("=" * 60)

# %% [markdown]
# ---
# ## 4. Loading & Preprocessing the CIFAR-10 Dataset

# %%
(x_train, y_train), (x_val, y_val), (x_test, y_test) = load_cifar10_data(
    val_split=0.1, normalize=True, standardize=True
)

batch_size = 128
train_ds = create_tf_dataset(x_train, y_train, batch_size=batch_size, augment=True, shuffle=True)
val_ds = create_tf_dataset(x_val, y_val, batch_size=batch_size, augment=False, shuffle=False)
test_ds = create_tf_dataset(x_test, y_test, batch_size=batch_size, augment=False, shuffle=False)

print(f"Training dataset:   {x_train.shape[0]} samples")
print(f"Validation dataset: {x_val.shape[0]} samples")
print(f"Test dataset:       {x_test.shape[0]} samples")

# %% [markdown]
# ---
# ## 5. Training Models & Persisting Weights
#
# We train models with standard optimization and store their weights into the `weights/` directory for later reuse.

# %%
weights_dir = "weights"
os.makedirs(weights_dir, exist_ok=True)

# Example: Training ResNet-20
resnet = models["ResNet-20"]
resnet.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss=losses.SparseCategoricalCrossentropy(),
    metrics=[metrics.SparseCategoricalAccuracy(name="accuracy")],
)

epochs = 10
print(f"--- Training ResNet-20 ({epochs} epochs) ---")
res_history = resnet.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=1)

# Save ResNet weights
resnet_weights_path = os.path.join(weights_dir, "resnet20_cifar10.weights.h5")
save_resnet_weights(resnet, resnet_weights_path)
print(f"✓ Saved ResNet-20 weights to: {resnet_weights_path}")

# %% [markdown]
# ---
# ## 6. Verifying Weight Reloading & Model Reusability
#
# We verify that saved weights can be loaded into fresh model instances for downstream inference or fine-tuning:

# %%
fresh_resnet = build_resnet20(input_shape=(32, 32, 3), num_classes=10)
load_resnet_weights(fresh_resnet, resnet_weights_path)
fresh_resnet.compile(
    loss=losses.SparseCategoricalCrossentropy(),
    metrics=[metrics.SparseCategoricalAccuracy(name="accuracy")],
)

test_loss, test_acc = fresh_resnet.evaluate(test_ds, verbose=0)
print(f"✓ Reloaded ResNet-20 Test Accuracy: {test_acc * 100:.2f}% | Test Loss: {test_loss:.4f}")

# %% [markdown]
# ---
# ## 7. Comparative Visualizations & Weight Inspection

# %%
histories = {
    "ResNet-20": res_history.history,
}
fig_dyn = plot_training_dynamics(histories)
plt.show()

fig_weights = plot_weight_distributions(models["ResNet-20"], models["Plain-20"])
plt.show()
