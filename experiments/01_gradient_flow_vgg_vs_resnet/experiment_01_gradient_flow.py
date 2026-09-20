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
# # Experiment 01: Gradient Propagation & Vanishing Gradients (Plain VGG vs ResNet)
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jorgvt/CVMIAX/blob/main/experiments/01_gradient_flow_vgg_vs_resnet/experiment_01_gradient_flow.ipynb)
#
# **Pedagogical Objective:**
# In deep neural networks for computer vision, stacking more convolutional layers theoretically increases representational capacity. However, sequentially stacked feedforward networks historically suffered from **the degradation problem**: as depth grew, training accuracy dropped severely due to optimization failure.
#
# In this interactive notebook, we empirically demonstrate and visualize:
# 1. **Vanishing Gradients in Plain Networks**: Gradients decay exponentially through sequential layers via the chain rule.
# 2. **Gradient Highways in Residual Networks (ResNet)**: Additive identity shortcuts ($x_{l+1} = \mathcal{F}(x_l) + x_l$) preserve gradient flow directly back to early layers.
# 3. **The Degradation Problem in Training**: Comparing optimization of shallow vs. deep models on CIFAR-10.

# %% [markdown]
# ## 1. Mathematical Formulation
#
# ### 1.1 Plain Network Gradient Propagation (Multiplicative Decay)
# In a standard feedforward convolutional network without skip connections:
# $$x_{l} = \sigma(W_l x_{l-1} + b_l)$$
#
# Applying the chain rule to compute the gradient of loss $\mathcal{L}$ with respect to early activation $x_l$ from a deep layer $x_L$:
# $$\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L} \prod_{k=l}^{L-1} \frac{\partial x_{k+1}}{\partial x_k} = \frac{\partial \mathcal{L}}{\partial x_L} \prod_{k=l}^{L-1} \left( W_{k+1}^T \cdot \text{diag}(\sigma'(z_k)) \right)$$
#
# As the depth $(L - l)$ increases, if the singular values of the transition matrices are below 1, the gradient magnitude decays exponentially:
# $$\lim_{L - l \to \infty} \left\| \frac{\partial \mathcal{L}}{\partial x_l} \right\| \to 0$$
#
# ---
#
# ### 1.2 Residual Networks (Additive Identity Highway)
# In a ResNet building block with an identity shortcut:
# $$x_{l+1} = x_l + \mathcal{F}(x_l, W_l)$$
#
# Unrolling recursively up to layer $L$:
# $$x_L = x_l + \sum_{i=l}^{L-1} \mathcal{F}(x_i, W_i)$$
#
# Taking the partial derivative with respect to $x_l$:
# $$\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L} \frac{\partial x_L}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L} \left( \mathbf{I} + \frac{\partial}{\partial x_l} \sum_{i=l}^{L-1} \mathcal{F}(x_i, W_i) \right)$$
#
# Even if the residual gradient term vanishes, the identity matrix $\mathbf{I}$ ensures that **the gradient $\frac{\partial \mathcal{L}}{\partial x_L}$ is transmitted directly to layer $l$ without decay**.

# %% [markdown]
# ## 2. Environment Setup & Dependencies

# %%
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
from keras import layers

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")
print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")

# %% [markdown]
# ## 3. Architecture Definitions: Plain VGG vs ResNet
#
# We define parameterized models where layer depths and channel widths are strictly identical, isolating the identity skip connections as the single experimental variable.

# %%
def build_plain_vgg(
    input_shape=(32, 32, 3),
    num_classes=10,
    num_blocks_per_stage=[4, 4, 4],
    filters_per_stage=[32, 64, 128],
    use_batchnorm=False,
    name="plain_vgg",
):
    inputs = keras.Input(shape=input_shape, name="input_image")
    x = inputs

    layer_idx = 0
    for stage_idx, (num_blocks, num_filters) in enumerate(zip(num_blocks_per_stage, filters_per_stage)):
        for block_idx in range(num_blocks):
            layer_idx += 1
            x = layers.Conv2D(
                filters=num_filters,
                kernel_size=3,
                padding="same",
                kernel_initializer="he_normal",
                name=f"stage{stage_idx}_conv{block_idx}_layer{layer_idx}",
            )(x)
            if use_batchnorm:
                x = layers.BatchNormalization(name=f"stage{stage_idx}_bn{block_idx}_layer{layer_idx}")(x)
            x = layers.ReLU(name=f"stage{stage_idx}_relu{block_idx}_layer{layer_idx}")(x)

        if stage_idx < len(num_blocks_per_stage) - 1:
            x = layers.MaxPooling2D(pool_size=2, strides=2, name=f"stage{stage_idx}_pool")(x)

    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classifier_dense")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name=name)


def build_resnet(
    input_shape=(32, 32, 3),
    num_classes=10,
    num_blocks_per_stage=[4, 4, 4],
    filters_per_stage=[32, 64, 128],
    use_batchnorm=False,
    name="resnet",
):
    inputs = keras.Input(shape=input_shape, name="input_image")
    x = inputs

    layer_idx = 0
    for stage_idx, (num_blocks, num_filters) in enumerate(zip(num_blocks_per_stage, filters_per_stage)):
        for block_idx in range(num_blocks):
            layer_idx += 1
            shortcut = x

            residual = layers.Conv2D(
                filters=num_filters,
                kernel_size=3,
                padding="same",
                kernel_initializer="he_normal",
                name=f"stage{stage_idx}_conv{block_idx}_layer{layer_idx}",
            )(x)
            if use_batchnorm:
                residual = layers.BatchNormalization(name=f"stage{stage_idx}_bn{block_idx}_layer{layer_idx}")(residual)

            if shortcut.shape[-1] != num_filters:
                shortcut = layers.Conv2D(
                    filters=num_filters,
                    kernel_size=1,
                    padding="same",
                    kernel_initializer="he_normal",
                    name=f"stage{stage_idx}_shortcut_conv{block_idx}_layer{layer_idx}",
                )(shortcut)

            x = layers.Add(name=f"stage{stage_idx}_add{block_idx}_layer{layer_idx}")([residual, shortcut])
            x = layers.ReLU(name=f"stage{stage_idx}_relu{block_idx}_layer{layer_idx}")(x)

        if stage_idx < len(num_blocks_per_stage) - 1:
            x = layers.MaxPooling2D(pool_size=2, strides=2, name=f"stage{stage_idx}_pool")(x)

    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classifier_dense")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name=name)


def create_model_pair(total_depth=24, input_shape=(32, 32, 3), num_classes=10, use_batchnorm=False):
    blocks_per_stage = total_depth // 3
    num_blocks = [blocks_per_stage, blocks_per_stage, total_depth - 2 * blocks_per_stage]
    filters = [32, 64, 128]

    plain = build_plain_vgg(input_shape, num_classes, num_blocks, filters, use_batchnorm, f"Plain_VGG_{total_depth}")
    res = build_resnet(input_shape, num_classes, num_blocks, filters, use_batchnorm, f"ResNet_{total_depth}")
    return plain, res

# %% [markdown]
# ## 4. Gradient Flow Measurement & Visualization
#
# We now compute the gradient norm $\|\nabla_{W} \mathcal{L}\|_2$ for each convolutional layer's weights using `tf.GradientTape`.

# %%
def compute_layerwise_gradients(model, x_batch, y_batch, loss_fn=keras.losses.CategoricalCrossentropy()):
    x_tensor = tf.convert_to_tensor(x_batch, dtype=tf.float32)
    y_tensor = tf.convert_to_tensor(y_batch, dtype=tf.float32)

    with tf.GradientTape() as tape:
        preds = model(x_tensor, training=False)
        loss = loss_fn(y_tensor, preds)

    conv_layers = [l for l in model.layers if isinstance(l, layers.Conv2D) and "shortcut" not in l.name]
    conv_weights = [l.kernel for l in conv_layers]
    grads = tape.gradient(loss, conv_weights)

    layer_names = [l.name for l in conv_layers]
    grad_norms = [float(tf.norm(g).numpy()) if g is not None else 0.0 for g in grads]
    return layer_names, grad_norms

# %% [markdown]
# ### 4.1 Layer-by-Layer Gradient Norm (Depth = 36)
#
# Let's inspect how the gradient decays as it travels backward from the final layer (right) to the first input layer (left).

# %%
depth = 36
batch_size = 64
x_sample = np.random.randn(batch_size, 32, 32, 3).astype(np.float32)
y_sample = keras.utils.to_categorical(np.random.randint(0, 10, size=(batch_size,)), num_classes=10)

plain_model, resnet_model = create_model_pair(total_depth=depth, use_batchnorm=False)
_, plain_grads = compute_layerwise_gradients(plain_model, x_sample, y_sample)
_, resnet_grads = compute_layerwise_gradients(resnet_model, x_sample, y_sample)

plt.figure(figsize=(14, 5))
layer_indices = list(range(1, depth + 1))
plt.plot(layer_indices, plain_grads, "o--", color="#d9534f", linewidth=2, label=f"Plain VGG-{depth}")
plt.plot(layer_indices, resnet_grads, "s-", color="#0275d8", linewidth=2, label=f"ResNet-{depth}")
plt.yscale("log")
plt.xlabel("Layer Index (1 = First Conv Layer near Input, N = Deepest Layer)", fontsize=11)
plt.ylabel(r"Gradient L2 Norm $\|\nabla_W \mathcal{L}\|_2$ (Log Scale)", fontsize=11)
plt.title(f"Layer-by-Layer Gradient Magnitude Comparison (Depth = {depth})", fontsize=13, fontweight="bold")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend(fontsize=11)
plt.show()

# %% [markdown]
# ### 4.2 Vanishing Gradient vs Total Depth Scaling
#
# Next, we measure the gradient received by the **very first layer** ($W_1$) as total network depth grows from 6 to 48 layers.

# %%
depths = [6, 12, 18, 24, 30, 36, 48]
plain_first_layer_grads = []
resnet_first_layer_grads = []

for d in depths:
    p_mod, r_mod = create_model_pair(total_depth=d, use_batchnorm=False)
    _, p_g = compute_layerwise_gradients(p_mod, x_sample, y_sample)
    _, r_g = compute_layerwise_gradients(r_mod, x_sample, y_sample)
    plain_first_layer_grads.append(p_g[0])
    resnet_first_layer_grads.append(r_g[0])

plt.figure(figsize=(10, 5))
plt.plot(depths, plain_first_layer_grads, "o--", color="#d9534f", linewidth=2.5, markersize=8, label="Plain VGG (Input Layer Gradient)")
plt.plot(depths, resnet_first_layer_grads, "s-", color="#0275d8", linewidth=2.5, markersize=8, label="ResNet (Input Layer Gradient)")
plt.yscale("log")
plt.xlabel("Total Network Depth (Number of Conv Layers)", fontsize=11)
plt.ylabel(r"First Layer Gradient Norm $\|\nabla_{W_1} \mathcal{L}\|_2$ (Log Scale)", fontsize=11)
plt.title("Vanishing Gradient Effect vs Network Depth", fontsize=13, fontweight="bold")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend(fontsize=11)
plt.show()

# %% [markdown]
# ## 5. The Degradation Problem: Training on CIFAR-10
#
# Finally, let's train a shallow (12 layers) and deep (30 layers) model for Plain VGG and ResNet on CIFAR-10 to observe how the gradient vanishing translates into an optimization failure.

# %%
from dataset import get_cifar10_subset

(x_tr, y_tr), (x_va, y_va) = get_cifar10_subset(num_train=8000, num_val=1500)

def train_model(model, epochs=8, lr=1e-3):
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(x_tr, y_tr, validation_data=(x_va, y_va), epochs=epochs, batch_size=128, verbose=0)
    return history.history

print("Training Plain VGG-12...")
h_p12 = train_model(build_plain_vgg(num_blocks_per_stage=[4, 4, 4]))
print("Training Plain VGG-30...")
h_p30 = train_model(build_plain_vgg(num_blocks_per_stage=[10, 10, 10]))
print("Training ResNet-12...")
h_r12 = train_model(build_resnet(num_blocks_per_stage=[4, 4, 4]))
print("Training ResNet-30...")
h_r30 = train_model(build_resnet(num_blocks_per_stage=[10, 10, 10]))

# %% [markdown]
# ### 5.1 Training Loss & Validation Accuracy Comparison

# %%
epochs_range = range(1, len(h_p12["loss"]) + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss
ax1.plot(epochs_range, h_p12["loss"], "r--", label="Plain-12 Loss")
ax1.plot(epochs_range, h_p30["loss"], "r-", linewidth=2.5, label="Plain-30 Loss")
ax1.plot(epochs_range, h_r12["loss"], "b--", label="ResNet-12 Loss")
ax1.plot(epochs_range, h_r30["loss"], "b-", linewidth=2.5, label="ResNet-30 Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Training Loss: Plain vs ResNet", fontweight="bold")
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend()

# Accuracy
ax2.plot(epochs_range, h_p12["val_accuracy"], "r--", label="Plain-12 Val Acc")
ax2.plot(epochs_range, h_p30["val_accuracy"], "r-", linewidth=2.5, label="Plain-30 Val Acc")
ax2.plot(epochs_range, h_r12["val_accuracy"], "b--", label="ResNet-12 Val Acc")
ax2.plot(epochs_range, h_r30["val_accuracy"], "b-", linewidth=2.5, label="ResNet-30 Val Acc")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Validation Accuracy")
ax2.set_title("Validation Accuracy: Plain vs ResNet", fontweight="bold")
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Summary & Key Takeaways
#
# 1. **Gradient Preservation**: In ResNet, the identity mapping guarantees an additive gradient highway ($\mathbf{I}$ term), preventing early layer gradients from vanishing.
# 2. **Scalability with Depth**: Deeper ResNets (e.g. ResNet-30) train stably and outperform shallow counterparts, whereas Plain-30 fails to optimize effectively.
# 3. **Architectural Insight**: Residual connections enable training networks with hundreds of layers without gradient vanishing.
