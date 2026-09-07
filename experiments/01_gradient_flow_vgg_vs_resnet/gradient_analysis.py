import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras

from models import build_plain_vgg, build_resnet, create_model_pair


def compute_layerwise_gradients(model, x_batch, y_batch, loss_fn=keras.losses.CategoricalCrossentropy()):
    """
    Computes the L2 norm of the gradient of the loss with respect to
    the weights of each convolutional layer in the model.

    Returns:
        layer_names: List of layer names (ordered from input to output)
        grad_norms: List of float gradient L2 norms for each layer's kernel
    """
    x_tensor = tf.convert_to_tensor(x_batch, dtype=tf.float32)
    y_tensor = tf.convert_to_tensor(y_batch, dtype=tf.float32)

    with tf.GradientTape() as tape:
        preds = model(x_tensor, training=False)
        loss = loss_fn(y_tensor, preds)

    # Filter only Conv2D kernel weights
    conv_layers = [l for l in model.layers if isinstance(l, keras.layers.Conv2D) and "shortcut" not in l.name]
    conv_weights = [l.kernel for l in conv_layers]
    grads = tape.gradient(loss, conv_weights)

    layer_names = [l.name for l in conv_layers]
    grad_norms = []
    for g in grads:
        if g is not None:
            norm = tf.norm(g).numpy()
            grad_norms.append(float(norm))
        else:
            grad_norms.append(0.0)

    return layer_names, grad_norms


def plot_layerwise_gradients(depth=36, output_path="gradient_norm_by_layer.png"):
    """
    Compares the layer-by-layer gradient magnitude for Plain VGG vs ResNet at a given depth.
    """
    print(f"\n[1/3] Computing Layer-wise Gradient Norms for Depth={depth}...")
    np.random.seed(42)
    tf.random.set_seed(42)

    # Generate synthetic input batch for reproducible gradient measurement
    batch_size = 64
    x_sample = np.random.randn(batch_size, 32, 32, 3).astype(np.float32)
    y_sample = keras.utils.to_categorical(np.random.randint(0, 10, size=(batch_size,)), num_classes=10)

    plain_model, resnet_model = create_model_pair(total_depth=depth, use_batchnorm=False)

    _, plain_grads = compute_layerwise_gradients(plain_model, x_sample, y_sample)
    _, resnet_grads = compute_layerwise_gradients(resnet_model, x_sample, y_sample)

    layers_idx = np.arange(1, len(plain_grads) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Log scale plot
    ax1.plot(layers_idx, plain_grads, "o-", color="#d9534f", linewidth=2.2, label=f"Plain VGG-{depth} (No Shortcuts)")
    ax1.plot(layers_idx, resnet_grads, "s-", color="#0275d8", linewidth=2.2, label=f"ResNet-{depth} (Residual Shortcuts)")
    ax1.set_yscale("log")
    ax1.set_xlabel("Layer Index (1 = Earliest Input Layer $\\rightarrow$ Output)", fontsize=12)
    ax1.set_ylabel("Gradient Norm $\|\\nabla_W \\mathcal{L}\|_2$ (Log Scale)", fontsize=12)
    ax1.set_title("Layer-wise Gradient Magnitude (Log Scale)", fontsize=14, fontweight="bold")
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    ax1.legend(fontsize=11)

    # Linear scale plot
    ax2.plot(layers_idx, plain_grads, "o-", color="#d9534f", linewidth=2.2, label=f"Plain VGG-{depth}")
    ax2.plot(layers_idx, resnet_grads, "s-", color="#0275d8", linewidth=2.2, label=f"ResNet-{depth}")
    ax2.set_xlabel("Layer Index (1 = Earliest Input Layer $\\rightarrow$ Output)", fontsize=12)
    ax2.set_ylabel("Gradient Norm $\|\\nabla_W \\mathcal{L}\|_2$ (Linear Scale)", fontsize=12)
    ax2.set_title("Layer-wise Gradient Magnitude (Linear Scale)", fontsize=14, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=11)

    plt.suptitle(
        f"Gradient Propagation Comparison: Plain VGG vs ResNet ({depth} Layers)",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_gradient_vs_depth(depths=[6, 12, 18, 24, 30, 36, 48], output_path="gradient_at_first_layer_vs_depth.png"):
    """
    Plots the gradient magnitude arriving at the very first conv layer as network depth increases.
    """
    print(f"\n[2/3] Analyzing First-Layer Gradient Norm as Depth Increases {depths}...")
    np.random.seed(42)
    tf.random.set_seed(42)

    batch_size = 64
    x_sample = np.random.randn(batch_size, 32, 32, 3).astype(np.float32)
    y_sample = keras.utils.to_categorical(np.random.randint(0, 10, size=(batch_size,)), num_classes=10)

    plain_first_layer_grads = []
    resnet_first_layer_grads = []

    for d in depths:
        plain_model, resnet_model = create_model_pair(total_depth=d, use_batchnorm=False)
        _, p_grads = compute_layerwise_gradients(plain_model, x_sample, y_sample)
        _, r_grads = compute_layerwise_gradients(resnet_model, x_sample, y_sample)

        plain_first_layer_grads.append(p_grads[0])
        resnet_first_layer_grads.append(r_grads[0])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        depths,
        plain_first_layer_grads,
        "o--",
        color="#d9534f",
        linewidth=2.5,
        markersize=8,
        label="Plain VGG (Input Layer Gradient)",
    )
    ax.plot(
        depths,
        resnet_first_layer_grads,
        "s-",
        color="#0275d8",
        linewidth=2.5,
        markersize=8,
        label="ResNet (Input Layer Gradient)",
    )

    ax.set_yscale("log")
    ax.set_xlabel("Total Network Depth (Number of Conv Layers)", fontsize=13)
    ax.set_ylabel("First Layer Gradient Norm $\|\\nabla_{W_1} \\mathcal{L}\|_2$ (Log Scale)", fontsize=13)
    ax.set_title("Vanishing Gradient Effect vs Network Depth", fontsize=15, fontweight="bold")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    plot_layerwise_gradients(depth=36, output_path=os.path.join(out_dir, "gradient_norm_by_layer.png"))
    plot_gradient_vs_depth(depths=[6, 12, 18, 24, 30, 36, 48], output_path=os.path.join(out_dir, "gradient_at_first_layer_vs_depth.png"))
