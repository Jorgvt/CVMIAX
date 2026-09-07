import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras

from models import create_model_pair


def get_cifar10_subset(num_train=10000, num_val=2000):
    """
    Loads and preprocesses CIFAR-10 data subset for fast, clear demonstration.
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    # Subsample for fast training run in course demonstrations
    x_train_sub = x_train[:num_train]
    y_train_sub = y_train[:num_train]
    x_val_sub = x_test[:num_val]
    y_val_sub = y_test[:num_val]

    return (x_train_sub, y_train_sub), (x_val_sub, y_val_sub)


def train_and_record(model, train_data, val_data, epochs=15, batch_size=128, lr=1e-3):
    """
    Compiles and trains a given model, returning history metrics.
    """
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        train_data[0],
        train_data[1],
        validation_data=val_data,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )
    return history.history


def run_training_experiment(shallow_depth=12, deep_depth=36, epochs=15, output_path="training_curves_comparison.png"):
    """
    Trains shallow vs deep Plain VGG and ResNet to showcase the degradation
    problem (plain deep model fails to train vs ResNet deep model).
    """
    print(f"\n[3/3] Running Training Degradation Experiment...")
    print(f"Comparing Shallow (Depth={shallow_depth}) vs Deep (Depth={deep_depth}) Plain & ResNet models...")

    train_data, val_data = get_cifar10_subset(num_train=10000, num_val=2000)

    plain_shallow, resnet_shallow = create_model_pair(total_depth=shallow_depth)
    plain_deep, resnet_deep = create_model_pair(total_depth=deep_depth)

    print("\nTraining Plain Shallow...")
    h_plain_shallow = train_and_record(plain_shallow, train_data, val_data, epochs=epochs)

    print("\nTraining ResNet Shallow...")
    h_resnet_shallow = train_and_record(resnet_shallow, train_data, val_data, epochs=epochs)

    print("\nTraining Plain Deep...")
    h_plain_deep = train_and_record(plain_deep, train_data, val_data, epochs=epochs)

    print("\nTraining ResNet Deep...")
    h_resnet_deep = train_and_record(resnet_deep, train_data, val_data, epochs=epochs)

    # Plot Training Loss and Validation Accuracy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    epochs_range = range(1, epochs + 1)

    # Training Loss
    ax1.plot(epochs_range, h_plain_shallow["loss"], "r--", linewidth=2, label=f"Plain VGG-{shallow_depth} (Loss)")
    ax1.plot(epochs_range, h_plain_deep["loss"], "r-", linewidth=2.5, label=f"Plain VGG-{deep_depth} (Loss)")
    ax1.plot(epochs_range, h_resnet_shallow["loss"], "b--", linewidth=2, label=f"ResNet-{shallow_depth} (Loss)")
    ax1.plot(epochs_range, h_resnet_deep["loss"], "b-", linewidth=2.5, label=f"ResNet-{deep_depth} (Loss)")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Training Loss", fontsize=12)
    ax1.set_title("Training Loss: Plain vs ResNet Across Depths", fontsize=14, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=10)

    # Validation Accuracy
    ax2.plot(epochs_range, h_plain_shallow["val_accuracy"], "r--", linewidth=2, label=f"Plain VGG-{shallow_depth} (Val Acc)")
    ax2.plot(epochs_range, h_plain_deep["val_accuracy"], "r-", linewidth=2.5, label=f"Plain VGG-{deep_depth} (Val Acc)")
    ax2.plot(epochs_range, h_resnet_shallow["val_accuracy"], "b--", linewidth=2, label=f"ResNet-{shallow_depth} (Val Acc)")
    ax2.plot(epochs_range, h_resnet_deep["val_accuracy"], "b-", linewidth=2.5, label=f"ResNet-{deep_depth} (Val Acc)")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Validation Accuracy", fontsize=12)
    ax2.set_title("Validation Accuracy: Plain vs ResNet Across Depths", fontsize=14, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=10)

    plt.suptitle("The Degradation Problem: Plain Networks vs Residual Networks", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    run_training_experiment(
        shallow_depth=12,
        deep_depth=30,
        epochs=10,
        output_path=os.path.join(out_dir, "training_curves_comparison.png"),
    )
