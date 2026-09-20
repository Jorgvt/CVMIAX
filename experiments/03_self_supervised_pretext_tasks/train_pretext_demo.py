"""
Training demonstration of a Self-Supervised Pretext Task (Rotation Prediction) on CIFAR-10.
Shows how a model trains completely without human annotation labels,
learning representation features purely from supervisory signal synthesized from data transformations.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras

from data_generators import create_rotation_batch
from models import build_rotation_model
from dataset import load_unlabeled_cifar10


def run_rotation_pretext_training(num_samples=5000, epochs=5, batch_size=64, output_dir="figures"):
    os.makedirs(output_dir, exist_ok=True)
    print("\n--- Training Self-Supervised Rotation Prediction Model ---")

    # 1. Load Unlabeled Images (we discard the original CIFAR labels!)
    x_train, x_test = load_unlabeled_cifar10(num_train=num_samples, num_test=1000)

    print(f"Loaded {len(x_train)} unlabeled images for self-supervised pretext training.")

    # 2. Synthesize Pretext Inputs and Labels (0°, 90°, 180°, 270°)
    x_ssl_train, y_ssl_train = create_rotation_batch(x_train)
    x_ssl_test, y_ssl_test = create_rotation_batch(x_test)

    # Shuffle training set
    shuffle_idx = np.random.permutation(len(x_ssl_train))
    x_ssl_train = x_ssl_train[shuffle_idx]
    y_ssl_train = y_ssl_train[shuffle_idx]

    y_ssl_train_cat = keras.utils.to_categorical(y_ssl_train, 4)
    y_ssl_test_cat = keras.utils.to_categorical(y_ssl_test, 4)

    print(f"Synthesized SSL dataset: {len(x_ssl_train)} samples across 4 rotation classes.")

    # 3. Build & Compile Pretext Model
    model = build_rotation_model(input_shape=(32, 32, 3), num_classes=4)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # 4. Train Pretext Model
    history = model.fit(
        x_ssl_train,
        y_ssl_train_cat,
        validation_data=(x_ssl_test, y_ssl_test_cat),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )

    # 5. Plot Pretext Training Accuracy Curve
    fig, ax = plt.subplots(figsize=(8, 4.5))
    epochs_range = range(1, epochs + 1)
    ax.plot(epochs_range, history.history["accuracy"], "o-", color="#1f77b4", linewidth=2.2, label="SSL Training Accuracy")
    ax.plot(epochs_range, history.history["val_accuracy"], "s-", color="#2ca02c", linewidth=2.2, label="SSL Validation Accuracy")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Rotation Classification Accuracy", fontsize=12)
    ax.set_title("Self-Supervised Pretext Training Curves (Rotation Prediction)", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=11)
    plt.tight_layout()
    curve_path = os.path.join(output_dir, "ssl_rotation_training_curve.png")
    fig.savefig(curve_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {curve_path}")

    # 6. Qualitative Test Predictions
    sample_indices = [10, 25, 42, 60]
    fig, axes = plt.subplots(len(sample_indices), 4, figsize=(12, 3 * len(sample_indices)))
    deg_names = ["0°", "90°", "180°", "270°"]

    for row_idx, sample_i in enumerate(sample_indices):
        raw_img = x_test[sample_i:sample_i+1]
        rot_imgs, true_labels = create_rotation_batch(raw_img)
        preds = model.predict(rot_imgs, verbose=0)
        pred_labels = np.argmax(preds, axis=1)

        for col_idx in range(4):
            ax = axes[row_idx, col_idx]
            ax.imshow(rot_imgs[col_idx])
            is_correct = pred_labels[col_idx] == true_labels[col_idx]
            status_color = "green" if is_correct else "red"
            ax.set_title(
                f"True: {deg_names[true_labels[col_idx]]} | Pred: {deg_names[pred_labels[col_idx]]}",
                fontsize=10,
                color=status_color,
                fontweight="bold",
            )
            ax.axis("off")

    plt.suptitle("Qualitative Self-Supervised Rotation Predictions on Unseen Test Images", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    pred_path = os.path.join(output_dir, "ssl_rotation_qualitative_predictions.png")
    fig.savefig(pred_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {pred_path}")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(current_dir, "figures")
    run_rotation_pretext_training(num_samples=4000, epochs=4, batch_size=64, output_dir=figures_dir)
