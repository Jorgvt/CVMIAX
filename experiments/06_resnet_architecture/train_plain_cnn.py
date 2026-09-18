"""
Independent training script for Plain-20 CNN on CIFAR-10.
Saves model weights upon completion for downstream comparisons.
"""

import os
import sys
import json
import time
from typing import Optional, Dict, Any, Tuple
import keras
from keras import optimizers, losses, metrics

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from plain_cnn import build_plain20, save_plain_weights
from dataset import load_cifar10_data, create_tf_dataset


def train_plain_cnn(
    epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    subset_samples: Optional[int] = None,
    weights_path: Optional[str] = None,
    history_path: Optional[str] = None,
) -> Tuple[keras.Model, Dict[str, Any]]:
    """
    Trains Plain-20 CNN on CIFAR-10 and saves its weights.
    """
    if weights_path is None:
        weights_path = os.path.join(current_dir, "weights", "plain20_cifar10.weights.h5")
    if history_path is None:
        history_path = os.path.join(current_dir, "weights", "plain_history.json")

    print("=" * 60)
    print(" 🚀 Training Plain-20 Forward CNN on CIFAR-10")
    print("=" * 60)

    # 1. Dataset
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_cifar10_data(
        val_split=0.1, normalize=True, standardize=True
    )
    if subset_samples is not None and subset_samples < len(x_train):
        x_train, y_train = x_train[:subset_samples], y_train[:subset_samples]
        x_val, y_val = x_val[:subset_samples // 5], y_val[:subset_samples // 5]
        x_test, y_test = x_test[:subset_samples // 5], y_test[:subset_samples // 5]
        print(f" [Demo Mode] Subsetting data to {x_train.shape[0]} training samples.")

    train_ds = create_tf_dataset(x_train, y_train, batch_size=batch_size, augment=True, shuffle=True)
    val_ds = create_tf_dataset(x_val, y_val, batch_size=batch_size, augment=False, shuffle=False)
    test_ds = create_tf_dataset(x_test, y_test, batch_size=batch_size, augment=False, shuffle=False)

    # 2. Build Model
    model = build_plain20(input_shape=(32, 32, 3), num_classes=10)
    print(f" Plain-20 Parameters: {model.count_params():,}")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=[metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    # 3. Train
    start_time = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=1,
    )
    duration = time.time() - start_time

    # 4. Evaluate & Save Weights
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"\n📊 Plain-20 Test Accuracy: {test_acc * 100:.2f}% | Test Loss: {test_loss:.4f} (Elapsed: {duration:.1f}s)")

    saved_path = save_plain_weights(model, weights_path)
    print(f"💾 Stored Plain-20 weights to: {saved_path}")

    # Save history json
    hist_dict = {
        "loss": [float(x) for x in history.history["loss"]],
        "val_loss": [float(x) for x in history.history.get("val_loss", [])],
        "accuracy": [float(x) for x in history.history.get("accuracy", [])],
        "val_accuracy": [float(x) for x in history.history.get("val_accuracy", [])],
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "duration_seconds": duration,
        "epochs": epochs,
    }
    os.makedirs(os.path.dirname(os.path.abspath(history_path)), exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(hist_dict, f, indent=2)

    return model, hist_dict


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Plain-20 CNN on CIFAR-10")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--quick-demo", action="store_true", help="Run on 5000 subset samples for quick test")
    args = parser.parse_args()

    subset = 5000 if args.quick_demo else None
    train_plain_cnn(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr, subset_samples=subset)
