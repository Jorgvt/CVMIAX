"""
Training comparison engine for Experiment 16.

Trains MobileNetV2 transfer learning models under different data augmentation
regimes, recording training trajectories, generalization gaps, and validation metrics.
"""

import os
import json
from typing import Dict, Any, List, Optional
import numpy as np
import tensorflow as tf
import keras

from dataset import load_cifar_subsample, create_tf_dataset
from augmentations import MixupLayer, get_augmentation_model
from models import build_transfer_learning_model


def run_training_experiment(
    epochs: int = 30,
    batch_size: int = 16,
    samples_per_class_train: int = 25,
    samples_per_class_val: int = 200,
    unfreeze_top_layers: int = 20,
    learning_rate: float = 1.5e-4,
    random_seed: int = 42,
    policies: Optional[List[str]] = None,
    save_dir: str = "outputs",
) -> Dict[str, Any]:
    """
    Executes comparative transfer learning training across multiple augmentation policies.

    Args:
        epochs: Number of training epochs per policy.
        batch_size: Mini-batch size.
        samples_per_class_train: Samples per class in train set (small to induce overfitting).
        samples_per_class_val: Samples per class in validation set.
        unfreeze_top_layers: Number of top backbone layers unfrozen for fine-tuning.
        learning_rate: Initial Adam learning rate.
        random_seed: Seed for reproducibility.
        policies: List of policy names to benchmark.
        save_dir: Directory to save history logs.

    Returns:
        Dictionary mapping policy names to history dictionaries and final metrics.
    """
    if policies is None:
        policies = ["none", "geometric", "photometric", "combined", "mixup"]

    os.makedirs(save_dir, exist_ok=True)

    # 1. Load subsampled dataset
    print(f"\n=======================================================")
    print(f"📦 Loading dataset (seed={random_seed})...")
    print(f"=======================================================")
    x_train, y_train, x_val, y_val, class_names = load_cifar_subsample(
        samples_per_class_train=samples_per_class_train,
        samples_per_class_val=samples_per_class_val,
        target_size=(96, 96),
        random_seed=random_seed,
    )
    num_classes = len(class_names)

    # 2. Validation dataset (fixed across all runs)
    val_dataset = create_tf_dataset(x_val, y_val, batch_size=batch_size, is_training=False)

    results = {}

    for policy in policies:
        print(f"\n" + "-" * 50)
        print(f"🚀 Training Transfer Learning Model with Policy: [{policy.upper()}]")
        print(f"-" * 50)

        # Set consistent seed before initializing each model
        tf.random.set_seed(random_seed)
        np.random.seed(random_seed)

        # Build model architecture
        model_aug_policy = "none" if policy == "mixup" else policy
        model = build_transfer_learning_model(
            input_shape=(96, 96, 3),
            num_classes=num_classes,
            aug_policy=model_aug_policy,
            unfreeze_top_layers=unfreeze_top_layers,
            dropout_rate=0.25,
            learning_rate=learning_rate,
        )

        # Train dataset preparation
        if policy == "mixup":
            # For mixup: combine geometric augmentation with batch-level mixup
            geom_layer = get_augmentation_model("geometric")
            mixup_layer = MixupLayer(alpha=0.2)
            train_raw_ds = create_tf_dataset(
                x_train, y_train, batch_size=batch_size, is_training=True
            )

            def apply_mixup_pipeline(images, labels):
                aug_images = geom_layer(images, training=True)
                return mixup_layer(aug_images, labels, training=True)

            train_dataset = train_raw_ds.map(
                apply_mixup_pipeline, num_parallel_calls=tf.data.AUTOTUNE
            )
        else:
            train_dataset = create_tf_dataset(
                x_train, y_train, batch_size=batch_size, is_training=True
            )

        # Learning rate schedule callback
        lr_callback = keras.callbacks.LearningRateScheduler(
            lambda epoch, lr: learning_rate * (0.95 ** (epoch // 5))
        )

        # Train model
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            verbose=1,
            callbacks=[lr_callback],
        )

        # Compute generalization metrics
        hist_dict = history.history
        train_acc = hist_dict["accuracy"]
        val_acc = hist_dict["val_accuracy"]
        train_loss = hist_dict["loss"]
        val_loss = hist_dict["val_loss"]

        acc_gap = [t - v for t, v in zip(train_acc, val_acc)]
        loss_gap = [v - t for t, v in zip(train_loss, val_loss)]

        results[policy] = {
            "policy": policy,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "acc_gap": acc_gap,
            "loss_gap": loss_gap,
            "final_train_acc": float(train_acc[-1]),
            "final_val_acc": float(val_acc[-1]),
            "best_val_acc": float(np.max(val_acc)),
            "final_train_loss": float(train_loss[-1]),
            "final_val_loss": float(val_loss[-1]),
            "final_acc_gap": float(acc_gap[-1]),
        }

        print(
            f"🎯 [{policy.upper()}] Final Train Acc: {train_acc[-1]:.4f} | "
            f"Final Val Acc: {val_acc[-1]:.4f} (Best: {np.max(val_acc):.4f}) | "
            f"Generalization Gap: {acc_gap[-1]:.4f}"
        )

    # Save results to JSON
    summary_path = os.path.join(save_dir, "training_history_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Saved full training experiment logs to: {summary_path}")

    return {
        "results": results,
        "class_names": class_names,
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
    }


if __name__ == "__main__":
    run_training_experiment(epochs=5, samples_per_class_train=20)
