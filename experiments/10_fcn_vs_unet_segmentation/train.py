"""
Training Pipeline for Semantic Segmentation:
Plain FCN vs U-Net with Automated Checkpointing and Persistence.

Trains both architectures under strictly controlled identical conditions
(same dataset splits, optimizer, loss function, learning rate, and batch size).
Saves model checkpoints and training histories to disk to avoid redundant retraining.
"""

from pathlib import Path
import json
from typing import Dict, Tuple, Any, Optional
import numpy as np
import tensorflow as tf
from tensorflow import keras

from dataset import get_dataset_splits, create_tf_dataset
from models import build_plain_fcn, build_unet

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"


def train_models(
    epochs: int = 25,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    num_train: int = 400,
    num_val: int = 100,
    num_test: int = 100,
    base_filters: int = 32,
    seed: int = 42,
    force_retrain: bool = False,
    checkpoint_dir: Optional[Path] = None,
    verbose: int = 1,
) -> Tuple[keras.Model, keras.Model, Dict[str, Any], Dict[str, Any], Tuple[Any, Any, Any]]:
    """
    Train both Plain FCN and U-Net models with automatic checkpoint saving/loading.

    Args:
        epochs: Number of training epochs.
        batch_size: Minibatch size.
        learning_rate: Adam optimizer initial learning rate.
        num_train: Number of synthetic training images.
        num_val: Number of validation images.
        num_test: Number of test images.
        base_filters: Initial convolution filter depth.
        seed: Random seed for reproducibility.
        force_retrain: If True, retrains and overwrites checkpoints. If False, loads saved weights if present.
        checkpoint_dir: Path to directory where checkpoints and histories are stored.
        verbose: Keras verbosity level (0, 1, or 2).

    Returns:
        fcn_model: Trained or loaded Plain FCN model.
        unet_model: Trained or loaded U-Net model.
        fcn_history: Training history dictionary for FCN.
        unet_history: Training history dictionary for U-Net.
        splits: ((x_train, y_train), (x_val, y_val), (x_test, y_test))
    """
    ckpt_dir = checkpoint_dir or CHECKPOINT_DIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    fcn_ckpt_path = ckpt_dir / "plain_fcn_best.weights.h5"
    unet_ckpt_path = ckpt_dir / "unet_best.weights.h5"
    fcn_hist_path = ckpt_dir / "plain_fcn_history.json"
    unet_hist_path = ckpt_dir / "unet_history.json"

    # 1. Generate Dataset
    print(f"Generating synthetic segmentation dataset (Train={num_train}, Val={num_val}, Test={num_test})...")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = get_dataset_splits(
        num_train=num_train,
        num_val=num_val,
        num_test=num_test,
        img_size=128,
        seed=seed,
    )

    train_ds = create_tf_dataset(x_train, y_train, batch_size=batch_size, shuffle=True, seed=seed)
    val_ds = create_tf_dataset(x_val, y_val, batch_size=batch_size, shuffle=False)

    # 2. Build Models
    fcn_model = build_plain_fcn(input_shape=(128, 128, 3), num_classes=4, base_filters=base_filters)
    unet_model = build_unet(input_shape=(128, 128, 3), num_classes=4, base_filters=base_filters)

    loss_fn = keras.losses.SparseCategoricalCrossentropy()

    fcn_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=["accuracy"],
    )

    unet_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=["accuracy"],
    )

    # Check if existing checkpoints should be loaded
    checkpoints_exist = (
        fcn_ckpt_path.exists()
        and unet_ckpt_path.exists()
        and fcn_hist_path.exists()
        and unet_hist_path.exists()
    )

    if checkpoints_exist and not force_retrain:
        print(f"\nFound existing trained checkpoints in {ckpt_dir}. Loading weights...")
        fcn_model.load_weights(str(fcn_ckpt_path))
        unet_model.load_weights(str(unet_ckpt_path))

        with open(fcn_hist_path, "r") as f:
            fcn_history = json.load(f)
        with open(unet_hist_path, "r") as f:
            unet_history = json.load(f)

        print("Loaded weights and histories successfully!")
        return (
            fcn_model,
            unet_model,
            fcn_history,
            unet_history,
            ((x_train, y_train), (x_val, y_val), (x_test, y_test)),
        )

    # 3. Train Plain FCN
    print("\n" + "=" * 60)
    print("Training 1/2: Plain FCN (Bottleneck-only, No Skip Connections)")
    print("=" * 60)
    fcn_callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(fcn_ckpt_path),
            save_weights_only=True,
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        )
    ]
    fcn_history_obj = fcn_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=fcn_callbacks,
        verbose=verbose,
    )
    # Load best weights
    fcn_model.load_weights(str(fcn_ckpt_path))
    fcn_history = {k: [float(v) for v in vals] for k, vals in fcn_history_obj.history.items()}
    with open(fcn_hist_path, "w") as f:
        json.dump(fcn_history, f, indent=2)

    # 4. Train U-Net
    print("\n" + "=" * 60)
    print("Training 2/2: U-Net (With Encoder-Decoder Skip Connections)")
    print("=" * 60)
    unet_callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(unet_ckpt_path),
            save_weights_only=True,
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        )
    ]
    unet_history_obj = unet_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=unet_callbacks,
        verbose=verbose,
    )
    # Load best weights
    unet_model.load_weights(str(unet_ckpt_path))
    unet_history = {k: [float(v) for v in vals] for k, vals in unet_history_obj.history.items()}
    with open(unet_hist_path, "w") as f:
        json.dump(unet_history, f, indent=2)

    print(f"\nModel checkpoints and training logs saved to {ckpt_dir}")

    return (
        fcn_model,
        unet_model,
        fcn_history,
        unet_history,
        ((x_train, y_train), (x_val, y_val), (x_test, y_test)),
    )


if __name__ == "__main__":
    fcn, unet, hist_fcn, hist_unet, splits = train_models(
        epochs=20,
        batch_size=16,
        num_train=300,
        num_val=80,
        num_test=80,
        force_retrain=True,
        verbose=1,
    )
    print("Training complete and checkpoints saved.")
