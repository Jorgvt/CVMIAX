"""Modular Training and Inference Pipeline for Time-to-Image Regime Classification.

Supports training across any transformation encoding (GAF, CWT, STFT, Fusion) and model architecture
(Custom ResNet-18 vs Pretrained Transfer Learning Backbone), with automatic weight persistence.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import tensorflow as tf
from tensorflow import keras

from dataset import get_encoded_datasets
from models import get_model, load_model_weights, save_model_weights

DEFAULT_WEIGHTS_DIR = Path("experiments/08_time_to_img/weights")


def train_or_load_regime_classifier(
    method: str = "fusion",
    model_type: str = "resnet18",
    epochs: int = 15,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    image_size: int = 128,
    cmap: str = "viridis",
    weights_dir: Union[str, Path] = DEFAULT_WEIGHTS_DIR,
    force_train: bool = False,
    verbose: int = 1,
) -> Tuple[keras.Model, Optional[keras.callbacks.History], Dict[str, Any]]:
    """Train or load a cached visual regime classifier for a specified 2D representation.

    Args:
        method: 'gaf', 'cwt', 'stft', or 'fusion'.
        model_type: 'resnet18' or 'mobilenet_v2'.
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        learning_rate: Initial learning rate.
        image_size: Image dimension (image_size x image_size).
        cmap: Colormap used for RGB transformation.
        weights_dir: Directory where model weights are cached.
        force_train: If True, retrains model even if saved weights already exist.
        verbose: Verbosity level (0, 1, or 2).

    Returns:
        model: Trained or loaded Keras model.
        history: Keras training history object (or None if loaded from disk).
        data_dict: Dictionary containing test datasets and labels for evaluation.
    """
    weights_path = Path(weights_dir)
    weights_path.mkdir(parents=True, exist_ok=True)
    checkpoint_file = weights_path / f"{method}_{model_type}.weights.h5"

    data_dict = get_encoded_datasets(
        method=method,
        image_size=image_size,
        cmap=cmap,
        batch_size=batch_size,
    )

    model = get_model(
        model_type=model_type,
        input_shape=(image_size, image_size, 3),
        num_classes=4,
        learning_rate=learning_rate,
    )

    if not force_train and checkpoint_file.exists():
        if verbose:
            print(f"--> Found cached weights at {checkpoint_file}. Loading without retraining.")
        load_model_weights(model, checkpoint_file)
        return model, None, data_dict

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=verbose,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=verbose,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_file),
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
            verbose=verbose,
        ),
    ]

    history = model.fit(
        data_dict["ds_train"],
        validation_data=data_dict["ds_val"],
        epochs=epochs,
        callbacks=callbacks,
        verbose=verbose,
    )

    # Save final best weights
    save_model_weights(model, checkpoint_file)
    if verbose:
        print(f"--> Saved best model weights to {checkpoint_file}")

    return model, history, data_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or Load Time-Series Image Regime Classifier")
    parser.add_argument("--method", type=str, default="fusion", choices=["gaf", "cwt", "stft", "fusion"])
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "mobilenet_v2"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--force_train", action="store_true", help="Force retraining even if cached weights exist")
    args = parser.parse_args()

    train_or_load_regime_classifier(
        method=args.method,
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        force_train=args.force_train,
    )
