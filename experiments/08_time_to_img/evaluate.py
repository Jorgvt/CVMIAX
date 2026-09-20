"""Evaluation and Interpretability Pipeline for Time-to-Image Regime Classification.

Includes:
1. Multi-class classification evaluation (Accuracy, F1, Confusion Matrices)
2. Grad-CAM (Gradient-Weighted Class Activation Mapping) for CNN feature map explainability.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

# Ensure experiment folder is in sys.path for sibling imports
_EXPERIMENT_DIR = str(Path(__file__).resolve().parent)
if _EXPERIMENT_DIR not in sys.path:
    sys.path.insert(0, _EXPERIMENT_DIR)

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix

from dataset import REGIME_NAMES


def evaluate_model(
    model: keras.Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """Evaluate model on test dataset and return detailed metrics.

    Args:
        model: Trained Keras model.
        x_test: 4D array of test images.
        y_test: 1D array of ground truth integer labels.

    Returns:
        metrics_dict: Dictionary containing test accuracy, macro F1, confusion matrix, and report.
    """
    y_probs = model.predict(x_test, verbose=0)
    y_preds = np.argmax(y_probs, axis=1)

    cm = confusion_matrix(y_test, y_preds, labels=[0, 1, 2, 3])
    report = classification_report(
        y_test,
        y_preds,
        labels=[0, 1, 2, 3],
        target_names=[REGIME_NAMES[i] for i in range(4)],
        output_dict=True,
        zero_division=0,
    )

    accuracy = float(np.mean(y_preds == y_test))
    macro_f1 = float(report["macro avg"]["f1-score"])

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
        "classification_report": report,
        "y_probs": y_probs,
        "y_preds": y_preds,
    }


def compute_gradcam(
    model: keras.Model,
    image: np.ndarray,
    target_class: Optional[int] = None,
    last_conv_layer_name: Optional[str] = None,
) -> np.ndarray:
    """Compute Grad-CAM heatmap for a single image sample.

    Args:
        model: Trained Keras CNN.
        image: Single image array of shape (H, W, 3).
        target_class: Integer class index to explain (if None, uses model's predicted class).
        last_conv_layer_name: Name of the last convolutional layer. If None, auto-detected.

    Returns:
        heatmap: 2D array of shape (H, W) normalized in [0, 1].
    """
    if image.ndim == 3:
        input_tensor = np.expand_dims(image, axis=0)
    else:
        input_tensor = image

    # Auto-detect last conv layer if not provided
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, keras.layers.Conv2D) or "conv" in layer.name.lower():
                last_conv_layer_name = layer.name
                break

    if last_conv_layer_name is None:
        raise ValueError("Could not automatically locate a convolutional layer in the model.")

    # Create gradient model
    grad_model = keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_tensor)
        if target_class is None:
            target_class = int(tf.argmax(predictions[0]))
        loss = predictions[:, target_class]

    # Compute gradients of class score with respect to feature maps
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU on heatmap to keep only features that positively correlate with the class
    heatmap = tf.maximum(heatmap, 0.0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap_np = heatmap.numpy()

    # Resize heatmap to input image dimensions
    from scipy.ndimage import zoom
    h, w = image.shape[:2]
    zoom_factors = (h / heatmap_np.shape[0], w / heatmap_np.shape[1])
    resized_heatmap = zoom(heatmap_np, zoom_factors, order=1)
    resized_heatmap = np.clip(resized_heatmap, 0.0, 1.0)

    return resized_heatmap
