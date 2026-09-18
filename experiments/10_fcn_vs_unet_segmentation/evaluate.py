"""
Evaluation and Quantitative Metrics for Semantic Segmentation.

Computes:
1. Overall Mean IoU (mIoU)
2. Per-Class IoU (Background, Rectangle, Circle, Triangle)
3. Boundary IoU (measures boundary sharpness and edge precision within a narrow boundary band)
4. Overall Pixel Accuracy
"""

from typing import Dict, List, Tuple
import numpy as np
import scipy.ndimage as ndimage
import tensorflow as tf
from tensorflow import keras

from dataset import NUM_CLASSES, CLASS_NAMES


def compute_iou_per_class(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> Dict[str, float]:
    """
    Compute Intersection over Union (IoU) for each class and mean IoU.

    Args:
        y_true: Ground-truth integer mask array of shape (N, H, W) or (H, W).
        y_pred: Predicted integer mask array of shape (N, H, W) or (H, W).
        num_classes: Total number of classes.

    Returns:
        Dictionary mapping class names to their IoU and 'mean_iou'.
    """
    y_true_flat = y_true.ravel()
    y_pred_flat = y_pred.ravel()

    ious = {}
    valid_ious = []

    for c in range(num_classes):
        true_c = (y_true_flat == c)
        pred_c = (y_pred_flat == c)

        intersection = np.logical_and(true_c, pred_c).sum()
        union = np.logical_or(true_c, pred_c).sum()

        if union == 0:
            iou = 1.0  # Perfect score if class is absent in both
        else:
            iou = float(intersection / union)

        ious[CLASS_NAMES[c]] = iou
        valid_ious.append(iou)

    ious["mean_iou"] = float(np.mean(valid_ious))
    ious["pixel_accuracy"] = float(np.mean(y_true_flat == y_pred_flat))
    return ious


def extract_boundary_mask(mask: np.ndarray, dilation_radius: int = 2) -> np.ndarray:
    """
    Extract a binary mask of shape boundaries using morphological gradient.

    Args:
        mask: 2D integer mask (H, W).
        dilation_radius: Pixel radius for boundary band width.

    Returns:
        binary mask (H, W) where True indicates boundary pixels.
    """
    # Structure element for dilation/erosion
    struct = ndimage.generate_binary_structure(2, 2)
    # Morphological gradient = dilation - erosion
    dilated = ndimage.binary_dilation(mask > 0, structure=struct, iterations=dilation_radius)
    eroded = ndimage.binary_erosion(mask > 0, structure=struct, iterations=dilation_radius)
    boundary = np.logical_xor(dilated, eroded)
    return boundary


def compute_boundary_iou(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dilation_radius: int = 2,
    num_classes: int = NUM_CLASSES,
) -> float:
    """
    Compute Boundary IoU to specifically evaluate edge sharpness and boundary localization.

    Args:
        y_true: Ground-truth integer masks of shape (N, H, W).
        y_pred: Predicted integer masks of shape (N, H, W).
        dilation_radius: Boundary band half-width in pixels.
        num_classes: Number of classes.

    Returns:
        Boundary IoU score averaged across samples.
    """
    sample_boundary_ious = []

    for i in range(len(y_true)):
        t_mask = y_true[i]
        p_mask = y_pred[i]

        t_bound = extract_boundary_mask(t_mask, dilation_radius=dilation_radius)
        p_bound = extract_boundary_mask(p_mask, dilation_radius=dilation_radius)

        intersection = np.logical_and(t_bound, p_bound).sum()
        union = np.logical_or(t_bound, p_bound).sum()

        if union == 0:
            sample_boundary_ious.append(1.0)
        else:
            sample_boundary_ious.append(float(intersection / union))

    return float(np.mean(sample_boundary_ious))


def evaluate_model(
    model: keras.Model,
    images: np.ndarray,
    masks: np.ndarray,
    batch_size: int = 16,
) -> Dict[str, float]:
    """
    Comprehensive evaluation of a trained segmentation model on a test set.

    Args:
        model: Trained Keras segmentation model.
        images: Input test images (N, H, W, 3).
        masks: Ground truth masks (N, H, W).
        batch_size: Batch size for model inference.

    Returns:
        Dictionary of quantitative evaluation metrics.
    """
    # Predict probabilities (N, H, W, num_classes)
    pred_probs = model.predict(images, batch_size=batch_size, verbose=0)
    # Argmax prediction (N, H, W)
    pred_masks = np.argmax(pred_probs, axis=-1)

    metrics = compute_iou_per_class(masks, pred_masks)
    metrics["boundary_iou"] = compute_boundary_iou(masks, pred_masks, dilation_radius=2)

    return metrics


if __name__ == "__main__":
    from dataset import get_dataset_splits
    from models import build_unet, build_plain_fcn

    print("Testing evaluate routines...")
    (x_tr, y_tr), (x_va, y_va), (x_te, y_te) = get_dataset_splits(20, 10, 10)
    model = build_plain_fcn()
    res = evaluate_model(model, x_te, y_te)
    print("Initial untransformed model metrics:", res)
