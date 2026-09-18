"""
Training Pipeline for Two-Stage R-CNN Object Detector.

Handles:
1. Extracting candidate proposal crops across all training images.
2. Balanced sampling of positive (foreground) vs hard negative (background) ROI crops.
3. Model training with learning rate scheduling, validation, and checkpoint saving/loading.
"""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import tensorflow as tf
from tensorflow import keras

try:
    from .dataset import generate_dataset
    from .models import build_rcnn_detector
    from .proposals import (
        extract_and_warp_crops,
        generate_sliding_window_proposals,
        match_proposals_to_ground_truth,
    )
except ImportError:
    from dataset import generate_dataset
    from models import build_rcnn_detector
    from proposals import (
        extract_and_warp_crops,
        generate_sliding_window_proposals,
        match_proposals_to_ground_truth,
    )


def prepare_rcnn_training_data(
    images: np.ndarray,
    annotations: List[Dict[str, np.ndarray]],
    crop_size: Tuple[int, int] = (32, 32),
    pos_thresh: float = 0.5,
    neg_thresh: float = 0.2,
    neg_to_pos_ratio: float = 4.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract, match, and balance ROI crops across a collection of images for R-CNN training.
    Includes both pure background crops (IoU = 0.0) and near-miss hard negative crops (0.0 < IoU < neg_thresh)
    to effectively suppress false positive box detections.

    Args:
        images: np.ndarray of shape (N_img, H, W, 3).
        annotations: List of dicts with 'boxes' (N, 4) and 'classes' (N,).
        crop_size: Canonical size to warp proposal crops to.
        pos_thresh: Minimum IoU with GT to count as positive.
        neg_thresh: Maximum IoU with GT to count as background.
        neg_to_pos_ratio: Ratio of negative to positive proposals to keep.
        seed: Random seed for reproducibility.

    Returns:
        X_crops: np.ndarray of shape (N_crops, 32, 32, 3) in float32.
        y_cls: np.ndarray of shape (N_crops,) with integer class IDs (0=bg, 1..C).
        y_bbox: np.ndarray of shape (N_crops, 4) with regression delta targets.
    """
    rng = np.random.default_rng(seed)
    all_crops = []
    all_labels = []
    all_deltas = []

    for i in range(len(images)):
        img = images[i]
        gt_boxes = annotations[i]["boxes"]
        gt_classes = annotations[i]["classes"]

        # 1. Generate candidate proposals
        proposals = generate_sliding_window_proposals(img_size=img.shape[0])

        # 2. Match proposals to GT
        labels, deltas, max_ious = match_proposals_to_ground_truth(
            proposals, gt_boxes, gt_classes, pos_thresh=pos_thresh, neg_thresh=neg_thresh
        )

        pos_indices = np.where(labels > 0)[0]
        neg_indices = np.where(labels == 0)[0]

        # Partition negatives into near-miss (0.0 < IoU < neg_thresh) and empty background (IoU == 0.0)
        near_miss_neg = np.where((labels == 0) & (max_ious > 0.0))[0]
        empty_neg = np.where((labels == 0) & (max_ious == 0.0))[0]

        # 3. Balanced sampling
        num_pos = len(pos_indices)
        if num_pos == 0:
            selected_neg = rng.choice(neg_indices, size=min(8, len(neg_indices)), replace=False)
            selected_indices = selected_neg
        else:
            total_neg_needed = int(num_pos * neg_to_pos_ratio)
            # Prioritize near-misses (50%) + pure empty backgrounds (50%)
            half_neg = total_neg_needed // 2

            sampled_near = []
            if len(near_miss_neg) > 0:
                sampled_near = rng.choice(
                    near_miss_neg,
                    size=min(half_neg, len(near_miss_neg)),
                    replace=False
                )

            rem_needed = total_neg_needed - len(sampled_near)
            sampled_empty = []
            if len(empty_neg) > 0 and rem_needed > 0:
                sampled_empty = rng.choice(
                    empty_neg,
                    size=min(rem_needed, len(empty_neg)),
                    replace=False
                )

            selected_neg = np.concatenate([sampled_near, sampled_empty]).astype(int)
            selected_indices = np.concatenate([pos_indices, selected_neg])

        if len(selected_indices) == 0:
            continue

        selected_proposals = proposals[selected_indices]
        selected_crops = extract_and_warp_crops(img, selected_proposals, crop_size=crop_size)

        all_crops.append(selected_crops)
        all_labels.append(labels[selected_indices])
        all_deltas.append(deltas[selected_indices])

    X_crops = np.concatenate(all_crops, axis=0)
    y_cls = np.concatenate(all_labels, axis=0)
    y_bbox = np.concatenate(all_deltas, axis=0)

    # Shuffle dataset
    shuffle_indices = rng.permutation(len(X_crops))
    X_crops = X_crops[shuffle_indices]
    y_cls = y_cls[shuffle_indices]
    y_bbox = y_bbox[shuffle_indices]

    return X_crops, y_cls, y_bbox


def train_rcnn_detector(
    X_train: np.ndarray,
    y_cls_train: np.ndarray,
    y_bbox_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_cls_val: Optional[np.ndarray] = None,
    y_bbox_val: Optional[np.ndarray] = None,
    epochs: int = 25,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    num_classes: int = 4,
    checkpoint_path: Optional[str] = None,
) -> Tuple[keras.Model, keras.callbacks.History]:
    """
    Train the R-CNN Stage 2 detector model with learning rate scheduling and checkpoint saving.

    Returns:
        model: Trained SimpleRCNNModel instance.
        history: Keras training history object.
    """
    model = build_rcnn_detector(
        input_shape=(32, 32, 3),
        num_classes=num_classes,
        reg_weight=1.0,
    )

    # Cosine decay schedule for smooth convergence across epochs
    total_steps = (len(X_train) // batch_size) * epochs
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=learning_rate,
        decay_steps=max(1, total_steps),
        alpha=0.05,
    )
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer)

    val_data = None
    if X_val is not None and y_cls_val is not None and y_bbox_val is not None:
        val_data = (X_val, (y_cls_val, y_bbox_val))

    callbacks = []
    if checkpoint_path:
        os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)

    history = model.fit(
        X_train,
        (y_cls_train, y_bbox_train),
        validation_data=val_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    if checkpoint_path:
        model.save_weights(checkpoint_path)
        print(f"[Checkpoint] Model weights successfully saved to: {checkpoint_path}")

    return model, history


def load_or_train_rcnn_detector(
    checkpoint_path: str,
    X_train: Optional[np.ndarray] = None,
    y_cls_train: Optional[np.ndarray] = None,
    y_bbox_train: Optional[np.ndarray] = None,
    X_val: Optional[np.ndarray] = None,
    y_cls_val: Optional[np.ndarray] = None,
    y_bbox_val: Optional[np.ndarray] = None,
    force_retrain: bool = False,
    epochs: int = 25,
    num_classes: int = 4,
) -> Tuple[keras.Model, Optional[keras.callbacks.History]]:
    """
    Load model weights from checkpoint if exists; otherwise train and save.
    """
    model = build_rcnn_detector(
        input_shape=(32, 32, 3),
        num_classes=num_classes,
        reg_weight=1.0,
    )
    # Forward dummy batch to initialize weights variables before loading
    dummy_input = np.zeros((1, 32, 32, 3), dtype=np.float32)
    model(dummy_input)

    if os.path.exists(checkpoint_path) and not force_retrain:
        print(f"[Checkpoint] Loading existing pre-trained weights from: {checkpoint_path}")
        model.load_weights(checkpoint_path)
        return model, None

    if X_train is None or y_cls_train is None or y_bbox_train is None:
        raise ValueError(f"Checkpoint {checkpoint_path} not found and no training data provided.")

    print(f"[Checkpoint] Training detector for {epochs} epochs...")
    model, history = train_rcnn_detector(
        X_train=X_train,
        y_cls_train=y_cls_train,
        y_bbox_train=y_bbox_train,
        X_val=X_val,
        y_cls_val=y_cls_val,
        y_bbox_val=y_bbox_val,
        epochs=epochs,
        checkpoint_path=checkpoint_path,
        num_classes=num_classes,
    )
    return model, history
