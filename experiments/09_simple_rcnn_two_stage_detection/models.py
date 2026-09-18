"""
Lightweight CNN Architecture & Multi-Task Loss for Two-Stage Object Detection (R-CNN Stage 2).

Consists of a shared convolutional feature extractor with dual heads:
1. Classification Head (Softmax over C+1 classes including background).
2. Bounding Box Regressor Head (Linear 4-delta offsets dx, dy, dw, dh).
"""

from typing import Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_rcnn_detector(
    input_shape: Tuple[int, int, int] = (32, 32, 3),
    num_classes: int = 4,  # 0: background, 1: circle, 2: rectangle, 3: triangle
    reg_weight: float = 1.0,
) -> keras.Model:
    """
    Build a compact multi-task CNN for proposal classification and bounding box delta regression.

    Args:
        input_shape: Resolution of warped ROI crops (H, W, C).
        num_classes: Total number of classes (including background at index 0).
        reg_weight: Loss weight multiplier for bbox regression head.

    Returns:
        Keras Model with inputs (32, 32, 3) and outputs [class_output, bbox_output].
    """
    inputs = keras.Input(shape=input_shape, name="roi_crop_input")

    # Shared Feature Extractor Backbone
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)  # -> 16x16

    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)  # -> 8x8

    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)  # -> 128

    # Shared dense embedding
    shared_feat = layers.Dense(64, activation="relu", name="shared_fc")(x)
    shared_feat = layers.Dropout(0.2, name="dropout")(shared_feat)

    # Head 1: Classification (Softmax over C+1 classes)
    class_output = layers.Dense(
        num_classes, activation="softmax", name="class_output"
    )(shared_feat)

    # Head 2: Bounding Box Regression (4 deltas: tx, ty, tw, th)
    bbox_output = layers.Dense(
        4, activation="linear", name="bbox_output"
    )(shared_feat)

    model = SimpleRCNNModel(
        inputs=inputs,
        outputs=[class_output, bbox_output],
        reg_weight=reg_weight,
        name="simple_rcnn_detector",
    )
    return model


class SimpleRCNNModel(keras.Model):
    """
    Keras Model subclass that implements the multi-task loss with foreground-only bbox regression.
    """

    def __init__(self, *args, reg_weight: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.reg_weight = reg_weight
        self.loss_tracker = keras.metrics.Mean(name="total_loss")
        self.cls_loss_tracker = keras.metrics.Mean(name="cls_loss")
        self.reg_loss_tracker = keras.metrics.Mean(name="reg_loss")
        self.cls_acc_tracker = keras.metrics.SparseCategoricalAccuracy(name="cls_acc")

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.cls_loss_tracker,
            self.reg_loss_tracker,
            self.cls_acc_tracker,
        ]

    def compute_custom_loss(
        self, y_true_cls, y_true_bbox, y_pred_cls, y_pred_bbox
    ):
        """
        Compute multi-task loss:
        - Classification loss (SparseCategoricalCrossentropy) on ALL proposals.
        - Bounding Box Regression loss (Smooth L1 / Huber) ONLY on positive proposals (label > 0).
        """
        # 1. Classification Loss (all proposals)
        cls_loss = keras.losses.sparse_categorical_crossentropy(
            y_true_cls, y_pred_cls
        )
        cls_loss = tf.reduce_mean(cls_loss)

        # 2. Regression Loss (Foreground samples only: label > 0)
        pos_mask = tf.cast(y_true_cls > 0, tf.float32)  # Shape (B,)
        num_pos = tf.reduce_sum(pos_mask)

        # Huber loss per sample
        huber = keras.losses.Huber(delta=1.0, reduction="none")
        reg_losses = huber(y_true_bbox, y_pred_bbox)  # Shape (B,)
        weighted_reg_losses = reg_losses * pos_mask

        reg_loss = tf.where(
            num_pos > 0.0,
            tf.reduce_sum(weighted_reg_losses) / (num_pos + 1e-8),
            0.0,
        )

        total_loss = cls_loss + self.reg_weight * reg_loss
        return total_loss, cls_loss, reg_loss

    def train_step(self, data):
        x, (y_true_cls, y_true_bbox) = data

        with tf.GradientTape() as tape:
            y_pred_cls, y_pred_bbox = self(x, training=True)
            total_loss, cls_loss, reg_loss = self.compute_custom_loss(
                y_true_cls, y_true_bbox, y_pred_cls, y_pred_bbox
            )

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.loss_tracker.update_state(total_loss)
        self.cls_loss_tracker.update_state(cls_loss)
        self.reg_loss_tracker.update_state(reg_loss)
        self.cls_acc_tracker.update_state(y_true_cls, y_pred_cls)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, (y_true_cls, y_true_bbox) = data
        y_pred_cls, y_pred_bbox = self(x, training=False)
        total_loss, cls_loss, reg_loss = self.compute_custom_loss(
            y_true_cls, y_true_bbox, y_pred_cls, y_pred_bbox
        )

        self.loss_tracker.update_state(total_loss)
        self.cls_loss_tracker.update_state(cls_loss)
        self.reg_loss_tracker.update_state(reg_loss)
        self.cls_acc_tracker.update_state(y_true_cls, y_pred_cls)

        return {m.name: m.result() for m in self.metrics}
