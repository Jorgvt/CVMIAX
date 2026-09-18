"""Model Architectures and Weight Persistence for Time-Series Image Classification.

Includes:
1. Custom Lightweight ResNet-18 (built in pure Keras for pedagogical transparency)
2. Pretrained Transfer Learning Backbones (MobileNetV2 / ResNet50V2)
3. Weight saving and loading utilities for rapid reproduction and offline evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Tuple, Union
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def _residual_block(
    x: tf.Tensor,
    filters: int,
    stride: int = 1,
    name_prefix: str = "res",
) -> tf.Tensor:
    """Standard ResNet BasicBlock with skip connection."""
    shortcut = x

    # First convolution
    y = layers.Conv2D(
        filters,
        kernel_size=3,
        strides=stride,
        padding="same",
        use_bias=False,
        name=f"{name_prefix}_conv1",
    )(x)
    y = layers.BatchNormalization(name=f"{name_prefix}_bn1")(y)
    y = layers.ReLU(name=f"{name_prefix}_relu1")(y)

    # Second convolution
    y = layers.Conv2D(
        filters,
        kernel_size=3,
        strides=1,
        padding="same",
        use_bias=False,
        name=f"{name_prefix}_conv2",
    )(y)
    y = layers.BatchNormalization(name=f"{name_prefix}_bn2")(y)

    # Shortcut projection if dimensions change
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(
            filters,
            kernel_size=1,
            strides=stride,
            padding="same",
            use_bias=False,
            name=f"{name_prefix}_shortcut_conv",
        )(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name_prefix}_shortcut_bn")(shortcut)

    out = layers.Add(name=f"{name_prefix}_add")([y, shortcut])
    out = layers.ReLU(name=f"{name_prefix}_out_relu")(out)
    return out


def build_custom_resnet18(
    input_shape: Tuple[int, int, int] = (128, 128, 3),
    num_classes: int = 4,
    base_filters: int = 32,
    dropout_rate: float = 0.2,
) -> keras.Model:
    """Construct a lightweight, transparent ResNet-18 architecture in Keras."""
    inputs = layers.Input(shape=input_shape, name="input_image")

    # Initial Convolution & Max Pooling
    x = layers.Conv2D(
        base_filters,
        kernel_size=7,
        strides=2,
        padding="same",
        use_bias=False,
        name="conv1",
    )(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.ReLU(name="relu1")(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding="same", name="maxpool1")(x)

    # Stage 1 (2 blocks, 32 filters)
    x = _residual_block(x, base_filters, stride=1, name_prefix="stage1_b1")
    x = _residual_block(x, base_filters, stride=1, name_prefix="stage1_b2")

    # Stage 2 (2 blocks, 64 filters)
    x = _residual_block(x, base_filters * 2, stride=2, name_prefix="stage2_b1")
    x = _residual_block(x, base_filters * 2, stride=1, name_prefix="stage2_b2")

    # Stage 3 (2 blocks, 128 filters)
    x = _residual_block(x, base_filters * 4, stride=2, name_prefix="stage3_b1")
    x = _residual_block(x, base_filters * 4, stride=1, name_prefix="stage3_b2")

    # Stage 4 (2 blocks, 256 filters)
    x = _residual_block(x, base_filters * 8, stride=2, name_prefix="stage4_b1")
    x = _residual_block(x, base_filters * 8, stride=1, name_prefix="stage4_b2")

    # Global Average Pooling & Classification Head
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    if dropout_rate > 0.0:
        x = layers.Dropout(dropout_rate, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="regime_logits")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="Custom_ResNet18")
    return model


def build_pretrained_backbone(
    backbone_name: Literal["mobilenet_v2", "resnet50_v2"] = "mobilenet_v2",
    input_shape: Tuple[int, int, int] = (128, 128, 3),
    num_classes: int = 4,
    freeze_backbone: bool = False,
    dropout_rate: float = 0.3,
) -> keras.Model:
    """Build a transfer-learning model using pretrained ImageNet weights."""
    inputs = layers.Input(shape=input_shape, name="input_image")

    if backbone_name == "mobilenet_v2":
        x = keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)
        base = keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
        )
    elif backbone_name == "resnet50_v2":
        x = keras.applications.resnet_v2.preprocess_input(inputs * 255.0)
        base = keras.applications.ResNet50V2(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    base.trainable = not freeze_backbone
    features = base(x, training=not freeze_backbone)
    pooled = layers.GlobalAveragePooling2D(name="global_avg_pool")(features)
    if dropout_rate > 0.0:
        pooled = layers.Dropout(dropout_rate, name="head_dropout")(pooled)
    outputs = layers.Dense(num_classes, activation="softmax", name="regime_logits")(pooled)

    model = keras.Model(inputs=inputs, outputs=outputs, name=f"Pretrained_{backbone_name}")
    return model


def get_model(
    model_type: Literal["resnet18", "mobilenet_v2", "resnet50_v2"] = "resnet18",
    input_shape: Tuple[int, int, int] = (128, 128, 3),
    num_classes: int = 4,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """Model factory helper with standard compilation."""
    if model_type == "resnet18":
        model = build_custom_resnet18(input_shape=input_shape, num_classes=num_classes)
    else:
        model = build_pretrained_backbone(
            backbone_name=model_type,
            input_shape=input_shape,
            num_classes=num_classes,
            freeze_backbone=False,
        )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def save_model_weights(model: keras.Model, filepath: Union[str, Path]) -> Path:
    """Save model weights to disk."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Ensure extension is .weights.h5
    if not str(path).endswith(".weights.h5"):
        path = path.with_suffix(".weights.h5")
    model.save_weights(str(path))
    return path


def load_model_weights(model: keras.Model, filepath: Union[str, Path]) -> bool:
    """Load model weights if the file exists on disk.

    Returns True if weights were loaded successfully, False otherwise.
    """
    path = Path(filepath)
    if not str(path).endswith(".weights.h5") and not path.exists():
        path = path.with_suffix(".weights.h5")
    if path.exists():
        model.load_weights(str(path))
        return True
    return False
