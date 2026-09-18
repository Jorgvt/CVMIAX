"""
Traditional Fully Forward Convolutional Neural Network (Plain CNN) Architecture in Keras.

This module provides an isolated implementation of:
1. Plain Convolutional Block without shortcut connections.
2. Matched Plain CNN architectures (Plain-20, Plain-18) with identical layer depths and widths to ResNet.
3. Weight saving and loading utilities for Plain CNN.
"""

from typing import List, Tuple, Optional
import os
import keras
from keras import layers


def plain_block(
    x: keras.KerasTensor,
    filters: int,
    stride: int = 1,
    name_prefix: str = "plain_block",
) -> keras.KerasTensor:
    """
    Matched Plain Forward Block without skip/shortcut connection.
    
    Computes: y = ReLU(F(x))
    where F(x) = BN(Conv(ReLU(BN(Conv(x)))))
    
    Maintains identical convolutional layers and parameter count to a residual block,
    but forces signals to flow strictly sequentially without skip connections.
    
    Args:
        x: Input tensor.
        filters: Number of output filters.
        stride: Stride for the first convolutional layer.
        name_prefix: Prefix for layer names.
        
    Returns:
        Output tensor of the plain block.
    """
    # 1st Conv: 3x3 convolution with stride
    out = layers.Conv2D(
        filters=filters,
        kernel_size=3,
        strides=stride,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        name=f"{name_prefix}_conv1",
    )(x)
    out = layers.BatchNormalization(name=f"{name_prefix}_bn1")(out)
    out = layers.ReLU(name=f"{name_prefix}_relu1")(out)

    # 2nd Conv: 3x3 convolution with stride 1
    out = layers.Conv2D(
        filters=filters,
        kernel_size=3,
        strides=1,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        name=f"{name_prefix}_conv2",
    )(out)
    out = layers.BatchNormalization(name=f"{name_prefix}_bn2")(out)

    # Strictly feedforward activation (no shortcut addition!)
    out = layers.ReLU(name=f"{name_prefix}_relu2")(out)
    return out


def build_plain_cnn(
    input_shape: Tuple[int, int, int] = (32, 32, 3),
    num_classes: int = 10,
    stage_blocks: List[int] = [3, 3, 3],
    stage_filters: List[int] = [16, 32, 64],
    name: str = "Plain-20",
) -> keras.Model:
    """
    Builds a Plain Forward CNN strictly matched to the ResNet architecture.
    
    Has the identical depth, filter channels, and pooling/striding schedule,
    but omits all residual/skip connections.
    """
    inputs = keras.Input(shape=input_shape, name="input_image")

    # Initial Stem
    x = layers.Conv2D(
        filters=stage_filters[0],
        kernel_size=3,
        strides=1,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        name="stem_conv",
    )(inputs)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)

    # Plain Stages
    for stage_idx, (num_blocks, num_filters) in enumerate(zip(stage_blocks, stage_filters)):
        for block_idx in range(num_blocks):
            stride = 2 if (stage_idx > 0 and block_idx == 0) else 1
            block_name = f"stage{stage_idx + 1}_b{block_idx + 1}"
            x = plain_block(
                x,
                filters=num_filters,
                stride=stride,
                name_prefix=block_name,
            )

    # Classification Head
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        kernel_initializer="he_normal",
        name="classifier_head",
    )(x)

    return keras.Model(inputs=inputs, outputs=outputs, name=name)


def build_plain20(input_shape=(32, 32, 3), num_classes=10) -> keras.Model:
    """Builds standard Plain-20 CNN matched to ResNet-20 without residual shortcuts."""
    return build_plain_cnn(
        input_shape=input_shape,
        num_classes=num_classes,
        stage_blocks=[3, 3, 3],
        stage_filters=[16, 32, 64],
        name="Plain-20",
    )


def build_plain18(input_shape=(32, 32, 3), num_classes=10) -> keras.Model:
    """Builds matched 4-stage Plain-18 CNN without residual shortcuts."""
    return build_plain_cnn(
        input_shape=input_shape,
        num_classes=num_classes,
        stage_blocks=[2, 2, 2, 2],
        stage_filters=[64, 128, 256, 512],
        name="Plain-18",
    )


def save_plain_weights(model: keras.Model, filepath: str) -> str:
    """Saves Plain CNN model weights (.weights.h5) or full model (.keras)."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    if filepath.endswith(".weights.h5"):
        model.save_weights(filepath)
    else:
        model.save(filepath)
    return os.path.abspath(filepath)


def load_plain_weights(model: keras.Model, filepath: str) -> keras.Model:
    """Loads saved weights into a Plain CNN model."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Plain CNN weight file not found: {filepath}")
    
    if filepath.endswith(".weights.h5"):
        model.load_weights(filepath)
    elif filepath.endswith(".keras"):
        loaded = keras.models.load_model(filepath)
        model.set_weights(loaded.get_weights())
    else:
        model.load_weights(filepath)
    return model
