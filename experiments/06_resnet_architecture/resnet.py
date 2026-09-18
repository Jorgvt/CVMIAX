"""
Deep Residual Network (ResNet) Architecture in Keras.

This module provides an isolated, clean implementation of:
1. Residual Building Block (He et al., 2015) with identity and projection shortcuts.
2. Complete ResNet architectures (ResNet-20 for CIFAR, ResNet-18).
3. Weight saving and loading utilities for ResNet.
"""

from typing import List, Tuple, Optional
import os
import keras
from keras import layers


def residual_block(
    x: keras.KerasTensor,
    filters: int,
    stride: int = 1,
    name_prefix: str = "res_block",
) -> keras.KerasTensor:
    """
    Standard Residual Building Block (He et al., 2015).
    
    Computes: y = ReLU(F(x) + shortcut(x))
    where F(x) = BN(Conv(ReLU(BN(Conv(x)))))
    
    Args:
        x: Input tensor.
        filters: Number of output filters.
        stride: Stride for the first convolutional layer (for spatial downsampling).
        name_prefix: Prefix for layer names.
        
    Returns:
        Output tensor of the residual block.
    """
    input_channels = x.shape[-1]
    shortcut = x

    # --- Residual Branch F(x) ---
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

    # --- Shortcut Branch (Identity or 1x1 Projection) ---
    # If spatial dimensions decrease (stride > 1) or channel depth changes,
    # apply a 1x1 convolution with matching stride (Projection Shortcut).
    if stride != 1 or input_channels != filters:
        shortcut = layers.Conv2D(
            filters=filters,
            kernel_size=1,
            strides=stride,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
            name=f"{name_prefix}_shortcut_conv",
        )(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name_prefix}_shortcut_bn")(shortcut)

    # --- Additive Fusion: H(x) = F(x) + shortcut(x) ---
    out = layers.Add(name=f"{name_prefix}_add")([out, shortcut])
    out = layers.ReLU(name=f"{name_prefix}_relu2")(out)
    return out


def build_resnet(
    input_shape: Tuple[int, int, int] = (32, 32, 3),
    num_classes: int = 10,
    stage_blocks: List[int] = [3, 3, 3],
    stage_filters: List[int] = [16, 32, 64],
    name: str = "ResNet-20",
) -> keras.Model:
    """
    Builds a flexible Residual Network (ResNet).
    
    Args:
        input_shape: Input image dimensions (H, W, C).
        num_classes: Number of target classification classes.
        stage_blocks: Number of residual blocks in each stage (e.g. [3, 3, 3] for ResNet-20).
        stage_filters: Filter count for each stage (e.g. [16, 32, 64]).
        name: Name of the model.
        
    Returns:
        Keras Model instance.
    """
    inputs = keras.Input(shape=input_shape, name="input_image")

    # Initial Stem: 3x3 conv + BN + ReLU (optimized for CIFAR-sized images)
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

    # Residual Stages
    for stage_idx, (num_blocks, num_filters) in enumerate(zip(stage_blocks, stage_filters)):
        for block_idx in range(num_blocks):
            # Downsample on the first block of stages > 0
            stride = 2 if (stage_idx > 0 and block_idx == 0) else 1
            block_name = f"stage{stage_idx + 1}_b{block_idx + 1}"
            x = residual_block(
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


def build_resnet20(input_shape=(32, 32, 3), num_classes=10) -> keras.Model:
    """Builds standard ResNet-20 for CIFAR (3 stages x 3 blocks = 6 convs/stage * 3 + 1 stem + 1 dense = 20 weight layers)."""
    return build_resnet(
        input_shape=input_shape,
        num_classes=num_classes,
        stage_blocks=[3, 3, 3],
        stage_filters=[16, 32, 64],
        name="ResNet-20",
    )


def build_resnet18(input_shape=(32, 32, 3), num_classes=10) -> keras.Model:
    """Builds 4-stage ResNet-18 adapted for small images (stage blocks: [2, 2, 2, 2], filters: [64, 128, 256, 512])."""
    return build_resnet(
        input_shape=input_shape,
        num_classes=num_classes,
        stage_blocks=[2, 2, 2, 2],
        stage_filters=[64, 128, 256, 512],
        name="ResNet-18",
    )


def save_resnet_weights(model: keras.Model, filepath: str) -> str:
    """Saves ResNet model weights (.weights.h5) or full model (.keras)."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    if filepath.endswith(".weights.h5"):
        model.save_weights(filepath)
    else:
        model.save(filepath)
    return os.path.abspath(filepath)


def load_resnet_weights(model: keras.Model, filepath: str) -> keras.Model:
    """Loads saved weights into a ResNet model."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"ResNet weight file not found: {filepath}")
    
    if filepath.endswith(".weights.h5"):
        model.load_weights(filepath)
    elif filepath.endswith(".keras"):
        loaded = keras.models.load_model(filepath)
        model.set_weights(loaded.get_weights())
    else:
        model.load_weights(filepath)
    return model
