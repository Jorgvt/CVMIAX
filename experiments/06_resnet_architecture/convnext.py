"""
ConvNeXt Architecture in Keras (Liu et al., 2022: 'A ConvNet for the 2020s').

Modernizes pure convolutional networks adopting design choices from Vision Transformers:
1. 7x7 Depthwise Convolutions (larger receptive field per block).
2. Inverted Bottleneck design (4x channel expansion in 1x1 convolutions).
3. Layer Normalization (channels-last) instead of Batch Normalization.
4. GELU non-linear activations.
5. Fewer activation and normalization layers per block.
6. Separate downsampling layers (2x2 conv with stride 2).
"""

from typing import List, Tuple, Optional
import os
import keras
from keras import layers


def convnext_block(
    x: keras.KerasTensor,
    dim: int,
    name_prefix: str = "convnext_block",
) -> keras.KerasTensor:
    """
    Standard ConvNeXt Block.
    
    1. Depthwise Conv 7x7 (padding='same')
    2. Layer Normalization
    3. Pointwise Conv 1x1 (expands to 4 * dim)
    4. GELU Activation
    5. Pointwise Conv 1x1 (projects back to dim)
    6. Residual Shortcut Addition: y = x + F(x)
    """
    shortcut = x

    # 1. 7x7 Depthwise Convolution
    out = layers.DepthwiseConv2D(
        kernel_size=7,
        padding="same",
        depthwise_initializer="he_normal",
        name=f"{name_prefix}_dw_conv",
    )(x)

    # 2. Layer Normalization
    out = layers.LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_ln")(out)

    # 3. Inverted Bottleneck (1x1 Conv -> 4 * dim)
    out = layers.Conv2D(
        filters=4 * dim,
        kernel_size=1,
        padding="same",
        kernel_initializer="he_normal",
        name=f"{name_prefix}_pw_conv1",
    )(out)

    # 4. GELU Activation
    out = layers.Activation("gelu", name=f"{name_prefix}_gelu")(out)

    # 5. Pointwise Conv back to dim
    out = layers.Conv2D(
        filters=dim,
        kernel_size=1,
        padding="same",
        kernel_initializer="he_normal",
        name=f"{name_prefix}_pw_conv2",
    )(out)

    # 6. Residual Connection
    out = layers.Add(name=f"{name_prefix}_add")([shortcut, out])
    return out


def build_convnext(
    input_shape: Tuple[int, int, int] = (32, 32, 3),
    num_classes: int = 10,
    stage_blocks: List[int] = [2, 2, 4, 2],
    stage_dims: List[int] = [32, 64, 128, 256],
    name: str = "ConvNeXt-Mini",
) -> keras.Model:
    """
    Builds a ConvNeXt architecture adapted for image classification.
    """
    inputs = keras.Input(shape=input_shape, name="input_image")

    # Stem: 3x3 Conv with stride 1 (for 32x32 images) + LayerNorm
    x = layers.Conv2D(
        filters=stage_dims[0],
        kernel_size=3,
        strides=1,
        padding="same",
        kernel_initializer="he_normal",
        name="stem_conv",
    )(inputs)
    x = layers.LayerNormalization(epsilon=1e-6, name="stem_ln")(x)

    # Stages
    for stage_idx, (num_blocks, dim) in enumerate(zip(stage_blocks, stage_dims)):
        # Downsampling layer between stages (except before stage 0)
        if stage_idx > 0:
            x = layers.LayerNormalization(epsilon=1e-6, name=f"stage{stage_idx + 1}_downsample_ln")(x)
            x = layers.Conv2D(
                filters=dim,
                kernel_size=2,
                strides=2,
                padding="valid",
                kernel_initializer="he_normal",
                name=f"stage{stage_idx + 1}_downsample_conv",
            )(x)

        # ConvNeXt Blocks
        for block_idx in range(num_blocks):
            block_name = f"stage{stage_idx + 1}_b{block_idx + 1}"
            x = convnext_block(x, dim=dim, name_prefix=block_name)

    # Head: Global Average Pooling + LayerNorm + Dense
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.LayerNormalization(epsilon=1e-6, name="head_ln")(x)
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        kernel_initializer="he_normal",
        name="classifier_head",
    )(x)

    return keras.Model(inputs=inputs, outputs=outputs, name=name)


def build_convnext_cifar(input_shape=(32, 32, 3), num_classes=10) -> keras.Model:
    """Standard ConvNeXt-CIFAR model."""
    return build_convnext(
        input_shape=input_shape,
        num_classes=num_classes,
        stage_blocks=[2, 2, 4, 2],
        stage_dims=[32, 64, 128, 256],
        name="ConvNeXt-CIFAR",
    )


def save_convnext_weights(model: keras.Model, filepath: str) -> str:
    """Saves ConvNeXt model weights to disk."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    if filepath.endswith(".weights.h5"):
        model.save_weights(filepath)
    else:
        model.save(filepath)
    return os.path.abspath(filepath)


def load_convnext_weights(model: keras.Model, filepath: str) -> keras.Model:
    """Loads saved ConvNeXt model weights."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"ConvNeXt weight file not found: {filepath}")
    if filepath.endswith(".weights.h5"):
        model.load_weights(filepath)
    elif filepath.endswith(".keras"):
        loaded = keras.models.load_model(filepath)
        model.set_weights(loaded.get_weights())
    else:
        model.load_weights(filepath)
    return model
