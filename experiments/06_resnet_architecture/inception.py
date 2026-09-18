"""
Inception (GoogLeNet) Architecture in Keras (Szegedy et al., 2015).

Implements the multi-scale Inception block:
1. Multi-branch parallel receptive fields (1x1, 3x3, 5x5 / stacked 3x3, and 3x3 MaxPooling).
2. 1x1 Convolutions for dimensionality reduction (bottlenecks) before expensive spatial convs.
3. Depth concatenation of all parallel branches.
"""

from typing import List, Tuple, Optional
import os
import keras
from keras import layers


def inception_block(
    x: keras.KerasTensor,
    f1x1: int,
    f3x3_reduce: int,
    f3x3: int,
    f5x5_reduce: int,
    f5x5: int,
    pool_proj: int,
    name_prefix: str = "inception",
) -> keras.KerasTensor:
    """
    Standard Inception Building Block with 4 parallel branches.
    
    Args:
        x: Input tensor.
        f1x1: Number of filters for direct 1x1 conv branch.
        f3x3_reduce: Number of filters for 1x1 reduction before 3x3 conv.
        f3x3: Number of filters for 3x3 conv branch.
        f5x5_reduce: Number of filters for 1x1 reduction before 5x5 branch.
        f5x5: Number of filters for 5x5 branch (implemented as 2 stacked 3x3 convs).
        pool_proj: Number of filters for 1x1 projection after 3x3 max pooling.
        name_prefix: Prefix for layer names.
        
    Returns:
        Concatenated multi-scale feature tensor.
    """
    # Branch 1: Direct 1x1 Conv
    b1 = layers.Conv2D(
        filters=f1x1,
        kernel_size=1,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        name=f"{name_prefix}_b1_conv1x1",
    )(x)
    b1 = layers.BatchNormalization(name=f"{name_prefix}_b1_bn")(b1)
    b1 = layers.ReLU(name=f"{name_prefix}_b1_relu")(b1)

    # Branch 2: 1x1 Conv Reduction -> 3x3 Conv
    b2 = layers.Conv2D(
        filters=f3x3_reduce,
        kernel_size=1,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        name=f"{name_prefix}_b2_reduce",
    )(x)
    b2 = layers.BatchNormalization(name=f"{name_prefix}_b2_reduce_bn")(b2)
    b2 = layers.ReLU(name=f"{name_prefix}_b2_reduce_relu")(b2)

    b2 = layers.Conv2D(
        filters=f3x3,
        kernel_size=3,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        name=f"{name_prefix}_b2_conv3x3",
    )(b2)
    b2 = layers.BatchNormalization(name=f"{name_prefix}_b2_bn")(b2)
    b2 = layers.ReLU(name=f"{name_prefix}_b2_relu")(b2)

    # Branch 3: 1x1 Conv Reduction -> 5x5 Conv (factorized into two 3x3 convs)
    b3 = layers.Conv2D(
        filters=f5x5_reduce,
        kernel_size=1,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        name=f"{name_prefix}_b3_reduce",
    )(x)
    b3 = layers.BatchNormalization(name=f"{name_prefix}_b3_reduce_bn")(b3)
    b3 = layers.ReLU(name=f"{name_prefix}_b3_reduce_relu")(b3)

    b3 = layers.Conv2D(
        filters=f5x5,
        kernel_size=3,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        name=f"{name_prefix}_b3_conv3x3_1",
    )(b3)
    b3 = layers.BatchNormalization(name=f"{name_prefix}_b3_bn1")(b3)
    b3 = layers.ReLU(name=f"{name_prefix}_b3_relu1")(b3)

    b3 = layers.Conv2D(
        filters=f5x5,
        kernel_size=3,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        name=f"{name_prefix}_b3_conv3x3_2",
    )(b3)
    b3 = layers.BatchNormalization(name=f"{name_prefix}_b3_bn2")(b3)
    b3 = layers.ReLU(name=f"{name_prefix}_b3_relu2")(b3)

    # Branch 4: 3x3 Max Pooling -> 1x1 Conv Projection
    b4 = layers.MaxPooling2D(
        pool_size=3,
        strides=1,
        padding="same",
        name=f"{name_prefix}_b4_pool",
    )(x)
    b4 = layers.Conv2D(
        filters=pool_proj,
        kernel_size=1,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        name=f"{name_prefix}_b4_proj",
    )(b4)
    b4 = layers.BatchNormalization(name=f"{name_prefix}_b4_proj_bn")(b4)
    b4 = layers.ReLU(name=f"{name_prefix}_b4_proj_relu")(b4)

    # Concatenate all branch channels
    out = layers.Concatenate(axis=-1, name=f"{name_prefix}_concat")([b1, b2, b3, b4])
    return out


def build_inception(
    input_shape: Tuple[int, int, int] = (32, 32, 3),
    num_classes: int = 10,
    name: str = "Inception-Mini",
) -> keras.Model:
    """
    Builds a compact Inception network adapted for CIFAR images.
    """
    inputs = keras.Input(shape=input_shape, name="input_image")

    # Initial Stem: 3x3 Conv + BN + ReLU
    x = layers.Conv2D(
        filters=32,
        kernel_size=3,
        strides=1,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        name="stem_conv",
    )(inputs)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)

    # Stage 1: Inception 3a, 3b + Downsample
    x = inception_block(x, f1x1=16, f3x3_reduce=16, f3x3=32, f5x5_reduce=8, f5x5=16, pool_proj=16, name_prefix="inc_3a")
    x = inception_block(x, f1x1=32, f3x3_reduce=32, f3x3=48, f5x5_reduce=16, f5x5=24, pool_proj=16, name_prefix="inc_3b")
    x = layers.MaxPooling2D(pool_size=2, strides=2, name="pool_stage1")(x)

    # Stage 2: Inception 4a, 4b + Downsample
    x = inception_block(x, f1x1=48, f3x3_reduce=48, f3x3=64, f5x5_reduce=16, f5x5=32, pool_proj=32, name_prefix="inc_4a")
    x = inception_block(x, f1x1=64, f3x3_reduce=48, f3x3=64, f5x5_reduce=16, f5x5=32, pool_proj=32, name_prefix="inc_4b")
    x = layers.MaxPooling2D(pool_size=2, strides=2, name="pool_stage2")(x)

    # Head: Global Average Pooling + Dropout + Dense
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dropout(0.2, name="head_dropout")(x)
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        kernel_initializer="he_normal",
        name="classifier_head",
    )(x)

    return keras.Model(inputs=inputs, outputs=outputs, name=name)


def build_inception_cifar(input_shape=(32, 32, 3), num_classes=10) -> keras.Model:
    """Standard Inception-CIFAR model."""
    return build_inception(
        input_shape=input_shape,
        num_classes=num_classes,
        name="Inception-CIFAR",
    )


def save_inception_weights(model: keras.Model, filepath: str) -> str:
    """Saves Inception model weights to disk."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    if filepath.endswith(".weights.h5"):
        model.save_weights(filepath)
    else:
        model.save(filepath)
    return os.path.abspath(filepath)


def load_inception_weights(model: keras.Model, filepath: str) -> keras.Model:
    """Loads saved Inception model weights."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Inception weight file not found: {filepath}")
    if filepath.endswith(".weights.h5"):
        model.load_weights(filepath)
    elif filepath.endswith(".keras"):
        loaded = keras.models.load_model(filepath)
        model.set_weights(loaded.get_weights())
    else:
        model.load_weights(filepath)
    return model
