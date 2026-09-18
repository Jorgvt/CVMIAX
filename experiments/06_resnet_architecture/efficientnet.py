"""
EfficientNet Architecture in Keras.

Implements the Mobile Inverted Bottleneck Convolution (MBConv) block (Tan & Le, 2019):
1. Inverted Bottleneck (1x1 expansion).
2. Depthwise Separable Convolution (3x3 depthwise).
3. Squeeze-and-Excitation (SE) Channel Attention.
4. Linear 1x1 projection back to target channel depth.
5. Residual shortcut when dimensions match.
"""

from typing import List, Tuple, Optional
import os
import keras
from keras import layers


def squeeze_and_excitation_block(
    x: keras.KerasTensor,
    in_channels: int,
    se_ratio: float = 0.25,
    name_prefix: str = "se",
) -> keras.KerasTensor:
    """
    Squeeze-and-Excitation (SE) channel-wise attention block.
    
    1. Squeeze: Global Average Pooling compresses (H, W, C) -> (1, 1, C).
    2. Excitation: Two 1x1 convolutions with Swish and Sigmoid produce per-channel weights.
    3. Scale: Element-wise multiply input activations with excitation weights.
    """
    reduced_channels = max(1, int(in_channels * se_ratio))
    
    se = layers.GlobalAveragePooling2D(keepdims=True, name=f"{name_prefix}_gap")(x)
    se = layers.Conv2D(
        filters=reduced_channels,
        kernel_size=1,
        padding="same",
        use_bias=True,
        activation="swish",
        name=f"{name_prefix}_reduce",
    )(se)
    se = layers.Conv2D(
        filters=in_channels,
        kernel_size=1,
        padding="same",
        use_bias=True,
        activation="sigmoid",
        name=f"{name_prefix}_expand",
    )(se)
    return layers.Multiply(name=f"{name_prefix}_scale")([x, se])


def mbconv_block(
    x: keras.KerasTensor,
    in_filters: int,
    out_filters: int,
    expand_ratio: int = 4,
    stride: int = 1,
    se_ratio: float = 0.25,
    name_prefix: str = "mbconv",
) -> keras.KerasTensor:
    """
    Mobile Inverted Bottleneck Convolution (MBConv) Block.
    
    Args:
        x: Input tensor.
        in_filters: Number of input channels.
        out_filters: Number of output channels.
        expand_ratio: Expansion factor for intermediate inverted bottleneck.
        stride: Stride for depthwise convolution (spatial downsampling).
        se_ratio: Squeeze-and-excitation channel reduction factor.
        name_prefix: Prefix for layer names.
    """
    shortcut = x
    expanded_filters = in_filters * expand_ratio

    # 1. Expansion Phase (1x1 Conv) - omitted if expand_ratio == 1
    if expand_ratio != 1:
        x = layers.Conv2D(
            filters=expanded_filters,
            kernel_size=1,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
            name=f"{name_prefix}_expand_conv",
        )(x)
        x = layers.BatchNormalization(name=f"{name_prefix}_expand_bn")(x)
        x = layers.Activation("swish", name=f"{name_prefix}_expand_swish")(x)

    # 2. Depthwise Convolution (3x3)
    x = layers.DepthwiseConv2D(
        kernel_size=3,
        strides=stride,
        padding="same",
        use_bias=False,
        depthwise_initializer="he_normal",
        name=f"{name_prefix}_dw_conv",
    )(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_dw_bn")(x)
    x = layers.Activation("swish", name=f"{name_prefix}_dw_swish")(x)

    # 3. Squeeze and Excitation
    if se_ratio > 0:
        x = squeeze_and_excitation_block(
            x,
            in_channels=expanded_filters,
            se_ratio=se_ratio,
            name_prefix=f"{name_prefix}_se",
        )

    # 4. Linear Projection Phase (1x1 Conv - NO non-linear activation)
    x = layers.Conv2D(
        filters=out_filters,
        kernel_size=1,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        name=f"{name_prefix}_project_conv",
    )(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_project_bn")(x)

    # 5. Residual Shortcut Connection (only when shape and stride match)
    if stride == 1 and in_filters == out_filters:
        x = layers.Add(name=f"{name_prefix}_add")([shortcut, x])

    return x


def build_efficientnet(
    input_shape: Tuple[int, int, int] = (32, 32, 3),
    num_classes: int = 10,
    stage_blocks: List[int] = [1, 2, 2, 3],
    stage_filters: List[int] = [16, 24, 40, 80],
    expand_ratios: List[int] = [1, 6, 6, 6],
    name: str = "EfficientNet-Mini",
) -> keras.Model:
    """
    Builds a compact EfficientNet adapted for small image classification.
    """
    inputs = keras.Input(shape=input_shape, name="input_image")

    # Initial Stem: 3x3 Conv + BN + Swish
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
    x = layers.Activation("swish", name="stem_swish")(x)

    in_f = stage_filters[0]
    for stage_idx, (num_blocks, out_f, exp_r) in enumerate(
        zip(stage_blocks, stage_filters, expand_ratios)
    ):
        for block_idx in range(num_blocks):
            stride = 2 if (stage_idx > 0 and block_idx == 0) else 1
            block_name = f"stage{stage_idx + 1}_b{block_idx + 1}"
            x = mbconv_block(
                x,
                in_filters=in_f,
                out_filters=out_f,
                expand_ratio=exp_r,
                stride=stride,
                se_ratio=0.25,
                name_prefix=block_name,
            )
            in_f = out_f

    # Head: 1x1 Conv Expansion + GAP + Dense
    x = layers.Conv2D(
        filters=out_f * 4,
        kernel_size=1,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        name="head_conv",
    )(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Activation("swish", name="head_swish")(x)

    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        kernel_initializer="he_normal",
        name="classifier_head",
    )(x)

    return keras.Model(inputs=inputs, outputs=outputs, name=name)


def build_efficientnet_cifar(input_shape=(32, 32, 3), num_classes=10) -> keras.Model:
    """Standard EfficientNet-CIFAR model."""
    return build_efficientnet(
        input_shape=input_shape,
        num_classes=num_classes,
        stage_blocks=[1, 2, 2, 3],
        stage_filters=[16, 24, 40, 80],
        expand_ratios=[1, 6, 6, 6],
        name="EfficientNet-CIFAR",
    )


def save_efficientnet_weights(model: keras.Model, filepath: str) -> str:
    """Saves EfficientNet weights to disk."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    if filepath.endswith(".weights.h5"):
        model.save_weights(filepath)
    else:
        model.save(filepath)
    return os.path.abspath(filepath)


def load_efficientnet_weights(model: keras.Model, filepath: str) -> keras.Model:
    """Loads saved EfficientNet weights."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"EfficientNet weight file not found: {filepath}")
    if filepath.endswith(".weights.h5"):
        model.load_weights(filepath)
    elif filepath.endswith(".keras"):
        loaded = keras.models.load_model(filepath)
        model.set_weights(loaded.get_weights())
    else:
        model.load_weights(filepath)
    return model
