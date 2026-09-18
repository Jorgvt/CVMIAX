"""
Modular Keras Model Architectures for Semantic Segmentation:
Plain Fully Convolutional Network (FCN / Encoder-Decoder) vs U-Net (with Skip Connections).

Pedagogical Core Idea:
Both models share the EXACT SAME encoder backbone, receptive field, and decoder depth.
The single structural difference is whether the decoder features are concatenated with
high-resolution spatial feature maps from the encoder (U-Net) or reconstructed solely
from the low-resolution bottleneck (Plain FCN).
"""

from typing import Tuple
import tensorflow as tf
from tensorflow import keras
from keras import layers


def conv_block(
    x: tf.Tensor,
    filters: int,
    kernel_size: int = 3,
    use_batch_norm: bool = True,
    name_prefix: str = "conv_block",
) -> tf.Tensor:
    """
    Standard double convolution block: [Conv2D -> BN -> ReLU] x 2.
    """
    # First Conv Layer
    x = layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        kernel_initializer="he_normal",
        name=f"{name_prefix}_conv1",
    )(x)
    if use_batch_norm:
        x = layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
    x = layers.Activation("relu", name=f"{name_prefix}_relu1")(x)

    # Second Conv Layer
    x = layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        kernel_initializer="he_normal",
        name=f"{name_prefix}_conv2",
    )(x)
    if use_batch_norm:
        x = layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)
    x = layers.Activation("relu", name=f"{name_prefix}_relu2")(x)

    return x


def build_segmentation_model(
    input_shape: Tuple[int, int, int] = (128, 128, 3),
    num_classes: int = 4,
    use_skip_connections: bool = True,
    base_filters: int = 32,
    use_batch_norm: bool = True,
    name: str = "segmentation_model",
) -> keras.Model:
    """
    Constructs a semantic segmentation model.

    Args:
        input_shape: Input image dimensions (H, W, C).
        num_classes: Number of categorical output segmentation classes.
        use_skip_connections:
            If True -> U-Net architecture (concatenates encoder skips).
            If False -> Plain FCN / Encoder-Decoder (bottleneck only).
        base_filters: Number of filters in the initial stage (doubled at each level).
        use_batch_norm: Whether to apply BatchNormalization.
        name: Name of the Keras model.

    Returns:
        keras.Model ready for compilation and training.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # =========================================================================
    # ENCODER (Downsampling Path: Contracting Path)
    # =========================================================================
    # Stage 1: 128x128 -> 64x64
    f1 = base_filters
    e1 = conv_block(inputs, f1, use_batch_norm=use_batch_norm, name_prefix="enc1")
    p1 = layers.MaxPooling2D(pool_size=(2, 2), name="enc1_pool")(e1)

    # Stage 2: 64x64 -> 32x32
    f2 = base_filters * 2
    e2 = conv_block(p1, f2, use_batch_norm=use_batch_norm, name_prefix="enc2")
    p2 = layers.MaxPooling2D(pool_size=(2, 2), name="enc2_pool")(e2)

    # Stage 3: 32x32 -> 16x16
    f3 = base_filters * 4
    e3 = conv_block(p2, f3, use_batch_norm=use_batch_norm, name_prefix="enc3")
    p3 = layers.MaxPooling2D(pool_size=(2, 2), name="enc3_pool")(e3)

    # =========================================================================
    # BOTTLENECK
    # =========================================================================
    # 16x16 -> 16x16
    f_b = base_filters * 8
    b = conv_block(p3, f_b, use_batch_norm=use_batch_norm, name_prefix="bottleneck")

    # =========================================================================
    # DECODER (Upsampling Path: Expanding Path)
    # =========================================================================
    # Stage 3 Up: 16x16 -> 32x32
    d3_up = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="dec3_upsample")(b)
    if use_skip_connections:
        d3_in = layers.Concatenate(axis=-1, name="dec3_skip_concat")([d3_up, e3])
    else:
        d3_in = d3_up
    d3 = conv_block(d3_in, f3, use_batch_norm=use_batch_norm, name_prefix="dec3")

    # Stage 2 Up: 32x32 -> 64x64
    d2_up = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="dec2_upsample")(d3)
    if use_skip_connections:
        d2_in = layers.Concatenate(axis=-1, name="dec2_skip_concat")([d2_up, e2])
    else:
        d2_in = d2_up
    d2 = conv_block(d2_in, f2, use_batch_norm=use_batch_norm, name_prefix="dec2")

    # Stage 1 Up: 64x64 -> 128x128
    d1_up = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="dec1_upsample")(d2)
    if use_skip_connections:
        d1_in = layers.Concatenate(axis=-1, name="dec1_skip_concat")([d1_up, e1])
    else:
        d1_in = d1_up
    d1 = conv_block(d1_in, f1, use_batch_norm=use_batch_norm, name_prefix="dec1")

    # =========================================================================
    # OUTPUT HEAD
    # =========================================================================
    outputs = layers.Conv2D(
        filters=num_classes,
        kernel_size=(1, 1),
        padding="same",
        activation="softmax",
        dtype="float32",
        name="segmentation_output",
    )(d1)

    model = keras.Model(inputs=inputs, outputs=outputs, name=name)
    return model


def build_plain_fcn(
    input_shape: Tuple[int, int, int] = (128, 128, 3),
    num_classes: int = 4,
    base_filters: int = 32,
    use_batch_norm: bool = True,
) -> keras.Model:
    """Builds Plain FCN / Encoder-Decoder without skip connections."""
    return build_segmentation_model(
        input_shape=input_shape,
        num_classes=num_classes,
        use_skip_connections=False,
        base_filters=base_filters,
        use_batch_norm=use_batch_norm,
        name="Plain_FCN_No_Skips",
    )


def build_unet(
    input_shape: Tuple[int, int, int] = (128, 128, 3),
    num_classes: int = 4,
    base_filters: int = 32,
    use_batch_norm: bool = True,
) -> keras.Model:
    """Builds U-Net architecture with skip connections."""
    return build_segmentation_model(
        input_shape=input_shape,
        num_classes=num_classes,
        use_skip_connections=True,
        base_filters=base_filters,
        use_batch_norm=use_batch_norm,
        name="UNet_With_Skips",
    )


if __name__ == "__main__":
    plain_fcn = build_plain_fcn()
    unet = build_unet()
    print("=== Plain FCN (No Skips) ===")
    print(f"Total params: {plain_fcn.count_params():,}")
    print("\n=== U-Net (With Skips) ===")
    print(f"Total params: {unet.count_params():,}")
