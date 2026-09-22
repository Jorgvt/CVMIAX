"""
Model architectures and transfer learning building blocks for Experiment 16.

Implements MobileNetV2 transfer learning pipelines with modular fine-tuning
depth configurations and optional integrated augmentation layers.
"""

from typing import Tuple, Optional
import keras
from keras import layers
from augmentations import get_augmentation_model


def build_transfer_learning_model(
    input_shape: Tuple[int, int, int] = (96, 96, 3),
    num_classes: int = 5,
    aug_policy: str = "none",
    unfreeze_top_layers: int = 20,
    dropout_rate: float = 0.3,
    l2_reg: float = 1e-4,
    learning_rate: float = 1e-4,
) -> keras.Model:
    """
    Builds a Transfer Learning model using an ImageNet pre-trained MobileNetV2 backbone.

    Args:
        input_shape: Input image dimensions (H, W, C).
        num_classes: Number of target classification classes.
        aug_policy: Augmentation policy ('none', 'geometric', 'photometric', 'combined').
        unfreeze_top_layers: Number of top backbone layers to unfreeze for fine-tuning.
            If 0, backbone is completely frozen (pure linear probe / feature extractor).
        dropout_rate: Dropout probability before the classification head.
        l2_reg: Weight decay parameter for dense head kernel.
        learning_rate: Initial learning rate for Adam optimizer.

    Returns:
        Compiled Keras Model ready for training and evaluation.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # 1. Data Augmentation Stage (active during training=True only)
    aug_block = get_augmentation_model(aug_policy)
    x = aug_block(inputs)

    # 2. Input Scaling to MobileNetV2 range [-1, 1] from [0, 1]
    x = layers.Rescaling(scale=2.0, offset=-1.0, name="rescale_to_mobilenet")(x)

    # 3. Pre-trained Feature Extractor Backbone
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )

    # Configure fine-tuning depth
    if unfreeze_top_layers > 0:
        base_model.trainable = True
        # Freeze all layers except the top N layers
        for layer in base_model.layers[:-unfreeze_top_layers]:
            layer.trainable = False
        print(
            f" Backbone fine-tuning active: {unfreeze_top_layers} top layers trainable out of {len(base_model.layers)}."
        )
    else:
        base_model.trainable = False
        print(" Backbone fully frozen (feature extraction mode).")

    x = base_model(x)

    # 4. Classification Head
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="head_batch_norm")(x)
    if dropout_rate > 0.0:
        x = layers.Dropout(dropout_rate, name="head_dropout")(x)

    regularizer = keras.regularizers.l2(l2_reg) if l2_reg > 0 else None
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        kernel_regularizer=regularizer,
        name="classification_head",
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=f"mobilenetv2_{aug_policy}")

    # Compile with Categorical Crossentropy and Adam optimizer
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.0),
        metrics=["accuracy"],
    )

    return model
