"""Keras models for Full-Reference (FR-IQA) and No-Reference (NR-IQA) assessment.

Implements:
1. Full-Reference Siamese Multi-Scale Difference CNN:
   - Extracts hierarchical feature representations from reference and distorted images.
   - Computes multi-level perceptual difference maps Delta phi = |phi(I_ref) - phi(I_dist)|.
   - Regresses subjective Mean Opinion Score (MOS).
2. No-Reference (Blind) CNN:
   - Direct perceptual quality regression from the distorted image alone.
"""

from typing import Tuple
import keras
from keras import layers, Model, ops


def build_conv_block(
    filters: int,
    kernel_size: int = 3,
    name_prefix: str = "block",
) -> keras.Sequential:
    """Construct a standardized Conv-BN-ReLU feature extraction block."""
    return keras.Sequential(
        [
            layers.Conv2D(
                filters,
                kernel_size=kernel_size,
                padding="same",
                use_bias=False,
                name=f"{name_prefix}_conv",
            ),
            layers.BatchNormalization(name=f"{name_prefix}_bn"),
            layers.Activation("relu", name=f"{name_prefix}_relu"),
            layers.MaxPooling2D(pool_size=2, strides=2, name=f"{name_prefix}_pool"),
        ],
        name=name_prefix,
    )


def build_fr_iqa_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
) -> Model:
    """Build a Full-Reference Siamese Multi-Scale Difference IQA Model.

    Architecture:
        1. Shared hierarchical convolutional feature extractors (Stage 1: 32, Stage 2: 64, Stage 3: 128).
        2. Multi-stage perceptual difference computation:
           Delta_1 = |phi_1(ref) - phi_1(dist)|
           Delta_2 = |phi_2(ref) - phi_2(dist)|
           Delta_3 = |phi_3(ref) - phi_3(dist)|
        3. Global pooling across all feature difference stages.
        4. MLP regression head to predict scalar MOS score.

    Inputs:
        dict: {"reference": Tensor(B, H, W, 3), "distorted": Tensor(B, H, W, 3)}

    Outputs:
        Tensor(B, 1): Predicted MOS score.
    """
    ref_input = layers.Input(shape=input_shape, name="reference")
    dist_input = layers.Input(shape=input_shape, name="distorted")

    # Shared stage 1 (Low-level edge/texture differences)
    stage1 = build_conv_block(32, kernel_size=3, name_prefix="stage1")
    # Shared stage 2 (Mid-level structural distortions)
    stage2 = build_conv_block(64, kernel_size=3, name_prefix="stage2")
    # Shared stage 3 (High-level semantic & regional artifacts)
    stage3 = build_conv_block(128, kernel_size=3, name_prefix="stage3")

    # Forward reference through stages
    ref_f1 = stage1(ref_input)
    ref_f2 = stage2(ref_f1)
    ref_f3 = stage3(ref_f2)

    # Forward distorted through identical shared stages
    dist_f1 = stage1(dist_input)
    dist_f2 = stage2(dist_f1)
    dist_f3 = stage3(dist_f2)

    # Compute absolute multi-scale difference feature maps
    diff_f1 = layers.Lambda(
        lambda tensors: ops.abs(tensors[0] - tensors[1]),
        name="diff_stage1",
    )([ref_f1, dist_f1])

    diff_f2 = layers.Lambda(
        lambda tensors: ops.abs(tensors[0] - tensors[1]),
        name="diff_stage2",
    )([ref_f2, dist_f2])

    diff_f3 = layers.Lambda(
        lambda tensors: ops.abs(tensors[0] - tensors[1]),
        name="diff_stage3",
    )([ref_f3, dist_f3])

    # Global spatial pooling per stage
    pool_f1 = layers.GlobalAveragePooling2D(name="gap_diff1")(diff_f1)
    pool_f2 = layers.GlobalAveragePooling2D(name="gap_diff2")(diff_f2)
    pool_f3 = layers.GlobalAveragePooling2D(name="gap_diff3")(diff_f3)

    # Concatenate multi-scale difference descriptors
    merged_features = layers.Concatenate(name="merged_diff_features")(
        [pool_f1, pool_f2, pool_f3]
    )

    # Quality regression MLP head
    x = layers.Dense(128, activation="relu", name="fc1")(merged_features)
    x = layers.Dropout(0.2, name="dropout1")(x)
    x = layers.Dense(64, activation="relu", name="fc2")(x)
    x = layers.Dropout(0.1, name="dropout2")(x)
    output = layers.Dense(1, name="mos_pred")(x)

    model = Model(
        inputs={"reference": ref_input, "distorted": dist_input},
        outputs=output,
        name="FR_IQA_Siamese_CNN",
    )
    return model


def build_nr_iqa_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
) -> Model:
    """Build a No-Reference (Blind) IQA Model.

    Predicts perceptual quality MOS directly from the distorted image without
    access to a reference pristine image.

    Inputs:
        Tensor(B, H, W, 3): Distorted image.

    Outputs:
        Tensor(B, 1): Predicted MOS score.
    """
    img_input = layers.Input(shape=input_shape, name="distorted_input")

    stage1 = build_conv_block(32, kernel_size=3, name_prefix="nr_stage1")
    stage2 = build_conv_block(64, kernel_size=3, name_prefix="nr_stage2")
    stage3 = build_conv_block(128, kernel_size=3, name_prefix="nr_stage3")

    x = stage1(img_input)
    x = stage2(x)
    x = stage3(x)

    x = layers.GlobalAveragePooling2D(name="nr_gap")(x)
    x = layers.Dense(128, activation="relu", name="nr_fc1")(x)
    x = layers.Dropout(0.2, name="nr_dropout1")(x)
    x = layers.Dense(64, activation="relu", name="nr_fc2")(x)
    x = layers.Dropout(0.1, name="nr_dropout2")(x)
    output = layers.Dense(1, name="nr_mos_pred")(x)

    model = Model(inputs=img_input, outputs=output, name="NR_IQA_Blind_CNN")
    return model


def get_fr_spatial_diff_extractor(
    fr_model: Model,
) -> Model:
    """Extract intermediate spatial difference activation maps for visualization."""
    return Model(
        inputs=fr_model.inputs,
        outputs=[
            fr_model.get_layer("diff_stage1").output,
            fr_model.get_layer("diff_stage2").output,
            fr_model.get_layer("diff_stage3").output,
        ],
        name="FR_Spatial_Diff_Extractor",
    )
