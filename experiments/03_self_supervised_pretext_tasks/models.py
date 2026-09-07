"""
Keras Architectures for Self-Supervised Pretext Tasks.

Provides clean model definitions for:
1. RotationClassifier: 4-way classification (0°, 90°, 180°, 270°)
2. JigsawSolver: Shared multi-tower Siamese CNN for tile permutation prediction
3. ColorizationUNet: Encoder-Decoder architecture predicting RGB from Grayscale
"""

import keras
from keras import layers


def build_rotation_model(input_shape=(32, 32, 3), num_classes=4):
    """
    Standard CNN backbone for 4-way rotation classification.
    """
    inputs = keras.Input(shape=input_shape, name="rotated_image_input")
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D(name="representation_features")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="rotation_prediction")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="rotation_prediction_model")


def build_jigsaw_solver(tile_shape=(16, 16, 3), num_tiles=4, num_permutations=8):
    """
    Siamese multi-tower architecture for solving Jigsaw puzzles.
    A single shared CNN processes each tile independently, then their feature
    embeddings are concatenated to predict the permutation class.
    """
    # 1. Shared feature extractor tower
    tile_input = keras.Input(shape=tile_shape, name="single_tile_input")
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(tile_input)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    shared_tower = keras.Model(inputs=tile_input, outputs=x, name="shared_tile_tower")

    # 2. Multi-input puzzle model
    puzzle_inputs = [
        keras.Input(shape=tile_shape, name=f"tile_input_{i}") for i in range(num_tiles)
    ]
    tile_embeddings = [shared_tower(inp) for inp in puzzle_inputs]

    # 3. Permutation classification head
    concat_features = layers.Concatenate(name="combined_tile_features")(tile_embeddings)
    h = layers.Dense(128, activation="relu")(concat_features)
    h = layers.Dropout(0.3)(h)
    outputs = layers.Dense(num_permutations, activation="softmax", name="permutation_prediction")(h)

    return keras.Model(inputs=puzzle_inputs, outputs=outputs, name="jigsaw_solver_model")


def build_colorization_unet(input_shape=(64, 64, 1), output_channels=3):
    """
    Lightweight Encoder-Decoder architecture for image colorization.
    Input: Grayscale luminance (1 channel)
    Output: Reconstructed full color RGB (3 channels) with sigmoid activation [0, 1].
    """
    inputs = keras.Input(shape=input_shape, name="grayscale_input")

    # Encoder
    e1 = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    e1 = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(e1)
    p1 = layers.MaxPooling2D((2, 2))(e1)

    e2 = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(p1)
    e2 = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(e2)
    p2 = layers.MaxPooling2D((2, 2))(e2)

    # Bottleneck
    b = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(p2)
    b = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(b)

    # Decoder with Skip Connections
    u2 = layers.UpSampling2D((2, 2))(b)
    c2 = layers.Concatenate()([u2, e2])
    d2 = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(c2)

    u1 = layers.UpSampling2D((2, 2))(d2)
    c1 = layers.Concatenate()([u1, e1])
    d1 = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(c1)

    outputs = layers.Conv2D(output_channels, (1, 1), padding="same", activation="sigmoid", name="color_output")(d1)

    return keras.Model(inputs=inputs, outputs=outputs, name="colorization_model")
