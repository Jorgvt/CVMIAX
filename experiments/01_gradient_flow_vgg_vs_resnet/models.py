import keras
from keras import layers


def build_plain_vgg(
    input_shape=(32, 32, 3),
    num_classes=10,
    num_blocks_per_stage=[3, 3, 3],
    filters_per_stage=[32, 64, 128],
    use_batchnorm=False,
    name="plain_vgg",
):
    """
    Builds a plain sequential VGG-style CNN without residual connections.
    
    Args:
        input_shape: Shape of input image tensor.
        num_classes: Number of output classes.
        num_blocks_per_stage: List specifying how many Conv layers per resolution stage.
        filters_per_stage: List specifying channel width per resolution stage.
        use_batchnorm: Whether to include BatchNormalization.
        name: Model name.
    """
    inputs = keras.Input(shape=input_shape, name="input_image")
    x = inputs

    layer_idx = 0
    for stage_idx, (num_blocks, num_filters) in enumerate(
        zip(num_blocks_per_stage, filters_per_stage)
    ):
        for block_idx in range(num_blocks):
            layer_idx += 1
            x = layers.Conv2D(
                filters=num_filters,
                kernel_size=3,
                padding="same",
                kernel_initializer="he_normal",
                name=f"stage{stage_idx}_conv{block_idx}_layer{layer_idx}",
            )(x)
            if use_batchnorm:
                x = layers.BatchNormalization(name=f"stage{stage_idx}_bn{block_idx}_layer{layer_idx}")(x)
            x = layers.ReLU(name=f"stage{stage_idx}_relu{block_idx}_layer{layer_idx}")(x)

        # Downsample at the end of stage (except the last stage)
        if stage_idx < len(num_blocks_per_stage) - 1:
            x = layers.MaxPooling2D(pool_size=2, strides=2, name=f"stage{stage_idx}_pool")(x)

    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classifier_dense")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name=name)


def build_resnet(
    input_shape=(32, 32, 3),
    num_classes=10,
    num_blocks_per_stage=[3, 3, 3],
    filters_per_stage=[32, 64, 128],
    use_batchnorm=False,
    name="resnet",
):
    """
    Builds a ResNet model with identical layer counts and widths to Plain VGG,
    incorporating identity shortcut connections: F(x) + x.
    
    Args:
        input_shape: Shape of input image tensor.
        num_classes: Number of output classes.
        num_blocks_per_stage: List specifying how many residual blocks per resolution stage.
        filters_per_stage: List specifying channel width per resolution stage.
        use_batchnorm: Whether to include BatchNormalization.
        name: Model name.
    """
    inputs = keras.Input(shape=input_shape, name="input_image")
    x = inputs

    layer_idx = 0
    for stage_idx, (num_blocks, num_filters) in enumerate(
        zip(num_blocks_per_stage, filters_per_stage)
    ):
        for block_idx in range(num_blocks):
            layer_idx += 1
            shortcut = x

            # Main branch
            residual = layers.Conv2D(
                filters=num_filters,
                kernel_size=3,
                padding="same",
                kernel_initializer="he_normal",
                name=f"stage{stage_idx}_conv{block_idx}_layer{layer_idx}",
            )(x)
            if use_batchnorm:
                residual = layers.BatchNormalization(
                    name=f"stage{stage_idx}_bn{block_idx}_layer{layer_idx}"
                )(residual)

            # Match shortcut channel dimension if needed
            if shortcut.shape[-1] != num_filters:
                shortcut = layers.Conv2D(
                    filters=num_filters,
                    kernel_size=1,
                    padding="same",
                    kernel_initializer="he_normal",
                    name=f"stage{stage_idx}_shortcut_conv{block_idx}_layer{layer_idx}",
                )(shortcut)

            # Residual addition: x_{l+1} = ReLU(F(x_l) + x_l)
            x = layers.Add(name=f"stage{stage_idx}_add{block_idx}_layer{layer_idx}")(
                [residual, shortcut]
            )
            x = layers.ReLU(name=f"stage{stage_idx}_relu{block_idx}_layer{layer_idx}")(x)

        # Downsample at the end of stage (except the last stage)
        if stage_idx < len(num_blocks_per_stage) - 1:
            x = layers.MaxPooling2D(pool_size=2, strides=2, name=f"stage{stage_idx}_pool")(x)

    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classifier_dense")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name=name)


def create_model_pair(total_depth=24, input_shape=(32, 32, 3), num_classes=10, use_batchnorm=False):
    """
    Utility helper to instantiate a matched Plain CNN and ResNet with equal depth.
    Depth is distributed across 3 stages.
    """
    blocks_per_stage = total_depth // 3
    num_blocks = [blocks_per_stage, blocks_per_stage, total_depth - 2 * blocks_per_stage]
    filters = [32, 64, 128]

    plain_model = build_plain_vgg(
        input_shape=input_shape,
        num_classes=num_classes,
        num_blocks_per_stage=num_blocks,
        filters_per_stage=filters,
        use_batchnorm=use_batchnorm,
        name=f"Plain_VGG_Depth{total_depth}",
    )

    resnet_model = build_resnet(
        input_shape=input_shape,
        num_classes=num_classes,
        num_blocks_per_stage=num_blocks,
        filters_per_stage=filters,
        use_batchnorm=use_batchnorm,
        name=f"ResNet_Depth{total_depth}",
    )

    return plain_model, resnet_model
