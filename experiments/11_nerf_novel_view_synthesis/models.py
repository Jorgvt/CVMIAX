"""
NeRF Architecture with Positional Encoding and Radiance Field MLP in Keras.

Implements high-frequency Fourier positional embeddings (Rahaman et al., Tancik et al.)
and compact coordinate-based MLPs predicting volume density and RGB emission.
"""

from typing import Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class PositionalEncoding(layers.Layer):
    """
    Fourier Positional Encoding layer for coordinate-based neural representations.

    Maps low-dimensional coordinates p into a higher-dimensional hypersphere of frequencies:
        gamma(p) = [p, sin(2^0 * pi * p), cos(2^0 * pi * p), ..., sin(2^{L-1} * pi * p), cos(2^{L-1} * pi * p)]

    Mitigates the spectral bias of deep MLPs (Neural Tangent Kernel behavior), enabling
    the network to fit high-frequency geometry and sharp texture details.
    """

    def __init__(
        self,
        num_freqs: int = 6,
        include_input: bool = True,
        max_freq_log2: int = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.max_freq_log2 = (num_freqs - 1) if max_freq_log2 is None else max_freq_log2

    def build(self, input_shape):
        if self.num_freqs > 0:
            freq_bands = 2.0 ** tf.linspace(0.0, float(self.max_freq_log2), self.num_freqs)
            # Store as non-trainable constant
            self.freq_bands = tf.constant(freq_bands, dtype=tf.float32)
        else:
            self.freq_bands = None
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Args:
            inputs: Tensor of shape [..., D] containing spatial coordinates.

        Returns:
            Encoded tensor of shape [..., D * (2 * num_freqs + (1 if include_input else 0))].
        """
        if self.num_freqs <= 0:
            return inputs

        outputs = [inputs] if self.include_input else []
        for freq in tf.unstack(self.freq_bands):
            outputs.append(tf.sin(inputs * freq * tf.constant(3.141592653589793, dtype=tf.float32)))
            outputs.append(tf.cos(inputs * freq * tf.constant(3.141592653589793, dtype=tf.float32)))

        return tf.concat(outputs, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_freqs": self.num_freqs,
            "include_input": self.include_input,
            "max_freq_log2": self.max_freq_log2,
        })
        return config


def build_tiny_nerf(
    num_freqs: int = 6,
    hidden_dim: int = 128,
    num_layers: int = 4,
    include_skips: bool = False,
) -> keras.Model:
    """
    Build a compact, pedagogical Tiny NeRF MLP in Keras.

    Predicts volume density sigma >= 0 (via ReLU) and RGB emission c in [0, 1] (via Sigmoid)
    from continuous 3D spatial coordinates x in R^3.

    Args:
        num_freqs: Number of octave frequencies in positional encoding (0 disables PE).
        hidden_dim: Number of hidden units per dense layer.
        num_layers: Total number of dense hidden layers.
        include_skips: Whether to inject coordinate embeddings into intermediate layers.

    Returns:
        Compiled or uncompiled Keras Model: inputs [..., 3] -> outputs [..., 4] (RGB + Sigma).
    """
    inputs = layers.Input(shape=(3,), name="input_xyz")
    
    # 1. High-frequency Fourier Positional Encoding
    if num_freqs > 0:
        x_enc = PositionalEncoding(num_freqs=num_freqs, name="positional_encoding")(inputs)
    else:
        x_enc = inputs

    # 2. Fully-Connected Radiance & Density Backbone
    x = x_enc
    skip_layer_idx = num_layers // 2

    for i in range(num_layers):
        x = layers.Dense(hidden_dim, activation="relu", name=f"dense_{i+1}")(x)
        if include_skips and i == skip_layer_idx:
            x = layers.Concatenate(name=f"skip_concat_{i+1}")([x, x_enc])

    # 3. Output Emission & Density Heads
    # Output 4 channels: [RGB (3 channels), Density sigma (1 channel)]
    # We use a combined dense layer or separate heads:
    rgb = layers.Dense(3, activation="sigmoid", name="rgb_head")(x)
    sigma = layers.Dense(1, activation="relu", name="density_head")(x)

    outputs = layers.Concatenate(name="rgb_sigma_out")([rgb, sigma])
    return keras.Model(inputs=inputs, outputs=outputs, name=f"TinyNeRF_L{num_freqs}_D{hidden_dim}")
