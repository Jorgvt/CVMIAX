"""Dual-Encoder Vision and Text models for CLIP contrastive learning in Keras 3."""

from typing import Tuple
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class PositionalEmbedding(layers.Layer):
    """Adds learnable positional embeddings to token representations."""

    def __init__(self, seq_length: int, hidden_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.seq_length = seq_length
        self.pos_emb = layers.Embedding(input_dim=seq_length, output_dim=hidden_dim)

    def call(self, x):
        positions = tf.range(start=0, limit=self.seq_length, delta=1)
        return x + self.pos_emb(positions)


def build_vision_encoder(
    input_shape: Tuple[int, int, int] = (64, 64, 3),
    embedding_dim: int = 64,
    name: str = "vision_encoder",
) -> keras.Model:
    """Build a lightweight convolutional vision encoder with UnitNormalization projection."""
    inputs = layers.Input(shape=input_shape, name="image_input")

    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)  # 32x32

    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)  # 16x16

    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)  # 128

    # Multimodal projection head
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(embedding_dim, activation=None, name="image_projection")(x)
    # L2 unit normalization for cosine similarity
    normalized_outputs = layers.UnitNormalization(axis=-1, name="image_l2_norm")(outputs)

    model = keras.Model(inputs=inputs, outputs=normalized_outputs, name=name)
    return model


def build_text_encoder(
    vocab_size: int = 100,
    seq_length: int = 8,
    embedding_dim: int = 64,
    hidden_dim: int = 64,
    name: str = "text_encoder",
) -> keras.Model:
    """Build a lightweight Transformer-based text encoder with UnitNormalization projection."""
    inputs = layers.Input(shape=(seq_length,), dtype="int32", name="text_input")

    # Token and Positional Embeddings
    token_emb = layers.Embedding(input_dim=vocab_size, output_dim=hidden_dim, mask_zero=False)(inputs)
    x = PositionalEmbedding(seq_length=seq_length, hidden_dim=hidden_dim)(token_emb)

    # Mini Transformer Encoder Block
    attn_out = layers.MultiHeadAttention(num_heads=2, key_dim=hidden_dim // 2)(x, x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn_out)

    ffn = keras.Sequential([
        layers.Dense(hidden_dim * 2, activation="relu"),
        layers.Dense(hidden_dim),
    ])
    x = layers.LayerNormalization(epsilon=1e-6)(x + ffn(x))

    # Global representation pooling
    x = layers.GlobalAveragePooling1D()(x)

    # Multimodal projection head
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(embedding_dim, activation=None, name="text_projection")(x)
    # L2 unit normalization for cosine similarity
    normalized_outputs = layers.UnitNormalization(axis=-1, name="text_l2_norm")(outputs)

    model = keras.Model(inputs=inputs, outputs=normalized_outputs, name=name)
    return model


class CLIPDualEncoder(keras.Model):
    """End-to-End CLIP model implementing dual encoders with learnable temperature scaling."""

    def __init__(
        self,
        vision_encoder: keras.Model,
        text_encoder: keras.Model,
        init_temperature: float = 0.07,
        max_temperature_scale: float = 100.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.max_temperature_scale = max_temperature_scale

        # Learnable logit scale initialized to log(1 / init_temperature)
        init_logit_scale = float(np.log(1.0 / init_temperature))
        self.logit_scale = self.add_weight(
            name="logit_scale",
            shape=(),
            initializer=tf.constant_initializer(init_logit_scale),
            trainable=True,
            dtype="float32",
        )

        # Loss and metric trackers
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.i2t_acc_tracker = keras.metrics.Mean(name="i2t_acc")
        self.t2i_acc_tracker = keras.metrics.Mean(name="t2i_acc")

    @property
    def metrics(self):
        return [self.loss_tracker, self.i2t_acc_tracker, self.t2i_acc_tracker]

    def call(self, inputs, training=False):
        """Forward pass computing similarity logits matrix."""
        images, texts = inputs
        img_embeddings = self.vision_encoder(images, training=training)
        txt_embeddings = self.text_encoder(texts, training=training)

        # Bound temperature to prevent numerical instability (scale <= 100)
        clamped_logit_scale = tf.clip_by_value(
            self.logit_scale, 0.0, float(np.log(self.max_temperature_scale))
        )
        temperature_scale = tf.math.exp(clamped_logit_scale)

        # Pairwise cosine similarity matrix: [batch_size, batch_size]
        logits = tf.matmul(img_embeddings, txt_embeddings, transpose_b=True) * temperature_scale
        return logits

    def compute_clip_loss_and_acc(self, logits: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Compute symmetric cross-entropy loss and top-1 directional retrieval accuracy."""
        batch_size = tf.shape(logits)[0]
        ground_truth_labels = tf.range(batch_size)

        # Image-to-Text loss (row-wise cross-entropy)
        loss_i2t = keras.losses.sparse_categorical_crossentropy(
            ground_truth_labels, logits, from_logits=True
        )
        # Text-to-Image loss (column-wise cross-entropy on transposed matrix)
        loss_t2i = keras.losses.sparse_categorical_crossentropy(
            ground_truth_labels, tf.transpose(logits), from_logits=True
        )

        loss = tf.reduce_mean((loss_i2t + loss_t2i) / 2.0)

        # Image-to-Text top-1 retrieval accuracy
        pred_i2t = tf.argmax(logits, axis=1, output_type=tf.int32)
        acc_i2t = tf.reduce_mean(tf.cast(tf.equal(pred_i2t, ground_truth_labels), tf.float32))

        # Text-to-Image top-1 retrieval accuracy
        pred_t2i = tf.argmax(tf.transpose(logits), axis=1, output_type=tf.int32)
        acc_t2i = tf.reduce_mean(tf.cast(tf.equal(pred_t2i, ground_truth_labels), tf.float32))

        return loss, acc_i2t, acc_t2i

    def train_step(self, data):
        images, texts = data

        with tf.GradientTape() as tape:
            logits = self((images, texts), training=True)
            loss, acc_i2t, acc_t2i = self.compute_clip_loss_and_acc(logits)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.loss_tracker.update_state(loss)
        self.i2t_acc_tracker.update_state(acc_i2t)
        self.t2i_acc_tracker.update_state(acc_t2i)

        return {
            "loss": self.loss_tracker.result(),
            "i2t_acc": self.i2t_acc_tracker.result(),
            "t2i_acc": self.t2i_acc_tracker.result(),
            "temperature": tf.math.exp(self.logit_scale),
        }

    def test_step(self, data):
        images, texts = data
        logits = self((images, texts), training=False)
        loss, acc_i2t, acc_t2i = self.compute_clip_loss_and_acc(logits)

        self.loss_tracker.update_state(loss)
        self.i2t_acc_tracker.update_state(acc_i2t)
        self.t2i_acc_tracker.update_state(acc_t2i)

        return {
            "loss": self.loss_tracker.result(),
            "i2t_acc": self.i2t_acc_tracker.result(),
            "t2i_acc": self.t2i_acc_tracker.result(),
            "temperature": tf.math.exp(self.logit_scale),
        }

    def encode_images(self, images: tf.Tensor) -> tf.Tensor:
        """Encode images into normalized embeddings."""
        return self.vision_encoder(images, training=False)

    def encode_texts(self, texts: tf.Tensor) -> tf.Tensor:
        """Encode tokenized texts into normalized embeddings."""
        return self.text_encoder(texts, training=False)
