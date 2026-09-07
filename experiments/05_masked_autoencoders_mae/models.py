"""
Masked Autoencoder (MAE) Vision Transformer Architecture in Keras.

Implements the asymmetric ViT-MAE design (He et al., CVPR 2022):
- Heavy Encoder: processes ONLY unmasked visible patches (e.g. 25% of image).
- Lightweight Decoder: processes full token sequence (encoded visible tokens + learnable [MASK] tokens + positional embeddings).
"""

import numpy as np
import tensorflow as tf
import keras
from keras import layers


def transformer_block(embed_dim, num_heads, mlp_dim, dropout=0.0, name="transformer_block"):
    """Standard Vision Transformer Block with Pre-LayerNorm."""
    inputs = keras.Input(shape=(None, embed_dim))
    
    # Self-Attention sub-layer
    norm1 = layers.LayerNormalization(epsilon=1e-6)(inputs)
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=dropout)(norm1, norm1)
    x1 = layers.Add()([inputs, attn])

    # MLP sub-layer
    norm2 = layers.LayerNormalization(epsilon=1e-6)(x1)
    mlp = layers.Dense(mlp_dim, activation="gelu")(norm2)
    mlp = layers.Dropout(dropout)(mlp)
    mlp = layers.Dense(embed_dim)(mlp)
    mlp = layers.Dropout(dropout)(mlp)
    outputs = layers.Add()([x1, mlp])

    return keras.Model(inputs=inputs, outputs=outputs, name=name)


class MaskedAutoencoderViT(keras.Model):
    """
    Asymmetric Vision Transformer Masked Autoencoder (ViT-MAE).
    """
    def __init__(
        self,
        image_shape=(32, 32, 3),
        patch_size=4,
        enc_dim=128,
        enc_depth=4,
        enc_heads=4,
        dec_dim=64,
        dec_depth=2,
        dec_heads=4,
        mlp_ratio=4,
        mask_ratio=0.75,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_shape = image_shape
        self.patch_size = patch_size
        self.patch_dim = patch_size * patch_size * image_shape[-1]
        self.num_patches = (image_shape[0] // patch_size) * (image_shape[1] // patch_size)
        self.mask_ratio = mask_ratio
        self.num_visible = int(self.num_patches * (1.0 - mask_ratio))

        # --- Encoder Layers ---
        self.patch_proj = layers.Dense(enc_dim, name="patch_projection")
        self.enc_pos_embed = self.add_weight(
            name="enc_pos_embed",
            shape=(1, self.num_patches, enc_dim),
            initializer=keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
        )
        self.enc_blocks = [
            transformer_block(enc_dim, enc_heads, enc_dim * mlp_ratio, name=f"enc_block_{i}")
            for i in range(enc_depth)
        ]
        self.enc_norm = layers.LayerNormalization(epsilon=1e-6, name="enc_norm")

        # --- Decoder Layers ---
        self.enc_to_dec_proj = layers.Dense(dec_dim, name="enc_to_dec_proj")
        self.mask_token = self.add_weight(
            name="mask_token",
            shape=(1, 1, dec_dim),
            initializer=keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
        )
        self.dec_pos_embed = self.add_weight(
            name="dec_pos_embed",
            shape=(1, self.num_patches, dec_dim),
            initializer=keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
        )
        self.dec_blocks = [
            transformer_block(dec_dim, dec_heads, dec_dim * mlp_ratio, name=f"dec_block_{i}")
            for i in range(dec_depth)
        ]
        self.dec_norm = layers.LayerNormalization(epsilon=1e-6, name="dec_norm")
        # Predict pixel values for each patch: patch_size * patch_size * channels
        self.pixel_pred = layers.Dense(self.patch_dim, name="pixel_prediction")

    def forward_encoder(self, patches, shuffle_indices):
        """
        Processes ONLY the unmasked visible patches.
        """
        # Linear projection of flattened patches: shape (B, N, D_enc)
        x = self.patch_proj(patches)
        # Add positional embedding
        x = x + self.enc_pos_embed

        # Gather visible tokens using shuffle_indices: shape (B, N_vis, D_enc)
        b = tf.shape(x)[0]
        batch_idx = tf.repeat(tf.range(b, dtype=tf.int32)[:, None], self.num_visible, axis=1)
        vis_idx = tf.cast(shuffle_indices[:, :self.num_visible], tf.int32)
        indices = tf.stack([batch_idx, vis_idx], axis=-1)
        x_vis = tf.gather_nd(x, indices)

        # Run Encoder Transformer Blocks strictly on visible tokens
        for blk in self.enc_blocks:
            x_vis = blk(x_vis)
        x_vis = self.enc_norm(x_vis)

        return x_vis

    def forward_decoder(self, x_vis, restore_indices):
        """
        Reconstructs full image patches from encoded visible tokens + [MASK] tokens.
        """
        # Project encoder features to decoder dimension
        x_vis = self.enc_to_dec_proj(x_vis)

        # Append learnable [MASK] tokens for masked positions
        b = tf.shape(x_vis)[0]
        num_masked = self.num_patches - self.num_visible
        mask_tokens = tf.repeat(self.mask_token, b, axis=0)
        mask_tokens = tf.repeat(mask_tokens, num_masked, axis=1)

        # Concatenate: [visible_tokens, mask_tokens] -> shape (B, N, D_dec)
        x_full = tf.concat([x_vis, mask_tokens], axis=1)

        # Unshuffle to restore original 2D spatial patch order
        batch_idx = tf.repeat(tf.range(b, dtype=tf.int32)[:, None], self.num_patches, axis=1)
        restore_idx = tf.cast(restore_indices, tf.int32)
        indices = tf.stack([batch_idx, restore_idx], axis=-1)
        x_ordered = tf.gather_nd(x_full, indices)

        # Add decoder positional embeddings to all patches
        x_ordered = x_ordered + self.dec_pos_embed

        # Run Decoder Transformer Blocks
        for blk in self.dec_blocks:
            x_ordered = blk(x_ordered)
        x_ordered = self.dec_norm(x_ordered)

        # Predict patch pixel values: shape (B, N, P*P*C)
        pred_patches = self.pixel_pred(x_ordered)
        return pred_patches

    def call(self, inputs, training=False):
        """
        Forward pass: inputs = (patches, shuffle_indices, restore_indices).
        """
        patches, shuffle_indices, restore_indices = inputs
        x_vis = self.forward_encoder(patches, shuffle_indices)
        pred_patches = self.forward_decoder(x_vis, restore_indices)
        return pred_patches

    def compute_mae_loss(self, target_patches, pred_patches, mask):
        """
        Computes MSE loss exclusively over the masked patches.
        mask: (B, N), where 1 = masked, 0 = visible.
        """
        loss_per_patch = tf.reduce_mean(tf.square(target_patches - pred_patches), axis=-1)
        # Apply mask: average only over masked patches
        masked_loss = tf.reduce_sum(loss_per_patch * mask) / (tf.reduce_sum(mask) + 1e-6)
        return masked_loss
