"""
Masked Autoencoder (MAE) Data & Patch Utilities.

Provides utilities for:
1. Converting images to non-overlapping patches and back.
2. Generating high-ratio random masking (e.g., 75% masked, 25% visible).
3. Reconstructing full composite images from patch predictions.
"""

import numpy as np
import tensorflow as tf
import keras


def patchify(images, patch_size=4):
    """
    Divides a batch of images of shape (B, H, W, C) into non-overlapping patches.

    Args:
        images: Array or Tensor of shape (B, H, W, C).
        patch_size: Integer patch dimension P.

    Returns:
        patches: Array of shape (B, num_patches, patch_size * patch_size * C)
                 where num_patches = (H // P) * (W // P).
    """
    images = np.asarray(images)
    b, h, w, c = images.shape
    assert h % patch_size == 0 and w % patch_size == 0, "Image dimensions must be divisible by patch_size."

    num_patches_h = h // patch_size
    num_patches_w = w // patch_size
    num_patches = num_patches_h * num_patches_w

    # Reshape to (B, num_patches_h, patch_size, num_patches_w, patch_size, C)
    patches = images.reshape(b, num_patches_h, patch_size, num_patches_w, patch_size, c)
    # Transpose to (B, num_patches_h, num_patches_w, patch_size, patch_size, C)
    patches = patches.transpose(0, 1, 3, 2, 4, 5)
    # Flatten patches to (B, num_patches, patch_size * patch_size * C)
    patches = patches.reshape(b, num_patches, patch_size * patch_size * c)

    return patches


def unpatchify(patches, image_shape=(32, 32, 3), patch_size=4):
    """
    Reconstructs image tensor from flattened patches.

    Args:
        patches: Array of shape (B, num_patches, patch_size * patch_size * C).
        image_shape: Tuple (H, W, C).
        patch_size: Integer patch dimension P.

    Returns:
        images: Array of shape (B, H, W, C).
    """
    patches = np.asarray(patches)
    b, num_patches, patch_dim = patches.shape
    h, w, c = image_shape
    num_patches_h = h // patch_size
    num_patches_w = w // patch_size

    patches = patches.reshape(b, num_patches_h, num_patches_w, patch_size, patch_size, c)
    patches = patches.transpose(0, 1, 3, 2, 4, 5)
    images = patches.reshape(b, h, w, c)

    return images


def random_masking(patches, mask_ratio=0.75, seed=None):
    """
    Applies random masking to a sequence of patches.

    Args:
        patches: Array of shape (B, num_patches, patch_dim).
        mask_ratio: Fraction of patches to mask (e.g. 0.75).
        seed: Optional random seed.

    Returns:
        visible_patches: Array of shape (B, num_visible, patch_dim).
        mask: Binary array of shape (B, num_patches), where 0 = visible, 1 = masked.
        restore_indices: Indices to restore canonical patch ordering in the decoder.
    """
    if seed is not None:
        np.random.seed(seed)

    b, num_patches, patch_dim = patches.shape
    num_masked = int(num_patches * mask_ratio)
    num_visible = num_patches - num_masked

    # Generate random noise for shuffling
    noise = np.random.rand(b, num_patches)
    # Sort noise to obtain random shuffle indices
    shuffle_indices = np.argsort(noise, axis=1)
    # Restore indices (inverse permutation)
    restore_indices = np.argsort(shuffle_indices, axis=1)

    # Keep only the first num_visible patches
    visible_indices = shuffle_indices[:, :num_visible]
    
    # Gather visible patches
    batch_indices = np.arange(b)[:, None]
    visible_patches = patches[batch_indices, visible_indices]

    # Binary mask: 0 for visible, 1 for masked
    mask = np.ones((b, num_patches), dtype=np.float32)
    mask[batch_indices, visible_indices] = 0.0

    return visible_patches, mask, restore_indices, shuffle_indices


def create_masked_view_image(image, mask_ratio=0.75, patch_size=4, mask_color=(0.1, 0.1, 0.1)):
    """
    Helper to create a visualization of an image with masked patches replaced by a gray color.
    """
    img_batch = np.expand_dims(image, axis=0)
    patches = patchify(img_batch, patch_size=patch_size)
    _, mask, _, _ = random_masking(patches, mask_ratio=mask_ratio, seed=42)

    # Reconstruct with mask color on masked patches
    masked_patches = patches.copy()
    mask_patch_val = np.tile(np.array(mask_color, dtype=np.float32), (patch_size * patch_size))
    for i in range(mask.shape[1]):
        if mask[0, i] == 1.0:
            masked_patches[0, i] = mask_patch_val

    masked_img = unpatchify(masked_patches, image_shape=image.shape, patch_size=patch_size)[0]
    return np.clip(masked_img, 0.0, 1.0), mask[0]
