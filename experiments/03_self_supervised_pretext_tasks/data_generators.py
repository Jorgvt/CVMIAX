"""
Pretext Task Data Generation Utilities for Self-Supervised Learning.

This module provides clear, pedagogical functions to create the inputs and labels
for three foundational self-supervised pretext tasks:
1. Rotation Prediction (Gidaris et al., 2018)
2. Jigsaw Puzzle Solving (Noroozi & Favaro, 2016)
3. Image Colorization (Zhang et al., 2016)
"""

import numpy as np
import tensorflow as tf
import keras


# =========================================================================
# 1. Rotation Prediction Pretext Task
# =========================================================================

def create_rotation_batch(images):
    """
    Given a batch of images of shape (N, H, W, C), generates 4 rotated copies
    for each image (0°, 90°, 180°, 270°) and their corresponding discrete labels.

    Args:
        images: Array or Tensor of shape (N, H, W, C) in range [0, 1] or [0, 255].

    Returns:
        rotated_images: Tensor of shape (4*N, H, W, C).
        labels: Tensor of shape (4*N,) containing integer labels in {0, 1, 2, 3}:
                0 -> 0 degrees
                1 -> 90 degrees (counter-clockwise)
                2 -> 180 degrees
                3 -> 270 degrees
    """
    images = np.asarray(images)
    n = len(images)

    rot0 = images
    rot90 = np.rot90(images, k=1, axes=(1, 2))
    rot180 = np.rot90(images, k=2, axes=(1, 2))
    rot270 = np.rot90(images, k=3, axes=(1, 2))

    # Stack all rotations: shape (4*N, H, W, C)
    rotated_images = np.concatenate([rot0, rot90, rot180, rot270], axis=0)

    # Labels: 0 for 0°, 1 for 90°, 2 for 180°, 3 for 270°
    labels = np.concatenate([
        np.zeros(n, dtype=np.int32),
        np.ones(n, dtype=np.int32),
        np.full(n, 2, dtype=np.int32),
        np.full(n, 3, dtype=np.int32),
    ], axis=0)

    return rotated_images, labels


# =========================================================================
# 2. Jigsaw Puzzle Pretext Task
# =========================================================================

# Pre-selected set of representative permutations for a 3x3 grid (9 tiles)
# In standard Jigsaw (Noroozi & Favaro 2016), a subset of 64 or 100 permutations
# with high Hamming distance is selected.
PREDEFINED_3x3_PERMUTATIONS = np.array([
    [0, 1, 2, 3, 4, 5, 6, 7, 8],  # 0: Identity (original order)
    [8, 7, 6, 5, 4, 3, 2, 1, 0],  # 1: Reverse order
    [1, 2, 0, 4, 5, 3, 7, 8, 6],  # 2: Column shift
    [3, 4, 5, 6, 7, 8, 0, 1, 2],  # 3: Row shift
    [6, 3, 0, 7, 4, 1, 8, 5, 2],  # 4: Transpose / diagonal swap
    [2, 5, 8, 1, 4, 7, 0, 3, 6],  # 5: Counter-transpose
    [4, 0, 1, 2, 3, 5, 6, 7, 8],  # 6: Center-first shuffle
    [8, 0, 7, 1, 6, 2, 5, 3, 4],  # 7: Alternating outside-in
])

PREDEFINED_2x2_PERMUTATIONS = np.array([
    [0, 1, 2, 3],  # 0: Identity
    [1, 0, 3, 2],  # 1: Horizontal swap
    [2, 3, 0, 1],  # 2: Vertical swap
    [3, 2, 1, 0],  # 3: Diagonal reverse
    [1, 2, 3, 0],  # 4: Cyclic shift left
    [3, 0, 1, 2],  # 5: Cyclic shift right
    [0, 3, 2, 1],  # 6: Secondary diagonal swap
    [2, 0, 3, 1],  # 7: Complex permutation
])


def extract_tiles(image, grid_size=3, tile_gap=2):
    """
    Divides a single image of shape (H, W, C) into a grid of tiles (e.g. 3x3 = 9 tiles).
    Optionally applies a small gap between tiles to prevent low-level pixel continuity shortcuts.

    Returns:
        tiles: List of NumPy arrays, each of shape (tile_h, tile_w, C) of length grid_size*grid_size.
    """
    h, w, _ = image.shape
    tile_h = h // grid_size
    tile_w = w // grid_size
    tiles = []

    for r in range(grid_size):
        for c in range(grid_size):
            r_start = r * tile_h + tile_gap
            r_end = (r + 1) * tile_h - tile_gap
            c_start = c * tile_w + tile_gap
            c_end = (c + 1) * tile_w - tile_gap

            tile = image[r_start:r_end, c_start:c_end]
            tiles.append(tile)

    return tiles


def assemble_jigsaw_image(shuffled_tiles, grid_size=3):
    """
    Reconstructs a 2D composite image from a list of shuffled tiles for visualization.
    """
    tile_h, tile_w, ch = shuffled_tiles[0].shape
    composite = np.zeros((grid_size * tile_h, grid_size * tile_w, ch), dtype=shuffled_tiles[0].dtype)

    idx = 0
    for r in range(grid_size):
        for c in range(grid_size):
            composite[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w] = shuffled_tiles[idx]
            idx += 1

    return composite


def create_jigsaw_sample(image, permutation_idx, permutations_table=PREDEFINED_3x3_PERMUTATIONS, grid_size=3):
    """
    Given a single image, extracts tiles, permutes them according to permutations_table[permutation_idx],
    and returns the shuffled tile collage along with the target permutation class label.

    Returns:
        shuffled_composite: Image array displaying the shuffled puzzle.
        shuffled_tiles: Array of shape (num_tiles, tile_h, tile_w, C) suitable for a multi-tower CNN.
        label: Integer permutation index.
    """
    tiles = extract_tiles(image, grid_size=grid_size, tile_gap=0)
    perm = permutations_table[permutation_idx]
    shuffled_tiles = [tiles[i] for i in perm]
    shuffled_composite = assemble_jigsaw_image(shuffled_tiles, grid_size=grid_size)

    return shuffled_composite, np.array(shuffled_tiles), permutation_idx


# =========================================================================
# 3. Image Colorization Pretext Task
# =========================================================================

def create_colorization_pair(rgb_image):
    """
    Given an RGB image in range [0, 1] or [0, 255], creates:
    - Input: Grayscale image (1 channel or replicated 3 channels) representing luminance.
    - Target: Full color RGB image (or (a,b) chrominance channels).

    Returns:
        input_gray: Grayscale image of shape (H, W, 1).
        target_color: Full RGB image of shape (H, W, 3).
    """
    rgb_arr = np.asarray(rgb_image).astype(np.float32)
    if rgb_arr.max() > 1.0:
        rgb_arr = rgb_arr / 255.0

    # ITU-R 601-2 standard luminance formula: Y = 0.299 R + 0.587 G + 0.114 B
    gray = 0.299 * rgb_arr[:, :, 0] + 0.587 * rgb_arr[:, :, 1] + 0.114 * rgb_arr[:, :, 2]
    input_gray = np.expand_dims(gray, axis=-1)
    target_color = rgb_arr

    return input_gray, target_color
