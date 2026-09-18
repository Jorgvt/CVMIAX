"""
Dataset utilities for training ResNet and Plain CNN on CIFAR-10.
"""

from typing import Tuple, Optional
import numpy as np
import tensorflow as tf
import keras


def load_cifar10_data(
    val_split: float = 0.1,
    normalize: bool = True,
    standardize: bool = True,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Loads and splits CIFAR-10 into train, validation, and test sets.
    
    Args:
        val_split: Fraction of training set to allocate for validation.
        normalize: If True, scales pixel values from [0, 255] to [0, 1].
        standardize: If True, applies per-channel mean subtraction and std normalization.
        
    Returns:
        (x_train, y_train), (x_val, y_val), (x_test, y_test)
    """
    (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.cifar10.load_data()
    
    x_train_full = x_train_full.astype("float32")
    x_test = x_test.astype("float32")
    
    if normalize:
        x_train_full /= 255.0
        x_test /= 255.0

    if standardize:
        mean = np.mean(x_train_full, axis=(0, 1, 2), keepdims=True)
        std = np.std(x_train_full, axis=(0, 1, 2), keepdims=True) + 1e-7
        x_train_full = (x_train_full - mean) / std
        x_test = (x_test - mean) / std

    # Train / Validation split
    num_val = int(len(x_train_full) * val_split)
    indices = np.random.RandomState(42).permutation(len(x_train_full))
    
    val_idx, train_idx = indices[:num_val], indices[num_val:]
    
    x_train, y_train = x_train_full[train_idx], y_train_full[train_idx]
    x_val, y_val = x_train_full[val_idx], y_train_full[val_idx]
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def create_tf_dataset(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int = 64,
    augment: bool = False,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """
    Creates an optimized tf.data.Dataset with optional data augmentation.
    
    Augmentations follow standard CIFAR practice:
    - Random 4-pixel padding followed by 32x32 random crop
    - Random horizontal flip
    """
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(x), 10000), seed=42)
        
    if augment:
        def augment_fn(image, label):
            # Pad by 4 pixels on all sides
            padded = tf.image.resize_with_crop_or_pad(image, 32 + 8, 32 + 8)
            # Random crop back to 32x32
            cropped = tf.image.random_crop(padded, size=[32, 32, 3])
            # Random horizontal flip
            flipped = tf.image.random_flip_left_right(cropped)
            return flipped, label

        ds = ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
        
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
