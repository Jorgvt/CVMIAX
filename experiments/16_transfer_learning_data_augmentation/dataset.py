"""
Dataset utilities for Experiment 16: Transfer Learning & Data Augmentation Regularization.

Provides structured loading, stratified subsampling (to create an intentional small-sample
overfitting regime), preprocessing, and tf.data pipeline creation.
"""

from typing import Tuple, List, Optional, Dict
import numpy as np
import tensorflow as tf
import keras

# CIFAR-10 class labels
CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# Default selected 5 distinct classes for clear domain transfer demonstration
DEFAULT_CLASSES = [0, 1, 2, 5, 8]  # airplane, automobile, bird, dog, ship


def load_cifar_subsample(
    selected_classes: Optional[List[int]] = None,
    samples_per_class_train: int = 50,
    samples_per_class_val: int = 200,
    target_size: Tuple[int, int] = (96, 96),
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Loads and creates a controlled small-sample subset of CIFAR-10.
    
    Subsampling creates an empirical data-starved regime where high-capacity
    pre-trained transfer learning models readily overfit without regularization.
    
    Args:
        selected_classes: List of integer class indices to include (default: 5 classes).
        samples_per_class_train: Number of training images per class (e.g., 50).
        samples_per_class_val: Number of validation images per class (e.g., 200).
        target_size: Spatial target dimension (H, W) for transfer learning backbone.
        random_seed: Random seed for deterministic subsampling.

    Returns:
        (x_train, y_train, x_val, y_val, class_names)
        Images are float32 in range [0, 1], labels are one-hot encoded.
    """
    if selected_classes is None:
        selected_classes = DEFAULT_CLASSES

    num_classes = len(selected_classes)
    class_names = [CIFAR10_CLASSES[c] for c in selected_classes]
    class_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(selected_classes)}

    (raw_x_train, raw_y_train), (raw_x_test, raw_y_test) = keras.datasets.cifar10.load_data()
    raw_y_train = raw_y_train.flatten()
    raw_y_test = raw_y_test.flatten()

    rng = np.random.RandomState(random_seed)

    x_train_list, y_train_list = [], []
    x_val_list, y_val_list = [], []

    for orig_cls in selected_classes:
        new_cls = class_map[orig_cls]

        # Train indices
        train_indices = np.where(raw_y_train == orig_cls)[0]
        rng.shuffle(train_indices)
        chosen_train = train_indices[:samples_per_class_train]
        x_train_list.append(raw_x_train[chosen_train])
        y_train_list.append(np.full(len(chosen_train), new_cls))

        # Val indices
        val_indices = np.where(raw_y_test == orig_cls)[0]
        rng.shuffle(val_indices)
        chosen_val = val_indices[:samples_per_class_val]
        x_val_list.append(raw_x_test[chosen_val])
        y_val_list.append(np.full(len(chosen_val), new_cls))

    x_train = np.concatenate(x_train_list, axis=0).astype("float32") / 255.0
    y_train = np.concatenate(y_train_list, axis=0)
    x_val = np.concatenate(x_val_list, axis=0).astype("float32") / 255.0
    y_val = np.concatenate(y_val_list, axis=0)

    # Shuffle training set
    train_perm = rng.permutation(len(x_train))
    x_train, y_train = x_train[train_perm], y_train[train_perm]

    # Shuffle val set
    val_perm = rng.permutation(len(x_val))
    x_val, y_val = x_val[val_perm], y_val[val_perm]

    # Resize images to target backbone resolution if needed
    if target_size != (32, 32):
        print(f"Resizing images from 32x32 to {target_size}...")
        x_train = tf.image.resize(x_train, target_size, method="bilinear").numpy()
        x_val = tf.image.resize(x_val, target_size, method="bilinear").numpy()

    # One-hot encode targets
    y_train_one_hot = keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_val_one_hot = keras.utils.to_categorical(y_val, num_classes=num_classes)

    print(
        f"✅ Dataset prepared: {len(x_train)} train images ({samples_per_class_train}/class), "
        f"{len(x_val)} val images ({samples_per_class_val}/class) across {num_classes} classes: {class_names}"
    )

    return x_train, y_train_one_hot, x_val, y_val_one_hot, class_names


def create_tf_dataset(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    is_training: bool = True,
    shuffle_buffer: int = 1000,
) -> tf.data.Dataset:
    """
    Creates a high-throughput tf.data.Dataset.
    """
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    if is_training:
        dataset = dataset.shuffle(buffer_size=min(len(x), shuffle_buffer), seed=42)
    dataset = dataset.batch(batch_size, drop_remainder=is_training)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset
