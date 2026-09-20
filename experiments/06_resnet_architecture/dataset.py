"""
Dataset utilities for training ResNet and Plain CNN on CIFAR-10.
Supports loading CIFAR-10 from Google Drive via gdown with Keras dataset fallback.
"""

import os
import tarfile
import pickle
from typing import Tuple, Optional
import numpy as np
import tensorflow as tf
import keras

GDRIVE_CIFAR10_URL = "https://drive.google.com/file/d/1suK4voQ19zuO-usH_CFLPiMWVrDVxmGI/view?usp=drive_link"


def _load_batch(fpath: str, label_key: str = "labels") -> Tuple[np.ndarray, np.ndarray]:
    """
    Internal utility for parsing CIFAR-10 python batch files.
    """
    with open(fpath, "rb") as f:
        d = pickle.load(f, encoding="bytes")
        d_decoded = {k.decode("utf-8") if isinstance(k, bytes) else k: v for k, v in d.items()}
    data = d_decoded["data"]
    labels = d_decoded[label_key]
    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, np.array(labels)


def _load_cifar10_from_gdrive(
    data_dir: Optional[str] = None,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Downloads and extracts CIFAR-10 dataset from Google Drive using gdown.
    """
    import gdown

    if data_dir is None:
        data_dir = os.path.expanduser("~/.keras/datasets")
    os.makedirs(data_dir, exist_ok=True)

    extracted_path = os.path.join(data_dir, "cifar-10-batches-py")
    tar_path = os.path.join(data_dir, "cifar-10-python.tar.gz")

    # Check if batches already exist
    batches_exist = os.path.exists(extracted_path) and all(
        os.path.exists(os.path.join(extracted_path, f"data_batch_{i}")) for i in range(1, 6)
    ) and os.path.exists(os.path.join(extracted_path, "test_batch"))

    if not batches_exist:
        if not os.path.exists(tar_path):
            print("📥 Downloading CIFAR-10 dataset from Google Drive via gdown...")
            gdown.download(url=GDRIVE_CIFAR10_URL, output=tar_path, quiet=False)

        print("📦 Extracting CIFAR-10 archive...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=data_dir)

    num_train_samples = 50000
    x_train = np.empty((num_train_samples, 3, 32, 32), dtype="uint8")
    y_train = np.empty((num_train_samples,), dtype="uint8")

    for i in range(1, 6):
        fpath = os.path.join(extracted_path, f"data_batch_{i}")
        (
            x_train[(i - 1) * 10000 : i * 10000, :, :, :],
            y_train[(i - 1) * 10000 : i * 10000],
        ) = _load_batch(fpath)

    fpath = os.path.join(extracted_path, "test_batch")
    x_test, y_test = _load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1)).astype("uint8")
    y_test = np.reshape(y_test, (len(y_test), 1)).astype("uint8")

    if keras.backend.image_data_format() == "channels_last":
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)


def load_cifar10_data(
    val_split: float = 0.1,
    normalize: bool = True,
    standardize: bool = True,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Loads and splits CIFAR-10 into train, validation, and test sets.
    Attempts to download/load from Google Drive first, falling back to keras.datasets.cifar10.

    Args:
        val_split: Fraction of training set to allocate for validation.
        normalize: If True, scales pixel values from [0, 255] to [0, 1].
        standardize: If True, applies per-channel mean subtraction and std normalization.

    Returns:
        (x_train, y_train), (x_val, y_val), (x_test, y_test)
    """
    try:
        (x_train_full, y_train_full), (x_test, y_test) = _load_cifar10_from_gdrive()
    except Exception as exc:
        print(f"⚠️ Warning: Could not load CIFAR-10 from Google Drive ({exc}). Falling back to keras.datasets.cifar10.")
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
