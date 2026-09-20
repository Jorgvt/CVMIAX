"""
Central CIFAR-10 dataset loader and pipeline utilities for CVMIAX.
Loads CIFAR-10 from Google Drive via gdown/direct stream with fallback to Keras / HF datasets.
"""

import os
import re
import tarfile
import pickle
import urllib.request
import urllib.parse
import http.cookiejar
from typing import Tuple, Optional
import numpy as np
import tensorflow as tf
import keras

GDRIVE_FILE_ID = "1suK4voQ19zuO-usH_CFLPiMWVrDVxmGI"
GDRIVE_CIFAR10_URL = f"https://drive.google.com/file/d/{GDRIVE_FILE_ID}/view?usp=drive_link"


def _is_valid_gzip_archive(path: str, min_size_mb: float = 50.0) -> bool:
    """
    Checks if a file exists, has expected size, and starts with gzip magic bytes (0x1f 0x8b).
    """
    if not os.path.isfile(path):
        return False
    size_mb = os.path.getsize(path) / (1024 * 1024)
    if size_mb < min_size_mb:
        return False
    try:
        with open(path, "rb") as f:
            magic = f.read(2)
            return magic == b"\x1f\x8b"
    except Exception:
        return False


def _download_gdrive_direct(file_id: str, output_path: str) -> str:
    """
    Streams large files from Google Drive using standard library,
    handling Google's virus scan confirmation tokens and session cookies.
    """
    cookie_jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))
    opener.addheaders = [("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)")]

    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    resp = opener.open(url)

    content_type = resp.headers.get("Content-Type", "")
    if "text/html" in content_type:
        html = resp.read().decode("utf-8", errors="ignore")
        action_match = re.search(r'action=\"([^\"]+)\"', html)
        inputs = dict(re.findall(r'<input[^>]+name=\"([^\"]+)\"[^>]+value=\"([^\"]+)\"', html))
        if action_match:
            download_url = action_match.group(1) + "?" + urllib.parse.urlencode(inputs)
            resp = opener.open(download_url)
        else:
            confirm_match = re.search(r"confirm=([0-9A-Za-z_]+)", html)
            if confirm_match:
                confirm_token = confirm_match.group(1)
                download_url = f"https://drive.google.com/uc?export=download&confirm={confirm_token}&id={file_id}"
                resp = opener.open(download_url)
            else:
                raise RuntimeError("Could not parse Google Drive download confirmation page.")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    chunk_size = 1024 * 1024  # 1MB chunks
    with open(output_path, "wb") as f:
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)

    return output_path


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
    Downloads and extracts CIFAR-10 dataset from Google Drive with robust format validation.
    """
    if data_dir is None:
        data_dir = os.path.expanduser("~/.keras/datasets")
    os.makedirs(data_dir, exist_ok=True)

    extracted_path = os.path.join(data_dir, "cifar-10-batches-py")
    tar_path = os.path.join(data_dir, "cifar-10-python.tar.gz")

    # Check if batches already exist and are intact
    batches_exist = os.path.isdir(extracted_path) and all(
        os.path.isfile(os.path.join(extracted_path, f"data_batch_{i}")) for i in range(1, 6)
    ) and os.path.isfile(os.path.join(extracted_path, "test_batch"))

    if not batches_exist:
        # Check if existing tar archive is valid; remove if corrupted (e.g. HTML response)
        if os.path.exists(tar_path) and not _is_valid_gzip_archive(tar_path):
            print("⚠️ Removing invalid/corrupted cached archive...")
            try:
                os.remove(tar_path)
            except Exception:
                pass

        if not os.path.exists(tar_path):
            print("📥 Downloading CIFAR-10 dataset from Google Drive...")
            downloaded = False
            # Method 1: gdown with explicit ID
            try:
                import gdown
                gdown.download(id=GDRIVE_FILE_ID, output=tar_path, quiet=False)
                if _is_valid_gzip_archive(tar_path):
                    downloaded = True
                else:
                    if os.path.exists(tar_path):
                        os.remove(tar_path)
            except Exception as e:
                print(f"ℹ️ gdown attempt noted: {e}")

            # Method 2: Direct streaming fallback
            if not downloaded:
                print("📥 Using direct Google Drive streaming downloader...")
                _download_gdrive_direct(GDRIVE_FILE_ID, tar_path)

            if not _is_valid_gzip_archive(tar_path):
                if os.path.exists(tar_path):
                    os.remove(tar_path)
                raise ValueError("Downloaded file failed gzip integrity check (not a valid .tar.gz archive).")

        print("📦 Extracting CIFAR-10 archive...")
        try:
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=data_dir)
        except Exception:
            # Clean up corrupted file on extract failure so next run can retry
            if os.path.exists(tar_path):
                os.remove(tar_path)
            raise

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


def load_cifar10_raw(
    data_dir: Optional[str] = None,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Loads raw CIFAR-10 data as (50000, 32, 32, 3) uint8 arrays.
    Attempts Google Drive download first with Keras / HuggingFace fallback.

    Returns:
        (x_train, y_train), (x_test, y_test)
    """
    try:
        return _load_cifar10_from_gdrive(data_dir=data_dir)
    except Exception as exc:
        print(f"⚠️ Warning: Could not load CIFAR-10 from Google Drive ({exc}). Trying fallback...")

    # Fallback 1: Hugging Face datasets
    try:
        from datasets import load_dataset
        print("📥 Loading CIFAR-10 from Hugging Face datasets fallback...")
        ds = load_dataset("uoft-cs/cifar10")
        train_imgs = np.array(ds["train"]["img"], dtype="uint8")
        train_labels = np.array(ds["train"]["label"], dtype="uint8").reshape(-1, 1)
        test_imgs = np.array(ds["test"]["img"], dtype="uint8")
        test_labels = np.array(ds["test"]["label"], dtype="uint8").reshape(-1, 1)
        return (train_imgs, train_labels), (test_imgs, test_labels)
    except Exception as hf_exc:
        print(f"ℹ️ Hugging Face fallback noted: {hf_exc}")

    # Fallback 2: Keras datasets
    return keras.datasets.cifar10.load_data()


def load_cifar10_data(
    val_split: float = 0.1,
    normalize: bool = True,
    standardize: bool = True,
    data_dir: Optional[str] = None,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Loads and splits CIFAR-10 into train, validation, and test sets.

    Args:
        val_split: Fraction of training set to allocate for validation.
        normalize: If True, scales pixel values from [0, 255] to [0, 1].
        standardize: If True, applies per-channel mean subtraction and std normalization.
        data_dir: Optional custom directory to store/cache the dataset.

    Returns:
        (x_train, y_train), (x_val, y_val), (x_test, y_test)
    """
    (x_train_full, y_train_full), (x_test, y_test) = load_cifar10_raw(data_dir=data_dir)

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
    Creates an optimized tf.data.Dataset with optional standard CIFAR-10 data augmentation.

    Augmentations:
    - Random 4-pixel padding followed by 32x32 random crop
    - Random horizontal flip
    """
    ds = tf.data.Dataset.from_tensor_slices((x, y))

    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(x), 10000), seed=42)

    if augment:
        def augment_fn(image, label):
            padded = tf.image.resize_with_crop_or_pad(image, 32 + 8, 32 + 8)
            cropped = tf.image.random_crop(padded, size=[32, 32, 3])
            flipped = tf.image.random_flip_left_right(cropped)
            return flipped, label

        ds = ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
