"""
Dataset utilities for Experiment 16: Transfer Learning & Data Augmentation Regularization.

Provides structured loading, stratified subsampling (to create an intentional small-sample
overfitting regime), preprocessing, and tf.data pipeline creation.
"""

import os
import re
import tarfile
import pickle
import urllib.request
import urllib.parse
import http.cookiejar
from typing import Tuple, List, Optional, Dict
import numpy as np
import tensorflow as tf
import keras

GDRIVE_FILE_ID = "1suK4voQ19zuO-usH_CFLPiMWVrDVxmGI"
GDRIVE_UC_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"


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


def _download_with_gdown(url: str, output_path: str) -> bool:
    """
    Downloads file using gdown with url as positional argument (compatible with all gdown versions).
    """
    try:
        import gdown
        print(f"📥 Downloading CIFAR-10 from {url}...")
        gdown.download(url, output_path, quiet=False)
        return _is_valid_gzip_archive(output_path)
    except Exception as e:
        print(f"ℹ️ gdown download issue: {e}")
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

    batches_exist = os.path.isdir(extracted_path) and all(
        os.path.isfile(os.path.join(extracted_path, f"data_batch_{i}")) for i in range(1, 6)
    ) and os.path.isfile(os.path.join(extracted_path, "test_batch"))

    if not batches_exist:
        if os.path.exists(tar_path) and not _is_valid_gzip_archive(tar_path):
            print("⚠️ Removing invalid/corrupted cached archive...")
            try:
                os.remove(tar_path)
            except Exception:
                pass

        if not os.path.exists(tar_path):
            success = _download_with_gdown(GDRIVE_UC_URL, tar_path)
            if not success:
                if os.path.exists(tar_path):
                    try:
                        os.remove(tar_path)
                    except Exception:
                        pass
                print("📥 Using direct streaming downloader...")
                _download_gdrive_direct(GDRIVE_FILE_ID, tar_path)

            if not _is_valid_gzip_archive(tar_path):
                if os.path.exists(tar_path):
                    try:
                        os.remove(tar_path)
                    except Exception:
                        pass
                raise ValueError("Downloaded file is not a valid gzip file (cifar-10-python.tar.gz).")

        print("📦 Extracting CIFAR-10 archive...")
        try:
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=data_dir)
        except Exception:
            if os.path.exists(tar_path):
                try:
                    os.remove(tar_path)
                except Exception:
                    pass
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
    Attempts Google Drive download first with HuggingFace / Keras fallback.
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
    data_dir: Optional[str] = None,
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
        data_dir: Directory to cache downloaded CIFAR-10 data.

    Returns:
        (x_train, y_train, x_val, y_val, class_names)
        Images are float32 in range [0, 1], labels are one-hot encoded.
    """
    if selected_classes is None:
        selected_classes = DEFAULT_CLASSES

    num_classes = len(selected_classes)
    class_names = [CIFAR10_CLASSES[c] for c in selected_classes]
    class_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(selected_classes)}

    (raw_x_train, raw_y_train), (raw_x_test, raw_y_test) = load_cifar10_raw(data_dir=data_dir)
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
