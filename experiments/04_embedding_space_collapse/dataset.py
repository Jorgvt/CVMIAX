"""
Dataset utilities for Experiment 04: Embedding Space Collapse on CIFAR-10.
Self-contained loader downloading CIFAR-10 from Google Drive via gdown/direct stream with fallback.
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
import keras

GDRIVE_FILE_ID = "1suK4voQ19zuO-usH_CFLPiMWVrDVxmGI"
GDRIVE_UC_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"


def _is_valid_gzip_archive(path: str, min_size_mb: float = 50.0) -> bool:
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
    try:
        import gdown
        print(f"📥 Downloading CIFAR-10 from {url}...")
        gdown.download(url, output_path, quiet=False)
        return _is_valid_gzip_archive(output_path)
    except Exception as e:
        print(f"ℹ️ gdown note: {e}")
        return False


def _download_gdrive_direct(file_id: str, output_path: str) -> str:
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
    chunk_size = 1024 * 1024
    with open(output_path, "wb") as f:
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)

    return output_path


def _load_batch(fpath: str, label_key: str = "labels") -> Tuple[np.ndarray, np.ndarray]:
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
            print("⚠️ Removing invalid cached archive...")
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
    try:
        return _load_cifar10_from_gdrive(data_dir=data_dir)
    except Exception as exc:
        print(f"⚠️ Warning: Could not load CIFAR-10 from Google Drive ({exc}). Trying fallback...")

    try:
        from datasets import load_dataset
        ds = load_dataset("uoft-cs/cifar10")
        train_imgs = np.array(ds["train"]["img"], dtype="uint8")
        train_labels = np.array(ds["train"]["label"], dtype="uint8").reshape(-1, 1)
        test_imgs = np.array(ds["test"]["img"], dtype="uint8")
        test_labels = np.array(ds["test"]["label"], dtype="uint8").reshape(-1, 1)
        return (train_imgs, train_labels), (test_imgs, test_labels)
    except Exception as hf_exc:
        print(f"ℹ️ Hugging Face fallback note: {hf_exc}")

    return keras.datasets.cifar10.load_data()


def load_cifar10_unlabeled_normalized(num_samples: int = 4000) -> np.ndarray:
    """
    Loads unlabeled CIFAR-10 training images normalized to [0, 1].

    Args:
        num_samples: Number of samples to return.

    Returns:
        x_train array of shape (num_samples, 32, 32, 3) float32 in [0, 1].
    """
    (x_train, _), _ = load_cifar10_raw()
    return x_train[:num_samples].astype("float32") / 255.0
