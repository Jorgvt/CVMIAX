"""TID2008 Dataset loading, reference-independent splitting, and preprocessing.

This module provides data loading pipelines for Image Quality Assessment (IQA)
experiments using the TID2008 benchmark dataset hosted on Hugging Face (Jorgvt/TID2008).
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import tensorflow as tf
from datasets import load_dataset
from PIL import Image

# TID2008 Distortion Names mapped by distortion_id (1-17)
DISTORTION_NAMES: Dict[int, str] = {
    1: "Additive Gaussian noise",
    2: "Additive noise in color components",
    3: "Spatially correlated noise",
    4: "Masked noise",
    5: "High frequency noise",
    6: "Impulse noise",
    7: "Quantization noise",
    8: "Gaussian blur",
    9: "Image denoising artifacts",
    10: "JPEG compression",
    11: "JPEG2000 compression",
    12: "JPEG transmission errors",
    13: "JPEG2000 transmission errors",
    14: "Non eccentricity pattern noise",
    15: "Local block-wise distortions",
    16: "Mean shift (intensity shift)",
    17: "Contrast change",
}

# High-level taxonomy grouping for pedagogical analysis
DISTORTION_GROUPS: Dict[str, List[int]] = {
    "Noise": [1, 2, 3, 4, 5, 6, 7, 14],
    "Blur & Denoising": [8, 9],
    "Compression": [10, 11],
    "Transmission Errors": [12, 13, 15],
    "Color & Tone Shifts": [16, 17],
}


def get_distortion_group(distortion_id: int) -> str:
    """Return the pedagogical category name for a distortion ID."""
    for group_name, dist_ids in DISTORTION_GROUPS.items():
        if distortion_id in dist_ids:
            return group_name
    return "Other"


def load_tid2008_raw(
    dataset_name: str = "Jorgvt/TID2008",
) -> List[Dict]:
    """Load the raw dataset from Hugging Face Datasets.

    Returns:
        List of dictionaries with keys:
            - 'reference': PIL Image
            - 'distorted': PIL Image
            - 'mos': float
            - 'reference_id': int (1-25)
            - 'distortion_id': int (1-17)
            - 'distortion_intensity': int (1-4)
    """
    ds = load_dataset(dataset_name, split="train")
    return list(ds)


def split_tid2008_by_reference(
    data: List[Dict],
    train_ref_ids: Optional[List[int]] = None,
    val_ref_ids: Optional[List[int]] = None,
    test_ref_ids: Optional[List[int]] = None,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Perform content-independent (reference-independent) train/val/test split.

    Crucial in IQA: Never allow the same reference image in both train and test!
    TID2008 contains 25 reference images (IDs 1-25).

    Default split:
        - Train: Ref IDs 1..18 (18 refs, 72% / 1224 images)
        - Val:   Ref IDs 19..20 (2 refs, 8% / 136 images)
        - Test:  Ref IDs 21..25 (5 refs, 20% / 340 images)
    """
    if train_ref_ids is None:
        train_ref_ids = list(range(1, 19))
    if val_ref_ids is None:
        val_ref_ids = [19, 20]
    if test_ref_ids is None:
        test_ref_ids = list(range(21, 26))

    train_data = [item for item in data if item["reference_id"] in train_ref_ids]
    val_data = [item for item in data if item["reference_id"] in val_ref_ids]
    test_data = [item for item in data if item["reference_id"] in test_ref_ids]

    return train_data, val_data, test_data


def process_image_pair(
    ref_pil: Image.Image,
    dist_pil: Image.Image,
    target_size: Tuple[int, int] = (224, 224),
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert PIL images to float32 NumPy arrays in [0, 1] resized to target_size."""
    ref_resized = ref_pil.resize(target_size, Image.Resampling.BILINEAR)
    dist_resized = dist_pil.resize(target_size, Image.Resampling.BILINEAR)

    ref_arr = np.array(ref_resized, dtype=np.float32) / 255.0
    dist_arr = np.array(dist_resized, dtype=np.float32) / 255.0

    return ref_arr, dist_arr


def prepare_numpy_arrays(
    data: List[Dict],
    target_size: Tuple[int, int] = (224, 224),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract preprocessed NumPy arrays for fast vectorized training and evaluation.

    Returns:
        refs: (N, H, W, 3) float32 [0, 1]
        dists: (N, H, W, 3) float32 [0, 1]
        mos: (N,) float32
        ref_ids: (N,) int32
        dist_ids: (N,) int32
        intensities: (N,) int32
    """
    n = len(data)
    refs = np.zeros((n, target_size[0], target_size[1], 3), dtype=np.float32)
    dists = np.zeros((n, target_size[0], target_size[1], 3), dtype=np.float32)
    mos = np.zeros((n,), dtype=np.float32)
    ref_ids = np.zeros((n,), dtype=np.int32)
    dist_ids = np.zeros((n,), dtype=np.int32)
    intensities = np.zeros((n,), dtype=np.int32)

    for i, item in enumerate(data):
        ref_arr, dist_arr = process_image_pair(
            item["reference"], item["distorted"], target_size=target_size
        )
        refs[i] = ref_arr
        dists[i] = dist_arr
        mos[i] = float(item["mos"])
        ref_ids[i] = int(item["reference_id"])
        dist_ids[i] = int(item["distortion_id"])
        intensities[i] = int(item["distortion_intensity"])

    return refs, dists, mos, ref_ids, dist_ids, intensities


def build_tf_datasets(
    train_arrays: Tuple[np.ndarray, np.ndarray, np.ndarray],
    val_arrays: Tuple[np.ndarray, np.ndarray, np.ndarray],
    test_arrays: Tuple[np.ndarray, np.ndarray, np.ndarray],
    batch_size: int = 32,
    mode: str = "full_reference",
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Create optimized tf.data.Dataset pipelines for Full-Reference or No-Reference IQA.

    Args:
        train_arrays: (refs, dists, mos)
        val_arrays: (refs, dists, mos)
        test_arrays: (refs, dists, mos)
        batch_size: Batch size
        mode: "full_reference" (inputs: {"reference": ref, "distorted": dist})
              or "no_reference" (inputs: dist)

    Returns:
        (train_ds, val_ds, test_ds)
    """
    train_refs, train_dists, train_mos = train_arrays
    val_refs, val_dists, val_mos = val_arrays
    test_refs, test_dists, test_mos = test_arrays

    if mode == "full_reference":
        train_ds = tf.data.Dataset.from_tensor_slices(
            ({"reference": train_refs, "distorted": train_dists}, train_mos)
        )
        val_ds = tf.data.Dataset.from_tensor_slices(
            ({"reference": val_refs, "distorted": val_dists}, val_mos)
        )
        test_ds = tf.data.Dataset.from_tensor_slices(
            ({"reference": test_refs, "distorted": test_dists}, test_mos)
        )
    elif mode == "no_reference":
        train_ds = tf.data.Dataset.from_tensor_slices((train_dists, train_mos))
        val_ds = tf.data.Dataset.from_tensor_slices((val_dists, val_mos))
        test_ds = tf.data.Dataset.from_tensor_slices((test_dists, test_mos))
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'full_reference' or 'no_reference'.")

    train_ds = (
        train_ds.shuffle(buffer_size=len(train_mos), seed=42)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds
