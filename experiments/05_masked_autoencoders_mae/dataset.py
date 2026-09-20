"""
Dataset loader for Experiment 05: Masked Autoencoders (ViT-MAE).
Delegates to central cvmiax CIFAR-10 loader.
"""

import os
import sys
from typing import Tuple
import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
src_dir = os.path.join(repo_root, "src")
if os.path.exists(src_dir) and src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from cvmiax.datasets.cifar10 import load_cifar10_raw, load_cifar10_data, create_tf_dataset


def load_mae_cifar10(
    num_train: int = 20000,
    num_test: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads normalized unlabeled CIFAR-10 images for MAE pre-training.

    Args:
        num_train: Number of training images.
        num_test: Number of test images.

    Returns:
        (x_train, x_test) float32 in [0, 1]
    """
    (x_train, _), (x_test, _) = load_cifar10_raw()
    x_train = x_train[:num_train].astype("float32") / 255.0
    x_test = x_test[:num_test].astype("float32") / 255.0
    return x_train, x_test
