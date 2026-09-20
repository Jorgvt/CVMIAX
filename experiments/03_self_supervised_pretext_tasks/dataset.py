"""
Dataset loader for Experiment 03: Self-Supervised Pretext Tasks.
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


def load_unlabeled_cifar10(
    num_train: int = 5000,
    num_test: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads unlabeled normalized CIFAR-10 images for self-supervised training.

    Args:
        num_train: Number of training images to return.
        num_test: Number of test images to return.

    Returns:
        x_train, x_test (float32 images in [0, 1])
    """
    (x_train, _), (x_test, _) = load_cifar10_raw()
    x_train = x_train[:num_train].astype("float32") / 255.0
    x_test = x_test[:num_test].astype("float32") / 255.0
    return x_train, x_test
