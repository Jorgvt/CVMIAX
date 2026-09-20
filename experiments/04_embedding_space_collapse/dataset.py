"""
Dataset loader for Experiment 04: Embedding Space Collapse.
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
