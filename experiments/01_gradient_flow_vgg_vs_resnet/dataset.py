"""
Dataset loader for Experiment 01: Gradient Flow (VGG vs ResNet).
Delegates to central cvmiax CIFAR-10 loader.
"""

import os
import sys
from typing import Tuple
import numpy as np
import keras

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
src_dir = os.path.join(repo_root, "src")
if os.path.exists(src_dir) and src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from cvmiax.datasets.cifar10 import load_cifar10_raw, load_cifar10_data, create_tf_dataset


def get_cifar10_subset(
    num_train: int = 10000,
    num_val: int = 2000,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Loads and preprocesses CIFAR-10 data subset for fast, clear demonstration.
    """
    (x_train, y_train), (x_test, y_test) = load_cifar10_raw()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    # Subsample for fast training run in course demonstrations
    x_train_sub = x_train[:num_train]
    y_train_sub = y_train[:num_train]
    x_val_sub = x_test[:num_val]
    y_val_sub = y_test[:num_val]

    return (x_train_sub, y_train_sub), (x_val_sub, y_val_sub)
