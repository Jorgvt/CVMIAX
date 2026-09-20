"""
Dataset utilities for training ResNet and Plain CNN on CIFAR-10.
Delegates to central cvmiax CIFAR-10 loader (Google Drive with Keras fallback).
"""

import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
src_dir = os.path.join(repo_root, "src")
if os.path.exists(src_dir) and src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from cvmiax.datasets.cifar10 import (
    load_cifar10_raw,
    load_cifar10_data,
    create_tf_dataset,
    GDRIVE_CIFAR10_URL,
)

__all__ = [
    "load_cifar10_raw",
    "load_cifar10_data",
    "create_tf_dataset",
    "GDRIVE_CIFAR10_URL",
]
