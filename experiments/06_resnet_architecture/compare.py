"""
Comparative evaluation script for Deep Computer Vision Architectures:
ResNet, Plain CNN, EfficientNet, ConvNeXt, and Inception.
Loads stored model weights, evaluates accuracy, and produces comparison visualizations.
"""

import os
import sys
import json
from typing import Optional, Dict, Any
import numpy as np
import keras
from keras import losses, metrics

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from resnet import build_resnet20, load_resnet_weights
from plain_cnn import build_plain20, load_plain_weights
from efficientnet import build_efficientnet_cifar, load_efficientnet_weights
from convnext import build_convnext_cifar, load_convnext_weights
from inception import build_inception_cifar, load_inception_weights

from dataset import load_cifar10_data, create_tf_dataset
from visualize import (
    plot_architecture_comparison,
    plot_all_architectures_poster,
    plot_training_dynamics,
    plot_weight_distributions,
)


def run_comparison(
    weights_dir: str = os.path.join(current_dir, "weights"),
    figures_dir: str = os.path.join(current_dir, "figures"),
):
    """
    Loads saved model weights for all available architectures and plots side-by-side comparisons.
    """
    print("=" * 80)
    print(" 🔍 Comparing Pre-Trained Computer Vision Architectures on CIFAR-10")
    print("=" * 80)

    os.makedirs(figures_dir, exist_ok=True)

    # 1. Load Data for Test Evaluation
    _, _, (x_test, y_test) = load_cifar10_data(normalize=True, standardize=True)
    test_ds = create_tf_dataset(x_test, y_test, batch_size=128, augment=False, shuffle=False)

    # 2. Build Models & Load Weights
    models_dict = {
        "ResNet-20": (build_resnet20(input_shape=(32, 32, 3), num_classes=10), "resnet20_cifar10.weights.h5", load_resnet_weights, "resnet_history.json"),
        "Plain-20": (build_plain20(input_shape=(32, 32, 3), num_classes=10), "plain20_cifar10.weights.h5", load_plain_weights, "plain_history.json"),
        "EfficientNet-CIFAR": (build_efficientnet_cifar(input_shape=(32, 32, 3), num_classes=10), "efficientnet_cifar10.weights.h5", load_efficientnet_weights, "efficientnet_history.json"),
        "ConvNeXt-CIFAR": (build_convnext_cifar(input_shape=(32, 32, 3), num_classes=10), "convnext_cifar10.weights.h5", load_convnext_weights, "convnext_history.json"),
        "Inception-CIFAR": (build_inception_cifar(input_shape=(32, 32, 3), num_classes=10), "inception_cifar10.weights.h5", load_inception_weights, "inception_history.json"),
    }

    histories = {}
    print(f"\n{'Model':<22} | {'Parameters':<12} | {'Weights Status':<18} | {'Test Acc':<10} | {'Test Loss':<10}")
    print("-" * 80)

    for name, (model, weight_file, loader_fn, hist_file) in models_dict.items():
        weight_path = os.path.join(weights_dir, weight_file)
        hist_path = os.path.join(weights_dir, hist_file)
        
        status = "Not Found"
        acc_str = "N/A"
        loss_str = "N/A"

        if os.path.exists(weight_path):
            try:
                loader_fn(model, weight_path)
                model.compile(loss=losses.SparseCategoricalCrossentropy(), metrics=[metrics.SparseCategoricalAccuracy(name="accuracy")])
                test_loss, test_acc = model.evaluate(test_ds, verbose=0)
                status = "Loaded ✓"
                acc_str = f"{test_acc * 100:.2f}%"
                loss_str = f"{test_loss:.4f}"
            except Exception as e:
                status = f"Error: {e}"

        if os.path.exists(hist_path):
            try:
                with open(hist_path, "r") as f:
                    histories[name] = json.load(f)
            except Exception:
                pass

        print(f"{name:<22} | {model.count_params():<12,} | {status:<18} | {acc_str:<10} | {loss_str:<10}")

    # 3. Generate Visualizations
    print("\n🎨 Generating Comparative Visualizations...")
    
    # ResNet vs Plain Diagram
    arch_path = os.path.join(figures_dir, "01_architecture_plain_vs_resnet.png")
    plot_architecture_comparison(save_path=arch_path)

    # Master Architecture Poster covering all 5 architectures
    poster_path = os.path.join(figures_dir, "00_all_architectures_overview.png")
    plot_all_architectures_poster(save_path=poster_path)

    # Training Curves across all trained models
    if histories:
        dyn_path = os.path.join(figures_dir, "02_training_dynamics_comparison.png")
        plot_training_dynamics(histories, save_path=dyn_path)

    # Weight Distributions for ResNet vs Plain
    resnet_model = models_dict["ResNet-20"][0]
    plain_model = models_dict["Plain-20"][0]
    weights_path_fig = os.path.join(figures_dir, "03_weight_distributions.png")
    plot_weight_distributions(resnet_model, plain_model, save_path=weights_path_fig)

    print("\n✅ All comparative figures successfully generated in:", figures_dir)


if __name__ == "__main__":
    run_comparison()
