"""
Master execution script for Deep Computer Vision Model Architectures.
Trains models independently, persists weights, and generates comprehensive comparative visualizations.
"""

import os
import sys
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from train_resnet import train_resnet
from train_plain_cnn import train_plain_cnn
from train_efficientnet import train_efficientnet
from train_convnext import train_convnext
from train_inception import train_inception
from compare import run_comparison


def main():
    parser = argparse.ArgumentParser(description="Train and benchmark Deep Computer Vision Architectures")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "resnet", "plain", "efficientnet", "convnext", "inception"],
        help="Which model(s) to train",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs per model")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--quick-demo", action="store_true", help="Run on 5000 subset samples for rapid test")
    args = parser.parse_args()

    print("=" * 80)
    print(" 🚀 Computer Vision Architectures Benchmark Suite")
    print("=" * 80)

    subset = 5000 if args.quick_demo else None

    # Train selected models
    if args.model in ["all", "resnet"]:
        print("\n>>> [1/5] Training ResNet-20...")
        train_resnet(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr, subset_samples=subset)

    if args.model in ["all", "plain"]:
        print("\n>>> [2/5] Training Plain-20 Forward CNN...")
        train_plain_cnn(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr, subset_samples=subset)

    if args.model in ["all", "efficientnet"]:
        print("\n>>> [3/5] Training EfficientNet-CIFAR...")
        train_efficientnet(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr, subset_samples=subset)

    if args.model in ["all", "convnext"]:
        print("\n>>> [4/5] Training ConvNeXt-CIFAR...")
        train_convnext(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr, subset_samples=subset)

    if args.model in ["all", "inception"]:
        print("\n>>> [5/5] Training Inception-CIFAR...")
        train_inception(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr, subset_samples=subset)

    # Comparison & Visualizations
    print("\n>>> Running Comparative Evaluation & Generating Figures...")
    run_comparison()


if __name__ == "__main__":
    main()
