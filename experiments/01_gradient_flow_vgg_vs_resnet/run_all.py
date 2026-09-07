import os
import sys

# Ensure current directory is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from gradient_analysis import plot_layerwise_gradients, plot_gradient_vs_depth
from train_comparison import run_training_experiment


def main():
    print("===================================================================")
    print(" Experiment 01: Gradient Propagation in Plain VGG vs ResNet")
    print("===================================================================")

    # 1. Layer-wise gradient norm plot for depth=36
    layerwise_fig = os.path.join(current_dir, "gradient_norm_by_layer.png")
    plot_layerwise_gradients(depth=36, output_path=layerwise_fig)

    # 2. First-layer gradient magnitude vs depth scaling
    depth_fig = os.path.join(current_dir, "gradient_at_first_layer_vs_depth.png")
    plot_gradient_vs_depth(depths=[6, 12, 18, 24, 30, 36, 48], output_path=depth_fig)

    # 3. Training dynamics comparison (10 epochs on CIFAR-10 subset)
    training_fig = os.path.join(current_dir, "training_curves_comparison.png")
    run_training_experiment(shallow_depth=12, deep_depth=30, epochs=8, output_path=training_fig)

    print("\n[Done] All figures and benchmark results generated successfully.")


if __name__ == "__main__":
    main()
