import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from visualize_collapse import main as run_visualizations
from train_siamese_collapse_demo import run_collapse_experiment


def main():
    print("===================================================================")
    print(" Experiment 04: Embedding Space Collapse in Self-Supervised Learning")
    print("===================================================================")

    figures_dir = os.path.join(current_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Generate 2D/spectral collapse comparison figures & lecture slide
    run_visualizations()

    # 2. Run empirical Siamese training collapse demonstration on CIFAR-10
    run_collapse_experiment(steps=150, batch_size=64, output_dir=figures_dir)

    print("\n[Done] All collapse figures and empirical demonstrations generated successfully.")


if __name__ == "__main__":
    main()
