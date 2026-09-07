import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from visualize_tasks import main as run_visualizations
from train_pretext_demo import run_rotation_pretext_training


def main():
    print("===================================================================")
    print(" Experiment 03: Self-Supervised Learning Pretext Tasks")
    print(" (Rotation Prediction, Jigsaw Puzzles, Image Colorization)")
    print("===================================================================")

    figures_dir = os.path.join(current_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Generate Input & Label Creation Visualizations for all 3 tasks + Overview Poster
    run_visualizations()

    # 2. Run SSL Pretext Training Demo on CIFAR-10
    run_rotation_pretext_training(num_samples=4000, epochs=4, batch_size=64, output_dir=figures_dir)

    print("\n[Done] All SSL visualizations and training demos completed successfully.")


if __name__ == "__main__":
    main()
