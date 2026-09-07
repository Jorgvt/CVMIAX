import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from visualize_mae import main as run_visualizations
from train_mae_demo import run_mae_training_demo


def main():
    print("===================================================================")
    print(" Experiment 05: Masked Autoencoders (MAE) for Vision (He et al., 2022)")
    print("===================================================================")

    figures_dir = os.path.join(current_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Generate masking, information routing, and ratio comparison figures
    run_visualizations()

    # 2. Run MAE training demo on CIFAR-10
    run_mae_training_demo(num_samples=2500, epochs=4, batch_size=64, mask_ratio=0.75, output_dir=figures_dir)

    print("\n[Done] All MAE visualizations and training demonstrations completed successfully.")


if __name__ == "__main__":
    main()
