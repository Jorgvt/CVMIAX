# CVMIAX: Computer Vision & Deep Learning Experiments

A pedagogical repository containing experiments, proofs, and visualizations showcasing fundamental phenomena in deep learning computer vision architectures using **Keras** and **uv**.

---

## Experiments Index

### [01. Gradient Propagation & Vanishing Gradients: Plain VGG vs ResNet](file:///Users/jorgvt/Developer/CVMIAX/experiments/01_gradient_flow_vgg_vs_resnet/README.md)
- **Topic**: Demonstrates why deep feedforward sequential CNNs suffer from the vanishing gradient / degradation problem, and how residual shortcut connections $\mathcal{F}(x) + x$ create an additive gradient highway enabling arbitrary depth.
- **Key Visualizations**:
  - Layer-wise gradient norm decay (Log and Linear scales).
  - First-layer gradient magnitude collapse vs. network depth.
  - Training loss and validation accuracy curves showcasing the degradation problem on CIFAR-10.
- **Code**:
  - Models: [`models.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/01_gradient_flow_vgg_vs_resnet/models.py)
  - Gradient Analysis: [`gradient_analysis.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/01_gradient_flow_vgg_vs_resnet/gradient_analysis.py)
  - Training Comparison: [`train_comparison.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/01_gradient_flow_vgg_vs_resnet/train_comparison.py)
  - Master Runner: [`run_all.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/01_gradient_flow_vgg_vs_resnet/run_all.py)

---

### [02. Data Augmentation Guide: Good Defaults & Label-Altering Pitfalls](file:///Users/jorgvt/Developer/CVMIAX/experiments/02_data_augmentation_guide/README.md)
- **Topic**: Visual reference guide designed for course slides contrasting safe default augmentations (crops, flips, color jitter, Mixup/CutMix, geometric sync) with failure modes that alter semantic ground truth labels (text/anatomy flipping, missing small objects, diagnostic color destruction, extreme non-rigid distortion).
- **Key Slide Assets**:
  - [Good Defaults Slide Poster](file:///Users/jorgvt/Developer/CVMIAX/experiments/02_data_augmentation_guide/figures/slide_good_defaults_overview.png)
  - [Pitfalls & Failure Modes Slide Poster](file:///Users/jorgvt/Developer/CVMIAX/experiments/02_data_augmentation_guide/figures/slide_pitfalls_overview.png)
- **Code**:
  - Visualization Generator: [`generate_visualizations.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/02_data_augmentation_guide/generate_visualizations.py)

---

### [03. Self-Supervised Learning Pretext Tasks: Rotation, Jigsaw, and Colorization](file:///Users/jorgvt/Developer/CVMIAX/experiments/03_self_supervised_pretext_tasks/README.md)
- **Topic**: Visual and algorithmic walkthrough of how self-supervised pretext tasks automatically construct inputs and supervisory target labels from unlabeled images, and the visual inductive biases each task instills in the representation.
- **Key Slide Assets**:
  - [Master SSL Pretext Tasks Overview Poster](file:///Users/jorgvt/Developer/CVMIAX/experiments/03_self_supervised_pretext_tasks/figures/slide_ssl_pretext_tasks_overview.png)
  - [Rotation Prediction Task](file:///Users/jorgvt/Developer/CVMIAX/experiments/03_self_supervised_pretext_tasks/figures/01_rotation_prediction_task.png)
  - [Jigsaw Puzzle Solving Task](file:///Users/jorgvt/Developer/CVMIAX/experiments/03_self_supervised_pretext_tasks/figures/02_jigsaw_puzzle_task.png)
  - [Image Colorization Task](file:///Users/jorgvt/Developer/CVMIAX/experiments/03_self_supervised_pretext_tasks/figures/03_colorization_task.png)
- **Code**:
  - Data Generators: [`data_generators.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/03_self_supervised_pretext_tasks/data_generators.py)
  - Architectures: [`models.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/03_self_supervised_pretext_tasks/models.py)
  - Visualizer: [`visualize_tasks.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/03_self_supervised_pretext_tasks/visualize_tasks.py)
  - Training Demo: [`train_pretext_demo.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/03_self_supervised_pretext_tasks/train_pretext_demo.py)
  - Master Runner: [`run_all.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/03_self_supervised_pretext_tasks/run_all.py)

---

### [04. Embedding Space Collapse in Self-Supervised Learning](file:///Users/jorgvt/Developer/CVMIAX/experiments/04_embedding_space_collapse/README.md)
- **Topic**: Visual and mathematical examination of complete constant collapse vs. dimensional subspace collapse in Siamese representation learning, diagnostic metrics (Covariance heatmaps, SVD singular value spectra), and post-training weight zeroing.
- **Key Slide Assets**:
  - [Master Collapse Overview Poster](file:///Users/jorgvt/Developer/CVMIAX/experiments/04_embedding_space_collapse/figures/slide_embedding_collapse_overview.png)
  - [2D Latent Space Comparison](file:///Users/jorgvt/Developer/CVMIAX/experiments/04_embedding_space_collapse/figures/01_2d_embedding_space_comparison.png)
  - [Spectral & Covariance Matrix Diagnostics](file:///Users/jorgvt/Developer/CVMIAX/experiments/04_embedding_space_collapse/figures/02_spectral_analysis_and_covariance.png)
  - [Post-Training Weight Inspection](file:///Users/jorgvt/Developer/CVMIAX/experiments/04_embedding_space_collapse/figures/05_weight_distributions_and_norms.png)
  - [Empirical Siamese Training Curves](file:///Users/jorgvt/Developer/CVMIAX/experiments/04_embedding_space_collapse/figures/04_empirical_collapse_curves.png)
- **Code**:
  - Collapse Utilities: [`collapse_utils.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/04_embedding_space_collapse/collapse_utils.py)
  - Visualizer: [`visualize_collapse.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/04_embedding_space_collapse/visualize_collapse.py)
  - Empirical Siamese Training: [`train_siamese_collapse_demo.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/04_embedding_space_collapse/train_siamese_collapse_demo.py)
  - Master Runner: [`run_all.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/04_embedding_space_collapse/run_all.py)

---

### [05. Masked Autoencoders (MAE) for Vision (He et al., 2022)](file:///Users/jorgvt/Developer/CVMIAX/experiments/05_masked_autoencoders_mae/README.md)
- **Topic**: Demonstrates the masking mechanism (patch partitioning and 75% random masking), asymmetric ViT encoder-decoder computation, exact information routing, and qualitative inpainting reconstructions.
- **Key Slide Assets**:
  - [Master MAE Overview Poster](file:///Users/jorgvt/Developer/CVMIAX/experiments/05_masked_autoencoders_mae/figures/slide_mae_overview.png)
  - [Masking Mechanism & Information Flow](file:///Users/jorgvt/Developer/CVMIAX/experiments/05_masked_autoencoders_mae/figures/01_mae_patch_masking_mechanism.png)
  - [Masking Ratio Comparison (25% vs 50% vs 75% vs 90%)](file:///Users/jorgvt/Developer/CVMIAX/experiments/05_masked_autoencoders_mae/figures/02_mae_reconstruction_vs_mask_ratio.png)
  - [Information Breakdown (Encoder vs Decoder vs Loss)](file:///Users/jorgvt/Developer/CVMIAX/experiments/05_masked_autoencoders_mae/figures/03_information_used_by_model.png)
  - [CIFAR-10 Inpainting Reconstructions](file:///Users/jorgvt/Developer/CVMIAX/experiments/05_masked_autoencoders_mae/figures/04_mae_cifar_reconstructions.png)
- **Code**:
  - Patch & Mask Utilities: [`mae_utils.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/05_masked_autoencoders_mae/mae_utils.py)
  - ViT-MAE Architecture: [`models.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/05_masked_autoencoders_mae/models.py)
  - Visualizer: [`visualize_mae.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/05_masked_autoencoders_mae/visualize_mae.py)
  - Training Demo: [`train_mae_demo.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/05_masked_autoencoders_mae/train_mae_demo.py)
  - Master Runner: [`run_all.py`](file:///Users/jorgvt/Developer/CVMIAX/experiments/05_masked_autoencoders_mae/run_all.py)

---

## Interactive Notebooks & Google Colab Access

Every experiment includes a paired master script (in Jupytext `py:percent` format) and a standalone Jupyter Notebook with 1-click **Open in Colab** badges:

| Experiment | Interactive Notebook | Google Colab |
| :--- | :--- | :--- |
| **01. Gradient Propagation (ResNet vs VGG)** | [`experiment_01_gradient_flow.ipynb`](file:///Users/jorgvt/Developer/CVMIAX/experiments/01_gradient_flow_vgg_vs_resnet/experiment_01_gradient_flow.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jorgvt/CVMIAX/blob/main/experiments/01_gradient_flow_vgg_vs_resnet/experiment_01_gradient_flow.ipynb) |
| **02. Data Augmentation Guide & Pitfalls** | [`experiment_02_data_augmentation.ipynb`](file:///Users/jorgvt/Developer/CVMIAX/experiments/02_data_augmentation_guide/experiment_02_data_augmentation.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jorgvt/CVMIAX/blob/main/experiments/02_data_augmentation_guide/experiment_02_data_augmentation.ipynb) |
| **03. SSL Pretext Tasks (Rotation, Jigsaw, Color)** | [`experiment_03_pretext_tasks.ipynb`](file:///Users/jorgvt/Developer/CVMIAX/experiments/03_self_supervised_pretext_tasks/experiment_03_pretext_tasks.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jorgvt/CVMIAX/blob/main/experiments/03_self_supervised_pretext_tasks/experiment_03_pretext_tasks.ipynb) |
| **04. Embedding Space Collapse in SSL** | [`experiment_04_embedding_collapse.ipynb`](file:///Users/jorgvt/Developer/CVMIAX/experiments/04_embedding_space_collapse/experiment_04_embedding_collapse.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jorgvt/CVMIAX/blob/main/experiments/04_embedding_space_collapse/experiment_04_embedding_collapse.ipynb) |
| **05. Masked Autoencoders (MAE)** | [`experiment_05_masked_autoencoders.ipynb`](file:///Users/jorgvt/Developer/CVMIAX/experiments/05_masked_autoencoders_mae/experiment_05_masked_autoencoders.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jorgvt/CVMIAX/blob/main/experiments/05_masked_autoencoders_mae/experiment_05_masked_autoencoders.ipynb) |

---

## Getting Started & Development Workflow

### 1. Environment Setup
Clone the repository and install all dependencies (including developer tools) using `uv`:
```bash
# Clone the repository
git clone https://github.com/jorgvt/CVMIAX.git
cd CVMIAX

# Install Python environment and dependencies
uv sync --dev
```

### 2. Install Git Pre-Commit Hooks (Automatic Notebook Sync)
To ensure that `.py` master scripts and paired `.ipynb` notebooks stay in lockstep automatically during `git commit`, install the pre-commit hook:
```bash
uv run pre-commit install
```
* With this installed, whenever you modify an `experiment_*.py` script and run `git commit`, the paired `.ipynb` notebook is automatically updated and staged for you.

### 3. Running Modular Experiment Scripts
You can run any experiment's full demonstration suite via standard CLI commands:
```bash
# Experiment 01: Gradient Propagation & Vanishing Gradients (ResNet vs VGG)
uv run python experiments/01_gradient_flow_vgg_vs_resnet/run_all.py

# Experiment 02: Data Augmentation Guide & Pitfalls
uv run python experiments/02_data_augmentation_guide/generate_visualizations.py

# Experiment 03: Self-Supervised Learning Pretext Tasks
uv run python experiments/03_self_supervised_pretext_tasks/run_all.py

# Experiment 04: Embedding Space Collapse in SSL
uv run python experiments/04_embedding_space_collapse/run_all.py

# Experiment 05: Masked Autoencoders (MAE)
uv run python experiments/05_masked_autoencoders_mae/run_all.py
```

### 4. Interactive Notebooks & Jupytext Synchronization
* **Interactive Editing**: You can open and edit the master `.py` scripts directly in VS Code (with native cell code lenses `# %%`) or JupyterLab.
* **Manual Synchronization**: To manually sync all paired scripts and notebooks across the repository:
  ```bash
  uv run jupytext --sync experiments/**/*.py
  ```
* **Run Pre-Commit Checks Across All Files**:
  ```bash
  uv run pre-commit run --all-files
  ```
* **CI Verification**: GitHub Actions automatically runs `jupytext --sync` and verifies zero uncommitted diffs on every Pull Request.

