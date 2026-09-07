# Experiment 05: Masked Autoencoders (MAE) for Vision (He et al., CVPR 2022)

## 1. Pedagogical Overview & Intuition

In Natural Language Processing (BERT), masked language modeling masks **15%** of word tokens to learn rich contextual representations. When adapting this concept to Computer Vision, **Masked Autoencoders (MAE)** mask an astonishing **75%** of image patches.

MAE is built on three core pillars:
1. **High Masking Ratio (75%)**: Eliminates local spatial redundancy, forcing the model to infer missing structures from high-level holistic semantic context rather than low-level pixel interpolation.
2. **Asymmetric Vision Transformer Architecture**: The heavy Encoder processes **only the 25% unmasked visible patches**, yielding massive computational savings ($\approx 75\%$ reduction in encoder FLOPs and memory).
3. **Lightweight Decoder**: A smaller Transformer reconstructs the full pixel values from the encoded visible tokens, shared learnable `[MASK]` tokens, and 2D positional embeddings.

---

## 2. Step-by-Step Mechanics: How Images are Masked & Reconstructed

```
Original Image (X) ──► [ Patch Partition ] ──► (N Patches of size P x P)
                                 │
                                 ▼
                     [ Random 75% Masking ]
                    ┌────────────┴────────────┐
                    ▼                         ▼
            25% Visible Patches       75% Masked Patches
                    │                         │
                    ▼                         │ (Never enters Encoder!)
          [ ViT Encoder ]                     │
                    │                         │
                    ▼                         ▼
         [ 25 Encoded Tokens ]  +  [ 75 [MASK] Tokens ]  +  [ 2D Positional Embeddings ]
                                         │
                                         ▼
                                 [ ViT Decoder ]
                                         │
                                         ▼
                             [ Reconstructed Patches ]
                                         │
                                         ▼
                 [ Loss: MSE evaluated ONLY on Masked Patches ]
```

---

## 3. Information Routing Breakdown (What the Model Uses)

### 3.1 What the ENCODER Uses
- **Inputs**: Strictly the **25% visible patches** ($x_{\text{vis}} \in \mathbb{R}^{N_{\text{vis}} \times (P^2 C)}$).
- **Positional Coordinates**: 2D positional embeddings added to each visible patch token so the self-attention layers know where each visible fragment is located in 2D space.
- **Critical Design Feature**: Masked patches **never enter the encoder**. No `[MASK]` tokens exist in the encoder, saving substantial compute and allowing massive scaling.

---

### 3.2 What the DECODER Uses
- **Encoded Visible Tokens**: The high-level semantic representation vectors output by the encoder.
- **Shared Learnable `[MASK]` Tokens**: A single learned parameter vector $\mathbf{e}_{\text{mask}} \in \mathbb{R}^{D_{\text{dec}}}$ replicated for all 75 masked positions.
- **Full 2D Positional Embeddings**: Added to **all $N$ tokens** (both visible and mask tokens), informing the decoder of the exact spatial coordinates of every patch to be reconstructed.

---

### 3.3 What the LOSS Function Uses
The loss is computed as Mean Squared Error (MSE) in pixel space **strictly over the masked patches**:
$$\mathcal{L}_{\text{MAE}} = \frac{1}{N_{\text{masked}}} \sum_{i \in \text{Masked}} \| x_i - \hat{x}_i \|_2^2$$

Because visible patches are already observed by the network, penalizing only the masked patches prevents trivial loss gaming and focuses optimization entirely on inpainting.

---

## 4. Visual Walkthrough & Empirical Figures

### 4.1 Masking Mechanism & Information Flow
![Masking Mechanism](file:///Users/jorgvt/Developer/CVMIAX/experiments/05_masked_autoencoders_mae/figures/01_mae_patch_masking_mechanism.png)

- **Panel 1**: Original image divided into a $10 \times 10$ patch grid.
- **Panel 2**: 75% random masking, leaving only 25 visible patches (highlighted in green).
- **Panel 3**: Asymmetric information routing diagram.
- **Panel 4**: Inpainted reconstruction where the model predicts missing pixel content.

---

### 4.2 Why 75% Masking Ratio? The Semantic Threshold
![Masking Ratio Comparison](file:///Users/jorgvt/Developer/CVMIAX/experiments/05_masked_autoencoders_mae/figures/02_mae_reconstruction_vs_mask_ratio.png)

- **25% Masking**: Too easy; nearby pixels leak identical texture, enabling trivial local interpolation without understanding object semantics.
- **75% Masking (The Sweet Spot)**: Eliminates spatial redundancy and forces global semantic reasoning (e.g. recognizing that a dog head requires a body and legs below).
- **90% Masking**: Too degraded; lacks sufficient contextual clues.

---

### 4.3 Information Used by Model
![Information Used by Model](file:///Users/jorgvt/Developer/CVMIAX/experiments/05_masked_autoencoders_mae/figures/03_information_used_by_model.png)

---

### 4.4 Qualitative Inpainting on CIFAR-10 Test Images
![CIFAR-10 Inpainting](file:///Users/jorgvt/Developer/CVMIAX/experiments/05_masked_autoencoders_mae/figures/04_mae_cifar_reconstructions.png)

Columns displayed:
1. **Original Ground Truth Image**
2. **75% Masked Input** (what the encoder actually receives)
3. **Full Model Prediction**
4. **Inpainting Composite** (original visible patches + model predicted masked patches).

- **Loss Convergence Curve**: [`mae_training_loss_curve.png`](file:///Users/jorgvt/Developer/CVMIAX/experiments/05_masked_autoencoders_mae/figures/mae_training_loss_curve.png) shows the steady descent of normalized MSE loss across 12 epochs on 25,000 images with Cosine Decay learning rate.

---

## 5. Master Lecture Slide Poster

![Master MAE Overview Poster](file:///Users/jorgvt/Developer/CVMIAX/experiments/05_masked_autoencoders_mae/figures/slide_mae_overview.png)

---

## 6. How to Run

```bash
# Run all visualizations and the CIFAR-10 training demo
uv run python experiments/05_masked_autoencoders_mae/run_all.py

# Run visualization generator individually
uv run python experiments/05_masked_autoencoders_mae/visualize_mae.py
```
