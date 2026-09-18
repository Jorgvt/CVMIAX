"""
visualize_gaussian_pyramid.py

A standalone, publication-quality visualization suite for Gaussian and Laplacian
Pyramids in Classical Computer Vision (without deep learning), showcasing:
1. `gaussian_pyramid_01_core_scale_space.png`: Multi-scale Gaussian Pyramid decomposition,
   actual spatial dimensions, upsampled visual comparison, and 2D Fourier frequency spectra.
2. `gaussian_pyramid_02_aliasing_and_sampling.png`: Aliasing vs Anti-Aliasing analysis
   (Nyquist-Shannon Theorem, spatial Moiré patterns, and spectral folding).
3. `gaussian_pyramid_03_laplacian_pyramid_and_reconstruction.png`: Laplacian Pyramid band-pass
   decomposition, octave-by-octave detail isolation, and exact lossless reconstruction.
4. `gaussian_pyramid_04_multiresolution_blending.png`: Burt & Adelson Multiresolution Pyramid
   Blending vs Naive Cut-and-Paste vs Direct Alpha Blending.

Usage:
    uv run python visualize_gaussian_pyramid.py
"""

from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
import scipy.ndimage as ndimage
import scipy.signal as signal
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
from matplotlib.colors import Normalize

# ---------------------------------------------------------------------------
# Visual Styling Constants & Color Palette
# ---------------------------------------------------------------------------
COLOR_BG_WHITE = "#FFFFFF"
COLOR_TEXT_MAIN = "#2C3E50"
COLOR_TEXT_MUTED = "#7F8C8D"
COLOR_GRID = "#BDC3C7"

# Category color accents
COLOR_ORIGINAL = "#2C3E50"      # Slate Navy
COLOR_GAUSSIAN = "#2980B9"      # Blue (Low-Pass / Smoothing)
COLOR_LAPLACIAN = "#8E44AD"     # Purple (Band-Pass / High Frequency)
COLOR_ALIASING = "#C0392B"      # Crimson / Red (Aliasing / Artifacts)
COLOR_ANTIALIAS = "#27AE60"     # Emerald Green (Correct Anti-Aliasing)
COLOR_BLEND = "#D35400"         # Deep Orange (Multiresolution Blending)
COLOR_TEXTURE = "#16A085"       # Teal / Synthesis Accent

FONT_FAMILY = "sans-serif"

plt.rcParams.update({
    "font.family": FONT_FAMILY,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.titlesize": 15,
    "figure.facecolor": COLOR_BG_WHITE,
    "axes.facecolor": COLOR_BG_WHITE,
    "savefig.facecolor": COLOR_BG_WHITE,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# ---------------------------------------------------------------------------
# Benchmark Test Images Synthesis
# ---------------------------------------------------------------------------
def generate_rich_scene(size: int = 512) -> np.ndarray:
    """
    Generates a high-resolution, multi-frequency scene ideal for pyramid testing:
    - Multi-scale geometric structures
    - Concentric chirp zone-plate (high to low frequency sweep)
    - Directional texture gratings
    - Smooth continuous gradients and naturalistic step edges
    """
    img = np.zeros((size, size), dtype=np.float32)
    y, x = np.mgrid[0:size, 0:size]
    cy, cx = size / 2, size / 2

    # 1. Base gradient
    img += 0.3 * (x + y) / (2 * size)

    # 2. Concentric zone plate (variable radial frequency chirp) in Top-Left quadrant
    r = np.sqrt((y - size * 0.28)**2 + (x - size * 0.28)**2)
    mask_tl = (y < size * 0.52) & (x < size * 0.52)
    zone_pattern = 0.5 + 0.5 * np.cos(0.045 * (r**1.45))
    img[mask_tl] = 0.15 + 0.7 * zone_pattern[mask_tl]

    # 3. High-frequency stripe textures in Top-Right quadrant
    mask_tr = (y < size * 0.5) & (x >= size * 0.5)
    diag_stripes = 0.5 + 0.5 * np.sin(0.35 * (x + y))
    vert_stripes = 0.5 + 0.5 * np.sin(0.55 * x)
    img[mask_tr & (y < size * 0.25)] = 0.25 + 0.55 * diag_stripes[mask_tr & (y < size * 0.25)]
    img[mask_tr & (y >= size * 0.25)] = 0.25 + 0.55 * vert_stripes[mask_tr & (y >= size * 0.25)]

    # 4. Checkerboard pattern in Bottom-Left quadrant
    mask_bl = (y >= size * 0.5) & (x < size * 0.5)
    check_size = size // 24
    checker = ((x // check_size) % 2 == (y // check_size) % 2).astype(np.float32)
    img[mask_bl] = 0.2 + 0.6 * checker[mask_bl]

    # 5. Geometric object (circle with hole + step edge bar) in Bottom-Right quadrant
    r_br = np.sqrt((y - size * 0.74)**2 + (x - size * 0.74)**2)
    mask_br = (y >= size * 0.5) & (x >= size * 0.5)
    disc = ((r_br < size * 0.18) & (r_br > size * 0.05)).astype(np.float32)
    img[mask_br] = 0.15 + 0.75 * disc[mask_br]

    # Add dividing crossbars
    img[np.abs(y - cy) < 2] = 0.95
    img[np.abs(x - cx) < 2] = 0.95

    return np.clip(img, 0.0, 1.0)


def generate_chirp_texture(size: int = 512) -> np.ndarray:
    """Generates a pure 2D radial chirp zone plate for aliasing demonstrations."""
    y, x = np.mgrid[-size // 2 : size // 2, -size // 2 : size // 2]
    r = np.sqrt(x**2 + y**2)
    # Frequency increases quadratically with radius
    chirp = 0.5 + 0.5 * np.cos(np.pi * (r / (size / 2))**2 * (size / 5))
    return np.clip(chirp, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Burt-Adelson Pyramid Kernel & Reduction/Expansion Operations
# ---------------------------------------------------------------------------
def get_burt_adelson_kernel(a: float = 0.4) -> np.ndarray:
    """
    Standard Burt-Adelson (1983) 5x5 separable binomial generating kernel.
    w = [1/4 - a/2, 1/4, a, 1/4, 1/4 - a/2]
    For a = 0.4, w = [0.05, 0.25, 0.4, 0.25, 0.05] (Gaussian-like).
    """
    w1d = np.array([0.25 - a / 2.0, 0.25, a, 0.25, 0.25 - a / 2.0], dtype=np.float32)
    w2d = np.outer(w1d, w1d)
    return w2d / np.sum(w2d)


def reduce_level(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    REDUCE operation: Convolve with low-pass filter and subsample by 2.
    G_k = REDUCE(G_{k-1}) = (G_{k-1} * w)[::2, ::2]
    """
    blurred = ndimage.convolve(image, kernel, mode="reflect")
    return blurred[::2, ::2]


def expand_level(image: np.ndarray, kernel: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """
    EXPAND operation: Upsample by factor of 2 (interleaving zeros) and convolve with 4*w.
    EXPAND(G_k) = 4 * (Upsample_2(G_k) * w)
    """
    h_out, w_out = target_shape
    upsampled = np.zeros((h_out, w_out), dtype=np.float32)
    # Zero interleave
    upsampled[::2, ::2] = image[: (h_out + 1) // 2, : (w_out + 1) // 2]
    expanded = 4.0 * ndimage.convolve(upsampled, kernel, mode="reflect")
    return expanded


def build_gaussian_pyramid(image: np.ndarray, levels: int = 5, a: float = 0.4) -> List[np.ndarray]:
    """Constructs a Gaussian Pyramid with `levels` octaves."""
    kernel = get_burt_adelson_kernel(a)
    pyramid = [image.copy()]
    curr = image.copy()
    for _ in range(1, levels):
        curr = reduce_level(curr, kernel)
        pyramid.append(curr)
    return pyramid


def build_laplacian_pyramid(gaussian_pyr: List[np.ndarray], a: float = 0.4) -> List[np.ndarray]:
    """
    Constructs a Laplacian Pyramid from a Gaussian Pyramid:
    L_k = G_k - EXPAND(G_{k+1}) for k in [0, N-2]
    L_{N-1} = G_{N-1} (coarsest base residual)
    """
    kernel = get_burt_adelson_kernel(a)
    laplacian_pyr = []
    num_levels = len(gaussian_pyr)
    for k in range(num_levels - 1):
        expanded = expand_level(gaussian_pyr[k + 1], kernel, target_shape=gaussian_pyr[k].shape)
        laplacian_pyr.append(gaussian_pyr[k] - expanded)
    laplacian_pyr.append(gaussian_pyr[-1].copy())  # Coarsest residual
    return laplacian_pyr


def reconstruct_from_laplacian_pyramid(laplacian_pyr: List[np.ndarray], a: float = 0.4) -> np.ndarray:
    """
    Exact inverse reconstruction from a Laplacian Pyramid:
    G_{N-1} = L_{N-1}
    G_k = L_k + EXPAND(G_{k+1})
    """
    kernel = get_burt_adelson_kernel(a)
    curr = laplacian_pyr[-1].copy()
    for k in range(len(laplacian_pyr) - 2, -1, -1):
        expanded = expand_level(curr, kernel, target_shape=laplacian_pyr[k].shape)
        curr = laplacian_pyr[k] + expanded
    return curr


def compute_fft_magnitude(image: np.ndarray) -> np.ndarray:
    """Computes log-magnitude 2D Fourier spectrum centered at DC."""
    fft = np.fft.fftshift(np.fft.fft2(image))
    magnitude = np.log1p(np.abs(fft))
    return (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)


# ---------------------------------------------------------------------------
# FIGURE 1: Core Gaussian Pyramid & Multi-Scale Scale-Space
# ---------------------------------------------------------------------------
def plot_figure_01_core_scale_space(out_path: Path):
    """
    Figure 1: Comprehensive pedagogical Gaussian Pyramid breakdown.
    - Row 1: Physical scale cascade (actual diminishing pixel sizes)
    - Row 2: Zoomed / Normalized view highlighting blur and frequency loss
    - Row 3: 2D Fourier Frequency Spectra demonstrating progressive band-limiting
    """
    fig = plt.figure(figsize=(19, 13), facecolor=COLOR_BG_WHITE)
    
    # Header title & subtitle
    fig.suptitle(
        "The Gaussian Pyramid: Multi-Scale Image Representation & Scale Space",
        fontsize=18,
        fontweight="bold",
        color=COLOR_TEXT_MAIN,
        y=0.98,
    )
    fig.text(
        0.5,
        0.952,
        r"Iterative low-pass filtering and spatial subsampling: $G_k = \mathrm{REDUCE}(G_{k-1}) = (G_{k-1} * w)\downarrow_2$ (Burt & Adelson, 1983)",
        ha="center",
        fontsize=12,
        color=COLOR_TEXT_MUTED,
    )

    image = generate_rich_scene(512)
    pyr = build_gaussian_pyramid(image, levels=5, a=0.4)
    level_names = ["Level 0 ($G_0$)", "Level 1 ($G_1$)", "Level 2 ($G_2$)", "Level 3 ($G_3$)", "Level 4 ($G_4$)"]
    shapes = [p.shape for p in pyr]

    # Grid layout: 3 rows, 5 columns
    gs = fig.add_gridspec(3, 5, left=0.135, right=0.98, top=0.91, bottom=0.05, hspace=0.35, wspace=0.25)

    # Row headers
    row_labels = [
        "Row 1: Physical Resolution\n(Actual Diminishing Grid)",
        "Row 2: Spatial Scale-Space\n(Normalized View Size)",
        "Row 3: Frequency Domain\n(2D FFT Log Spectrum)"
    ]

    for row_idx, r_label in enumerate(row_labels):
        fig.text(
            0.015,
            0.78 - row_idx * 0.31,
            r_label,
            va="center",
            ha="left",
            fontsize=10.0,
            fontweight="bold",
            color=COLOR_TEXT_MAIN,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F4F6F7", edgecolor=COLOR_GRID, lw=1.2),
        )

    # Row 1: Actual proportional sizes inside a fixed 512x512 canvas
    for i in range(5):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor(COLOR_BG_WHITE)
        
        # Place in center of a 512x512 white canvas
        canvas = np.ones((512, 512), dtype=np.float32)
        cur_img = pyr[i]
        h, w = cur_img.shape
        y_start = (512 - h) // 2
        x_start = (512 - w) // 2
        canvas[y_start:y_start + h, x_start:x_start + w] = cur_img
        
        ax.imshow(canvas, cmap="gray", vmin=0, vmax=1)
        
        # Border around actual content
        rect = patches.Rectangle((x_start, y_start), w, h, linewidth=1.5, edgecolor=COLOR_GAUSSIAN, facecolor="none")
        ax.add_patch(rect)
        
        ax.set_title(f"{level_names[i]}\nSize: {shapes[i][0]}×{shapes[i][1]} px", fontsize=11, fontweight="bold", color=COLOR_GAUSSIAN)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(COLOR_GRID)
            spine.set_linewidth(1.0)
            
        if i < 4:
            ax.set_xlabel(r"$\downarrow 2 \circ \mathrm{Blur}$", fontsize=11, color=COLOR_TEXT_MUTED, fontweight="bold")

    # Row 2: Scaled up to full frame to inspect octave blurring & loss of fine details
    for i in range(5):
        ax = fig.add_subplot(gs[1, i])
        ax.set_facecolor(COLOR_BG_WHITE)
        
        # Nearest neighbor interpolation to show pixelation at coarse scales
        ax.imshow(pyr[i], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        
        scale_factor = 2**i
        sigma_eff = f"{scale_factor:.1f}"
        ax.set_title(fr"Scale $\sigma \approx {sigma_eff}$ ({shapes[i][0]}×{shapes[i][1]})", fontsize=11, fontweight="bold", color=COLOR_TEXT_MAIN)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(COLOR_GAUSSIAN)
            spine.set_linewidth(1.5)
        
        if i == 0:
            ax.set_xlabel("Full Frequency Bandwidth", fontsize=9.5, color=COLOR_TEXT_MUTED)
        elif i == 1:
            ax.set_xlabel("Minor High-Freq Attenuation", fontsize=9.5, color=COLOR_TEXT_MUTED)
        elif i == 2:
            ax.set_xlabel("Mid-Frequencies Suppressed", fontsize=9.5, color=COLOR_TEXT_MUTED)
        elif i == 3:
            ax.set_xlabel("Only Coarse Edges Retained", fontsize=9.5, color=COLOR_TEXT_MUTED)
        elif i == 4:
            ax.set_xlabel("DC & Global Shading Only", fontsize=9.5, color=COLOR_TEXT_MUTED)

    # Row 3: 2D FFT Log-Magnitude
    for i in range(5):
        ax = fig.add_subplot(gs[2, i])
        ax.set_facecolor(COLOR_BG_WHITE)
        
        fft_mag = compute_fft_magnitude(pyr[i])
        im = ax.imshow(fft_mag, cmap="magma", vmin=0, vmax=1)
        
        ax.set_title(f"Spectrum $|\mathcal{{F}}(G_{i})|$\nNyquist: $f_s/{2**i}$", fontsize=10.5, fontweight="bold", color=COLOR_LAPLACIAN)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(COLOR_LAPLACIAN)
            spine.set_linewidth(1.2)
        ax.set_xlabel(f"BW: $0 \dots \pi/{2**i}$ rad", fontsize=9.5, color=COLOR_TEXT_MUTED)

    plt.savefig(out_path, dpi=300, facecolor=COLOR_BG_WHITE)
    plt.close()
    print(f"  [1/4] Core Gaussian Pyramid created: {out_path}")


# ---------------------------------------------------------------------------
# FIGURE 2: Aliasing vs. Anti-Aliasing (Nyquist-Shannon Theorem)
# ---------------------------------------------------------------------------
def plot_figure_02_aliasing_and_sampling(out_path: Path):
    """
    Figure 2: Explains the fundamental mathematical justification for Gaussian
    blurring before subsampling: preventing aliasing and spectral folding.
    """
    fig = plt.figure(figsize=(18, 12), facecolor=COLOR_BG_WHITE)
    
    fig.suptitle(
        "Why Blur Before Downsampling? Aliasing vs. Gaussian Anti-Aliasing",
        fontsize=17,
        fontweight="bold",
        color=COLOR_TEXT_MAIN,
        y=0.98,
    )
    fig.text(
        0.5,
        0.952,
        r"Nyquist-Shannon Theorem: Frequencies above $f_s/2$ fold back into lower frequencies as spurious Moiré patterns without low-pass filtering.",
        ha="center",
        fontsize=11.5,
        color=COLOR_TEXT_MUTED,
    )

    chirp = generate_chirp_texture(512)
    kernel = get_burt_adelson_kernel(a=0.4)

    # 1. Direct naive subsampling without blur (factor 4)
    naive_down_2 = chirp[::2, ::2]
    naive_down_4 = chirp[::4, ::4]

    # 2. Gaussian anti-aliased subsampling (factor 4)
    aa_down_1 = reduce_level(chirp, kernel)
    aa_down_2 = reduce_level(aa_down_1, kernel)

    gs = fig.add_gridspec(2, 4, left=0.06, right=0.98, top=0.91, bottom=0.07, hspace=0.32, wspace=0.25)

    # Panel 1: Original High-Frequency Zone Plate
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(chirp, cmap="gray", vmin=0, vmax=1)
    ax1.set_title("Original Zone Plate\n($512 \\times 512$, Continuous Chirp)", fontweight="bold", color=COLOR_ORIGINAL)
    ax1.set_xticks([]); ax1.set_yticks([])
    ax1.set_xlabel("High radial spatial frequencies", fontsize=9.5, color=COLOR_TEXT_MUTED)

    # Panel 2: Naive Downsample (4x) - Aliased
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(naive_down_4, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax2.set_title("Naive Subsampling ($4\\times$ Down)\nWITHOUT Gaussian Pre-Filter", fontweight="bold", color=COLOR_ALIASING)
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.set_xlabel("Severe False Moiré Patterns!", fontsize=9.5, fontweight="bold", color=COLOR_ALIASING)
    for spine in ax2.spines.values():
        spine.set_color(COLOR_ALIASING); spine.set_linewidth(2.0)

    # Panel 3: Anti-Aliased Gaussian Downsample (4x)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(aa_down_2, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax3.set_title("Gaussian Anti-Aliased ($4\\times$ Down)\nWITH $\mathrm{REDUCE} = \downarrow 4 \\circ G_\\sigma$", fontweight="bold", color=COLOR_ANTIALIAS)
    ax3.set_xticks([]); ax3.set_yticks([])
    ax3.set_xlabel("Clean bandlimited attenuation", fontsize=9.5, fontweight="bold", color=COLOR_ANTIALIAS)
    for spine in ax3.spines.values():
        spine.set_color(COLOR_ANTIALIAS); spine.set_linewidth(2.0)

    # Panel 4: Direct Difference Map / Error Artifacts
    ax4 = fig.add_subplot(gs[0, 3])
    # Upsample both to compare artifacts directly
    diff = np.abs(naive_down_4.astype(np.float32) - aa_down_2.astype(np.float32))
    im4 = ax4.imshow(diff, cmap="hot", vmin=0, vmax=0.8)
    ax4.set_title("Aliasing Error Magnitude\n$|I_{\\mathrm{naive}} - I_{\\mathrm{antialiased}}|$", fontweight="bold", color=COLOR_ALIASING)
    ax4.set_xticks([]); ax4.set_yticks([])
    ax4.set_xlabel("Spurious Energy / Hallucinated Patterns", fontsize=9.5, color=COLOR_TEXT_MUTED)
    cbar4 = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    cbar4.ax.tick_params(labelsize=8)

    # Row 2: Spectral & 1D Profile Evidence
    # Panel 5: Original FFT Spectrum
    ax5 = fig.add_subplot(gs[1, 0])
    fft_orig = compute_fft_magnitude(chirp)
    ax5.imshow(fft_orig, cmap="magma", vmin=0, vmax=1)
    ax5.set_title("Original 2D Spectrum\nFull Frequency Circle", fontweight="bold", color=COLOR_ORIGINAL)
    ax5.set_xticks([]); ax5.set_yticks([])
    ax5.set_xlabel("High bandwidth extending to edge", fontsize=9.5, color=COLOR_TEXT_MUTED)

    # Panel 6: Naive FFT Spectrum (Folded / Aliased)
    ax6 = fig.add_subplot(gs[1, 1])
    fft_naive = compute_fft_magnitude(naive_down_4)
    ax6.imshow(fft_naive, cmap="magma", vmin=0, vmax=1)
    ax6.set_title("Naive Spectrum (Aliased)\nSpectral Overlap & Folding", fontweight="bold", color=COLOR_ALIASING)
    ax6.set_xticks([]); ax6.set_yticks([])
    ax6.set_xlabel("High-frequencies folded into baseband", fontsize=9.5, color=COLOR_ALIASING)
    for spine in ax6.spines.values():
        spine.set_color(COLOR_ALIASING); spine.set_linewidth(1.8)

    # Panel 7: Anti-Aliased FFT Spectrum (Cleanly Bandlimited)
    ax7 = fig.add_subplot(gs[1, 2])
    fft_aa = compute_fft_magnitude(aa_down_2)
    ax7.imshow(fft_aa, cmap="magma", vmin=0, vmax=1)
    ax7.set_title("Anti-Aliased Spectrum\nBandlimited Low-Pass Disk", fontweight="bold", color=COLOR_ANTIALIAS)
    ax7.set_xticks([]); ax7.set_yticks([])
    ax7.set_xlabel("Zero spectral overlap / No folding", fontsize=9.5, color=COLOR_ANTIALIAS)
    for spine in ax7.spines.values():
        spine.set_color(COLOR_ANTIALIAS); spine.set_linewidth(1.8)

    # Panel 8: 1D Radial Cross-Section Profile
    ax8 = fig.add_subplot(gs[1, 3])
    mid_idx_naive = naive_down_4.shape[0] // 2
    mid_idx_aa = aa_down_2.shape[0] // 2
    x_axis = np.arange(naive_down_4.shape[1])
    
    ax8.plot(x_axis, naive_down_4[mid_idx_naive, :], color=COLOR_ALIASING, lw=1.8, label="Naive (Aliased Spurious Peaks)")
    ax8.plot(x_axis, aa_down_2[mid_idx_aa, :], color=COLOR_ANTIALIAS, lw=2.2, label="Gaussian Anti-Aliased (Smooth)")
    ax8.set_title("1D Spatial Intensity Profile\nCenter Cross-Section Slice", fontweight="bold", color=COLOR_TEXT_MAIN)
    ax8.set_xlabel("Subsampled Pixel Coordinate (x)", fontsize=9.5, color=COLOR_TEXT_MAIN)
    ax8.set_ylabel("Intensity", fontsize=9.5, color=COLOR_TEXT_MAIN)
    ax8.set_ylim(-0.1, 1.1)
    ax8.grid(True, linestyle="--", alpha=0.5, color=COLOR_GRID)
    ax8.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    for spine in ax8.spines.values():
        spine.set_color(COLOR_GRID)

    plt.savefig(out_path, dpi=300, facecolor=COLOR_BG_WHITE)
    plt.close()
    print(f"  [2/4] Aliasing & Sampling figure created: {out_path}")


# ---------------------------------------------------------------------------
# FIGURE 3: Laplacian Pyramid & Lossless Reconstruction
# ---------------------------------------------------------------------------
def plot_figure_03_laplacian_pyramid(out_path: Path):
    """
    Figure 3: Detailed pedagogical breakdown of the Laplacian Pyramid:
    - Band-pass decomposition: L_k = G_k - EXPAND(G_{k+1})
    - Visualizing fine edges, intermediate textures, and coarse residual
    - Step-by-step exact mathematical reconstruction pipeline with numerical verification
    """
    fig = plt.figure(figsize=(19, 13), facecolor=COLOR_BG_WHITE)
    
    fig.suptitle(
        "The Laplacian Pyramid: Band-Pass Decomposition & Lossless Reconstruction",
        fontsize=18,
        fontweight="bold",
        color=COLOR_TEXT_MAIN,
        y=0.98,
    )
    fig.text(
        0.5,
        0.952,
        r"Band-pass detail: $L_k = G_k - \mathrm{EXPAND}(G_{k+1})$. Lossless reconstruction: $G_k = L_k + \mathrm{EXPAND}(G_{k+1})$",
        ha="center",
        fontsize=12,
        color=COLOR_TEXT_MUTED,
    )

    image = generate_rich_scene(512)
    g_pyr = build_gaussian_pyramid(image, levels=4, a=0.4)
    l_pyr = build_laplacian_pyramid(g_pyr, a=0.4)
    recon = reconstruct_from_laplacian_pyramid(l_pyr, a=0.4)

    # Calculate reconstruction error
    recon_error = np.abs(image - recon)
    max_err = np.max(recon_error)

    gs = fig.add_gridspec(3, 4, left=0.135, right=0.98, top=0.91, bottom=0.05, hspace=0.36, wspace=0.25)

    # Row 1: Gaussian Pyramid Stages (G0 -> G1 -> G2 -> G3)
    # Row 2: Laplacian Pyramid Band-Pass Details (L0, L1, L2, G3_residual)
    # Row 3: Step-by-Step Inverse Reconstruction Pipeline
    
    row_labels = [
        "Row 1: Gaussian Pyramid\n$G_k = \\mathrm{REDUCE}(G_{k-1})$",
        "Row 2: Laplacian Pyramid\n$L_k = G_k - \\mathrm{EXPAND}(G_{k+1})$",
        "Row 3: Exact Reconstruction\n$G_k = L_k + \\mathrm{EXPAND}(G_{k+1})$"
    ]

    for row_idx, r_label in enumerate(row_labels):
        fig.text(
            0.015,
            0.78 - row_idx * 0.31,
            r_label,
            va="center",
            ha="left",
            fontsize=10.0,
            fontweight="bold",
            color=COLOR_TEXT_MAIN,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F4F6F7", edgecolor=COLOR_GRID, lw=1.2),
        )

    # Plot Row 1: Gaussian Pyramid
    for i in range(4):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor(COLOR_BG_WHITE)
        ax.imshow(g_pyr[i], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Gaussian Level $G_{i}$\n({g_pyr[i].shape[0]}×{g_pyr[i].shape[1]} px)", fontsize=11, fontweight="bold", color=COLOR_GAUSSIAN)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(COLOR_GAUSSIAN); spine.set_linewidth(1.3)
        if i < 3:
            ax.set_xlabel(r"$\downarrow 2 \text{ Blur + Subsample}$", fontsize=9.5, color=COLOR_TEXT_MUTED)
        else:
            ax.set_xlabel("Coarsest Base Residual", fontsize=9.5, color=COLOR_GAUSSIAN, fontweight="bold")

    # Plot Row 2: Laplacian Pyramid (Diverging Colormap coolwarm centered at 0)
    for i in range(4):
        ax = fig.add_subplot(gs[1, i])
        ax.set_facecolor(COLOR_BG_WHITE)
        
        if i < 3:
            # Symmetrize color scale around 0
            vmax = max(0.01, np.percentile(np.abs(l_pyr[i]), 99))
            im = ax.imshow(l_pyr[i], cmap="coolwarm", vmin=-vmax, vmax=vmax)
            ax.set_title(f"Laplacian Level $L_{i}$ (Band-Pass)\n({l_pyr[i].shape[0]}×{l_pyr[i].shape[1]} px)", fontsize=11, fontweight="bold", color=COLOR_LAPLACIAN)
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=7.5)
        else:
            # Coarsest level is just the lowest-pass Gaussian base residual
            im = ax.imshow(l_pyr[i], cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"Residual Base $L_3 = G_3$\n({l_pyr[i].shape[0]}×{l_pyr[i].shape[1]} px)", fontsize=11, fontweight="bold", color=COLOR_LAPLACIAN)
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=7.5)

        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(COLOR_LAPLACIAN); spine.set_linewidth(1.3)
            
        if i == 0:
            ax.set_xlabel("Highest Frequency Edges & Noise", fontsize=9, color=COLOR_TEXT_MUTED)
        elif i == 1:
            ax.set_xlabel("Mid-Frequency Textures & Contours", fontsize=9, color=COLOR_TEXT_MUTED)
        elif i == 2:
            ax.set_xlabel("Low-Frequency Structural Outlines", fontsize=9, color=COLOR_TEXT_MUTED)
        elif i == 3:
            ax.set_xlabel("Global DC / Average Illumination", fontsize=9, color=COLOR_TEXT_MUTED)

    # Plot Row 3: Step-by-step reconstruction
    # 3.0: Start at G3
    # 3.1: Reconstruct G2 = L2 + EXPAND(G3)
    # 3.2: Reconstruct G1 = L1 + EXPAND(G2)
    # 3.3: Final Reconstructed Image G0 & Residual Error Badge
    kernel = get_burt_adelson_kernel(0.4)
    step_g3 = l_pyr[3].copy()
    step_g2 = l_pyr[2] + expand_level(step_g3, kernel, l_pyr[2].shape)
    step_g1 = l_pyr[1] + expand_level(step_g2, kernel, l_pyr[1].shape)
    step_g0 = l_pyr[0] + expand_level(step_g1, kernel, l_pyr[0].shape)

    reconstruct_steps = [step_g3, step_g2, step_g1, step_g0]
    step_titles = [
        "1. Base Level: $G_3 = L_3$\n($64 \\times 64$)",
        "2. Step 1: $G_2 = L_2 + \\mathrm{EXP}(G_3)$\n($128 \\times 128$)",
        "3. Step 2: $G_1 = L_1 + \\mathrm{EXP}(G_2)$\n($256 \\times 256$)",
        "4. Final: $G_0 = L_0 + \\mathrm{EXP}(G_1)$\n($512 \\times 512$ - Perfect Recall)",
    ]

    for i in range(4):
        ax = fig.add_subplot(gs[2, i])
        ax.set_facecolor(COLOR_BG_WHITE)
        ax.imshow(reconstruct_steps[i], cmap="gray", vmin=0, vmax=1)
        ax.set_title(step_titles[i], fontsize=11, fontweight="bold", color=COLOR_TEXT_MAIN)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(COLOR_TEXT_MAIN if i < 3 else COLOR_ANTIALIAS)
            spine.set_linewidth(1.3 if i < 3 else 2.0)

        if i == 3:
            ax.set_xlabel(f"Max Absolute Error $L_\\infty < {max_err:.2e}$", fontsize=9.5, fontweight="bold", color=COLOR_ANTIALIAS)
        else:
            ax.set_xlabel(r"$\uparrow 2 \text{ Upsample + Add Band-pass}$", fontsize=9, color=COLOR_TEXT_MUTED)

    plt.savefig(out_path, dpi=300, facecolor=COLOR_BG_WHITE)
    plt.close()
    print(f"  [3/4] Laplacian Pyramid figure created: {out_path}")


# ---------------------------------------------------------------------------
# FIGURE 4: Multiresolution Pyramid Blending (Burt-Adelson 1983)
# ---------------------------------------------------------------------------
def plot_figure_04_pyramid_blending(out_path: Path):
    """
    Figure 4: Demonstrates Burt & Adelson's multiresolution image blending
    (the 'apple + orange' / 'multiscale spline' paradigm).
    - Source Image A (Textured stripes / organic pattern)
    - Source Image B (Checkerboard / geometric pattern)
    - Hard cut vs Naive alpha blend vs Multiresolution Laplacian blend
    - Step-by-step breakdown across pyramid octaves showing why it prevents ghosting and sharp seams.
    """
    fig = plt.figure(figsize=(19, 13), facecolor=COLOR_BG_WHITE)
    
    fig.suptitle(
        "Multiresolution Image Splining: Seamless Blending via Laplacian Pyramids",
        fontsize=18,
        fontweight="bold",
        color=COLOR_TEXT_MAIN,
        y=0.98,
    )
    fig.text(
        0.5,
        0.952,
        r"Burt & Adelson (1983): High-frequency edges blend over a narrow spatial width, while low frequencies blend smoothly across a broad window.",
        ha="center",
        fontsize=11.5,
        color=COLOR_TEXT_MUTED,
    )

    size = 512
    y, x = np.mgrid[0:size, 0:size]

    # Image A: Concentric / radial textured rings & soft gradients
    r = np.sqrt((x - size * 0.4)**2 + (y - size * 0.5)**2)
    img_a = 0.5 + 0.45 * np.cos(0.08 * r) * np.exp(-r / (size * 0.8)) + 0.1 * (x / size)
    img_a = np.clip(img_a, 0.0, 1.0).astype(np.float32)

    # Image B: High-contrast directional diamond checkerboard & fine stripes
    rot_x = (x + y) / np.sqrt(2)
    rot_y = (-x + y) / np.sqrt(2)
    check_b = ((rot_x // 20) % 2 == (rot_y // 20) % 2).astype(np.float32)
    img_b = 0.2 + 0.6 * check_b + 0.15 * np.sin(0.4 * x)
    img_b = np.clip(img_b, 0.0, 1.0).astype(np.float32)

    # Mask M: Step split along diagonal / vertical center
    mask = (x >= size // 2).astype(np.float32)

    # 1. Naive Cut & Paste
    naive_cut = img_a * (1.0 - mask) + img_b * mask

    # 2. Naive Linear Alpha Blend (blurred mask directly applied in spatial domain)
    smooth_mask = ndimage.gaussian_filter(mask, sigma=30)
    naive_alpha = img_a * (1.0 - smooth_mask) + img_b * smooth_mask

    # 3. Burt-Adelson Multiresolution Pyramid Blend
    levels = 5
    kernel = get_burt_adelson_kernel(0.4)
    g_pyr_a = build_gaussian_pyramid(img_a, levels=levels, a=0.4)
    g_pyr_b = build_gaussian_pyramid(img_b, levels=levels, a=0.4)
    g_pyr_m = build_gaussian_pyramid(mask, levels=levels, a=0.4)

    l_pyr_a = build_laplacian_pyramid(g_pyr_a, a=0.4)
    l_pyr_b = build_laplacian_pyramid(g_pyr_b, a=0.4)

    # Blend each band-pass level using the corresponding Gaussian mask octave
    l_pyr_blend = []
    for k in range(levels):
        w_m = g_pyr_m[k]
        # Blend: L_blend = (1 - M)*L_A + M*L_B
        l_blended = l_pyr_a[k] * (1.0 - w_m) + l_pyr_b[k] * w_m
        l_pyr_blend.append(l_blended)

    # Reconstruct blended composite
    pyramid_blend = reconstruct_from_laplacian_pyramid(l_pyr_blend, a=0.4)
    pyramid_blend = np.clip(pyramid_blend, 0.0, 1.0)

    # Layout: 2 rows of 4 panels
    # Row 1: Source A, Source B, Mask, Naive Cut
    # Row 2: Naive Alpha (Ghosting), Pyramid Blended (Seamless), Zoom Comparison, Octave Detail
    gs = fig.add_gridspec(2, 4, left=0.06, right=0.98, top=0.91, bottom=0.06, hspace=0.32, wspace=0.25)

    # Panel 1: Image A
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_a, cmap="gray", vmin=0, vmax=1)
    ax1.set_title("Source Image A ($I_A$)\n(Radial Wave Texture)", fontweight="bold", color=COLOR_TEXT_MAIN)
    ax1.set_xticks([]); ax1.set_yticks([])

    # Panel 2: Image B
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(img_b, cmap="gray", vmin=0, vmax=1)
    ax2.set_title("Source Image B ($I_B$)\n(Checkerboard Diamond Texture)", fontweight="bold", color=COLOR_TEXT_MAIN)
    ax2.set_xticks([]); ax2.set_yticks([])

    # Panel 3: Binary Mask M
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(mask, cmap="gray", vmin=0, vmax=1)
    ax3.set_title("Binary Splitting Mask ($M$)\n(Hard Boundary at $x=256$)", fontweight="bold", color=COLOR_TEXT_MAIN)
    ax3.set_xticks([]); ax3.set_yticks([])

    # Panel 4: Naive Cut and Paste
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(naive_cut, cmap="gray", vmin=0, vmax=1)
    ax4.set_title("Naive Direct Cut & Paste\n$I_{\\mathrm{cut}} = (1-M)I_A + MI_B$", fontweight="bold", color=COLOR_ALIASING)
    ax4.set_xticks([]); ax4.set_yticks([])
    ax4.set_xlabel("Severe Sharp Edge Seam Artifact!", fontsize=9.5, fontweight="bold", color=COLOR_ALIASING)
    for spine in ax4.spines.values():
        spine.set_color(COLOR_ALIASING); spine.set_linewidth(1.8)

    # Panel 5: Naive Spatial Alpha Blend (Single-Scale Gaussian Mask)
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.imshow(naive_alpha, cmap="gray", vmin=0, vmax=1)
    ax5.set_title("Single-Scale Alpha Blending\n$I_{\\mathrm{alpha}} = (1-G_\\sigma*M)I_A + (G_\\sigma*M)I_B$", fontweight="bold", color=COLOR_ALIASING)
    ax5.set_xticks([]); ax5.set_yticks([])
    ax5.set_xlabel("Noticeable 'Ghosting' / Double-Exposure", fontsize=9.5, fontweight="bold", color=COLOR_ALIASING)
    for spine in ax5.spines.values():
        spine.set_color(COLOR_ALIASING); spine.set_linewidth(1.8)

    # Panel 6: Laplacian Pyramid Blending (Multiresolution Spline)
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.imshow(pyramid_blend, cmap="gray", vmin=0, vmax=1)
    ax6.set_title("Laplacian Pyramid Blend\n(Burt & Adelson Multiresolution Spline)", fontweight="bold", color=COLOR_ANTIALIAS)
    ax6.set_xticks([]); ax6.set_yticks([])
    ax6.set_xlabel("No Seam & No Ghosting Artifacts!", fontsize=9.5, fontweight="bold", color=COLOR_ANTIALIAS)
    for spine in ax6.spines.values():
        spine.set_color(COLOR_ANTIALIAS); spine.set_linewidth(2.2)

    # Panel 7: Boundary Zoom Comparison (Naive Cut vs Alpha vs Pyramid)
    ax7 = fig.add_subplot(gs[1, 2])
    # Extract vertical slice around boundary x in [220, 292], y in [200, 312]
    crop_cut = naive_cut[200:312, 220:292]
    crop_alpha = naive_alpha[200:312, 220:292]
    crop_pyr = pyramid_blend[200:312, 220:292]
    
    # Concatenate side by side with thin separators
    h_c, w_c = crop_cut.shape
    sep = np.ones((h_c, 3), dtype=np.float32)
    triptych = np.hstack([crop_cut, sep, crop_alpha, sep, crop_pyr])
    
    ax7.imshow(triptych, cmap="gray", vmin=0, vmax=1)
    ax7.set_title("Seam Close-Up ($112 \\times 72$ px)\nCut  |  Alpha  |  Pyramid", fontweight="bold", color=COLOR_TEXT_MAIN)
    ax7.set_xticks([])
    ax7.set_yticks([])
    ax7.set_xlabel("Left: Hard Seam | Mid: Ghost | Right: Seamless", fontsize=9.0, color=COLOR_TEXT_MUTED)
    for spine in ax7.spines.values():
        spine.set_color(COLOR_GRID)

    # Panel 8: Band-Pass Blended Laplacian Level L_1
    ax8 = fig.add_subplot(gs[1, 3])
    # Show L_1 blended octave
    vmax_l = max(0.01, np.percentile(np.abs(l_pyr_blend[1]), 99))
    im8 = ax8.imshow(l_pyr_blend[1], cmap="coolwarm", vmin=-vmax_l, vmax=vmax_l)
    ax8.set_title("Blended Laplacian Octave $L_{\\mathrm{blend}, 1}$\nScale-Dependent Frequency Merging", fontweight="bold", color=COLOR_LAPLACIAN)
    ax8.set_xticks([]); ax8.set_yticks([])
    ax8.set_xlabel(r"$L_{\mathrm{blend}, k} = (1-G_{M,k})L_{A,k} + G_{M,k}L_{B,k}$", fontsize=9.0, color=COLOR_LAPLACIAN)
    cbar8 = plt.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04)
    cbar8.ax.tick_params(labelsize=8)
    for spine in ax8.spines.values():
        spine.set_color(COLOR_LAPLACIAN); spine.set_linewidth(1.5)

    plt.savefig(out_path, dpi=300, facecolor=COLOR_BG_WHITE)
    plt.close()
    print(f"  [4/4] Pyramid Blending figure created: {out_path}")


# ---------------------------------------------------------------------------
# FIGURE 5: 3D Stacked Pyramid Geometry & Dual-Pyramid Flowchart
# ---------------------------------------------------------------------------
def plot_figure_05_pyramid_hierarchy_overview(out_path: Path):
    """
    Figure 5: Master architectural overview:
    - Left: 3D isometric stacked pyramid structure showing diminishing image plates
      hovering in scale-space with projection edges connecting to the apex.
    - Right: Complete mathematical dataflow diagram contrasting REDUCE (analysis)
      and EXPAND (synthesis) with Laplacian residual extraction.
    """
    fig = plt.figure(figsize=(19, 11), facecolor=COLOR_BG_WHITE)

    fig.suptitle(
        "Pyramid Architecture: 3D Scale-Space Geometry & Dual-Pyramid Dataflow",
        fontsize=18,
        fontweight="bold",
        color=COLOR_TEXT_MAIN,
        y=0.96,
    )
    fig.text(
        0.5,
        0.915,
        r"Scale-space representation builds an octave hierarchy of octave-halved resolutions: $N \times N \rightarrow \frac{N}{2} \times \frac{N}{2} \dots$",
        ha="center",
        fontsize=11.5,
        color=COLOR_TEXT_MUTED,
    )

    image = generate_rich_scene(256)
    pyr = build_gaussian_pyramid(image, levels=4, a=0.4)

    # 3D axes for isometric stacked pyramid
    ax_3d = fig.add_axes([0.02, 0.18, 0.44, 0.72], projection="3d", facecolor=COLOR_BG_WHITE)
    ax_3d.set_facecolor(COLOR_BG_WHITE)

    # Plot each pyramid level as a horizontal textured plane in 3D
    levels = 4
    z_spacings = [0.0, 1.2, 2.4, 3.6]
    half_widths = [1.0, 0.5, 0.25, 0.125]
    level_colors = [COLOR_GAUSSIAN, COLOR_GAUSSIAN, COLOR_GAUSSIAN, COLOR_GAUSSIAN]

    # Plot the 4 apex connecting guide lines
    apex_z = 4.8
    for corner_x, corner_y in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        ax_3d.plot(
            [corner_x, 0],
            [corner_y, 0],
            [0, apex_z],
            color=COLOR_GRID,
            linestyle=":",
            linewidth=1.2,
            alpha=0.7,
        )

    for i in range(levels):
        z = z_spacings[i]
        hw = half_widths[i]
        cur_img = pyr[i]
        h_i, w_i = cur_img.shape

        # Create meshgrid for current plate
        x_p = np.linspace(-hw, hw, w_i)
        y_p = np.linspace(-hw, hw, h_i)
        xx, yy = np.meshgrid(x_p, y_p)
        zz = np.full_like(xx, z)

        # Normalize grayscale image to RGB for 3D surface
        img_rgb = cm.gray(cur_img)[:, :, :3]

        ax_3d.plot_surface(
            xx,
            yy,
            zz,
            facecolors=img_rgb,
            rstride=max(1, w_i // 64),
            cstride=max(1, h_i // 64),
            shade=False,
            alpha=0.95,
        )

        # Border wireframe around the plate
        border_x = [-hw, hw, hw, -hw, -hw]
        border_y = [-hw, -hw, hw, hw, -hw]
        border_z = [z, z, z, z, z]
        ax_3d.plot(border_x, border_y, border_z, color=COLOR_GAUSSIAN, linewidth=2.0)

    # 3D view styling
    ax_3d.view_init(elev=24, azim=-55)
    ax_3d.set_xlim(-1.2, 1.2)
    ax_3d.set_ylim(-1.2, 1.2)
    ax_3d.set_zlim(-0.2, 4.8)
    ax_3d.set_axis_off()
    ax_3d.set_title("3D Scale-Space Octave Cascade", fontsize=13, fontweight="bold", color=COLOR_GAUSSIAN, pad=10)

    # Level Badges Underneath 3D Plot
    badge_x = np.linspace(0.04, 0.42, 4)
    badge_texts = [
        "Level 0 ($G_0$)\n$256\\times 256$ px\nScale $\\sigma = 1.0$",
        "Level 1 ($G_1$)\n$128\\times 128$ px\nScale $\\sigma = 2.0$",
        "Level 2 ($G_2$)\n$64\\times 64$ px\nScale $\\sigma = 4.0$",
        "Level 3 ($G_3$)\n$32\\times 32$ px\nScale $\\sigma = 8.0$"
    ]
    for bx, btxt in zip(badge_x, badge_texts):
        fig.text(
            bx,
            0.065,
            btxt,
            ha="center",
            va="bottom",
            fontsize=9.0,
            fontweight="bold",
            color=COLOR_TEXT_MAIN,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F4F6F7", edgecolor=COLOR_GAUSSIAN, lw=1.2),
        )

    # Right side: 2D Flowchart & Mathematics
    ax_flow = fig.add_axes([0.48, 0.08, 0.49, 0.82], facecolor=COLOR_BG_WHITE)
    ax_flow.set_facecolor(COLOR_BG_WHITE)
    ax_flow.set_xlim(0, 100)
    ax_flow.set_ylim(0, 100)
    ax_flow.axis("off")

    # Draw Flowchart Boxes
    def draw_box(x, y, w, h, text, color, header=""):
        rect = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=1.0", facecolor="#FFFFFF", edgecolor=color, linewidth=2.0
        )
        ax_flow.add_patch(rect)
        if header:
            ax_flow.text(x + w / 2, y + h - 4, header, ha="center", va="top", fontsize=10.5, fontweight="bold", color=color)
            ax_flow.text(x + w / 2, y + (h - 4) / 2, text, ha="center", va="center", fontsize=9.0, color=COLOR_TEXT_MAIN)
        else:
            ax_flow.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5, fontweight="bold", color=COLOR_TEXT_MAIN)

    # 1. Burt-Adelson Generating Kernel Box (Top)
    draw_box(
        5, 80, 90, 16,
        r"$w(m, n) = \hat{w}(m)\hat{w}(n), \quad \hat{w} = [0.05, 0.25, 0.40, 0.25, 0.05]$" + "\n" +
        r"Separable $5\times 5$ Binomial Filter (Approximates Gaussian $\sigma \approx 1.06$, $\sum w = 1$)",
        COLOR_GAUSSIAN,
        header="1. Generating Low-Pass Smoothing Kernel $w(m, n)$"
    )

    # 2. Gaussian Pyramid Reduction Box (Middle-Left)
    draw_box(
        5, 48, 42, 26,
        r"$G_k(i, j) = \sum_{m=-2}^2 \sum_{n=-2}^2 w(m, n) G_{k-1}(2i+m, 2j+n)$" + "\n\n" +
        r"$\mathbf{Step\;1:}$ Spatial convolution with $w$" + "\n" +
        r"$\mathbf{Step\;2:}$ Subsample every 2nd pixel ($\downarrow 2$)",
        COLOR_GAUSSIAN,
        header="2. Analysis: REDUCE ($G_k$)"
    )

    # 3. Laplacian Band-Pass Extraction Box (Middle-Right)
    draw_box(
        53, 48, 42, 26,
        r"$L_k = G_k - \mathrm{EXPAND}(G_{k+1})$" + "\n\n" +
        r"$\mathbf{Band\text{-}Pass:}$ Stores details lost during reduction." + "\n" +
        r"$\mathbf{Entropy:}$ Compact, sparse, decorrelated representation.",
        COLOR_LAPLACIAN,
        header="3. Band-Pass: Laplacian ($L_k$)"
    )

    # 4. Expansion & Synthesis Box (Bottom-Left)
    draw_box(
        5, 12, 42, 28,
        r"$\mathrm{EXPAND}(G_k) = 4 \sum_{m,n} w(m,n) G_k\left(\frac{i-m}{2}, \frac{j-n}{2}\right)$" + "\n\n" +
        r"$\mathbf{Step\;1:}$ Interleave zeros (upsample $\uparrow 2$)" + "\n" +
        r"$\mathbf{Step\;2:}$ Convolve with interpolation filter $4 \cdot w$",
        COLOR_TEXTURE,
        header="4. Synthesis: EXPAND"
    )

    # 5. Lossless Reconstruction Box (Bottom-Right)
    draw_box(
        53, 12, 42, 28,
        r"$G_k = L_k + \mathrm{EXPAND}(G_{k+1})$" + "\n\n" +
        r"$\mathbf{Base:}$ Start at coarsest level $G_{N-1} = L_{N-1}$" + "\n" +
        r"$\mathbf{Recurse:}$ Expand and add octave details up to $G_0$." + "\n" +
        r"$\mathbf{Result:}$ Exact, lossless image reconstruction.",
        COLOR_ANTIALIAS,
        header="5. Inversion & Reconstruction"
    )

    # Connecting Arrows
    ax_flow.annotate("", xy=(26, 48), xytext=(26, 80), arrowprops=dict(arrowstyle="->", color=COLOR_GAUSSIAN, lw=2))
    ax_flow.annotate("", xy=(53, 61), xytext=(47, 61), arrowprops=dict(arrowstyle="->", color=COLOR_LAPLACIAN, lw=2))
    ax_flow.annotate("", xy=(26, 40), xytext=(26, 48), arrowprops=dict(arrowstyle="->", color=COLOR_TEXTURE, lw=2))
    ax_flow.annotate("", xy=(53, 26), xytext=(47, 26), arrowprops=dict(arrowstyle="->", color=COLOR_ANTIALIAS, lw=2))

    plt.savefig(out_path, dpi=300, facecolor=COLOR_BG_WHITE)
    plt.close()
    print(f"  [5/5] Hierarchy Overview & Flowchart created: {out_path}")


# ---------------------------------------------------------------------------
# Main Execution Entry Point
# ---------------------------------------------------------------------------
def main():
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating Gaussian Pyramid Pedagogical Figures...")
    
    fig1_path = out_dir / "gaussian_pyramid_01_core_scale_space.png"
    plot_figure_01_core_scale_space(fig1_path)
    
    fig2_path = out_dir / "gaussian_pyramid_02_aliasing_and_sampling.png"
    plot_figure_02_aliasing_and_sampling(fig2_path)
    
    fig3_path = out_dir / "gaussian_pyramid_03_laplacian_pyramid_and_reconstruction.png"
    plot_figure_03_laplacian_pyramid(fig3_path)
    
    fig4_path = out_dir / "gaussian_pyramid_04_multiresolution_blending.png"
    plot_figure_04_pyramid_blending(fig4_path)

    fig5_path = out_dir / "gaussian_pyramid_05_hierarchy_and_flowchart.png"
    plot_figure_05_pyramid_hierarchy_overview(fig5_path)
    
    print(f"\nAll 5 Gaussian Pyramid figures generated successfully in ./{out_dir}/")


if __name__ == "__main__":
    main()

