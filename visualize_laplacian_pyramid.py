"""
visualize_laplacian_pyramid.py

A standalone, publication-quality visualization of the Laplacian Pyramid
in Classical Computer Vision (without deep learning), showcasing:
1. `laplacian_pyramid_01_core_scale_space.png`: Multi-scale Laplacian Pyramid
   band-pass decomposition, actual physical grid dimensions, normalized octave detail views,
   and 2D Fourier log-magnitude spectra.

Usage:
    uv run python visualize_laplacian_pyramid.py
"""

from pathlib import Path
from typing import List, Tuple
import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ---------------------------------------------------------------------------
# Visual Styling Constants & Color Palette
# ---------------------------------------------------------------------------
COLOR_BG_WHITE = "#FFFFFF"
COLOR_TEXT_MAIN = "#2C3E50"
COLOR_TEXT_MUTED = "#7F8C8D"
COLOR_GRID = "#BDC3C7"

# Category color accents
COLOR_GAUSSIAN = "#2980B9"      # Blue (Low-Pass / Smoothing)
COLOR_LAPLACIAN = "#8E44AD"     # Purple (Band-Pass / High Frequency)
COLOR_RESIDUAL = "#16A085"      # Teal (Coarsest Base Residual)

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
# Benchmark Test Scene Generation
# ---------------------------------------------------------------------------
def generate_rich_scene(size: int = 512) -> np.ndarray:
    """
    Generates a high-resolution, multi-frequency scene ideal for multiresolution pyramid testing:
    - Background gradient
    - Concentric chirp zone plate (variable radial frequency)
    - High-frequency diagonal and vertical texture gratings
    - Checkerboard pattern
    - Geometric shapes with sharp step edges
    """
    img = np.zeros((size, size), dtype=np.float32)
    y, x = np.mgrid[0:size, 0:size]
    cy, cx = size // 2, size // 2

    # 1. Base gradient
    img += 0.35 + 0.15 * np.sin(2.0 * np.pi * y / size)

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

    # 5. Geometric object (circle with hole) in Bottom-Right quadrant
    r_br = np.sqrt((y - size * 0.74)**2 + (x - size * 0.74)**2)
    mask_br = (y >= size * 0.5) & (x >= size * 0.5)
    disc = ((r_br < size * 0.18) & (r_br > size * 0.05)).astype(np.float32)
    img[mask_br] = 0.15 + 0.75 * disc[mask_br]

    # Dividing crossbars
    img[np.abs(y - cy) < 2] = 0.95
    img[np.abs(x - cx) < 2] = 0.95

    return np.clip(img, 0.0, 1.0)


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
    laplacian_pyr.append(gaussian_pyr[-1].copy())
    return laplacian_pyr


def compute_fft_magnitude(image: np.ndarray) -> np.ndarray:
    """Computes log-magnitude 2D Fourier spectrum centered at DC."""
    fft = np.fft.fftshift(np.fft.fft2(image))
    magnitude = np.log1p(np.abs(fft))
    return (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)


# ---------------------------------------------------------------------------
# FIGURE: Core Laplacian Pyramid & Multi-Scale Band-Pass Decomposition
# ---------------------------------------------------------------------------
def plot_figure_01_laplacian_scale_space(out_path: Path):
    """
    Generates a 3-row pedagogical breakdown for the Laplacian Pyramid:
    - Row 1: Physical scale cascade (actual diminishing grid sizes)
    - Row 2: Zoomed / Normalized view highlighting band-pass detail isolation per octave
    - Row 3: 2D Fourier Frequency Spectra demonstrating octave band-pass frequency slicing
    """
    fig = plt.figure(figsize=(19, 13), facecolor=COLOR_BG_WHITE)

    # Header title & subtitle
    fig.suptitle(
        "The Laplacian Pyramid: Multi-Scale Band-Pass Decomposition & Residuals",
        fontsize=18,
        fontweight="bold",
        color=COLOR_TEXT_MAIN,
        y=0.98,
    )
    fig.text(
        0.5,
        0.952,
        r"Band-pass octave detail encoding: $L_k = G_k - \mathrm{EXPAND}(G_{k+1})$ for $k \in \{0, 1, 2, 3\}$, and coarsest low-pass base $G_4$ (Burt & Adelson, 1983)",
        ha="center",
        fontsize=12,
        color=COLOR_TEXT_MUTED,
    )

    image = generate_rich_scene(512)
    gaussian_pyr = build_gaussian_pyramid(image, levels=5, a=0.4)
    laplacian_pyr = build_laplacian_pyramid(gaussian_pyr, a=0.4)

    level_names = [
        r"Level 0 ($L_0$)",
        r"Level 1 ($L_1$)",
        r"Level 2 ($L_2$)",
        r"Level 3 ($L_3$)",
        r"Level 4 ($G_4$ Base)",
    ]
    shapes = [p.shape for p in laplacian_pyr]

    # Grid layout: 3 rows, 5 columns
    gs = fig.add_gridspec(3, 5, left=0.135, right=0.98, top=0.91, bottom=0.05, hspace=0.35, wspace=0.25)

    # Row headers
    row_labels = [
        "Row 1: Physical Resolution\n(Actual Diminishing Grid)",
        "Row 2: Band-Pass Scale-Space\n(Normalized Detail View)",
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

    # Helper function to normalize Laplacian level for display (zero centered at mid-gray 0.5)
    def normalize_laplacian_vis(arr: np.ndarray, is_base: bool) -> np.ndarray:
        if is_base:
            return np.clip(arr, 0.0, 1.0)
        max_abs = np.percentile(np.abs(arr), 99.5) + 1e-6
        scaled = 0.5 + 0.5 * (arr / max_abs)
        return np.clip(scaled, 0.0, 1.0)

    # Row 1: Actual proportional sizes inside a fixed 512x512 canvas
    for i in range(5):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor(COLOR_BG_WHITE)

        is_base = (i == 4)
        vis_img = normalize_laplacian_vis(laplacian_pyr[i], is_base=is_base)

        canvas = np.ones((512, 512), dtype=np.float32)
        h, w = vis_img.shape
        y_start = (512 - h) // 2
        x_start = (512 - w) // 2
        canvas[y_start:y_start + h, x_start:x_start + w] = vis_img

        ax.imshow(canvas, cmap="gray", vmin=0, vmax=1)

        border_color = COLOR_GAUSSIAN if is_base else COLOR_LAPLACIAN
        rect = patches.Rectangle((x_start, y_start), w, h, linewidth=1.5, edgecolor=border_color, facecolor="none")
        ax.add_patch(rect)

        ax.set_title(f"{level_names[i]}\nSize: {shapes[i][0]}×{shapes[i][1]} px", fontsize=11, fontweight="bold", color=border_color)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(COLOR_GRID)
            spine.set_linewidth(1.0)

        if i < 4:
            ax.set_xlabel(r"$G_k - \mathrm{EXPAND}(G_{k+1})$", fontsize=10.0, color=COLOR_LAPLACIAN, fontweight="bold")
        else:
            ax.set_xlabel(r"$\mathrm{Coarsest\ Low\text{-}Pass}$", fontsize=10.0, color=COLOR_GAUSSIAN, fontweight="bold")

    # Row 2: Scaled up to full frame to inspect band-pass details per octave
    subtitles_row2 = [
        "Finest Edges & Chirps (High-Pass)",
        "Medium-High Contours & Textures",
        "Medium Structural Boundaries",
        "Coarse Transitions & Large Blobs",
        "DC Component & Low-Pass Shading",
    ]

    for i in range(5):
        ax = fig.add_subplot(gs[1, i])
        ax.set_facecolor(COLOR_BG_WHITE)

        is_base = (i == 4)
        vis_img = normalize_laplacian_vis(laplacian_pyr[i], is_base=is_base)
        border_color = COLOR_GAUSSIAN if is_base else COLOR_LAPLACIAN

        ax.imshow(vis_img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")

        if not is_base:
            ax.set_title(fr"Band-Pass $L_{i}$ ({shapes[i][0]}×{shapes[i][1]})", fontsize=11, fontweight="bold", color=COLOR_LAPLACIAN)
        else:
            ax.set_title(fr"Low-Pass Base $G_4$ ({shapes[i][0]}×{shapes[i][1]})", fontsize=11, fontweight="bold", color=COLOR_GAUSSIAN)

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(border_color)
            spine.set_linewidth(1.5)

        ax.set_xlabel(subtitles_row2[i], fontsize=9.5, color=COLOR_TEXT_MUTED)

    # Row 3: 2D FFT Log-Magnitude Spectra
    freq_bands = [
        r"Band $[\frac{\pi}{2}, \pi]$ rad",
        r"Band $[\frac{\pi}{4}, \frac{\pi}{2}]$ rad",
        r"Band $[\frac{\pi}{8}, \frac{\pi}{4}]$ rad",
        r"Band $[\frac{\pi}{16}, \frac{\pi}{8}]$ rad",
        r"Base $[0, \frac{\pi}{16}]$ rad (Low-Pass)",
    ]

    for i in range(5):
        ax = fig.add_subplot(gs[2, i])
        ax.set_facecolor(COLOR_BG_WHITE)

        is_base = (i == 4)
        fft_mag = compute_fft_magnitude(laplacian_pyr[i])
        ax.imshow(fft_mag, cmap="magma", vmin=0, vmax=1)

        border_color = COLOR_GAUSSIAN if is_base else COLOR_LAPLACIAN
        if not is_base:
            ax.set_title(f"Spectrum $|\mathcal{{F}}(L_{i})|$\n(Band-Pass Octave)", fontsize=10.5, fontweight="bold", color=COLOR_LAPLACIAN)
        else:
            ax.set_title(f"Spectrum $|\mathcal{{F}}(G_4)|$\n(Low-Pass Residual)", fontsize=10.5, fontweight="bold", color=COLOR_GAUSSIAN)

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(border_color)
            spine.set_linewidth(1.2)
        ax.set_xlabel(freq_bands[i], fontsize=9.5, color=COLOR_TEXT_MUTED)

    plt.savefig(out_path, dpi=300, facecolor=COLOR_BG_WHITE)
    plt.close()
    print(f"Laplacian Pyramid Figure created: {out_path}")


def main():
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / "laplacian_pyramid_01_core_scale_space.png"
    plot_figure_01_laplacian_scale_space(fig_path)


if __name__ == "__main__":
    main()
