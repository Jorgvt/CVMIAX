"""
visualize_classical_filters.py

A standalone, publication-quality visualization suite for Classical Computer Vision
Spatial Filters (without deep learning), showcasing fundamental 2D image processing
kernels, spatial convolution, gradient/frequency analysis, and edge detection mechanics.

Generates 4 pedagogical figures:
1. `classical_filters_01_master_gallery.png`: 12-panel comprehensive gallery of classical filters
   applied to a multi-feature benchmark image (smoothing, gradients, 2nd derivatives, texture, Canny).
2. `classical_filters_02_kernel_matrices.png`: Discrete kernel anatomy with heatmaps, exact weights,
   and 3D impulse response surfaces for Box, Gaussian, Sobel, Laplacian, LoG, and Gabor kernels.
3. `classical_filters_03_noise_and_smoothing.png`: Noise reduction trade-offs (Gaussian vs. Salt & Pepper)
   comparing Box, Gaussian, Median, and Bilateral filters with 1D edge profile cross-sections.
4. `classical_filters_04_derivative_edge_detection.png`: 1D calculus intuition (f, f', f'') linked to
   2D edge detectors (Sobel, Laplacian, LoG, and Canny hysteresis stages).

Usage:
    uv run python visualize_classical_filters.py
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np
import scipy.ndimage as ndimage
import scipy.signal as signal
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ---------------------------------------------------------------------------
# Visual Styling Constants & Color Palette
# ---------------------------------------------------------------------------
COLOR_BG_DARK = "#1A252F"
COLOR_TEXT_MAIN = "#2C3E50"
COLOR_TEXT_MUTED = "#7F8C8D"
COLOR_GRID = "#BDC3C7"

# Category color accents
COLOR_ORIGINAL = "#2C3E50"      # Slate Navy
COLOR_LOWPASS = "#2980B9"       # Blue (Smoothing / Low-Pass)
COLOR_FIRST_DERIV = "#D35400"   # Deep Orange (1st Derivative / Gradient)
COLOR_SECOND_DERIV = "#8E44AD"  # Purple (2nd Derivative / Laplacian / LoG)
COLOR_TEXTURE = "#27AE60"       # Emerald Green (Gabor / Texture / Frequency)
COLOR_MULTISTAGE = "#C0392B"    # Crimson (Canny / Multi-stage)
COLOR_MEDIAN = "#16A085"        # Teal (Rank / Non-linear)

FONT_FAMILY = "sans-serif"

plt.rcParams.update({
    "font.family": FONT_FAMILY,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.titlesize": 15,
    "figure.facecolor": "#FFFFFF",
    "axes.facecolor": "#FFFFFF",
    "savefig.facecolor": "#FFFFFF",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# ---------------------------------------------------------------------------
# Benchmark Test Image Synthesis
# ---------------------------------------------------------------------------
def generate_benchmark_image(size: int = 384) -> np.ndarray:
    """
    Synthesizes a rich, multi-feature benchmark image containing:
    - Geometric shapes with sharp step edges (circle, rectangle, diamond)
    - High-frequency frequency patterns (zone plate / concentric rings, checkerboard)
    - Diagonal stripe gratings with varying frequencies
    - Smooth continuous gradients (shading)
    - Subtle textured regions and point impulses / blobs
    """
    img = np.zeros((size, size), dtype=np.float32)
    y_arr = np.arange(size)
    x_arr = np.arange(size)
    xx, yy = np.meshgrid(x_arr, y_arr)
    cy, cx = size / 2, size / 2

    # 1. Smooth background gradient (diagonal)
    img += 0.25 * (xx + yy) / (2 * size)

    # 2. Concentric ring zone plate / Siemens star texture (top-left quadrant)
    r1 = np.sqrt((yy - size * 0.28)**2 + (xx - size * 0.28)**2)
    mask_q1 = (yy < size * 0.52) & (xx < size * 0.52)
    zone_pattern = 0.5 + 0.5 * np.cos(0.04 * (r1**1.5))
    img[mask_q1] = 0.2 + 0.6 * zone_pattern[mask_q1]

    # 3. Oriented high-frequency gratings (top-right quadrant)
    mask_q2 = (yy < size * 0.5) & (xx >= size * 0.5)
    diag_freq = np.sin(0.25 * (xx + yy)) * 0.5 + 0.5
    vert_freq = np.sin(0.4 * xx) * 0.5 + 0.5
    img[mask_q2 & (yy < size * 0.25)] = 0.3 + 0.5 * diag_freq[mask_q2 & (yy < size * 0.25)]
    img[mask_q2 & (yy >= size * 0.25)] = 0.3 + 0.5 * vert_freq[mask_q2 & (yy >= size * 0.25)]

    # 4. Checkerboard texture patch (bottom-left quadrant)
    cb_size = 12
    cb = ((xx // cb_size) % 2) ^ ((yy // cb_size) % 2)
    mask_cb = (yy >= size * 0.55) & (yy < size * 0.9) & (xx >= size * 0.08) & (xx < size * 0.42)
    img[mask_cb] = 0.2 + 0.6 * cb[mask_cb]

    # 5. Sharp Geometric Solid Shapes with high contrast (bottom-right quadrant)
    # Circle
    r_circle = np.sqrt((yy - size * 0.75)**2 + (xx - size * 0.72)**2)
    img[r_circle < size * 0.16] = 0.95
    # Inner dark hole
    img[r_circle < size * 0.06] = 0.15

    # 6. Central sharp tilted rectangle
    rect_mask = (np.abs((xx - cx) * 0.707 + (yy - cy) * 0.707) < size * 0.12) & \
                (np.abs(-(xx - cx) * 0.707 + (yy - cy) * 0.707) < size * 0.06)
    img[rect_mask] = 0.88

    # 7. Fine lines / wire edges
    img[int(size * 0.49):int(size * 0.51), int(size * 0.05):int(size * 0.95)] = 1.0
    img[int(size * 0.05):int(size * 0.95), int(size * 0.49):int(size * 0.51)] = 0.05

    # 8. Point impulses / small isolated blobs
    for bx, by in [(int(size * 0.65), int(size * 0.58)),
                   (int(size * 0.85), int(size * 0.58)),
                   (int(size * 0.75), int(size * 0.60))]:
        r_b = np.sqrt((yy - by)**2 + (xx - bx)**2)
        img[r_b < 4] = 1.0

    return np.clip(img, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Filter Implementations & Kernels
# ---------------------------------------------------------------------------
def get_box_kernel(ksize: int = 3) -> np.ndarray:
    """Discrete 2D Box / Mean Averaging Kernel."""
    return np.ones((ksize, ksize), dtype=np.float32) / (ksize * ksize)


def get_gaussian_kernel(ksize: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Discrete 2D Gaussian Kernel."""
    ax = np.arange(-ksize // 2 + 1., ksize // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return kernel / np.sum(kernel)


def get_sobel_kernels() -> Tuple[np.ndarray, np.ndarray]:
    """Sobel Horizontal (Gx) and Vertical (Gy) 3x3 Derivative Kernels."""
    kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float32)
    return kx, ky


def get_laplacian_kernel() -> np.ndarray:
    """Discrete 2D 3x3 Laplacian (4-neighbor) Kernel."""
    return np.array([[ 0,  1,  0],
                     [ 1, -4,  1],
                     [ 0,  1,  0]], dtype=np.float32)


def get_log_kernel(ksize: int = 9, sigma: float = 1.4) -> np.ndarray:
    """
    Discrete Laplacian of Gaussian (LoG / Mexican Hat) Kernel.
    LoG(x,y) = -1 / (pi * sigma^4) * [1 - (x^2 + y^2)/(2 sigma^2)] * exp(-(x^2 + y^2)/(2 sigma^2))
    """
    ax = np.arange(-ksize // 2 + 1., ksize // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    r2 = xx**2 + yy**2
    s2 = sigma**2
    kernel = -(1.0 - r2 / (2.0 * s2)) * np.exp(-r2 / (2.0 * s2))
    # Zero-mean normalization
    kernel = kernel - np.mean(kernel)
    return kernel


def get_gabor_kernel(ksize: int = 15, sigma: float = 3.0, theta: float = np.pi / 4,
                     lambd: float = 6.0, gamma: float = 0.5, psi: float = 0.0) -> np.ndarray:
    """
    2D Gabor Wavelet Kernel (Real component).
    Models simple cells in the primary visual cortex (V1) as oriented sinusoidal gratings in Gaussian envelopes.
    """
    ax = np.arange(-ksize // 2 + 1., ksize // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    x_theta = xx * np.cos(theta) + yy * np.sin(theta)
    y_theta = -xx * np.sin(theta) + yy * np.cos(theta)
    envelope = np.exp(-(x_theta**2 + (gamma * y_theta)**2) / (2.0 * sigma**2))
    carrier = np.cos(2.0 * np.pi * x_theta / lambd + psi)
    kernel = envelope * carrier
    return kernel - np.mean(kernel)


# ---------------------------------------------------------------------------
# Spatial Filter Application Functions
# ---------------------------------------------------------------------------
def apply_box_filter(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    kernel = get_box_kernel(ksize)
    return ndimage.convolve(img, kernel, mode="reflect")


def apply_gaussian_filter(img: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    return ndimage.gaussian_filter(img, sigma=sigma, mode="reflect")


def apply_median_filter(img: np.ndarray, size: int = 5) -> np.ndarray:
    return ndimage.median_filter(img, size=size, mode="reflect")


def apply_bilateral_filter(img: np.ndarray, sigma_s: float = 3.0, sigma_r: float = 0.15, radius: int = 5) -> np.ndarray:
    """
    Edge-preserving Bilateral Filter combining spatial domain Gaussian with range/intensity Gaussian.
    """
    h, w = img.shape
    out = np.zeros_like(img)
    pad = radius
    padded = np.pad(img, pad, mode="reflect")

    # Spatial weights
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    spatial_weights = np.exp(-(x**2 + y**2) / (2.0 * sigma_s**2))

    for i in range(h):
        for j in range(w):
            patch = padded[i:i + 2 * radius + 1, j:j + 2 * radius + 1]
            range_weights = np.exp(-((patch - img[i, j])**2) / (2.0 * sigma_r**2))
            weights = spatial_weights * range_weights
            out[i, j] = np.sum(patch * weights) / (np.sum(weights) + 1e-8)
    return out


def apply_sobel(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Computes Sobel Gx, Gy, Gradient Magnitude, and Direction."""
    kx, ky = get_sobel_kernels()
    gx = ndimage.convolve(img, kx, mode="reflect")
    gy = ndimage.convolve(img, ky, mode="reflect")
    magnitude = np.hypot(gx, gy)
    direction = np.arctan2(gy, gx)
    return gx, gy, magnitude, direction


def apply_laplacian(img: np.ndarray) -> np.ndarray:
    kernel = get_laplacian_kernel()
    return ndimage.convolve(img, kernel, mode="reflect")


def apply_log(img: np.ndarray, sigma: float = 1.8, ksize: int = 11) -> np.ndarray:
    kernel = get_log_kernel(ksize=ksize, sigma=sigma)
    return ndimage.convolve(img, kernel, mode="reflect")


def apply_unsharp_mask(img: np.ndarray, sigma: float = 1.5, strength: float = 1.5) -> np.ndarray:
    blurred = ndimage.gaussian_filter(img, sigma=sigma, mode="reflect")
    mask = img - blurred
    sharpened = img + strength * mask
    return np.clip(sharpened, 0.0, 1.0)


def apply_gabor(img: np.ndarray, theta: float = np.pi / 4) -> np.ndarray:
    kernel = get_gabor_kernel(ksize=17, sigma=3.0, theta=theta, lambd=5.0)
    return ndimage.convolve(img, kernel, mode="reflect")


def apply_canny(img: np.ndarray, sigma: float = 1.4, low_thresh: float = 0.10, high_thresh: float = 0.25) -> np.ndarray:
    """
    Full 4-stage classical Canny Edge Detector:
    1. Gaussian Smoothing
    2. Sobel Gradient Magnitude and Quantized Direction
    3. Non-Maximum Suppression (Thinning)
    4. Hysteresis Double Thresholding and Edge Tracking
    """
    # 1. Smooth
    smoothed = ndimage.gaussian_filter(img, sigma=sigma, mode="reflect")

    # 2. Gradient
    gx, gy, mag, direct = apply_sobel(smoothed)
    # Normalize magnitude
    if mag.max() > 0:
        mag = mag / mag.max()

    # 3. Non-Maximum Suppression
    h, w = img.shape
    nms = np.zeros((h, w), dtype=np.float32)
    # Convert angles to degrees [0, 180)
    angle = np.rad2deg(direct) % 180

    pad_mag = np.pad(mag, 1, mode="constant")
    for i in range(h):
        for j in range(w):
            pi, pj = i + 1, j + 1
            a = angle[i, j]
            # 0 deg (Horizontal edge / vertical gradient)
            if (0 <= a < 22.5) or (157.5 <= a <= 180):
                q = pad_mag[pi, pj + 1]
                r = pad_mag[pi, pj - 1]
            # 45 deg (Diagonal)
            elif 22.5 <= a < 67.5:
                q = pad_mag[pi + 1, pj - 1]
                r = pad_mag[pi - 1, pj + 1]
            # 90 deg (Vertical edge / horizontal gradient)
            elif 67.5 <= a < 112.5:
                q = pad_mag[pi + 1, pj]
                r = pad_mag[pi - 1, pj]
            # 135 deg (Diagonal)
            else:
                q = pad_mag[pi - 1, pj - 1]
                r = pad_mag[pi + 1, pj + 1]

            if pad_mag[pi, pj] >= q and pad_mag[pi, pj] >= r:
                nms[i, j] = pad_mag[pi, pj]

    # 4. Double Thresholding & Hysteresis
    strong = nms >= high_thresh
    weak = (nms >= low_thresh) & (~strong)

    # 8-connectivity edge tracking
    edges = strong.copy()
    labels, num_features = ndimage.label(nms >= low_thresh, structure=np.ones((3, 3)))
    for label_idx in range(1, num_features + 1):
        component = (labels == label_idx)
        if np.any(strong & component):
            edges |= component

    return edges.astype(np.float32)


# ---------------------------------------------------------------------------
# Figure 1: Master Gallery of Classical Spatial Filters
# ---------------------------------------------------------------------------
def plot_master_gallery(output_dir: Path) -> Path:
    """
    Creates a 12-panel comprehensive master gallery comparing classical filters
    on the rich benchmark test image with color badges, formulas, and visual intuitions.
    """
    img = generate_benchmark_image(384)

    # Compute filter outputs
    box_out = apply_box_filter(img, ksize=7)
    gauss_out = apply_gaussian_filter(img, sigma=2.5)
    median_out = apply_median_filter(img, size=5)
    gx, gy, mag, _ = apply_sobel(img)
    lap_out = apply_laplacian(img)
    log_out = apply_log(img, sigma=1.8, ksize=13)
    unsharp_out = apply_unsharp_mask(img, sigma=1.5, strength=2.0)
    gabor_diag = apply_gabor(img, theta=np.pi / 4)
    canny_out = apply_canny(img, sigma=1.2, low_thresh=0.08, high_thresh=0.22)

    # Panel definitions: (title, image, cmap, vmin, vmax, category, cat_color, formula)
    panels = [
        ("1. Original Input Image", img, "gray", 0.0, 1.0, "INPUT", COLOR_ORIGINAL, r"$I(x, y)$ benchmark pattern"),
        ("2. Box / Mean Filter (7x7)", box_out, "gray", 0.0, 1.0, "LOW-PASS", COLOR_LOWPASS, r"$K = \frac{1}{49} \cdot \mathbf{1}_{7\times 7}$ (Uniform blur)"),
        (r"3. Gaussian Blur ($\sigma=2.5$)", gauss_out, "gray", 0.0, 1.0, "LOW-PASS", COLOR_LOWPASS, r"$G(x,y) \propto \exp(-(x^2+y^2)/2\sigma^2)$"),
        ("4. Median Filter (5x5)", median_out, "gray", 0.0, 1.0, "NON-LINEAR", COLOR_MEDIAN, r"$\mathrm{Rank\ filter:}\ \mathrm{median}(I_\Omega)$"),
        (r"5. Sobel $G_x$ (Horizontal Gradient)", gx, "coolwarm", -1.5, 1.5, "1ST DERIV", COLOR_FIRST_DERIV, r"$G_x \approx \partial I / \partial x$ (Vertical edges)"),
        (r"6. Sobel $G_y$ (Vertical Gradient)", gy, "coolwarm", -1.5, 1.5, "1ST DERIV", COLOR_FIRST_DERIV, r"$G_y \approx \partial I / \partial y$ (Horizontal edges)"),
        ("7. Sobel Gradient Magnitude", mag, "inferno", 0.0, 1.8, "1ST DERIV", COLOR_FIRST_DERIV, r"$|\nabla I| = \sqrt{G_x^2 + G_y^2}$"),
        (r"8. Laplacian ($\nabla^2 I$)", lap_out, "bwr", -1.5, 1.5, "2ND DERIV", COLOR_SECOND_DERIV, r"$\nabla^2 I = \partial^2 I/\partial x^2 + \partial^2 I/\partial y^2$"),
        ("9. Laplacian of Gaussian (LoG)", log_out, "bwr", -0.5, 0.5, "2ND DERIV", COLOR_SECOND_DERIV, r"$\mathrm{LoG} = \nabla^2(G_\sigma * I)$ (Blob detector)"),
        ("10. Unsharp Masking", unsharp_out, "gray", 0.0, 1.0, "ENHANCE", COLOR_LOWPASS, r"$I_{\mathrm{sharp}} = I + \alpha(I - G_\sigma * I)$"),
        (r"11. Gabor Filter ($\theta=45^\circ$)", gabor_diag, "PuOr", -1.0, 1.0, "FREQUENCY", COLOR_TEXTURE, r"$G_\sigma(x',y') \cos(2\pi x' / \lambda)$"),
        ("12. Canny Edge Detector", canny_out, "gray", 0.0, 1.0, "MULTI-STAGE", COLOR_MULTISTAGE, r"$\mathrm{NMS} + \mathrm{Hysteresis}(\tau_{\mathrm{low}}, \tau_{\mathrm{high}})$"),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(19, 15.5), gridspec_kw={"hspace": 0.35, "wspace": 0.20})
    fig.patch.set_facecolor("#FFFFFF")
    axes = axes.flatten()

    for idx, (title, data, cmap, vmin, vmax, cat_name, cat_color, formula) in enumerate(panels):
        ax = axes[idx]
        ax.set_facecolor("#FFFFFF")
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])

        # Title & Category Badge
        ax.set_title(title, fontsize=12, fontweight="bold", color=COLOR_TEXT_MAIN, pad=16)

        # Badge pill
        bbox_props = dict(boxstyle="round,pad=0.3", fc=cat_color, ec="none", alpha=0.92)
        ax.text(0.03, 0.94, cat_name, transform=ax.transAxes, fontsize=8.5,
                fontweight="bold", color="white", va="top", ha="left", bbox=bbox_props)

        # Mathematical Formula sub-label
        ax.set_xlabel(formula, fontsize=9.5, color=COLOR_TEXT_MAIN, labelpad=8)

        # Border styling
        for spine in ax.spines.values():
            spine.set_color(cat_color)
            spine.set_linewidth(1.8 if idx > 0 else 2.2)

    plt.suptitle("Classical Computer Vision: Spatial Convolution Filters & Feature Extractors",
                 fontsize=17, fontweight="bold", color=COLOR_TEXT_MAIN, y=0.985)
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.965])

    out_path = output_dir / "classical_filters_01_master_gallery.png"
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close()
    return out_path


# ---------------------------------------------------------------------------
# Figure 2: Discrete Kernel Anatomy & 3D Impulse Responses
# ---------------------------------------------------------------------------
def plot_kernel_anatomy(output_dir: Path) -> Path:
    """
    Visualizes 6 key classical kernels with discrete weight matrix heatmaps,
    numeric annotations, and 3D continuous impulse response surfaces.
    """
    kernels = [
        ("Box / Mean (3x3)", get_box_kernel(3), r"$K = \frac{1}{9} \cdot \mathbf{1}_{3\times 3}$", "All positive, uniform weights → Low-pass blurring"),
        (r"Gaussian (5x5, $\sigma=1.0$)", get_gaussian_kernel(5, sigma=1.0), r"$G(x,y) \propto \exp(-(x^2+y^2)/2\sigma^2)$", "Bell-shaped, isotropic, rotation-invariant smoothing"),
        ("Sobel $K_x$ (3x3)", get_sobel_kernels()[0], r"$K_x = [-1, 0, 1]^T * [1, 2, 1]$", "Central difference + smoothing → Vertical edge detection"),
        ("Laplacian (3x3)", get_laplacian_kernel(), r"$\nabla^2 \approx \Delta_x^2 + \Delta_y^2$", "Isotropic 2nd derivative → Zero-crossings at edges"),
        ("Laplacian of Gaussian (LoG 9x9)", get_log_kernel(9, sigma=1.4), r"$\nabla^2 G_\sigma(x,y)$ (Mexican Hat)", "Combines Gaussian noise suppression with blob detection"),
        (r"Gabor Wavelet ($\theta=45^\circ$)", get_gabor_kernel(15, sigma=2.5, theta=np.pi/4, lambd=5.0), r"$\cos(2\pi x' / \lambda) \exp(-(x'^2+\gamma^2 y'^2)/2\sigma^2)$", "Frequency & orientation selectivity ($V_1$ receptive field)"),
    ]

    fig = plt.figure(figsize=(19, 13))
    fig.patch.set_facecolor("#FFFFFF")

    for i, (name, k, formula, note) in enumerate(kernels):
        # Left sub-panel: 2D Matrix Heatmap with numbers
        ax_2d = fig.add_subplot(3, 4, 2 * i + 1)
        ax_2d.set_facecolor("#FFFFFF")

        vmax = max(abs(k.min()), abs(k.max()))
        vmin = -vmax if k.min() < -1e-4 else 0.0
        cmap = "coolwarm" if k.min() < -1e-4 else "Blues"

        im = ax_2d.imshow(k, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax_2d.set_title(f"{name}\n{formula}", fontsize=11, fontweight="bold", color=COLOR_TEXT_MAIN, pad=8)
        ax_2d.set_xticks([])
        ax_2d.set_yticks([])

        # Annotate numbers if small kernel
        if k.shape[0] <= 5:
            for r in range(k.shape[0]):
                for c in range(k.shape[1]):
                    val = k[r, c]
                    text_str = f"{val:+.2f}" if abs(val) < 1.0 else f"{int(val):+d}"
                    color = "white" if abs(val) > 0.4 * vmax else "black"
                    ax_2d.text(c, r, text_str, ha="center", va="center", fontsize=8.5, fontweight="bold", color=color)

        ax_2d.set_xlabel(note, fontsize=9.0, color=COLOR_TEXT_MUTED, labelpad=6)

        # Right sub-panel: 3D Surface / Mesh of the continuous impulse response
        ax_3d = fig.add_subplot(3, 4, 2 * i + 2, projection="3d")
        ax_3d.set_facecolor("#FFFFFF")

        # Smooth mesh for 3D viewing
        h_k, w_k = k.shape
        x_grid, y_grid = np.meshgrid(np.arange(w_k) - (w_k - 1) / 2.0, np.arange(h_k) - (h_k - 1) / 2.0)
        surf = ax_3d.plot_surface(x_grid, y_grid, k, cmap=cmap, edgecolor="k", linewidth=0.25, alpha=0.92, antialiased=True)
        ax_3d.set_title("3D Spatial Profile", fontsize=10, color=COLOR_TEXT_MAIN, fontweight="bold", pad=2)
        ax_3d.view_init(elev=32, azim=-55)
        ax_3d.set_xticks([])
        ax_3d.set_yticks([])
        ax_3d.set_zticks([])
        ax_3d.xaxis.pane.fill = False
        ax_3d.yaxis.pane.fill = False
        ax_3d.zaxis.pane.fill = False

    plt.suptitle("Classical Filter Anatomy: Discrete Kernel Matrices & Spatial Impulse Responses",
                 fontsize=17, fontweight="bold", color=COLOR_TEXT_MAIN, y=0.985)
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.96])

    out_path = output_dir / "classical_filters_02_kernel_matrices.png"
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close()
    return out_path


# ---------------------------------------------------------------------------
# Figure 3: Noise Robustness, Denoising & Edge Preservation
# ---------------------------------------------------------------------------
def plot_noise_and_smoothing(output_dir: Path) -> Path:
    """
    Compares the behavior of Box, Gaussian, Median, and Bilateral filters
    on Gaussian noise vs. Salt-and-Pepper noise, with 1D intensity cross-sections.
    """
    # Clean step edge scene with texture
    size = 200
    clean = np.zeros((size, size), dtype=np.float32)
    clean[:, size // 2:] = 1.0  # Step edge

    # Add Gaussian noise
    np.random.seed(42)
    gauss_noise = np.clip(clean + np.random.normal(0.0, 0.18, clean.shape), 0.0, 1.0)

    # Add Salt & Pepper noise
    sp_noise = clean.copy()
    sp_prob = 0.10
    num_noise = int(sp_prob * size * size)
    # Salt (1.0)
    salt_y = np.random.randint(0, size, num_noise // 2)
    salt_x = np.random.randint(0, size, num_noise // 2)
    sp_noise[salt_y, salt_x] = 1.0
    # Pepper (0.0)
    pep_y = np.random.randint(0, size, num_noise // 2)
    pep_x = np.random.randint(0, size, num_noise // 2)
    sp_noise[pep_y, pep_x] = 0.0

    # Filters to compare
    filters = [
        ("Box Blur (5x5)", lambda im: apply_box_filter(im, ksize=5), COLOR_LOWPASS),
        ("Gaussian ($\sigma=2$)", lambda im: apply_gaussian_filter(im, sigma=2.0), COLOR_LOWPASS),
        ("Median (5x5)", lambda im: apply_median_filter(im, size=5), COLOR_MEDIAN),
        ("Bilateral ($r=4$)", lambda im: apply_bilateral_filter(im, sigma_s=3.0, sigma_r=0.2, radius=4), COLOR_TEXTURE),
    ]

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#FFFFFF")
    gs = fig.add_gridspec(3, 5, height_ratios=[1.0, 1.0, 0.85], hspace=0.32, wspace=0.2)

    # Row 0: Gaussian Noise Denoising
    ax_noisy_g = fig.add_subplot(gs[0, 0])
    ax_noisy_g.imshow(gauss_noise, cmap="gray", vmin=0, vmax=1)
    ax_noisy_g.set_title("Input: Gaussian Noise\n($\sigma_{\mathrm{noise}}=0.18$)", fontsize=11, fontweight="bold", color=COLOR_FIRST_DERIV)
    ax_noisy_g.set_xticks([]); ax_noisy_g.set_yticks([])

    for j, (fname, ffunc, fcolor) in enumerate(filters):
        res = ffunc(gauss_noise)
        ax = fig.add_subplot(gs[0, j + 1])
        ax.imshow(res, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{fname}\non Gaussian Noise", fontsize=10.5, fontweight="bold", color=fcolor)
        ax.set_xticks([]); ax.set_yticks([])

    # Row 1: Salt & Pepper Noise Denoising
    ax_noisy_sp = fig.add_subplot(gs[1, 0])
    ax_noisy_sp.imshow(sp_noise, cmap="gray", vmin=0, vmax=1)
    ax_noisy_sp.set_title("Input: Salt & Pepper\n($p_{\mathrm{impulse}}=10\%$)", fontsize=11, fontweight="bold", color=COLOR_MULTISTAGE)
    ax_noisy_sp.set_xticks([]); ax_noisy_sp.set_yticks([])

    for j, (fname, ffunc, fcolor) in enumerate(filters):
        res = ffunc(sp_noise)
        ax = fig.add_subplot(gs[1, j + 1])
        ax.imshow(res, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{fname}\non Salt & Pepper", fontsize=10.5, fontweight="bold", color=fcolor)
        ax.set_xticks([]); ax.set_yticks([])

    # Row 2: 1D Cross-Section Profile across Step Edge (Row 100)
    ax_prof_g = fig.add_subplot(gs[2, :2])
    ax_prof_g.set_facecolor("#FFFFFF")
    ax_prof_g.plot(clean[100, :], "k--", lw=2, label="Clean Ground Truth")
    ax_prof_g.plot(gauss_noise[100, :], color="#BDC3C7", alpha=0.5, label="Gaussian Noisy")
    ax_prof_g.plot(apply_gaussian_filter(gauss_noise, 2.0)[100, :], color=COLOR_LOWPASS, lw=2.2, label="Gaussian (Edge blurred)")
    ax_prof_g.plot(apply_bilateral_filter(gauss_noise, 3.0, 0.2, 4)[100, :], color=COLOR_TEXTURE, lw=2.2, label="Bilateral (Edge preserved)")
    ax_prof_g.set_title("1D Intensity Profile: Gaussian Noise Denoising", fontsize=11, fontweight="bold", color=COLOR_TEXT_MAIN)
    ax_prof_g.set_xlabel("Horizontal Pixel Position $x$", fontsize=10)
    ax_prof_g.set_ylabel("Pixel Intensity", fontsize=10)
    ax_prof_g.set_ylim(-0.1, 1.2)
    ax_prof_g.grid(True, linestyle="--", alpha=0.5)
    ax_prof_g.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

    ax_prof_sp = fig.add_subplot(gs[2, 2:])
    ax_prof_sp.set_facecolor("#FFFFFF")
    ax_prof_sp.plot(clean[100, :], "k--", lw=2, label="Clean Ground Truth")
    ax_prof_sp.plot(sp_noise[100, :], color="#BDC3C7", alpha=0.6, label="Salt & Pepper Noisy")
    ax_prof_sp.plot(apply_gaussian_filter(sp_noise, 2.0)[100, :], color=COLOR_LOWPASS, lw=2.2, label="Gaussian (Impulses smeared!)")
    ax_prof_sp.plot(apply_median_filter(sp_noise, 5)[100, :], color=COLOR_MEDIAN, lw=2.2, label="Median (Impulses removed & sharp edge!)")
    ax_prof_sp.set_title("1D Intensity Profile: Salt & Pepper Noise Denoising", fontsize=11, fontweight="bold", color=COLOR_TEXT_MAIN)
    ax_prof_sp.set_xlabel("Horizontal Pixel Position $x$", fontsize=10)
    ax_prof_sp.set_ylabel("Pixel Intensity", fontsize=10)
    ax_prof_sp.set_ylim(-0.1, 1.2)
    ax_prof_sp.grid(True, linestyle="--", alpha=0.5)
    ax_prof_sp.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

    plt.suptitle("Smoothing & Denoising Dynamics: Linear (Gaussian/Box) vs. Non-Linear (Median/Bilateral)",
                 fontsize=15, fontweight="bold", color=COLOR_TEXT_MAIN, y=0.98)

    out_path = output_dir / "classical_filters_03_noise_and_smoothing.png"
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close()
    return out_path


# ---------------------------------------------------------------------------
# Figure 4: Calculus of Edge Detection & Multi-Stage Pipelines
# ---------------------------------------------------------------------------
def plot_derivative_edge_detection(output_dir: Path) -> Path:
    """
    Demonstrates the mathematical mechanics of edge detection:
    - 1D step edge: Signal f(x) -> 1st derivative f'(x) peak -> 2nd derivative f''(x) zero-crossing
    - Impact of noise and smoothing before differentiation
    - 2D Canny Edge Detector 4-stage pipeline visualization.
    """
    fig = plt.figure(figsize=(18, 10.5))
    fig.patch.set_facecolor("#FFFFFF")
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.0], hspace=0.28, wspace=0.24)

    # --- Top Row: 1D Calculus of Edge Detection ---
    x = np.linspace(-10, 10, 500)
    # Smooth step edge model (Sigmoid)
    clean_step = 1.0 / (1.0 + np.exp(-1.5 * x))
    np.random.seed(99)
    noise = np.random.normal(0, 0.04, len(x))
    noisy_step = clean_step + noise

    # Smooth noisy signal (sufficient Gaussian filtering to illustrate robust gradient)
    smoothed_step = ndimage.gaussian_filter1d(noisy_step, sigma=14)

    # Numerical derivatives
    d1_clean = np.gradient(clean_step, x)
    d1_noisy = np.gradient(noisy_step, x)
    d1_smooth = np.gradient(smoothed_step, x)

    d2_clean = np.gradient(d1_clean, x)
    d2_smooth = np.gradient(d1_smooth, x)

    # Subplot 1: Signal f(x)
    ax_f = fig.add_subplot(gs[0, 0])
    ax_f.set_facecolor("#FFFFFF")
    ax_f.plot(x, clean_step, "k-", lw=2.2, label="Clean Step $f(x)$")
    ax_f.plot(x, noisy_step, color="#BDC3C7", lw=1.0, label="Noisy $f(x) + \epsilon$")
    ax_f.plot(x, smoothed_step, color=COLOR_LOWPASS, lw=2.2, label=r"Smoothed $G_\sigma * f$")
    ax_f.axvline(0, color="gray", linestyle=":", alpha=0.7)
    ax_f.set_title("1. Intensity Profile $f(x)$", fontsize=11, fontweight="bold", color=COLOR_TEXT_MAIN)
    ax_f.set_xlabel("$x$ coordinate", fontsize=9.5)
    ax_f.grid(True, linestyle="--", alpha=0.4)
    ax_f.legend(loc="upper left", fontsize=8.5)

    # Subplot 2: 1st Derivative f'(x) (Sobel concept)
    ax_d1 = fig.add_subplot(gs[0, 1])
    ax_d1.set_facecolor("#FFFFFF")
    ax_d1.plot(x, d1_noisy, color="#E74C3C", alpha=0.4, lw=0.8, label="Noisy $f'(x)$ (Unstable!)")
    ax_d1.plot(x, d1_clean, "k--", lw=1.8, label="Clean $f'(x)$")
    ax_d1.plot(x, d1_smooth, color=COLOR_FIRST_DERIV, lw=2.2, label=r"$\frac{\partial}{\partial x}(G_\sigma * f)$ (Sobel)")
    ax_d1.axvline(0, color=COLOR_FIRST_DERIV, linestyle=":", lw=1.5)
    ax_d1.scatter([0], [d1_smooth[len(x)//2]], color=COLOR_FIRST_DERIV, s=60, zorder=5)
    ax_d1.text(0.5, d1_smooth.max() * 0.85, "Extremum (Peak)\n→ Edge Location", fontsize=8.5, fontweight="bold", color=COLOR_FIRST_DERIV)
    ax_d1.set_title("2. 1st Derivative: Peaks → Edges", fontsize=11, fontweight="bold", color=COLOR_FIRST_DERIV)
    ax_d1.set_xlabel("$x$ coordinate", fontsize=9.5)
    ax_d1.grid(True, linestyle="--", alpha=0.4)
    ax_d1.legend(loc="upper right", fontsize=8.5)

    # Subplot 3: 2nd Derivative f''(x) (Laplacian / LoG concept)
    ax_d2 = fig.add_subplot(gs[0, 2])
    ax_d2.set_facecolor("#FFFFFF")
    ax_d2.plot(x, d2_clean, "k--", lw=1.8, label="Clean $f''(x)$")
    ax_d2.plot(x, d2_smooth, color=COLOR_SECOND_DERIV, lw=2.2, label=r"$\nabla^2 (G_\sigma * f)$ (LoG)")
    ax_d2.axhline(0, color="gray", linestyle="-", lw=1)
    ax_d2.axvline(0, color=COLOR_SECOND_DERIV, linestyle=":", lw=1.5)
    ax_d2.scatter([0], [0], color=COLOR_SECOND_DERIV, s=70, zorder=5)
    ax_d2.text(0.5, 0.05, "Zero-Crossing\n→ Inflection Point", fontsize=8.5, fontweight="bold", color=COLOR_SECOND_DERIV)
    ax_d2.set_title("3. 2nd Derivative: Zero-Crossings", fontsize=11, fontweight="bold", color=COLOR_SECOND_DERIV)
    ax_d2.set_xlabel("$x$ coordinate", fontsize=9.5)
    ax_d2.grid(True, linestyle="--", alpha=0.4)
    ax_d2.legend(loc="upper right", fontsize=8.5)

    # Subplot 4: Summary Theory Table / Flowchart
    ax_th = fig.add_subplot(gs[0, 3])
    ax_th.set_facecolor("#FFFFFF")
    ax_th.axis("off")
    summary_text = (
        r"$\mathbf{Core\ Edge\ Detection\ Math:}$" "\n\n"
        r"$\bullet\ \mathbf{1^{st}\ Derivative\ (Gradient):}$" "\n"
        r"   $\nabla I = [\,\partial I/\partial x,\, \partial I/\partial y\,]^T$" "\n"
        r"   $\|\nabla I\| = \sqrt{G_x^2 + G_y^2}$" "\n"
        r"   $\theta = \arctan2(G_y, G_x)$" "\n"
        r"   $\rightarrow\ \mathrm{Edges\ are\ local\ maxima}$" "\n\n"
        r"$\bullet\ \mathbf{2^{nd}\ Derivative\ (Laplacian):}$" "\n"
        r"   $\nabla^2 I = \partial^2 I/\partial x^2 + \partial^2 I/\partial y^2$" "\n"
        r"   $\rightarrow\ \mathrm{Edges\ are\ zero-crossings}$" "\n\n"
        r"$\bullet\ \mathbf{Key\ Takeaway:}$" "\n"
        r"   Differentiation amplifies noise!" "\n"
        r"   $\therefore$ Always apply Gaussian" "\n"
        r"   smoothing before differentiation."
    )
    bbox_info = dict(boxstyle="round,pad=0.6", fc="#FFFFFF", ec="#BDC3C7", lw=1.5)
    ax_th.text(0.05, 0.95, summary_text, transform=ax_th.transAxes, fontsize=9.5,
               va="top", ha="left", color=COLOR_TEXT_MAIN, bbox=bbox_info)

    # --- Bottom Row: Canny Edge Detector 4-Stage Breakdown ---
    test_img = generate_benchmark_image(300)

    # Step 1: Gaussian Smoothed
    step1_smooth = ndimage.gaussian_filter(test_img, sigma=1.4)
    # Step 2: Sobel Magnitude
    gx, gy, mag, direct = apply_sobel(step1_smooth)
    mag_norm = mag / (mag.max() + 1e-8)
    # Step 3: Non-Maximum Suppression (NMS)
    h, w = test_img.shape
    nms = np.zeros((h, w), dtype=np.float32)
    angle = np.rad2deg(direct) % 180
    pad_mag = np.pad(mag_norm, 1, mode="constant")
    for i in range(h):
        for j in range(w):
            pi, pj = i + 1, j + 1
            a = angle[i, j]
            if (0 <= a < 22.5) or (157.5 <= a <= 180):
                q, r = pad_mag[pi, pj + 1], pad_mag[pi, pj - 1]
            elif 22.5 <= a < 67.5:
                q, r = pad_mag[pi + 1, pj - 1], pad_mag[pi - 1, pj + 1]
            elif 67.5 <= a < 112.5:
                q, r = pad_mag[pi + 1, pj], pad_mag[pi - 1, pj]
            else:
                q, r = pad_mag[pi - 1, pj - 1], pad_mag[pi + 1, pj + 1]
            if pad_mag[pi, pj] >= q and pad_mag[pi, pj] >= r:
                nms[i, j] = pad_mag[pi, pj]

    # Step 4: Final Hysteresis
    canny_final = apply_canny(test_img, sigma=1.4, low_thresh=0.08, high_thresh=0.22)

    canny_steps = [
        ("Canny Stage 1: Gaussian Smoothing", step1_smooth, "gray", "Suppresses high-frequency noise"),
        ("Canny Stage 2: Gradient Magnitude", mag_norm, "inferno", r"$|\nabla I| = \sqrt{G_x^2 + G_y^2}$ (Thick edges)"),
        ("Canny Stage 3: Non-Max Suppression", nms, "inferno", "1-pixel edge thinning along gradient angle"),
        ("Canny Stage 4: Hysteresis Thresholding", canny_final, "gray", "Double threshold + 8-way edge connectivity"),
    ]

    for col, (stitle, sdata, scmap, snote) in enumerate(canny_steps):
        ax_s = fig.add_subplot(gs[1, col])
        ax_s.imshow(sdata, cmap=scmap)
        ax_s.set_title(stitle, fontsize=10.5, fontweight="bold", color=COLOR_MULTISTAGE)
        ax_s.set_xlabel(snote, fontsize=8.5, color=COLOR_TEXT_MUTED, labelpad=5)
        ax_s.set_xticks([]); ax_s.set_yticks([])
        for spine in ax_s.spines.values():
            spine.set_color(COLOR_MULTISTAGE)
            spine.set_linewidth(1.5)

    plt.suptitle("Calculus of Edge Detection: From 1D Derivative Principles to Multi-Stage Canny Pipeline",
                 fontsize=15, fontweight="bold", color=COLOR_TEXT_MAIN, y=0.98)

    out_path = output_dir / "classical_filters_04_derivative_edge_detection.png"
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close()
    return out_path


def get_sharpen_kernel() -> np.ndarray:
    """Discrete 2D 3x3 High-Pass Sharpening Kernel (Center Boost)."""
    return np.array([[ 0, -1,  0],
                     [-1,  5, -1],
                     [ 0, -1,  0]], dtype=np.float32)


# ---------------------------------------------------------------------------
# Figure 5: Paired 2-Row Visualization (Top: 2D Filters, Bottom: Filtered Outputs)
# ---------------------------------------------------------------------------
def plot_kernels_and_outputs_paired(output_dir: Path) -> Path:
    """
    Creates a clean, pedagogical 2-row visualization:
    - Top row: 2D Classical Spatial Filter Kernels (Identity, Box, Gaussian, Sobel X, Sobel Y, Laplacian, LoG, Gabor, Sharpen)
    - Bottom row: Resulting Convolved Output Images (I * h) for a representative sample input image.
    """
    sample_img = generate_benchmark_image(300)

    # 1. Identity / Input Reference
    k_ident = np.zeros((3, 3), dtype=np.float32)
    k_ident[1, 1] = 1.0
    out_ident = sample_img.copy()

    # 2. Box / Mean Filter
    k_box = get_box_kernel(ksize=5)
    out_box = ndimage.convolve(sample_img, k_box, mode="reflect")

    # 3. Gaussian Blur
    k_gauss = get_gaussian_kernel(ksize=9, sigma=1.8)
    out_gauss = ndimage.gaussian_filter(sample_img, sigma=2.0, mode="reflect")

    # 4. Sobel X
    kx, ky = get_sobel_kernels()
    out_sobel_x = ndimage.convolve(sample_img, kx, mode="reflect")

    # 5. Sobel Y
    out_sobel_y = ndimage.convolve(sample_img, ky, mode="reflect")

    # 6. Laplacian
    k_lap = get_laplacian_kernel()
    out_lap = ndimage.convolve(sample_img, k_lap, mode="reflect")

    # 7. Laplacian of Gaussian (LoG)
    k_log = get_log_kernel(ksize=11, sigma=1.6)
    out_log = ndimage.convolve(sample_img, k_log, mode="reflect")

    # 8. Gabor Wavelet (theta = 45 deg)
    k_gabor = get_gabor_kernel(ksize=15, sigma=2.5, theta=np.pi / 4, lambd=5.0)
    out_gabor = ndimage.convolve(sample_img, k_gabor, mode="reflect")

    # 9. Sharpening (Center Boost)
    k_sharpen = get_sharpen_kernel()
    out_sharpen = np.clip(ndimage.convolve(sample_img, k_sharpen, mode="reflect"), 0.0, 1.0)

    # Column specifications: (Top Title, Kernel, Kernel Colormap, Output Img, Output Colormap, vmin_k, vmax_k, vmin_o, vmax_o, Formula, Bottom Title, Category Color)
    items = [
        ("Identity (Input)", k_ident, "Blues", out_ident, "gray", 0.0, 1.0, 0.0, 1.0, r"$\delta(x,y)$", "Original Image", COLOR_ORIGINAL),
        ("Box Blur (5x5)", k_box, "Blues", out_box, "gray", 0.0, 0.04, 0.0, 1.0, r"$\frac{1}{25}\mathbf{1}_{5\times 5}$", "Mean Averaged", COLOR_LOWPASS),
        (r"Gaussian ($\sigma=1.8$)", k_gauss, "Blues", out_gauss, "gray", 0.0, k_gauss.max(), 0.0, 1.0, r"$G_\sigma(x,y)$", "Gaussian Blurred", COLOR_LOWPASS),
        (r"Sobel $K_x$", kx, "coolwarm", out_sobel_x, "coolwarm", -2.0, 2.0, -1.5, 1.5, r"$\partial I / \partial x$", "Vertical Edges ($G_x$)", COLOR_FIRST_DERIV),
        (r"Sobel $K_y$", ky, "coolwarm", out_sobel_y, "coolwarm", -2.0, 2.0, -1.5, 1.5, r"$\partial I / \partial y$", "Horizontal Edges ($G_y$)", COLOR_FIRST_DERIV),
        (r"Laplacian ($\nabla^2$)", k_lap, "coolwarm", out_lap, "bwr", -4.0, 4.0, -1.5, 1.5, r"$\nabla^2 I$", "Zero-Crossings", COLOR_SECOND_DERIV),
        ("LoG (Mexican Hat)", k_log, "coolwarm", out_log, "bwr", -abs(k_log).max(), abs(k_log).max(), -0.4, 0.4, r"$\nabla^2 G_\sigma$", "Blob & Scale Map", COLOR_SECOND_DERIV),
        (r"Gabor ($\theta=45^\circ$)", k_gabor, "PuOr", out_gabor, "PuOr", -abs(k_gabor).max(), abs(k_gabor).max(), -0.8, 0.8, r"$G_\sigma \cos(\omega x')$", "Diagonal Texture", COLOR_TEXTURE),
        ("Sharpening", k_sharpen, "coolwarm", out_sharpen, "gray", -2.0, 5.0, 0.0, 1.0, r"$I + \alpha(I - G*I)$", "Edge Enhanced", COLOR_LOWPASS),
    ]

    num_cols = len(items)
    fig, axes = plt.subplots(2, num_cols, figsize=(25, 8.2), gridspec_kw={"hspace": 0.36, "wspace": 0.28})
    fig.patch.set_facecolor("#FFFFFF")

    # Add row section badges on the left margin
    fig.text(0.015, 0.72, "TOP ROW:\n2D Spatial Filter\nKernels $h(u,v)$", fontsize=11, fontweight="bold",
             color=COLOR_TEXT_MAIN, va="center", ha="left", bbox=dict(boxstyle="round,pad=0.5", fc="#FFFFFF", ec="#BDC3C7", lw=1.5))

    fig.text(0.015, 0.28, "BOTTOM ROW:\nFiltered Output\nImages $I_{\mathrm{out}} = I * h$", fontsize=11, fontweight="bold",
             color=COLOR_TEXT_MAIN, va="center", ha="left", bbox=dict(boxstyle="round,pad=0.5", fc="#FFFFFF", ec="#BDC3C7", lw=1.5))

    for col_idx, (top_title, kernel, cmap_k, out_img, cmap_o, vmin_k, vmax_k, vmin_o, vmax_o, formula, bot_title, cat_color) in enumerate(items):
        ax_top = axes[0, col_idx]
        ax_bot = axes[1, col_idx]

        # Top Row: Kernel Visualization
        ax_top.set_facecolor("#FFFFFF")
        im_k = ax_top.imshow(kernel, cmap=cmap_k, vmin=vmin_k, vmax=vmax_k, interpolation="nearest")
        ax_top.set_title(top_title, fontsize=11, fontweight="bold", color=COLOR_TEXT_MAIN, pad=10)
        ax_top.set_xticks([]); ax_top.set_yticks([])
        ax_top.set_xlabel(formula, fontsize=9.5, color=COLOR_TEXT_MAIN, labelpad=5)

        # Annotate small 3x3 kernels with numeric matrix weights
        if kernel.shape[0] == 3 and kernel.shape[1] == 3:
            for r in range(3):
                for c in range(3):
                    val = kernel[r, c]
                    text_str = f"{int(val):+d}" if abs(val) >= 1.0 else f"{val:.1f}"
                    text_col = "white" if abs(val) >= 0.5 * max(abs(vmin_k), abs(vmax_k)) else "black"
                    ax_top.text(c, r, text_str, ha="center", va="center", fontsize=8.5, fontweight="bold", color=text_col)

        # Border styling for top row
        for spine in ax_top.spines.values():
            spine.set_color(cat_color)
            spine.set_linewidth(2.0)

        # Bottom Row: Output Image Visualization
        ax_bot.set_facecolor("#FFFFFF")
        im_o = ax_bot.imshow(out_img, cmap=cmap_o, vmin=vmin_o, vmax=vmax_o, interpolation="nearest")
        ax_bot.set_title(bot_title, fontsize=10.5, fontweight="bold", color=cat_color, pad=8)
        ax_bot.set_xticks([]); ax_bot.set_yticks([])
        ax_bot.set_xlabel(f"Convolved ({out_img.shape[0]}x{out_img.shape[1]})", fontsize=8.5, color=COLOR_TEXT_MUTED, labelpad=4)

        # Border styling for bottom row
        for spine in ax_bot.spines.values():
            spine.set_color(cat_color)
            spine.set_linewidth(2.0)

    plt.suptitle("Classical 2D Spatial Filters: Convolution Kernels (Top) and Their Output Feature Responses (Bottom)",
                 fontsize=16, fontweight="bold", color=COLOR_TEXT_MAIN, y=0.985)
    plt.tight_layout(rect=[0.085, 0.02, 0.99, 0.95])

    out_path = output_dir / "classical_filters_05_kernels_and_outputs_paired.png"
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close()
    return out_path


# ---------------------------------------------------------------------------
# Main Execution Entry Point
# ---------------------------------------------------------------------------
def main():
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True, parents=True)

    print("Generating Classical Computer Vision Filter Pedagogical Figures...")
    p1 = plot_master_gallery(output_dir)
    print(f"  [1/5] Master Gallery created: {p1}")

    p2 = plot_kernel_anatomy(output_dir)
    print(f"  [2/5] Kernel Anatomy created: {p2}")

    p3 = plot_noise_and_smoothing(output_dir)
    print(f"  [3/5] Noise & Smoothing created: {p3}")

    p4 = plot_derivative_edge_detection(output_dir)
    print(f"  [4/5] Edge Detection Calculus created: {p4}")

    p5 = plot_kernels_and_outputs_paired(output_dir)
    print(f"  [5/5] Paired 2-Row Filters & Outputs created: {p5}")

    print("\nAll 5 classical filter figures generated successfully in ./outputs/")


if __name__ == "__main__":
    main()

