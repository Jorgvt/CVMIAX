"""
visualize_gratings_fourier.py

A standalone, publication-quality visualization suite for Gratings and Windowed Gratings
in both the Spatial (Pixel) Domain and the 2D Fourier (Frequency) Domain.

Covers fundamental mathematical principles of 2D Signal Processing and Classical Vision:
1. `gratings_01_fundamental_sinusoids.png`:
   - Spatial frequency mapping: f_0 -> radial distance from DC in Fourier plane.
   - Spatial orientation mapping: theta -> orthogonal spectral peak orientation.
   - Phase offset mapping: phi -> shift in spatial domain, invariant magnitude spectrum |F|,
     linear anti-symmetric phase spectrum arg(F).
2. `gratings_02_waveform_families.png`:
   - Sinusoid, Square (Ronchi), Triangle, Plaid (Superposition), Radial Pinwheel (Siemens Star),
     and Chirp (Quadratic Phase Zone Plate).
   - Harmonic analysis (odd harmonics 1/k decay), linearity theorem, and dispersion.
3. `gratings_03_windowing_and_convolution.png`:
   - Step-by-step Convolution Theorem proof: F{s(x,y) * w(x,y)} = S(u,v) ** W(u,v).
   - Window geometry comparison: Gaussian (Gabor), Rectangular (Hard Box), Hann (Apodization),
     and Circular (Airy Disk).
   - Sinc spectral leakage / side lobes vs smooth Gaussian decay.
4. `gratings_04_gabor_wavelets_and_uncertainty.png`:
   - Heisenberg-Gabor Uncertainty Principle: Delta_x * Delta_u >= 1 / (4*pi).
   - Spatial spread sigma trade-off: Spatial localization vs Frequency resolution.
   - Spatial aspect ratio gamma: Orientation selectivity in Fourier space.
   - Quadrature pairs: Even (Cosine / Line detector) vs Odd (Sine / Edge detector) and Energy Envelope.
5. `gratings_05_3d_surfaces_and_spectral_landscapes.png`:
   - 3D topographic surface rendering of spatial waveforms and 2D Fourier log-magnitude landscapes.
   - Visualizing delta spikes vs smooth Gaussian hills vs Sinc grid ripples.

Usage:
    uv run python visualize_gratings_fourier.py
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
COLOR_BG_WHITE = "#FFFFFF"
COLOR_TEXT_MAIN = "#1A252F"
COLOR_TEXT_MUTED = "#5D6D7E"
COLOR_GRID = "#BDC3C7"

# Distinct thematic category accents
COLOR_SINUSOID = "#2980B9"     # Deep Blue (Sinusoidal pure tones)
COLOR_ORIENTATION = "#8E44AD"  # Violet / Purple (Oriented filters)
COLOR_PHASE = "#16A085"        # Teal (Phase domain)
COLOR_HARMONICS = "#D35400"    # Rich Orange (Harmonics / Square waves)
COLOR_WINDOW = "#C0392B"       # Crimson (Window envelopes)
COLOR_GABOR = "#27AE60"        # Emerald Green (Gabor / Wavelets)
COLOR_ACCENT_GOLD = "#F39C12"   # Golden Yellow

FONT_FAMILY = "sans-serif"

plt.rcParams.update({
    "font.family": FONT_FAMILY,
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "figure.titlesize": 14,
    "figure.facecolor": COLOR_BG_WHITE,
    "axes.facecolor": COLOR_BG_WHITE,
    "savefig.facecolor": COLOR_BG_WHITE,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# ---------------------------------------------------------------------------
# Core Fourier & Grating Mathematical Utilities
# ---------------------------------------------------------------------------
def compute_fourier_transform(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the 2D centered Discrete Fourier Transform, log-magnitude spectrum,
    and masked phase spectrum.

    Returns:
        F_shifted: Complex 2D array of centered DFT
        log_mag: Log-magnitude spectrum log(1 + |F|) normalized to [0, 1]
        phase: Phase spectrum arg(F) in [-pi, pi], masked where magnitude < 1e-4 * max_mag
    """
    F = np.fft.fft2(img)
    F_shifted = np.fft.fftshift(F)
    mag = np.abs(F_shifted)
    
    # Normalized log magnitude for visualization
    log_mag = np.log1p(mag)
    if np.max(log_mag) > 0:
        log_mag = log_mag / np.max(log_mag)
        
    phase = np.angle(F_shifted)
    # Mask negligible noise components to avoid chaotic phase patterns
    threshold = 1e-3 * np.max(mag)
    phase_masked = np.where(mag > threshold, phase, np.nan)
    
    return F_shifted, log_mag, phase_masked


def make_grid(size: int = 256, coord_range: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Creates symmetric centered spatial grid coordinates [-coord_range, coord_range]."""
    coords = np.linspace(-coord_range, coord_range, size, endpoint=False)
    x, y = np.meshgrid(coords, coords)
    return x, y


def generate_sinusoid_grating(
    size: int = 256,
    freq: float = 8.0,
    theta_deg: float = 0.0,
    phase_rad: float = 0.0
) -> np.ndarray:
    """
    Generates a 2D pure sinusoidal grating:
    s(x, y) = cos(2*pi*f_0*(x*cos(theta) + y*sin(theta)) + phi)
    """
    x, y = make_grid(size, coord_range=1.0)
    theta = np.radians(theta_deg)
    # Project coordinates along orientation vector
    x_rot = x * np.cos(theta) + y * np.sin(theta)
    return np.cos(2.0 * np.pi * freq * x_rot + phase_rad)


def generate_square_grating(
    size: int = 256,
    freq: float = 6.0,
    theta_deg: float = 0.0
) -> np.ndarray:
    """Generates a 2D square-wave (Ronchi) grating."""
    x, y = make_grid(size, coord_range=1.0)
    theta = np.radians(theta_deg)
    x_rot = x * np.cos(theta) + y * np.sin(theta)
    return np.sign(np.cos(2.0 * np.pi * freq * x_rot))


def generate_triangle_grating(
    size: int = 256,
    freq: float = 6.0,
    theta_deg: float = 0.0
) -> np.ndarray:
    """Generates a 2D triangle-wave grating with linear ramps."""
    x, y = make_grid(size, coord_range=1.0)
    theta = np.radians(theta_deg)
    x_rot = x * np.cos(theta) + y * np.sin(theta)
    # Normalized periodic triangle between -1 and 1
    t = freq * x_rot
    return 2.0 * np.abs(2.0 * (t - np.floor(t + 0.5))) - 1.0


def generate_plaid_grating(
    size: int = 256,
    freq1: float = 8.0,
    theta1_deg: float = 0.0,
    freq2: float = 8.0,
    theta2_deg: float = 90.0
) -> np.ndarray:
    """Generates a plaid grating formed by linear superposition of two sinusoids."""
    g1 = generate_sinusoid_grating(size, freq1, theta1_deg)
    g2 = generate_sinusoid_grating(size, freq2, theta2_deg)
    return 0.5 * (g1 + g2)


def generate_radial_pinwheel(size: int = 256, num_spokes: int = 16) -> np.ndarray:
    """Generates a polar radial pinwheel / Siemens star grating: cos(K * arctan2(y, x))."""
    x, y = make_grid(size, coord_range=1.0)
    angle = np.arctan2(y, x)
    return np.cos(num_spokes * angle)


def generate_chirp_zone_plate(size: int = 256, beta: float = 16.0) -> np.ndarray:
    """Generates a quadratic chirp / Fresnel zone plate grating: cos(pi * beta * (x^2 + y^2))."""
    x, y = make_grid(size, coord_range=1.0)
    r_sq = x**2 + y**2
    return np.cos(np.pi * beta * r_sq)


def generate_window(
    window_type: str = "gaussian",
    size: int = 256,
    sigma: float = 0.35,
    gamma: float = 1.0,
    theta_deg: float = 0.0,
    box_half_width: float = 0.4
) -> np.ndarray:
    """
    Generates 2D spatial window functions:
    'gaussian', 'rectangular', 'hann', 'tukey', 'circular'.
    """
    x, y = make_grid(size, coord_range=1.0)
    theta = np.radians(theta_deg)
    x_rot = x * np.cos(theta) + y * np.sin(theta)
    y_rot = -x * np.sin(theta) + y * np.cos(theta)

    if window_type == "gaussian":
        return np.exp(-(x_rot**2 + (gamma * y_rot)**2) / (2.0 * sigma**2))
    elif window_type == "rectangular":
        mask = (np.abs(x_rot) <= box_half_width) & (np.abs(y_rot) <= box_half_width)
        return mask.astype(np.float32)
    elif window_type == "hann":
        mask = (np.abs(x_rot) <= box_half_width) & (np.abs(y_rot) <= box_half_width)
        w_x = np.cos(np.pi * x_rot / (2.0 * box_half_width))**2
        w_y = np.cos(np.pi * y_rot / (2.0 * box_half_width))**2
        return np.where(mask, w_x * w_y, 0.0)
    elif window_type == "circular":
        r = np.sqrt(x**2 + y**2)
        return (r <= box_half_width).astype(np.float32)
    elif window_type == "tukey":
        # Tapered cosine with flat top
        r = np.sqrt(x**2 + y**2)
        r_inner = 0.5 * box_half_width
        r_outer = box_half_width
        w = np.zeros_like(r)
        w[r <= r_inner] = 1.0
        taper_mask = (r > r_inner) & (r <= r_outer)
        w[taper_mask] = 0.5 * (1.0 + np.cos(np.pi * (r[taper_mask] - r_inner) / (r_outer - r_inner)))
        return w
    else:
        raise ValueError(f"Unknown window type: {window_type}")


def generate_gabor(
    size: int = 256,
    freq: float = 8.0,
    theta_deg: float = 0.0,
    sigma: float = 0.35,
    gamma: float = 1.0,
    phase_rad: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a 2D Gabor wavelet in quadrature:
    - Real (Even / Cosine modulated)
    - Imaginary (Odd / Sine modulated)
    - Envelope (Gaussian window)
    """
    x, y = make_grid(size, coord_range=1.0)
    theta = np.radians(theta_deg)
    x_rot = x * np.cos(theta) + y * np.sin(theta)
    y_rot = -x * np.sin(theta) + y * np.cos(theta)

    envelope = np.exp(-(x_rot**2 + (gamma * y_rot)**2) / (2.0 * sigma**2))
    carrier_even = np.cos(2.0 * np.pi * freq * x_rot + phase_rad)
    carrier_odd = np.sin(2.0 * np.pi * freq * x_rot + phase_rad)

    gabor_even = envelope * carrier_even
    gabor_odd = envelope * carrier_odd

    return gabor_even, gabor_odd, envelope


# ---------------------------------------------------------------------------
# FIGURE 1: Fundamental Sinusoidal Gratings (Frequency, Orientation, Phase)
# ---------------------------------------------------------------------------
def plot_figure_1_fundamental_sinusoids(output_dir: Path):
    """
    Figure 1: Fundamental Sinusoids: Frequency, Orientation, and Phase Mapping.
    Deconstructs the 2D Fourier Transform of pure sinusoidal gratings across:
    1. Spatial Frequency f_0 variation (distance from DC in frequency plane)
    2. Spatial Orientation theta variation (rotation angle in frequency plane)
    3. Spatial Phase phi variation (invariance of |F|, antisymmetry of phase arg(F))
    """
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(
        3, 6,
        height_ratios=[1.0, 1.0, 1.15],
        hspace=0.40,
        wspace=0.32,
        left=0.04,
        right=0.96,
        top=0.91,
        bottom=0.05
    )

    fig.suptitle(
        "Fundamental 2D Sinusoidal Gratings: Spatial vs. Fourier Representation\n"
        r"Mathematical Basis: $s(x,y) = \cos(2\pi f_0 (x\cos\theta + y\sin\theta) + \phi) \longleftrightarrow "
        r"S(u,v) = \frac{1}{2} e^{i\phi}\delta(u-u_0, v-v_0) + \frac{1}{2} e^{-i\phi}\delta(u+u_0, v+v_0)$",
        fontsize=13.0,
        fontweight="bold",
        color=COLOR_TEXT_MAIN,
        y=0.975
    )

    size = 256
    zoom_limit = 50

    # -----------------------------------------------------------------------
    # Row 1: Frequency Variation (f_0 = 4, 10, 20 cycles/frame; theta = 0 deg)
    # -----------------------------------------------------------------------
    freqs = [4.0, 10.0, 20.0]
    freq_subtitles = ["Low Freq ($f_0=4$ cyc)", "Medium Freq ($f_0=10$ cyc)", "High Freq ($f_0=20$ cyc)"]

    for col_idx, (f, sub_t) in enumerate(zip(freqs, freq_subtitles)):
        grating = generate_sinusoid_grating(size=size, freq=f, theta_deg=0.0, phase_rad=0.0)
        _, log_mag, _ = compute_fourier_transform(grating)

        # Spatial Domain
        ax_sp = fig.add_subplot(gs[0, col_idx * 2])
        im_sp = ax_sp.imshow(grating, cmap="gray", vmin=-1, vmax=1, origin="lower")
        ax_sp.set_title(f"Pixel Domain: {sub_t}\n$T_x = 1/{int(f)}$ frame", fontsize=9.5, fontweight="bold", color=COLOR_SINUSOID)
        ax_sp.set_xticks([0, size // 2, size - 1])
        ax_sp.set_xticklabels(["-1", "0", "+1"])
        ax_sp.set_yticks([0, size // 2, size - 1])
        ax_sp.set_yticklabels(["-1", "0", "+1"])
        ax_sp.set_xlabel("x (norm)")
        ax_sp.set_ylabel("y (norm)")

        # Fourier Domain (Zoomed for peak clarity)
        ax_ft = fig.add_subplot(gs[0, col_idx * 2 + 1])
        im_ft = ax_ft.imshow(log_mag, cmap="inferno", origin="lower", extent=[-size // 2, size // 2, -size // 2, size // 2])
        ax_ft.set_title(f"Fourier Mag $|F(u,v)|$\n$u_0 = \\pm {int(f*2)}$, $v_0 = 0$", fontsize=9.5, fontweight="bold", color=COLOR_HARMONICS)
        ax_ft.axhline(0, color="cyan", linestyle=":", alpha=0.5, linewidth=0.8)
        ax_ft.axvline(0, color="cyan", linestyle=":", alpha=0.5, linewidth=0.8)
        ax_ft.set_xlim(-zoom_limit, zoom_limit)
        ax_ft.set_ylim(-zoom_limit, zoom_limit)
        ax_ft.set_xlabel("u (freq)")
        ax_ft.set_ylabel("v (freq)")

        # Draw circle showing radial distance from DC
        radius = f * 2.0  # normalized frequency mapping in 256 grid
        circle = patches.Circle((0, 0), radius, fill=False, edgecolor="cyan", linestyle="--", linewidth=1.2, alpha=0.85)
        ax_ft.add_patch(circle)
        ax_ft.scatter([radius, -radius], [0, 0], color="yellow", s=40, marker="o", edgecolors="black", zorder=5)
        ax_ft.annotate(f"r={int(radius)}", xy=(radius, 0), xytext=(radius + 5, 8),
                       arrowprops=dict(arrowstyle="->", color="white", lw=1),
                       color="white", fontsize=8, fontweight="bold")

    # -----------------------------------------------------------------------
    # Row 2: Orientation Variation (theta = 0, 45, 90 deg; f_0 = 10)
    # -----------------------------------------------------------------------
    thetas = [0.0, 45.0, 90.0]
    theta_subtitles = [r"$\theta = 0^\circ$ (Vertical Stripes)", r"$\theta = 45^\circ$ (Diagonal Stripes)", r"$\theta = 90^\circ$ (Horizontal Stripes)"]

    for col_idx, (th, sub_t) in enumerate(zip(thetas, theta_subtitles)):
        grating = generate_sinusoid_grating(size=size, freq=10.0, theta_deg=th, phase_rad=0.0)
        _, log_mag, _ = compute_fourier_transform(grating)

        # Spatial Domain
        ax_sp = fig.add_subplot(gs[1, col_idx * 2])
        im_sp = ax_sp.imshow(grating, cmap="gray", vmin=-1, vmax=1, origin="lower")
        ax_sp.set_title(f"Pixel Domain: {sub_t}\nOrientation $\\theta = {int(th)}^\\circ$", fontsize=9.5, fontweight="bold", color=COLOR_ORIENTATION)
        ax_sp.set_xticks([0, size // 2, size - 1])
        ax_sp.set_xticklabels(["-1", "0", "+1"])
        ax_sp.set_yticks([0, size // 2, size - 1])
        ax_sp.set_yticklabels(["-1", "0", "+1"])
        ax_sp.set_xlabel("x")
        ax_sp.set_ylabel("y")

        # Draw normal orientation arrow in spatial plot
        th_rad = np.radians(th)
        cx, cy = size // 2, size // 2
        dx, dy = 45 * np.cos(th_rad), 45 * np.sin(th_rad)
        ax_sp.annotate("", xy=(cx + dx, cy + dy), xytext=(cx - dx, cy - dy),
                       arrowprops=dict(arrowstyle="<->", color="red", lw=1.8))

        # Fourier Domain
        ax_ft = fig.add_subplot(gs[1, col_idx * 2 + 1])
        im_ft = ax_ft.imshow(log_mag, cmap="inferno", origin="lower", extent=[-size // 2, size // 2, -size // 2, size // 2])
        ax_ft.set_title(f"Fourier Mag $|F(u,v)|$\nSpike Angle $\\theta = {int(th)}^\\circ$", fontsize=9.5, fontweight="bold", color=COLOR_HARMONICS)
        ax_ft.axhline(0, color="cyan", linestyle=":", alpha=0.5, linewidth=0.8)
        ax_ft.axvline(0, color="cyan", linestyle=":", alpha=0.5, linewidth=0.8)
        ax_ft.set_xlim(-zoom_limit, zoom_limit)
        ax_ft.set_ylim(-zoom_limit, zoom_limit)
        ax_ft.set_xlabel("u")
        ax_ft.set_ylabel("v")

        # Ray through origin and peak markers
        ray_len = 45
        ax_ft.plot([-ray_len * np.cos(th_rad), ray_len * np.cos(th_rad)],
                   [-ray_len * np.sin(th_rad), ray_len * np.sin(th_rad)],
                   color="lime", linestyle="--", linewidth=1.2)
        u_p, v_p = 20 * np.cos(th_rad), 20 * np.sin(th_rad)
        ax_ft.scatter([u_p, -u_p], [v_p, -v_p], color="yellow", s=40, marker="o", edgecolors="black", zorder=5)

    # -----------------------------------------------------------------------
    # Row 3: Phase Variation & Fourier Phase Spectrum (phi = 0, pi/2, pi; f_0 = 8, theta = 0)
    # -----------------------------------------------------------------------
    phases = [0.0, np.pi / 2.0, np.pi]
    phase_subtitles = [r"$\phi = 0$ (Cosine / Even)", r"$\phi = \pi/2$ (Sine / Odd)", r"$\phi = \pi$ (Inverted / $-\cos$)"]

    for col_idx, (ph, sub_t) in enumerate(zip(phases, phase_subtitles)):
        grating = generate_sinusoid_grating(size=size, freq=8.0, theta_deg=0.0, phase_rad=ph)
        _, log_mag, phase_spec = compute_fourier_transform(grating)

        # Spatial Domain with 1D slice profile inset
        ax_sp = fig.add_subplot(gs[2, col_idx * 2])
        ax_sp.imshow(grating, cmap="gray", vmin=-1, vmax=1, origin="lower")
        ax_sp.set_title(f"Pixel Domain: {sub_t}\nSpatial Shift $\\Delta x = -\\phi/(2\\pi f)$", fontsize=9.5, fontweight="bold", color=COLOR_PHASE)
        ax_sp.set_xticks([0, size // 2, size - 1])
        ax_sp.set_xticklabels(["-1", "0", "+1"])
        ax_sp.set_yticks([0, size // 2, size - 1])
        ax_sp.set_yticklabels(["-1", "0", "+1"])
        ax_sp.set_xlabel("x")
        ax_sp.set_ylabel("y")

        # Inset 1D profile across center row
        profile = grating[size // 2, :]
        ax_inset = ax_sp.inset_axes([0.05, 0.05, 0.9, 0.28])
        ax_inset.plot(profile, color="cyan", linewidth=1.5)
        ax_inset.set_facecolor("#1A252FCC")
        ax_inset.set_ylim(-1.3, 1.3)
        ax_inset.set_xticks([])
        ax_inset.set_yticks([-1, 0, 1])
        ax_inset.tick_params(colors="white", labelsize=6.5)
        ax_inset.set_title("1D Center Row Slice", color="white", fontsize=7.5, pad=2)

        # Fourier Phase Spectrum
        ax_ph = fig.add_subplot(gs[2, col_idx * 2 + 1])
        im_ph = ax_ph.imshow(phase_spec, cmap="twilight_shifted", origin="lower",
                             vmin=-np.pi, vmax=np.pi, extent=[-size // 2, size // 2, -size // 2, size // 2])
        ax_ph.set_title(f"Fourier Phase $\\angle F(u,v)$\nPeak Phase $= \\pm {ph:.2f}$ rad", fontsize=9.5, fontweight="bold", color=COLOR_TEXT_MAIN)
        ax_ph.axhline(0, color="gray", linestyle=":", alpha=0.5, linewidth=0.8)
        ax_ph.axvline(0, color="gray", linestyle=":", alpha=0.5, linewidth=0.8)
        ax_ph.set_xlim(-zoom_limit, zoom_limit)
        ax_ph.set_ylim(-zoom_limit, zoom_limit)
        ax_ph.set_xlabel("u")
        ax_ph.set_ylabel("v")

        # Highlight peaks
        u_peak = 16  # 8 cyc * 2
        ax_ph.scatter([u_peak, -u_peak], [0, 0], s=70, facecolors='none', edgecolors='red', linewidth=2.0)
        ax_ph.annotate(r"$+\phi$", xy=(u_peak, 0), xytext=(u_peak + 5, 8), color="red", fontsize=8.5, fontweight="bold")
        ax_ph.annotate(r"$-\phi$", xy=(-u_peak, 0), xytext=(-u_peak - 18, 8), color="red", fontsize=8.5, fontweight="bold")

    out_path = output_dir / "gratings_01_fundamental_sinusoids.png"
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Generated: {out_path}")


# ---------------------------------------------------------------------------
# FIGURE 2: Non-Sinusoidal Grating Families & Harmonic Spectrum
# ---------------------------------------------------------------------------
def plot_figure_2_waveform_families(output_dir: Path):
    """
    Figure 2: Non-Sinusoidal Grating Families & Harmonics.
    Compares 6 distinctive grating waveforms:
    1. Pure Sinusoid (Single delta pair)
    2. Square Wave / Ronchi (Odd harmonics 1/k decay: f, 3f, 5f, 7f, ...)
    3. Triangle Wave (Harmonics 1/k^2 decay: f, 3f, 5f, ...)
    4. Plaid Grating (Linear superposition of orthogonal sinusoids: 4 delta peaks)
    5. Radial Pinwheel / Siemens Star (Radial/Angular spatial-frequency distribution)
    6. Linear Chirp / Fresnel Zone Plate (Broadband quadratic phase disk)
    """
    fig, axes = plt.subplots(6, 3, figsize=(15, 18), gridspec_kw={"width_ratios": [1.0, 1.0, 1.25], "hspace": 0.40, "wspace": 0.28})
    
    fig.suptitle(
        "Waveform Families & Harmonic Spectra in Spatial vs. Fourier Domains\n"
        r"Demonstrating Linearity $\mathcal{F}\{f+g\}=\mathcal{F}\{f\}+\mathcal{F}\{g\}$, Fourier Series Harmonics, and Spatial Dispersion",
        fontsize=13.0,
        fontweight="bold",
        color=COLOR_TEXT_MAIN,
        y=0.985
    )

    size = 256
    u_axis = np.fft.fftshift(np.fft.fftfreq(size, d=1.0)) * size

    # Waveform configurations
    waveforms = [
        ("Pure Sinusoid Grating",
         generate_sinusoid_grating(size=size, freq=8.0, theta_deg=0.0),
         "Single harmonic tone\n$S(u,v) = \\frac{1}{2}[\\delta(u-u_0) + \\delta(u+u_0)]$",
         COLOR_SINUSOID),
        
        ("Square Wave (Ronchi Grating)",
         generate_square_grating(size=size, freq=8.0, theta_deg=0.0),
         "Infinite odd harmonics with $1/k$ decay\n$s(x) = \\frac{4}{\\pi}\\sum_{k=1,3,5}\\frac{1}{k}\\sin(2\\pi k f_0 x)$",
         COLOR_HARMONICS),
        
        ("Triangle Wave Grating",
         generate_triangle_grating(size=size, freq=8.0, theta_deg=0.0),
         "Odd harmonics with $1/k^2$ fast decay\n$s(x) = \\frac{8}{\\pi^2}\\sum_{k=1,3,5}\\frac{(-1)^{(k-1)/2}}{k^2}\\sin(2\\pi k f_0 x)$",
         COLOR_ORIENTATION),
        
        ("Plaid Grating (Crossed Sinusoids)",
         generate_plaid_grating(size=size, freq1=8.0, theta1_deg=0.0, freq2=8.0, theta2_deg=90.0),
         "Linear Superposition Principle\n$\\mathcal{F}\\{s_1 + s_2\\} = \\mathcal{F}\\{s_1\\} + \\mathcal{F}\\{s_2\\}$ (4 discrete peaks)",
         COLOR_GABOR),
        
        ("Radial Pinwheel (Siemens Star)",
         generate_radial_pinwheel(size=size, num_spokes=16),
         "Radial-angular frequency\n$s(r,\\theta) = \\cos(K\\theta) \\longleftrightarrow$ Angular starburst spectrum",
         COLOR_PHASE),
        
        ("Quadratic Chirp (Zone Plate)",
         generate_chirp_zone_plate(size=size, beta=18.0),
         "Linear frequency sweep $f(r) \\propto r$\nBroadband flat spectral disk (Spatial-frequency dispersion)",
         COLOR_WINDOW)
    ]

    for row_idx, (name, img, desc, color_theme) in enumerate(waveforms):
        _, log_mag, _ = compute_fourier_transform(img)

        # 1. Pixel Domain
        ax_sp = axes[row_idx, 0]
        ax_sp.imshow(img, cmap="gray", vmin=-1, vmax=1, origin="lower")
        ax_sp.set_title(f"Pixel Domain: {name}", fontsize=9.5, fontweight="bold", color=color_theme)
        ax_sp.set_xticks([])
        ax_sp.set_yticks([])
        ax_sp.set_ylabel(f"Row {row_idx+1}", fontweight="bold", color=color_theme)

        # 2. 2D Fourier Log-Magnitude (Zoomed for peak sharpness)
        ax_ft = axes[row_idx, 1]
        ax_ft.imshow(log_mag, cmap="inferno", origin="lower", extent=[-size // 2, size // 2, -size // 2, size // 2])
        ax_ft.set_title("2D Fourier Log-Magnitude $\\log(1+|F|)$", fontsize=9.5, fontweight="bold", color=COLOR_TEXT_MAIN)
        ax_ft.axhline(0, color="cyan", linestyle=":", alpha=0.4, linewidth=0.7)
        ax_ft.axvline(0, color="cyan", linestyle=":", alpha=0.4, linewidth=0.7)
        ax_ft.set_xlim(-60, 60)
        ax_ft.set_ylim(-60, 60)
        ax_ft.set_xticks([-50, 0, 50])
        ax_ft.set_yticks([-50, 0, 50])

        # 3. 1D Cross-Section Spectrum & Theoretical Annotation
        ax_prof = axes[row_idx, 2]
        
        if row_idx in [0, 1, 2]:
            # Horizontal slice through DC (v = 0)
            center_slice = log_mag[size // 2, :]
            ax_prof.plot(u_axis, center_slice, color=color_theme, linewidth=1.8, label="Spectral Profile (v=0)")
            ax_prof.fill_between(u_axis, 0, center_slice, color=color_theme, alpha=0.25)
            ax_prof.set_xlim(-70, 70)
            ax_prof.set_ylim(-0.05, 1.05)
            ax_prof.set_xlabel("Frequency index u (cycles/frame)")
            ax_prof.set_ylabel("Normalized Log Mag")
            ax_prof.grid(True, linestyle=":", alpha=0.5)

            # Annotate harmonic peaks
            f0 = 16  # frequency in indices for 8 cyc
            if row_idx == 0:
                ax_prof.annotate(r"$\pm f_0$", xy=(f0, center_slice[size//2 + f0]), xytext=(f0+4, 0.8),
                                 arrowprops=dict(arrowstyle="->", color="black", lw=1.2), fontweight="bold")
            elif row_idx == 1:
                # Square wave: f0, 3f0, 5f0
                for h_idx, h_mult in enumerate([1, 3]):
                    h_pos = f0 * h_mult
                    if size//2 + h_pos < size:
                        ax_prof.annotate(f"{h_mult}" + r"$f_0$" + f"\n(1/{h_mult})",
                                         xy=(h_pos, center_slice[size//2 + h_pos]),
                                         xytext=(h_pos+2, 0.75 - h_idx*0.2),
                                         arrowprops=dict(arrowstyle="->", color="black", lw=1), fontsize=8, fontweight="bold")
            elif row_idx == 2:
                # Triangle wave: f0, 3f0
                ax_prof.annotate(r"$f_0$" + "\n(1.0)", xy=(f0, center_slice[size//2 + f0]), xytext=(f0+3, 0.8),
                                 arrowprops=dict(arrowstyle="->", color="black", lw=1.2), fontsize=8, fontweight="bold")
                h3 = f0 * 3
                if size//2 + h3 < size:
                    ax_prof.annotate(r"$3f_0$" + "\n(1/9)", xy=(h3, center_slice[size//2 + h3]), xytext=(h3+3, 0.45),
                                     arrowprops=dict(arrowstyle="->", color="black", lw=1), fontsize=8, fontweight="bold")

        elif row_idx == 3:
            # Plaid diagonal profile
            diag_slice = np.diag(log_mag)
            diag_u = np.linspace(-size//2, size//2, len(diag_slice))
            ax_prof.plot(u_axis, log_mag[size//2, :], color=COLOR_SINUSOID, linewidth=1.5, label="u-slice (v=0)")
            ax_prof.plot(u_axis, log_mag[:, size//2], color=COLOR_ORIENTATION, linewidth=1.5, linestyle="--", label="v-slice (u=0)")
            ax_prof.set_xlim(-60, 60)
            ax_prof.set_ylim(-0.05, 1.05)
            ax_prof.set_xlabel("Frequency index (cycles/frame)")
            ax_prof.set_ylabel("Normalized Log Mag")
            ax_prof.legend(fontsize=8, loc="upper right")
            ax_prof.grid(True, linestyle=":", alpha=0.5)

        elif row_idx == 4:
            # Radial Pinwheel angular energy
            angles = np.linspace(0, 2*np.pi, 360, endpoint=False)
            r_sample = 25.0
            cx, cy = size // 2, size // 2
            angular_vals = [log_mag[int(cy + r_sample*np.sin(a)), int(cx + r_sample*np.cos(a))] for a in angles]
            ax_prof.plot(np.degrees(angles), angular_vals, color=color_theme, linewidth=1.6)
            ax_prof.set_xlim(0, 360)
            ax_prof.set_ylim(-0.05, 1.05)
            ax_prof.set_xlabel("Orientation Angle (degrees)")
            ax_prof.set_ylabel("Spectral Magnitude at r=25")
            ax_prof.grid(True, linestyle=":", alpha=0.5)
            ax_prof.set_title("Angular Harmonic Distribution", fontsize=8.5, color=color_theme)

        elif row_idx == 5:
            # Chirp radial profile
            r_axis = np.arange(0, size//2)
            radial_prof = [log_mag[size//2, size//2 + r] for r in r_axis]
            ax_prof.plot(r_axis, radial_prof, color=color_theme, linewidth=1.6)
            ax_prof.fill_between(r_axis, 0, radial_prof, color=color_theme, alpha=0.25)
            ax_prof.set_xlim(0, 80)
            ax_prof.set_ylim(-0.05, 1.05)
            ax_prof.set_xlabel("Radial Frequency r (cycles/frame)")
            ax_prof.set_ylabel("Normalized Log Mag")
            ax_prof.grid(True, linestyle=":", alpha=0.5)
            ax_prof.set_title("Broadband Uniform Spectral Plateau", fontsize=8.5, color=color_theme)

        # Theoretical subtitle text box
        ax_prof.text(0.02, 0.95, desc, transform=ax_prof.transAxes,
                     fontsize=8.0, verticalalignment='top',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="#F8F9F9", edgecolor=COLOR_GRID, alpha=0.9))

    out_path = output_dir / "gratings_02_waveform_families.png"
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Generated: {out_path}")


# ---------------------------------------------------------------------------
# FIGURE 3: Windowed Gratings & The Convolution Theorem
# ---------------------------------------------------------------------------
def plot_figure_3_windowing_and_convolution(output_dir: Path):
    """
    Figure 3: Windowed Gratings & The 2D Convolution Theorem.
    Part A: Explicit step-by-step mathematical demonstration of:
            Spatial Multiplication:  g(x,y) = s(x,y) * w(x,y)
                    <== 2D Fourier Transform ==>
            Fourier Convolution:    G(u,v) = S(u,v) ** W(u,v)
    Part B: Comparison of 4 classic window apertures:
            1. Gaussian Window (Gabor patch, no ringing/sidelobes)
            2. Rectangular Box (Severe Sinc cross-artifacts / spectral leakage)
            3. Hann / Tukey Window (Smooth cosine apodization suppressing sidelobes)
            4. Circular Aperture (Airy disk Jinc ring pattern)
    """
    fig = plt.figure(figsize=(16, 17))
    gs = fig.add_gridspec(
        5, 5,
        height_ratios=[1.1, 1.0, 1.0, 1.0, 1.0],
        hspace=0.46,
        wspace=0.32,
        left=0.04,
        right=0.96,
        top=0.92,
        bottom=0.04
    )

    fig.suptitle(
        "Windowed Gratings: The 2D Convolution Theorem & Spectral Leakage Mechanics\n"
        r"$\mathcal{F}\{ s(x,y) \cdot w(x,y) \} = S(u,v) * W(u,v) = \frac{1}{2} W(u-u_0, v-v_0) + \frac{1}{2} W(u+u_0, v+v_0)$",
        fontsize=13.0,
        fontweight="bold",
        color=COLOR_TEXT_MAIN,
        y=0.975
    )

    size = 256
    u_axis = np.fft.fftshift(np.fft.fftfreq(size, d=1.0)) * size

    # -----------------------------------------------------------------------
    # Section A (Top Row): The Convolution Theorem Anatomy
    # -----------------------------------------------------------------------
    grating = generate_sinusoid_grating(size=size, freq=10.0, theta_deg=0.0)
    window = generate_window("gaussian", size=size, sigma=0.25)
    gabor = grating * window

    _, log_s, _ = compute_fourier_transform(grating)
    _, log_w, _ = compute_fourier_transform(window)
    _, log_g, _ = compute_fourier_transform(gabor)

    # 1. Carrier Grating s(x,y)
    ax_s = fig.add_subplot(gs[0, 0])
    ax_s.imshow(grating, cmap="gray", origin="lower")
    ax_s.set_title("1. Carrier Grating $s(x,y)$\n(Infinite Sinusoid)", fontsize=9, fontweight="bold", color=COLOR_SINUSOID)
    ax_s.set_xticks([])
    ax_s.set_yticks([])

    # 2. Window Envelope w(x,y)
    ax_w = fig.add_subplot(gs[0, 1])
    ax_w.imshow(window, cmap="magma", origin="lower")
    ax_w.set_title("2. Spatial Window $w(x,y)$\n(Gaussian Envelope)", fontsize=9, fontweight="bold", color=COLOR_WINDOW)
    ax_w.set_xticks([])
    ax_w.set_yticks([])

    # 3. Multiplied Windowed Grating g(x,y)
    ax_gw = fig.add_subplot(gs[0, 2])
    ax_gw.imshow(gabor, cmap="gray", origin="lower")
    ax_gw.set_title("3. Windowed Grating $g(x,y)$\n$g = s(x,y) \\cdot w(x,y)$", fontsize=9, fontweight="bold", color=COLOR_GABOR)
    ax_gw.set_xticks([])
    ax_gw.set_yticks([])

    # 4. Grating Fourier S(u,v) + Window Fourier W(u,v)
    ax_sw_ft = fig.add_subplot(gs[0, 3])
    # Composite overlay: Window at DC in red/yellow, Grating deltas in cyan
    ax_sw_ft.imshow(log_w, cmap="inferno", origin="lower", extent=[-size//2, size//2, -size//2, size//2])
    u0 = 20
    ax_sw_ft.scatter([u0, -u0], [0, 0], color="cyan", s=50, marker="x", linewidth=2.0, label="Delta Spikes $S(u,v)$")
    ax_sw_ft.set_title("4. Fourier Spectra $S(u,v)$ & $W(u,v)$\nWindow Spectrum at DC $(0,0)$", fontsize=9, fontweight="bold", color=COLOR_HARMONICS)
    ax_sw_ft.set_xlim(-60, 60)
    ax_sw_ft.set_ylim(-60, 60)
    ax_sw_ft.legend(fontsize=7.5, loc="upper right")

    # 5. Convolved Fourier G(u,v) = S * W
    ax_g_ft = fig.add_subplot(gs[0, 4])
    ax_g_ft.imshow(log_g, cmap="inferno", origin="lower", extent=[-size//2, size//2, -size//2, size//2])
    ax_g_ft.set_title("5. Convolved Spectrum $G(u,v)$\n$G(u,v) = S(u,v) * W(u,v)$", fontsize=9, fontweight="bold", color=COLOR_GABOR)
    ax_g_ft.set_xlim(-60, 60)
    ax_g_ft.set_ylim(-60, 60)
    ax_g_ft.annotate("Window Spectrum\nshifted to $\\pm f_0$", xy=(u0, 0), xytext=(u0 - 15, 25),
                     arrowprops=dict(arrowstyle="->", color="white", lw=1.2), color="white", fontsize=8, fontweight="bold")

    # -----------------------------------------------------------------------
    # Section B (Rows 1-4): Window Aperture Comparison & Sidelobe Leakage
    # -----------------------------------------------------------------------
    window_configs = [
        ("Gaussian Window (Gabor Filter)",
         "gaussian",
         dict(sigma=0.3),
         "Optimal time-frequency concentration (Heisenberg bound)\nZero sinc ripples; smooth Gaussian spectral falloff",
         COLOR_GABOR),
        
        ("Rectangular Box Aperture (Hard Cutoff)",
         "rectangular",
         dict(box_half_width=0.4),
         "Discontinuous hard boundaries produce 2D Sinc sidelobes\nHigh spectral leakage with prominent cross-artifacts",
         COLOR_HARMONICS),
        
        ("Hann / Apodization Window (Tapered Cosine)",
         "hann",
         dict(box_half_width=0.4),
         "Smooth cosine tapering to zero at boundary\nDrastically reduced sidelobes compared to Rectangular box",
         COLOR_ORIENTATION),
        
        ("Circular Pupil Aperture (Disk)",
         "circular",
         dict(box_half_width=0.4),
         "Sharp circular edge produces 2D Jinc (Airy disk) ring ripples\nCharacteristic of circular optical lenses",
         COLOR_WINDOW)
    ]

    for row_offset, (win_name, win_type, win_kwargs, explanation, col_theme) in enumerate(window_configs):
        r = row_offset + 1
        
        win = generate_window(win_type, size=size, **win_kwargs)
        win_grating = grating * win
        _, log_win, _ = compute_fourier_transform(win)
        _, log_wg, _ = compute_fourier_transform(win_grating)

        # 1. Window in Pixel Domain
        ax_w_pix = fig.add_subplot(gs[r, 0])
        ax_w_pix.imshow(win, cmap="magma", origin="lower")
        ax_w_pix.set_title(f"Window: {win_name.split('(')[0]}", fontsize=9, fontweight="bold", color=col_theme)
        ax_w_pix.set_xticks([])
        ax_w_pix.set_yticks([])

        # 2. Windowed Grating in Pixel Domain
        ax_wg_pix = fig.add_subplot(gs[r, 1])
        ax_wg_pix.imshow(win_grating, cmap="gray", origin="lower")
        ax_wg_pix.set_title("Windowed Grating $g(x,y)$", fontsize=9, fontweight="bold", color=COLOR_TEXT_MAIN)
        ax_wg_pix.set_xticks([])
        ax_wg_pix.set_yticks([])

        # 3. 2D Fourier Log-Magnitude of Window alone
        ax_w_spec = fig.add_subplot(gs[r, 2])
        ax_w_spec.imshow(log_win, cmap="inferno", origin="lower", extent=[-size//2, size//2, -size//2, size//2])
        ax_w_spec.set_title("Window Spectrum $W(u,v)$", fontsize=9, fontweight="bold", color=COLOR_TEXT_MAIN)
        ax_w_spec.set_xlim(-60, 60)
        ax_w_spec.set_ylim(-60, 60)

        # 4. 2D Fourier Log-Magnitude of Windowed Grating
        ax_wg_spec = fig.add_subplot(gs[r, 3])
        ax_wg_spec.imshow(log_wg, cmap="inferno", origin="lower", extent=[-size//2, size//2, -size//2, size//2])
        ax_wg_spec.set_title("Shifted Spectrum $G(u,v)$", fontsize=9, fontweight="bold", color=col_theme)
        ax_wg_spec.set_xlim(-60, 60)
        ax_wg_spec.set_ylim(-60, 60)

        # 5. 1D Cross-section Profile showing Sidelobes / Leakage
        ax_prof = fig.add_subplot(gs[r, 4])
        profile_slice = log_wg[size // 2, :]
        ax_prof.plot(u_axis, profile_slice, color=col_theme, linewidth=1.6)
        ax_prof.fill_between(u_axis, 0, profile_slice, color=col_theme, alpha=0.2)
        ax_prof.set_xlim(-50, 50)
        ax_prof.set_ylim(-0.05, 1.05)
        ax_prof.set_xlabel("u (freq)")
        ax_prof.set_ylabel("Log Mag")
        ax_prof.grid(True, linestyle=":", alpha=0.5)

        # Annotate leakage
        if win_type == "rectangular":
            ax_prof.annotate("Sinc Sidelobes\n(High Leakage)", xy=(35, profile_slice[size//2 + 35]), xytext=(22, 0.7),
                             arrowprops=dict(arrowstyle="->", color="red", lw=1), color="red", fontsize=7.5, fontweight="bold")
        elif win_type == "gaussian":
            ax_prof.annotate("Smooth Gaussian Decay\n(No Ringing)", xy=(25, profile_slice[size//2 + 25]), xytext=(15, 0.75),
                             arrowprops=dict(arrowstyle="->", color="green", lw=1), color="green", fontsize=7.5, fontweight="bold")
        elif win_type == "hann":
            ax_prof.annotate("Suppressed Sidelobes", xy=(25, profile_slice[size//2 + 25]), xytext=(15, 0.7),
                             arrowprops=dict(arrowstyle="->", color=COLOR_ORIENTATION, lw=1), color=COLOR_ORIENTATION, fontsize=7.5, fontweight="bold")

    out_path = output_dir / "gratings_03_windowing_and_convolution.png"
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Generated: {out_path}")


# ---------------------------------------------------------------------------
# FIGURE 4: Gabor Wavelets, Quadrature Pairs, & Uncertainty Principle
# ---------------------------------------------------------------------------
def plot_figure_4_gabor_wavelets_and_uncertainty(output_dir: Path):
    """
    Figure 4: Gabor Wavelets, Quadrature Pairs, & The Heisenberg-Gabor Uncertainty Principle.
    Deconstructs the mathematical properties of 2D Gabor filters:
    1. Heisenberg Uncertainty Principle trade-off: Spatial envelope size sigma vs Fourier spectral bandwidth.
    2. Spatial Aspect Ratio gamma (Ellipticity): Anisotropic orientation tuning.
    3. Quadrature pairs: Even (Cosine / Line) vs Odd (Sine / Edge) vs Complex Energy Envelope.
    """
    fig = plt.figure(figsize=(16, 13))
    gs = fig.add_gridspec(
        3, 4,
        height_ratios=[1.0, 1.0, 1.1],
        hspace=0.42,
        wspace=0.28,
        left=0.05,
        right=0.95,
        top=0.90,
        bottom=0.05
    )

    fig.suptitle(
        "Gabor Wavelets: Spatial-Frequency Localization, Uncertainty Principle, & Quadrature Pairs\n"
        r"$\text{Gabor}(x,y) = \exp\left(-\frac{x'^2 + \gamma^2 y'^2}{2\sigma^2}\right) \cdot \cos(2\pi f_0 x' + \phi), \quad "
        r"\text{Uncertainty Principle: } \Delta x \cdot \Delta u \geq \frac{1}{4\pi}$",
        fontsize=13.0,
        fontweight="bold",
        color=COLOR_TEXT_MAIN,
        y=0.970
    )

    size = 256
    u_axis = np.fft.fftshift(np.fft.fftfreq(size, d=1.0)) * size

    # -----------------------------------------------------------------------
    # Row 1: Heisenberg Uncertainty Principle: Sigma variation (0.12, 0.28, 0.55)
    # -----------------------------------------------------------------------
    sigmas = [0.12, 0.28, 0.55]
    sigma_titles = [
        r"Narrow $\sigma=0.12$ (High Spatial Precision)",
        r"Medium $\sigma=0.28$ (Balanced)",
        r"Broad $\sigma=0.55$ (High Frequency Precision)"
    ]

    for col_idx, (sig, sub_t) in enumerate(zip(sigmas, sigma_titles)):
        g_even, _, _ = generate_gabor(size=size, freq=8.0, theta_deg=0.0, sigma=sig, gamma=1.0)
        _, log_mag, _ = compute_fourier_transform(g_even)

        # Spatial Domain
        ax_sp = fig.add_subplot(gs[0, col_idx])
        ax_sp.imshow(g_even, cmap="seismic", vmin=-1, vmax=1, origin="lower")
        ax_sp.set_title(f"Pixel: {sub_t}\nEnvelope $\\sigma_x = {sig}$", fontsize=9, fontweight="bold", color=COLOR_GABOR)
        ax_sp.set_xticks([])
        ax_sp.set_yticks([])

        # Draw contour of Gaussian envelope at 1-sigma
        circ = patches.Circle((size // 2, size // 2), sig * (size // 2), fill=False, edgecolor="yellow", linestyle="--", linewidth=1.5)
        ax_sp.add_patch(circ)

    # 4th Column in Row 1: 1D Fourier bandwidth comparison across sigmas
    ax_bw = fig.add_subplot(gs[0, 3])
    colors = [COLOR_HARMONICS, COLOR_GABOR, COLOR_SINUSOID]
    for sig, col, lab in zip(sigmas, colors, [r"Narrow $\sigma=0.12$", r"Medium $\sigma=0.28$", r"Broad $\sigma=0.55$"]):
        g_even, _, _ = generate_gabor(size=size, freq=8.0, theta_deg=0.0, sigma=sig, gamma=1.0)
        _, log_mag, _ = compute_fourier_transform(g_even)
        center_prof = log_mag[size // 2, :]
        ax_bw.plot(u_axis, center_prof, color=col, linewidth=1.8, label=lab)
    
    ax_bw.set_xlim(-40, 40)
    ax_bw.set_ylim(-0.05, 1.05)
    ax_bw.set_title("Spectral Bandwidth $\\Delta u$ Trade-off\n(Inverse Fourier Scaling)", fontsize=9.5, fontweight="bold", color=COLOR_TEXT_MAIN)
    ax_bw.set_xlabel("u (cycles/frame)")
    ax_bw.set_ylabel("Log Mag")
    ax_bw.legend(fontsize=8, loc="upper right")
    ax_bw.grid(True, linestyle=":", alpha=0.5)

    # -----------------------------------------------------------------------
    # Row 2: Aspect Ratio (Gamma) Variation: Anisotropic Receptive Fields
    # -----------------------------------------------------------------------
    gammas = [0.4, 1.0, 2.5]
    gamma_titles = [
        r"Elongated ($\gamma=0.4$, narrow Fourier band)",
        r"Isotropic Circular ($\gamma=1.0$)",
        r"Compressed ($\gamma=2.5$, wide Fourier band)"
    ]

    for col_idx, (gam, sub_t) in enumerate(zip(gammas, gamma_titles)):
        g_even, _, _ = generate_gabor(size=size, freq=8.0, theta_deg=30.0, sigma=0.3, gamma=gam)
        _, log_mag, _ = compute_fourier_transform(g_even)

        # Spatial Domain
        ax_sp = fig.add_subplot(gs[1, col_idx])
        ax_sp.imshow(g_even, cmap="seismic", vmin=-1, vmax=1, origin="lower")
        ax_sp.set_title(f"Pixel: {sub_t}\nAspect Ratio $\\gamma = {gam}$", fontsize=9, fontweight="bold", color=COLOR_ORIENTATION)
        ax_sp.set_xticks([])
        ax_sp.set_yticks([])

    # 4th Column in Row 2: 2D Fourier Spectrum for anisotropic gamma
    g_aniso, _, _ = generate_gabor(size=size, freq=8.0, theta_deg=30.0, sigma=0.3, gamma=0.4)
    _, log_aniso, _ = compute_fourier_transform(g_aniso)
    ax_aniso = fig.add_subplot(gs[1, 3])
    ax_aniso.imshow(log_aniso, cmap="inferno", origin="lower", extent=[-size//2, size//2, -size//2, size//2])
    ax_aniso.set_title("2D Fourier Orientation Tuning\nElliptical Spectral Peaks", fontsize=9.5, fontweight="bold", color=COLOR_ORIENTATION)
    ax_aniso.set_xlim(-50, 50)
    ax_aniso.set_ylim(-50, 50)
    ax_aniso.set_xlabel("u")
    ax_aniso.set_ylabel("v")

    # -----------------------------------------------------------------------
    # Row 3: Quadrature Pairs (Even vs Odd vs Complex Energy Envelope)
    # -----------------------------------------------------------------------
    g_even, g_odd, envelope = generate_gabor(size=size, freq=8.0, theta_deg=0.0, sigma=0.3, gamma=1.0)
    energy = np.sqrt(g_even**2 + g_odd**2)

    quad_items = [
        ("Even Gabor (Cosine Modulated)", g_even, "seismic", "Symmetric Line / Bar Detector\n$g_{\\text{even}}(x) = w(x)\\cos(2\\pi f_0 x)$", COLOR_SINUSOID),
        ("Odd Gabor (Sine Modulated)", g_odd, "seismic", "Anti-symmetric Edge Detector\n$g_{\\text{odd}}(x) = w(x)\\sin(2\\pi f_0 x)$", COLOR_HARMONICS),
        ("Gaussian Spatial Envelope", envelope, "magma", "Window Envelope $w(x,y)$\n$\\exp(-(x^2+y^2)/2\\sigma^2)$", COLOR_WINDOW),
        ("Complex Energy Envelope", energy, "viridis", "Phase-Invariant Complex Cell Model\n$\\sqrt{g_{\\text{even}}^2 + g_{\\text{odd}}^2}$", COLOR_GABOR)
    ]

    for col_idx, (q_name, q_data, q_cmap, q_desc, q_col) in enumerate(quad_items):
        ax_q = fig.add_subplot(gs[2, col_idx])
        if q_cmap == "seismic":
            ax_q.imshow(q_data, cmap=q_cmap, vmin=-1, vmax=1, origin="lower")
        else:
            ax_q.imshow(q_data, cmap=q_cmap, vmin=0, vmax=1, origin="lower")
        ax_q.set_title(f"{q_name}", fontsize=9.5, fontweight="bold", color=q_col)
        ax_q.set_xticks([])
        ax_q.set_yticks([])

        # Inset 1D profile
        prof = q_data[size // 2, :]
        ax_ins = ax_q.inset_axes([0.05, 0.05, 0.9, 0.3])
        ax_ins.plot(prof, color="white" if q_cmap == "magma" else "cyan", linewidth=1.5)
        ax_ins.set_facecolor("#1A252FCC")
        ax_ins.set_ylim(-1.2 if q_cmap == "seismic" else -0.1, 1.2)
        ax_ins.set_xticks([])
        ax_ins.set_yticks([])
        ax_ins.set_title("1D Center Slice", color="white", fontsize=7, pad=1)

    out_path = output_dir / "gratings_04_gabor_wavelets_and_uncertainty.png"
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Generated: {out_path}")


# ---------------------------------------------------------------------------
# FIGURE 5: 3D Topographic Surfaces & Log-Magnitude Fourier Landscapes
# ---------------------------------------------------------------------------
def plot_figure_5_3d_surfaces_and_spectral_landscapes(output_dir: Path):
    """
    Figure 5: 3D Topographic Surfaces of Spatial Waveforms & 2D Fourier Spectra.
    Renders high-resolution 3D perspective surfaces to provide intuitive spatial depth:
    1. Infinite Pure Sinusoid vs Windowed Gabor Patch (3D delta spikes vs Gaussian bell hills)
    2. Hard Box Rectangular Aperture vs Smooth Gaussian Aperture (3D sinc ripple mesh vs smooth peak)
    3. Chirp Zone Plate 3D wave and circular plateau spectrum.
    """
    fig = plt.figure(figsize=(16, 12))
    
    fig.suptitle(
        "3D Topographic Landscapes: Spatial Waveforms & 2D Fourier Log-Spectra\n"
        "Visualizing Dirac Impulses, Sinc Sidelobe Valleys, and Gaussian Spectral Hills",
        fontsize=13.0,
        fontweight="bold",
        color=COLOR_TEXT_MAIN,
        y=0.97
    )

    size = 128
    x, y = make_grid(size, coord_range=1.0)
    
    # 3 Comparison cases
    cases = [
        ("Pure Sinusoid Grating",
         generate_sinusoid_grating(size=size, freq=5.0, theta_deg=0.0),
         "Infinite carrier wave -> Pair of Dirac impulse spikes"),
        
        ("Gabor Wavelet (Gaussian Windowed)",
         generate_gabor(size=size, freq=5.0, theta_deg=0.0, sigma=0.35)[0],
         "Gaussian envelope convolves impulses into smooth bell hills"),
        
        ("Rectangular Windowed Grating",
         generate_sinusoid_grating(size=size, freq=5.0, theta_deg=0.0) * generate_window("rectangular", size=size, box_half_width=0.4),
         "Sharp aperture edges create 2D Sinc ripples & leakage valleys")
    ]

    for idx, (title, img, desc) in enumerate(cases):
        _, log_mag, _ = compute_fourier_transform(img)

        # 3D Spatial Domain
        ax_sp_3d = fig.add_subplot(2, 3, idx + 1, projection="3d")
        surf_sp = ax_sp_3d.plot_surface(x, y, img, cmap="coolwarm", edgecolor="none", alpha=0.92, antialiased=True)
        ax_sp_3d.set_title(f"Spatial 3D: {title}", fontsize=9.5, fontweight="bold", color=COLOR_TEXT_MAIN, pad=10)
        ax_sp_3d.set_zlim(-1.5, 1.5)
        ax_sp_3d.view_init(elev=38, azim=-55)
        ax_sp_3d.set_xlabel("x", labelpad=-8, fontsize=8)
        ax_sp_3d.set_ylabel("y", labelpad=-8, fontsize=8)
        ax_sp_3d.set_zlabel("Intensity", labelpad=-8, fontsize=8)
        ax_sp_3d.tick_params(labelsize=7, pad=-2)

        # 3D Fourier Log-Magnitude Landscape
        ax_ft_3d = fig.add_subplot(2, 3, idx + 4, projection="3d")
        u_sub = np.linspace(-size//2, size//2, size)
        uu, vv = np.meshgrid(u_sub, u_sub)
        
        surf_ft = ax_ft_3d.plot_surface(uu, vv, log_mag, cmap="inferno", edgecolor="none", alpha=0.92, antialiased=True)
        ax_ft_3d.set_title(f"Fourier 3D: Log-Magnitude\n{desc}", fontsize=8.5, fontweight="bold", color=COLOR_HARMONICS, pad=10)
        ax_ft_3d.set_xlim(-40, 40)
        ax_ft_3d.set_ylim(-40, 40)
        ax_ft_3d.set_zlim(0, 1.1)
        ax_ft_3d.view_init(elev=42, azim=-60)
        ax_ft_3d.set_xlabel("u", labelpad=-8, fontsize=8)
        ax_ft_3d.set_ylabel("v", labelpad=-8, fontsize=8)
        ax_ft_3d.set_zlabel("Log |F|", labelpad=-8, fontsize=8)
        ax_ft_3d.tick_params(labelsize=7, pad=-2)

    plt.subplots_adjust(left=0.03, right=0.97, top=0.90, bottom=0.05, hspace=0.30, wspace=0.15)

    out_path = output_dir / "gratings_05_3d_surfaces_and_spectral_landscapes.png"
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Generated: {out_path}")


# ---------------------------------------------------------------------------
# Main Execution Pipeline
# ---------------------------------------------------------------------------
def main():
    print("=" * 75)
    print("Starting Gratings & Fourier Domain Publication Visualization Suite")
    print("=" * 75)

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/5] Generating Figure 1: Fundamental Sinusoidal Gratings...")
    plot_figure_1_fundamental_sinusoids(output_dir)

    print("\n[2/5] Generating Figure 2: Non-Sinusoidal Grating Families & Harmonics...")
    plot_figure_2_waveform_families(output_dir)

    print("\n[3/5] Generating Figure 3: Windowed Gratings & The 2D Convolution Theorem...")
    plot_figure_3_windowing_and_convolution(output_dir)

    print("\n[4/5] Generating Figure 4: Gabor Wavelets, Quadrature Pairs, & Uncertainty Principle...")
    plot_figure_4_gabor_wavelets_and_uncertainty(output_dir)

    print("\n[5/5] Generating Figure 5: 3D Topographic Surfaces & Spectral Landscapes...")
    plot_figure_5_3d_surfaces_and_spectral_landscapes(output_dir)

    print("\n" + "=" * 75)
    print("All 5 publication-quality figures successfully rendered to outputs/")
    print("=" * 75)


if __name__ == "__main__":
    main()
