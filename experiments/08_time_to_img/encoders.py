"""Time-Series to Image Encoders for Financial Data.

Provides modular, mathematically sound transformation routines:
1. Gramian Angular Fields (GAF: GASF & GADF) via pyts
2. Continuous Wavelet Transform (CWT Scalograms) via PyWavelets (pywt)
3. Short-Time Fourier Transform (STFT Spectrograms) via scipy.signal
4. Multi-channel fusion (GASF Trend + GADF Dynamics + CWT Energy)
5. Continuous colormapping and robust global scaling.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple, Union
import numpy as np
import pywt
from pyts.image import GramianAngularField
from scipy import ndimage, signal
import matplotlib.pyplot as plt


def compute_gaf(
    data: np.ndarray,
    image_size: int = 128,
    method: Literal["summation", "difference"] = "summation",
    sample_range: Tuple[float, float] = (-1.0, 1.0),
) -> np.ndarray:
    """Compute Gramian Angular Field (GASF or GADF) for 1D time-series windows."""
    is_1d = data.ndim == 1
    x = data[np.newaxis, :] if is_1d else data

    gaf = GramianAngularField(
        image_size=image_size,
        method=method,
        sample_range=sample_range,
    )
    encoded = gaf.fit_transform(x)
    return encoded[0] if is_1d else encoded


def compute_cwt(
    data: np.ndarray,
    image_size: int = 128,
    wavelet: str = "cmor1.5-1.0",
    total_scales: int = 128,
    scale_spacing: Literal["linear", "geometric"] = "geometric",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Continuous Wavelet Transform (CWT) Scalogram and Cone of Influence."""
    length = len(data)
    if scale_spacing == "geometric":
        scales = np.geomspace(1.0, length / 2.0, num=total_scales)
    else:
        scales = np.linspace(1.0, length / 2.0, num=total_scales)

    coefficients, _ = pywt.cwt(data, scales, wavelet)
    power = np.abs(coefficients) ** 2

    # Cone of Influence (COI) line
    decay_constant = np.sqrt(2.0)
    coi_bounds = decay_constant * scales

    t = np.arange(length)
    coi_mask_raw = np.zeros((total_scales, length), dtype=bool)
    for i, bound in enumerate(coi_bounds):
        coi_mask_raw[i, :] = (t < bound) | (t > (length - 1 - bound))

    zoom_factors = (image_size / total_scales, image_size / length)
    scalogram = ndimage.zoom(power, zoom_factors, order=1)
    coi_mask = ndimage.zoom(coi_mask_raw.astype(float), zoom_factors, order=0) > 0.5

    return scalogram, coi_mask


def compute_stft(
    data: np.ndarray,
    image_size: int = 128,
    nperseg: int = 32,
    noverlap: int = 28,
    window: str = "hann",
) -> np.ndarray:
    """Compute Short-Time Fourier Transform (STFT) Log-Spectrogram."""
    length = len(data)
    nperseg = min(nperseg, length)
    if noverlap >= nperseg:
        noverlap = nperseg - 1

    _, _, zxx = signal.stft(
        data,
        nperseg=nperseg,
        noverlap=noverlap,
        window=window,
        boundary=None,
        padded=False,
    )
    power = np.abs(zxx) ** 2
    log_power = np.log1p(power * 1000.0)

    zoom_factors = (image_size / log_power.shape[0], image_size / log_power.shape[1])
    spectrogram = ndimage.zoom(log_power, zoom_factors, order=1)
    return spectrogram


def apply_colormap(
    matrix_2d: np.ndarray,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> np.ndarray:
    """Normalize a 2D matrix using global/local bounds and map through a matplotlib colormap."""
    min_val = vmin if vmin is not None else np.min(matrix_2d)
    max_val = vmax if vmax is not None else np.max(matrix_2d)

    if max_val - min_val > 1e-8:
        norm = np.clip((matrix_2d - min_val) / (max_val - min_val), 0.0, 1.0)
    else:
        norm = np.zeros_like(matrix_2d, dtype=np.float32)

    colormap = plt.get_cmap(cmap)
    rgba = colormap(norm)
    return rgba[:, :, :3].astype(np.float32)


def batch_encode_windows(
    windows_ret: np.ndarray,
    windows_cum: Optional[np.ndarray] = None,
    method: Literal["gaf", "cwt", "stft", "fusion"] = "cwt",
    image_size: int = 128,
    cmap: str = "viridis",
    vmax: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """Encode a batch of 1D financial windows into 3-channel RGB image representations.

    Args:
        windows_ret: 2D array of log returns (N, W).
        windows_cum: 2D array of cumulative returns (N, W). If None, calculated from returns.
        method: 'gaf', 'cwt', 'stft', or 'fusion'.
        image_size: Target square image dimension (image_size x image_size).
        cmap: Colormap for single-matrix representations.
        vmax: Precomputed global maximum scaling value (for consistent train/test scaling).

    Returns:
        images: 4D float32 array (N, image_size, image_size, 3).
        vmax_out: The global scaling value used.
    """
    n_samples = len(windows_ret)
    if windows_cum is None:
        windows_cum = np.cumsum(windows_ret, axis=1)

    images = np.zeros((n_samples, image_size, image_size, 3), dtype=np.float32)

    if method == "gaf":
        # Encode cumulative price trajectory with GASF
        gasf_2d = compute_gaf(windows_cum, image_size=image_size, method="summation")
        for i in range(n_samples):
            images[i] = apply_colormap(gasf_2d[i], cmap=cmap, vmin=-1.0, vmax=1.0)
        vmax_out = 1.0

    elif method == "cwt":
        raw_scalograms = []
        for i in range(n_samples):
            sc, _ = compute_cwt(windows_ret[i], image_size=image_size)
            raw_scalograms.append(np.log1p(sc * 1000.0))
        raw_arr = np.array(raw_scalograms, dtype=np.float32)

        vmax_out = vmax if vmax is not None else float(np.percentile(raw_arr, 99))
        for i in range(n_samples):
            images[i] = apply_colormap(raw_arr[i], cmap=cmap, vmin=0.0, vmax=vmax_out)

    elif method == "stft":
        raw_spectrograms = []
        for i in range(n_samples):
            sp = compute_stft(windows_ret[i], image_size=image_size)
            raw_spectrograms.append(sp)
        raw_arr = np.array(raw_spectrograms, dtype=np.float32)

        vmax_out = vmax if vmax is not None else float(np.percentile(raw_arr, 99))
        for i in range(n_samples):
            images[i] = apply_colormap(raw_arr[i], cmap=cmap, vmin=0.0, vmax=vmax_out)

    elif method == "fusion":
        # 3-channel composite:
        # Ch0: GASF (Cumulative price trend)
        # Ch1: GADF (Cumulative price difference dynamics)
        # Ch2: CWT Scalogram (Volatility energy)
        gasf_2d = compute_gaf(windows_cum, image_size=image_size, method="summation")
        gadf_2d = compute_gaf(windows_cum, image_size=image_size, method="difference")

        raw_scalograms = []
        for i in range(n_samples):
            sc, _ = compute_cwt(windows_ret[i], image_size=image_size)
            raw_scalograms.append(np.log1p(sc * 1000.0))
        raw_arr = np.array(raw_scalograms, dtype=np.float32)

        vmax_out = vmax if vmax is not None else float(np.percentile(raw_arr, 99))
        cwt_norm = np.clip(raw_arr / (vmax_out + 1e-8), 0.0, 1.0)
        gasf_norm = np.clip((gasf_2d + 1.0) / 2.0, 0.0, 1.0)
        gadf_norm = np.clip((gadf_2d + 1.0) / 2.0, 0.0, 1.0)

        for i in range(n_samples):
            images[i, :, :, 0] = gasf_norm[i]
            images[i, :, :, 1] = gadf_norm[i]
            images[i, :, :, 2] = cwt_norm[i]
    else:
        raise ValueError(f"Unknown encoding method: {method}")

    return images, vmax_out
