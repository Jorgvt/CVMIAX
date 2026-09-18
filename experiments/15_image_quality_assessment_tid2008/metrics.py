"""Image Quality Assessment (IQA) classical and statistical evaluation metrics.

Provides:
- Classical FR-IQA metrics: PSNR, SSIM
- Benchmark correlation metrics: SROCC, PLCC, KROCC, RMSE, MAE
- ITU-R / VQEG 4-parameter non-linear logistic regression mapping
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
from scipy import optimize, stats
import tensorflow as tf


def compute_psnr_single(
    ref: np.ndarray,
    dist: np.ndarray,
    max_val: float = 1.0,
) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR) between reference and distorted image.

    PSNR = 10 * log10(max_val^2 / MSE)
    """
    mse = np.mean((ref.astype(np.float64) - dist.astype(np.float64)) ** 2)
    if mse == 0:
        return 100.0  # Identical images
    return float(10.0 * np.log10((max_val ** 2) / mse))


def compute_psnr_batch(
    refs: np.ndarray,
    dists: np.ndarray,
    max_val: float = 1.0,
) -> np.ndarray:
    """Compute PSNR for a batch of images (N, H, W, C)."""
    n = len(refs)
    psnr_vals = np.zeros((n,), dtype=np.float64)
    for i in range(n):
        psnr_vals[i] = compute_psnr_single(refs[i], dists[i], max_val=max_val)
    return psnr_vals


def compute_ssim_single(
    ref: np.ndarray,
    dist: np.ndarray,
    max_val: float = 1.0,
) -> float:
    """Compute Structural Similarity Index (SSIM) using TensorFlow's tf.image.ssim."""
    ref_tensor = tf.convert_to_tensor(ref[np.newaxis, ...], dtype=tf.float32)
    dist_tensor = tf.convert_to_tensor(dist[np.newaxis, ...], dtype=tf.float32)
    ssim_val = tf.image.ssim(ref_tensor, dist_tensor, max_val=max_val)
    return float(ssim_val.numpy()[0])


def compute_ssim_batch(
    refs: np.ndarray,
    dists: np.ndarray,
    max_val: float = 1.0,
) -> np.ndarray:
    """Compute SSIM for a batch of images (N, H, W, C)."""
    refs_tensor = tf.convert_to_tensor(refs, dtype=tf.float32)
    dists_tensor = tf.convert_to_tensor(dists, dtype=tf.float32)
    ssim_vals = tf.image.ssim(refs_tensor, dists_tensor, max_val=max_val)
    return ssim_vals.numpy().astype(np.float64)


def logistic_4param(
    x: np.ndarray,
    beta1: float,
    beta2: float,
    beta3: float,
    beta4: float,
) -> np.ndarray:
    """4-parameter monotonic logistic function recommended by ITU-R/VQEG.

    f(x) = (beta1 - beta2) / (1 + exp((x - beta3) / |beta4|)) + beta2
    """
    return (beta1 - beta2) / (1.0 + np.exp((x - beta3) / (np.abs(beta4) + 1e-7))) + beta2


def fit_logistic_mapping(
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> np.ndarray:
    """Fit a 4-parameter logistic mapping to align predicted metrics with subjective MOS.

    Standard practice in IQA benchmarks to remove monotonic non-linearities before PLCC/RMSE.
    If the initial correlation is positive, fits monotonic logistic mapping; falls back
    to linear calibration if fitting is ill-conditioned.
    """
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)

    # Initial parameter estimates
    min_t, max_t = float(np.min(y_true)), float(np.max(y_true))
    mean_p, std_p = float(np.mean(y_pred)), float(np.std(y_pred)) + 1e-4

    beta0 = [max_t, min_t, mean_p, std_p]
    try:
        popt, _ = optimize.curve_fit(
            logistic_4param,
            y_pred,
            y_true,
            p0=beta0,
            bounds=([-2.0, -2.0, -np.inf, 1e-4], [12.0, 12.0, np.inf, np.inf]),
            maxfev=5000,
        )
        mapped_pred = logistic_4param(y_pred, *popt)
        # Verify that mapped predictions are valid and correlated
        if np.all(np.isfinite(mapped_pred)) and np.std(mapped_pred) > 1e-3:
            # Ensure mapped predictions retain positive rank correlation direction
            orig_srocc, _ = stats.spearmanr(y_true, y_pred)
            mapped_plcc, _ = stats.pearsonr(y_true, mapped_pred)
            if np.sign(mapped_plcc) == np.sign(orig_srocc):
                return mapped_pred
    except Exception:
        pass

    # Robust fallback: linear regression calibration y_mapped = a * y_pred + b
    try:
        slope, intercept, _, _, _ = stats.linregress(y_pred, y_true)
        return slope * y_pred + intercept
    except Exception:
        return y_pred



def compute_srocc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Spearman Rank-Order Correlation Coefficient (SROCC).

    Measures the monotonic relationship between predicted scores and MOS ground truth.
    In IQA, SROCC is invariant to monotonic transformations and evaluates ranking ability.
    """
    rho, _ = stats.spearmanr(y_true, y_pred)
    return float(rho) if not np.isnan(rho) else 0.0


def compute_krocc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Kendall Rank-Order Correlation Coefficient (KROCC).

    Measures relative pairwise order concordance.
    """
    tau, _ = stats.kendalltau(y_true, y_pred)
    return float(tau) if not np.isnan(tau) else 0.0


def compute_plcc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    apply_logistic_fit: bool = True,
) -> float:
    """Pearson Linear Correlation Coefficient (PLCC).

    Measures the linear prediction accuracy against MOS ground truth.
    """
    if apply_logistic_fit:
        pred = fit_logistic_mapping(y_pred, y_true)
    else:
        pred = y_pred

    r, _ = stats.pearsonr(y_true, pred)
    return float(r) if not np.isnan(r) else 0.0


def compute_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    apply_logistic_fit: bool = True,
) -> float:
    """Root Mean Squared Error (RMSE) between MOS and mapped predictions."""
    if apply_logistic_fit:
        pred = fit_logistic_mapping(y_pred, y_true)
    else:
        pred = y_pred

    return float(np.sqrt(np.mean((y_true - pred) ** 2)))


def compute_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    apply_logistic_fit: bool = True,
) -> float:
    """Mean Absolute Error (MAE) between MOS and mapped predictions."""
    if apply_logistic_fit:
        pred = fit_logistic_mapping(y_pred, y_true)
    else:
        pred = y_pred

    return float(np.mean(np.abs(y_true - pred)))


def evaluate_iqa_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    apply_logistic_fit: bool = True,
) -> Dict[str, Union[str, float]]:
    """Compute full IQA diagnostic metric suite.

    Returns dict with SROCC, KROCC, PLCC, RMSE, and MAE.
    """
    srocc = compute_srocc(y_true, y_pred)
    krocc = compute_krocc(y_true, y_pred)
    plcc = compute_plcc(y_true, y_pred, apply_logistic_fit=apply_logistic_fit)
    rmse = compute_rmse(y_true, y_pred, apply_logistic_fit=apply_logistic_fit)
    mae = compute_mae(y_true, y_pred, apply_logistic_fit=apply_logistic_fit)

    return {
        "Model": model_name,
        "SROCC": srocc,
        "KROCC": krocc,
        "PLCC": plcc,
        "RMSE": rmse,
        "MAE": mae,
    }
