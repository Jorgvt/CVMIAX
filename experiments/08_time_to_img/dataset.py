"""Dataset Generation and Causal Regime Labeling Pipeline for Time-to-Image.

Extracts sliding windows from multi-asset price series (SPY, QQQ, TLT, GLD, EURUSD)
and assigns strictly backward-looking regime labels without lookahead bias:
- Class 0: Low-Vol Bull (positive trend, subdued volatility)
- Class 1: Low-Vol Bear (steady downward drift, subdued volatility)
- Class 2: High-Vol Choppy (sideways / range-bound, elevated volatility)
- Class 3: Volatility Shock (extreme volatility spike / tail-risk event)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import tensorflow as tf

from encoders import batch_encode_windows

REGIME_NAMES: Dict[int, str] = {
    0: "Low-Vol Bull",
    1: "Low-Vol Bear",
    2: "High-Vol Choppy",
    3: "Volatility Shock",
}


def load_market_data(
    csv_path: Union[str, Path] = "experiments/08_time_to_img/data/market_data.csv",
) -> pd.DataFrame:
    """Load multi-asset historical price data."""
    path = Path(csv_path)
    if not path.exists():
        fallback_path = Path("experiments/08_time_to_img/data/sp500_historical.csv")
        if fallback_path.exists():
            path = fallback_path
        else:
            raise FileNotFoundError(f"Price data not found at {csv_path} or fallback.")

    df = pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

    return df


def extract_asset_windows(
    df_sub: pd.DataFrame,
    window_size: int = 128,
    step_size: int = 8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Extract sliding windows of returns, cumulative trajectories, vols, and returns for a sub-dataframe."""
    feature_cols = [c for c in df_sub.columns if c != "Date"]

    all_ret_windows = []
    all_cum_windows = []
    all_vols = []
    all_rets = []
    asset_sources = []

    for col in feature_cols:
        prices = df_sub[col].dropna().values.astype(float)
        if len(prices) < window_size + 1:
            continue
        log_ret = np.log(prices[1:] / prices[:-1])

        for s in range(0, len(log_ret) - window_size + 1, step_size):
            w_ret = log_ret[s : s + window_size]
            all_ret_windows.append(w_ret)
            all_cum_windows.append(np.cumsum(w_ret))
            all_vols.append(float(np.std(w_ret) * np.sqrt(252)))
            all_rets.append(float(np.sum(w_ret)))
            asset_sources.append(col)

    return (
        np.array(all_ret_windows, dtype=np.float32),
        np.array(all_cum_windows, dtype=np.float32),
        np.array(all_vols, dtype=np.float32),
        np.array(all_rets, dtype=np.float32),
        asset_sources,
    )


def compute_regime_labels(
    vols: np.ndarray,
    rets: np.ndarray,
    thresholds: Optional[Tuple[float, float, float]] = None,
    vol_shock_percentile: float = 75.0,
    vol_median_percentile: float = 45.0,
    return_threshold: float = 0.03,
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Assign 4-class backward-looking regime labels based on volatility and cumulative return."""
    if thresholds is None:
        vol_shock_th = float(np.percentile(vols, vol_shock_percentile))
        vol_med_th = float(np.percentile(vols, vol_median_percentile))
        ret_th = return_threshold
        thresholds = (vol_shock_th, vol_med_th, ret_th)
    else:
        vol_shock_th, vol_med_th, ret_th = thresholds

    labels = np.zeros(len(vols), dtype=np.int32)
    for i in range(len(vols)):
        v = vols[i]
        r = rets[i]
        if v >= vol_shock_th:
            labels[i] = 3  # Volatility Shock
        elif v < vol_med_th and r > ret_th:
            labels[i] = 0  # Low-Vol Bull
        elif v < vol_med_th and r < -ret_th:
            labels[i] = 1  # Low-Vol Bear
        else:
            labels[i] = 2  # High-Vol Choppy

    return labels, thresholds


def create_tf_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 16,
    is_training: bool = True,
    seed: int = 42,
) -> tf.data.Dataset:
    """Build tf.data.Dataset pipeline."""
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if is_training:
        dataset = dataset.shuffle(buffer_size=len(images), seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def get_encoded_datasets(
    method: str = "fusion",
    image_size: int = 128,
    cmap: str = "viridis",
    csv_path: Union[str, Path] = "experiments/08_time_to_img/data/market_data.csv",
    batch_size: int = 32,
    window_size: int = 128,
    step_size: int = 8,
    train_end_date: str = "2019-01-01",
    val_end_date: str = "2021-06-01",
) -> Dict[str, Union[tf.data.Dataset, np.ndarray, Tuple, List]]:
    """Load data, split chronologically by calendar date, assign regimes, encode, and return datasets."""
    df = load_market_data(csv_path)

    train_mask = df["Date"] < train_end_date
    val_mask = (df["Date"] >= train_end_date) & (df["Date"] < val_end_date)
    test_mask = df["Date"] >= val_end_date

    w_ret_tr, w_cum_tr, v_tr, r_tr, a_tr = extract_asset_windows(df[train_mask], window_size, step_size)
    w_ret_va, w_cum_va, v_va, r_va, a_va = extract_asset_windows(df[val_mask], window_size, step_size)
    w_ret_te, w_cum_te, v_te, r_te, a_te = extract_asset_windows(df[test_mask], window_size, step_size)

    y_tr, thresholds = compute_regime_labels(v_tr, r_tr)
    y_va, _ = compute_regime_labels(v_va, r_va, thresholds=thresholds)
    y_te, _ = compute_regime_labels(v_te, r_te, thresholds=thresholds)

    # Encode to images using global scale factor computed on training data
    x_tr, vmax = batch_encode_windows(w_ret_tr, w_cum_tr, method=method, image_size=image_size, cmap=cmap)
    x_va, _ = batch_encode_windows(w_ret_va, w_cum_va, method=method, image_size=image_size, cmap=cmap, vmax=vmax)
    x_te, _ = batch_encode_windows(w_ret_te, w_cum_te, method=method, image_size=image_size, cmap=cmap, vmax=vmax)

    ds_train = create_tf_dataset(x_tr, y_tr, batch_size=batch_size, is_training=True)
    ds_val = create_tf_dataset(x_va, y_va, batch_size=batch_size, is_training=False)
    ds_test = create_tf_dataset(x_te, y_te, batch_size=batch_size, is_training=False)

    return {
        "ds_train": ds_train,
        "ds_val": ds_val,
        "ds_test": ds_test,
        "x_train": x_tr,
        "y_train": y_tr,
        "x_val": x_va,
        "y_val": y_va,
        "x_test": x_te,
        "y_test": y_te,
        "windows_ret": w_ret_tr,
        "windows_cum": w_cum_tr,
        "labels": y_tr,
        "asset_sources": a_tr,
        "thresholds": thresholds,
    }
