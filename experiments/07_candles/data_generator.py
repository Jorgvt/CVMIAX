"""Dataset generation and OHLC data synthesis pipeline.

Supports:
- High-speed synthetic OHLC generation (Geometric Brownian Motion + jump diffusion + pattern injection)
- Real-world OHLC fetching via yfinance
- Automated dataset generation in YOLO format with train/val/test splits and data.yaml
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add experiments/07_candles to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from PIL import Image
import yaml

from patterns import (
    ID_TO_PATTERN,
    PATTERN_NAMES,
    PATTERN_TO_ID,
    PatternMatch,
    detect_all_patterns,
)
from renderer import BoundingBox, render_candlestick_chart


def generate_synthetic_ohlc(
    n_candles: int = 500,
    initial_price: float = 100.0,
    mu: float = 0.0002,
    sigma: float = 0.015,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Generate realistic OHLC series using Geometric Brownian Motion with intra-candle price dynamics."""
    if seed is not None:
        np.random.seed(seed)

    # 4 sub-steps per candle to create Open, High, Low, Close
    dt = 1.0 / 4.0
    total_steps = n_candles * 4
    shocks = np.random.normal(loc=(mu - 0.5 * sigma**2) * dt, scale=sigma * np.sqrt(dt), size=total_steps)

    log_prices = np.log(initial_price) + np.cumsum(shocks)
    prices = np.exp(log_prices).reshape((n_candles, 4))

    opens = prices[:, 0]
    closes = prices[:, 3]
    # High is max of sub-steps plus positive wick noise
    highs = np.max(prices, axis=1) * (1.0 + np.abs(np.random.normal(0, sigma * 0.4, size=n_candles)))
    highs = np.maximum(highs, np.maximum(opens, closes))
    # Low is min of sub-steps minus negative wick noise
    lows = np.min(prices, axis=1) * (1.0 - np.abs(np.random.normal(0, sigma * 0.4, size=n_candles)))
    lows = np.minimum(lows, np.minimum(opens, closes))

    volumes = np.random.lognormal(mean=10.0, sigma=0.8, size=n_candles)

    df = pd.DataFrame({
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "Volume": volumes,
    })
    return df


def fetch_real_ohlc(
    ticker: str = "SPY",
    period: str = "2y",
    interval: str = "1d",
) -> pd.DataFrame:
    """Fetch real historical market data via yfinance."""
    import yfinance as yf

    data = yf.download(ticker, period=period, interval=interval, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        # Flatten MultiIndex columns if yfinance returns multi-level
        data.columns = [col[0] for col in data.columns]

    df = data[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
    return df


def build_candlestick_dataset(
    output_dir: str | Path,
    num_images: int = 500,
    window_size: int = 30,
    image_size: int = 640,
    train_ratio: float = 0.70,
    val_ratio: float = 0.20,
    test_ratio: float = 0.10,
    include_yfinance: bool = True,
    seed: int = 42,
) -> Dict[str, int]:
    """Generate and write a complete YOLO-formatted candlestick dataset to disk.

    Creates:
        output_dir/
            ├── data.yaml
            ├── images/
            │   ├── train/
            │   ├── val/
            │   └── test/
            └── labels/
                ├── train/
                ├── val/
                └── test/

    Returns summary stats dict with image counts per split and per class.
    """
    output_path = Path(output_dir).resolve()
    for split in ["train", "val", "test"]:
        (output_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_path / "labels" / split).mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)
    stats: Dict[str, int] = {name: 0 for name in PATTERN_NAMES}
    stats["total_images"] = 0
    stats["train_images"] = 0
    stats["val_images"] = 0
    stats["test_images"] = 0

    # Collect source series
    ohlc_sources: List[pd.DataFrame] = []
    # 1. Real series if requested
    if include_yfinance:
        for ticker in ["SPY", "AAPL", "MSFT", "QQQ"]:
            try:
                df_real = fetch_real_ohlc(ticker, period="2y", interval="1d")
                if len(df_real) > window_size:
                    ohlc_sources.append(df_real)
            except Exception:
                pass

    # 2. Synthetic series
    for s in range(20):
        df_syn = generate_synthetic_ohlc(n_candles=600, seed=seed + s * 7)
        ohlc_sources.append(df_syn)

    # Sample windows containing patterns
    candidate_windows: List[Tuple[pd.DataFrame, List[PatternMatch], np.ndarray]] = []
    for df_source in ohlc_sources:
        patterns_all = detect_all_patterns(df_source)
        if not patterns_all:
            continue

        for p in patterns_all:
            # Center the window roughly around the pattern with random jitter
            center = (p.start_idx + p.end_idx) // 2
            half_win = window_size // 2
            jitter = np.random.randint(-5, 6)
            start = max(0, center - half_win + jitter)
            end = start + window_size

            if end <= len(df_source):
                sub_df = df_source.iloc[start:end].reset_index(drop=True)
                vol = sub_df["Volume"].values if "Volume" in sub_df.columns else None

                # Re-detect patterns within this exact sub-window to get local indices
                sub_patterns = detect_all_patterns(sub_df)
                if sub_patterns:
                    candidate_windows.append((sub_df, sub_patterns, vol))

    # Shuffle candidates
    np.random.shuffle(candidate_windows)

    # Subsample to target count
    selected_windows = candidate_windows[:num_images]

    # Split into train, val, test
    n_total = len(selected_windows)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    splits = (
        [("train", w) for w in selected_windows[:n_train]] +
        [("val", w) for w in selected_windows[n_train:n_train + n_val]] +
        [("test", w) for w in selected_windows[n_train + n_val:]]
    )

    for idx, (split, (sub_df, sub_patterns, vol)) in enumerate(splits):
        img_filename = f"candle_{idx:05d}.png"
        lbl_filename = f"candle_{idx:05d}.txt"

        img_rgb, bboxes = render_candlestick_chart(
            df=sub_df,
            patterns=sub_patterns,
            image_size=image_size,
            style="clean",
            volume=vol,
        )

        # Save image
        img_path = output_path / "images" / split / img_filename
        Image.fromarray(img_rgb).save(img_path)

        # Save YOLO labels
        lbl_path = output_path / "labels" / split / lbl_filename
        with open(lbl_path, "w") as f:
            for bbox in bboxes:
                f.write(bbox.to_yolo_str() + "\n")
                stats[ID_TO_PATTERN[bbox.class_id]] += 1

        stats[f"{split}_images"] += 1
        stats["total_images"] += 1

    # Write data.yaml for Ultralytics YOLO
    data_yaml_content = {
        "path": str(output_path),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(PATTERN_NAMES)},
    }

    with open(output_path / "data.yaml", "w") as f:
        yaml.dump(data_yaml_content, f, sort_keys=False)

    return stats
