"""Candlestick pattern recognition rules.

Implements pure-Python deterministic pattern recognizers for 8 standard candlestick patterns,
with optional delegation to TA-Lib if the C-binding is installed.

Taxonomy (8 classes):
0: Doji
1: Hammer
2: Inverted Hammer
3: Morning Star
4: Evening Star
5: Bullish Engulfing
6: Bearish Engulfing
7: Shooting Star
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# Class taxonomy
PATTERN_NAMES: List[str] = [
    "Doji",
    "Hammer",
    "Inverted Hammer",
    "Morning Star",
    "Evening Star",
    "Bullish Engulfing",
    "Bearish Engulfing",
    "Shooting Star",
]

PATTERN_TO_ID: Dict[str, int] = {name: i for i, name in enumerate(PATTERN_NAMES)}
ID_TO_PATTERN: Dict[int, str] = {i: name for i, name in enumerate(PATTERN_NAMES)}


@dataclass
class PatternMatch:
    """Represents a detected pattern occurrence in OHLC data."""
    start_idx: int
    end_idx: int
    class_id: int
    class_name: str
    price_min: float
    price_max: float


def compute_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute standard candle anatomical metrics: body, shadows, trend."""
    res = df.copy()
    open_p = res["Open"].values
    high_p = res["High"].values
    low_p = res["Low"].values
    close_p = res["Close"].values

    body = np.abs(close_p - open_p)
    body_top = np.maximum(open_p, close_p)
    body_bottom = np.minimum(open_p, close_p)
    upper_shadow = high_p - body_top
    lower_shadow = body_bottom - low_p
    candle_range = np.maximum(high_p - low_p, 1e-6)

    is_bullish = close_p > open_p
    is_bearish = close_p < open_p

    # Simple moving average for trend estimation (5-period EMA/SMA)
    sma5 = pd.Series(close_p).rolling(5, min_periods=1).mean().values

    res["body"] = body
    res["body_top"] = body_top
    res["body_bottom"] = body_bottom
    res["upper_shadow"] = upper_shadow
    res["lower_shadow"] = lower_shadow
    res["range"] = candle_range
    res["is_bullish"] = is_bullish
    res["is_bearish"] = is_bearish
    res["sma5"] = sma5
    return res


def detect_doji(df: pd.DataFrame) -> List[PatternMatch]:
    """Doji: Very small body relative to total range (Body <= 0.05 * Range)."""
    matches = []
    for i in range(len(df)):
        c = df.iloc[i]
        if c["body"] <= 0.05 * c["range"] and c["range"] > 0:
            matches.append(
                PatternMatch(
                    start_idx=i,
                    end_idx=i,
                    class_id=PATTERN_TO_ID["Doji"],
                    class_name="Doji",
                    price_min=float(c["Low"]),
                    price_max=float(c["High"]),
                )
            )
    return matches


def detect_hammer(df: pd.DataFrame) -> List[PatternMatch]:
    """Hammer: Small body near top, lower shadow >= 2 * body, minimal upper shadow, in downtrend."""
    matches = []
    for i in range(2, len(df)):
        c = df.iloc[i]
        is_small_body = c["body"] <= 0.35 * c["range"] and c["body"] > 0
        has_long_lower_shadow = c["lower_shadow"] >= 2.0 * c["body"]
        has_small_upper_shadow = c["upper_shadow"] <= 0.25 * c["body"] or c["upper_shadow"] <= 0.10 * c["range"]
        prev_downtrend = df.iloc[i - 1]["Close"] <= df.iloc[i - 2]["Close"] or c["Close"] < c["sma5"]

        if is_small_body and has_long_lower_shadow and has_small_upper_shadow and prev_downtrend:
            matches.append(
                PatternMatch(
                    start_idx=i,
                    end_idx=i,
                    class_id=PATTERN_TO_ID["Hammer"],
                    class_name="Hammer",
                    price_min=float(c["Low"]),
                    price_max=float(c["High"]),
                )
            )
    return matches


def detect_inverted_hammer(df: pd.DataFrame) -> List[PatternMatch]:
    """Inverted Hammer: Small body near bottom, upper shadow >= 2 * body, minimal lower shadow, in downtrend."""
    matches = []
    for i in range(2, len(df)):
        c = df.iloc[i]
        is_small_body = c["body"] <= 0.35 * c["range"] and c["body"] > 0
        has_long_upper_shadow = c["upper_shadow"] >= 2.0 * c["body"]
        has_small_lower_shadow = c["lower_shadow"] <= 0.25 * c["body"] or c["lower_shadow"] <= 0.10 * c["range"]
        prev_downtrend = df.iloc[i - 1]["Close"] <= df.iloc[i - 2]["Close"] or c["Close"] < c["sma5"]

        if is_small_body and has_long_upper_shadow and has_small_lower_shadow and prev_downtrend:
            matches.append(
                PatternMatch(
                    start_idx=i,
                    end_idx=i,
                    class_id=PATTERN_TO_ID["Inverted Hammer"],
                    class_name="Inverted Hammer",
                    price_min=float(c["Low"]),
                    price_max=float(c["High"]),
                )
            )
    return matches


def detect_shooting_star(df: pd.DataFrame) -> List[PatternMatch]:
    """Shooting Star: Small body near bottom, upper shadow >= 2 * body, minimal lower shadow, in uptrend."""
    matches = []
    for i in range(2, len(df)):
        c = df.iloc[i]
        is_small_body = c["body"] <= 0.35 * c["range"] and c["body"] > 0
        has_long_upper_shadow = c["upper_shadow"] >= 2.0 * c["body"]
        has_small_lower_shadow = c["lower_shadow"] <= 0.25 * c["body"] or c["lower_shadow"] <= 0.10 * c["range"]
        prev_uptrend = df.iloc[i - 1]["Close"] >= df.iloc[i - 2]["Close"] or c["Close"] > c["sma5"]

        if is_small_body and has_long_upper_shadow and has_small_lower_shadow and prev_uptrend:
            matches.append(
                PatternMatch(
                    start_idx=i,
                    end_idx=i,
                    class_id=PATTERN_TO_ID["Shooting Star"],
                    class_name="Shooting Star",
                    price_min=float(c["Low"]),
                    price_max=float(c["High"]),
                )
            )
    return matches


def detect_bullish_engulfing(df: pd.DataFrame) -> List[PatternMatch]:
    """Bullish Engulfing: 2-candle pattern. Candle 1 bearish, Candle 2 bullish and engulfs Candle 1 body."""
    matches = []
    for i in range(1, len(df)):
        c1 = df.iloc[i - 1]
        c2 = df.iloc[i]

        is_c1_bearish = c1["is_bearish"] or (c1["body"] < 0.05 * c1["range"] and c1["Close"] <= c1["Open"])
        is_c2_bullish = c2["is_bullish"]
        is_c2_larger = c2["body"] > c1["body"]

        engulfs_body = (c2["Open"] <= c1["Close"] + 1e-4) and (c2["Close"] >= c1["Open"] - 1e-4)

        if is_c1_bearish and is_c2_bullish and is_c2_larger and engulfs_body:
            p_min = min(float(c1["Low"]), float(c2["Low"]))
            p_max = max(float(c1["High"]), float(c2["High"]))
            matches.append(
                PatternMatch(
                    start_idx=i - 1,
                    end_idx=i,
                    class_id=PATTERN_TO_ID["Bullish Engulfing"],
                    class_name="Bullish Engulfing",
                    price_min=p_min,
                    price_max=p_max,
                )
            )
    return matches


def detect_bearish_engulfing(df: pd.DataFrame) -> List[PatternMatch]:
    """Bearish Engulfing: 2-candle pattern. Candle 1 bullish, Candle 2 bearish and engulfs Candle 1 body."""
    matches = []
    for i in range(1, len(df)):
        c1 = df.iloc[i - 1]
        c2 = df.iloc[i]

        is_c1_bullish = c1["is_bullish"] or (c1["body"] < 0.05 * c1["range"] and c1["Close"] >= c1["Open"])
        is_c2_bearish = c2["is_bearish"]
        is_c2_larger = c2["body"] > c1["body"]

        engulfs_body = (c2["Open"] >= c1["Close"] - 1e-4) and (c2["Close"] <= c1["Open"] + 1e-4)

        if is_c1_bullish and is_c2_bearish and is_c2_larger and engulfs_body:
            p_min = min(float(c1["Low"]), float(c2["Low"]))
            p_max = max(float(c1["High"]), float(c2["High"]))
            matches.append(
                PatternMatch(
                    start_idx=i - 1,
                    end_idx=i,
                    class_id=PATTERN_TO_ID["Bearish Engulfing"],
                    class_name="Bearish Engulfing",
                    price_min=p_min,
                    price_max=p_max,
                )
            )
    return matches


def detect_morning_star(df: pd.DataFrame) -> List[PatternMatch]:
    """Morning Star: 3-candle bullish reversal. 1: Long bear, 2: Small body star gapping down, 3: Long bull into C1."""
    matches = []
    for i in range(2, len(df)):
        c1 = df.iloc[i - 2]
        c2 = df.iloc[i - 1]
        c3 = df.iloc[i]

        c1_bearish = c1["is_bearish"] and c1["body"] >= 0.35 * c1["range"]
        c2_small = c2["body"] <= 0.45 * c1["body"]
        c2_gap = c2["body_top"] <= c1["body_bottom"] + 0.3 * c1["body"]
        c3_bullish = c3["is_bullish"] and c3["body"] >= 0.35 * c3["range"]
        c3_penetrates = c3["Close"] >= c1["Close"] + 0.35 * c1["body"]

        if c1_bearish and c2_small and c2_gap and c3_bullish and c3_penetrates:
            p_min = min(float(c1["Low"]), float(c2["Low"]), float(c3["Low"]))
            p_max = max(float(c1["High"]), float(c2["High"]), float(c3["High"]))
            matches.append(
                PatternMatch(
                    start_idx=i - 2,
                    end_idx=i,
                    class_id=PATTERN_TO_ID["Morning Star"],
                    class_name="Morning Star",
                    price_min=p_min,
                    price_max=p_max,
                )
            )
    return matches


def detect_evening_star(df: pd.DataFrame) -> List[PatternMatch]:
    """Evening Star: 3-candle bearish reversal. 1: Long bull, 2: Small body star gapping up, 3: Long bear into C1."""
    matches = []
    for i in range(2, len(df)):
        c1 = df.iloc[i - 2]
        c2 = df.iloc[i - 1]
        c3 = df.iloc[i]

        c1_bullish = c1["is_bullish"] and c1["body"] >= 0.35 * c1["range"]
        c2_small = c2["body"] <= 0.45 * c1["body"]
        c2_gap = c2["body_bottom"] >= c1["body_top"] - 0.3 * c1["body"]
        c3_bearish = c3["is_bearish"] and c3["body"] >= 0.35 * c3["range"]
        c3_penetrates = c3["Close"] <= c1["Close"] - 0.35 * c1["body"]

        if c1_bullish and c2_small and c2_gap and c3_bearish and c3_penetrates:
            p_min = min(float(c1["Low"]), float(c2["Low"]), float(c3["Low"]))
            p_max = max(float(c1["High"]), float(c2["High"]), float(c3["High"]))
            matches.append(
                PatternMatch(
                    start_idx=i - 2,
                    end_idx=i,
                    class_id=PATTERN_TO_ID["Evening Star"],
                    class_name="Evening Star",
                    price_min=p_min,
                    price_max=p_max,
                )
            )
    return matches


def filter_overlapping_patterns(matches: List[PatternMatch], iou_thresh: float = 0.4) -> List[PatternMatch]:
    """Filter out heavily overlapping bounding boxes, prioritizing multi-candle / higher-order patterns."""
    if not matches:
        return []

    # Sort by pattern span length descending (multi-candle first) then start_idx
    sorted_matches = sorted(
        matches,
        key=lambda m: (m.end_idx - m.start_idx, -m.class_id),
        reverse=True,
    )

    kept: List[PatternMatch] = []
    for candidate in sorted_matches:
        conflict = False
        for existing in kept:
            # Check 1D index overlap
            overlap_start = max(candidate.start_idx, existing.start_idx)
            overlap_end = min(candidate.end_idx, existing.end_idx)
            if overlap_start <= overlap_end:
                overlap_len = overlap_end - overlap_start + 1
                min_len = min(candidate.end_idx - candidate.start_idx + 1, existing.end_idx - existing.start_idx + 1)
                if overlap_len / min_len >= iou_thresh:
                    conflict = True
                    break
        if not conflict:
            kept.append(candidate)

    kept.sort(key=lambda m: (m.start_idx, m.end_idx))
    return kept


def detect_all_patterns(df: pd.DataFrame, filter_overlap: bool = True) -> List[PatternMatch]:
    """Detect all 8 candlestick patterns in the given OHLC dataframe."""
    featured_df = compute_candle_features(df)
    matches: List[PatternMatch] = []

    matches.extend(detect_doji(featured_df))
    matches.extend(detect_hammer(featured_df))
    matches.extend(detect_inverted_hammer(featured_df))
    matches.extend(detect_shooting_star(featured_df))
    matches.extend(detect_bullish_engulfing(featured_df))
    matches.extend(detect_bearish_engulfing(featured_df))
    matches.extend(detect_morning_star(featured_df))
    matches.extend(detect_evening_star(featured_df))

    if filter_overlap:
        matches = filter_overlapping_patterns(matches)

    matches.sort(key=lambda m: (m.start_idx, m.end_idx))
    return matches
