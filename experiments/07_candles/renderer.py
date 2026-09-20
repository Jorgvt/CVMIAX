"""Candlestick chart renderer with pixel-perfect YOLO bounding box calculation.

Supports:
- Clean rendering for dataset generation (axes/ticks/spines stripped)
- Precise mathematical bounding box computation in normalized [0, 1] YOLO format
- Configurable visual styles (clean, dark, grid, volume, watermark) for domain shift benchmarking
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from PIL import Image

from patterns import PatternMatch


@dataclass
class BoundingBox:
    """Normalized YOLO bounding box (0.0 to 1.0)."""
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def to_yolo_str(self) -> str:
        return f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"


def render_candlestick_chart(
    df: pd.DataFrame,
    patterns: Optional[List[PatternMatch]] = None,
    image_size: int = 640,
    style: str = "clean",
    volume: Optional[np.ndarray] = None,
    dpi: int = 100,
    bbox_padding: float = 0.02,
) -> Tuple[np.ndarray, List[BoundingBox]]:
    """Render a candlestick chart to a numpy RGB image and compute normalized YOLO bounding boxes.

    Args:
        df: DataFrame with ['Open', 'High', 'Low', 'Close'] columns for N candles.
        patterns: List of PatternMatch objects within this window.
        image_size: Square output resolution in pixels (e.g. 640).
        style: Style preset ('clean', 'dark', 'grid', 'noisy', 'volume').
        volume: Optional volume array.
        dpi: Matplotlib DPI.
        bbox_padding: Margin around bounding boxes (fraction of price range).

    Returns:
        image_rgb: (H, W, 3) uint8 numpy array.
        bboxes: List of BoundingBox objects in normalized YOLO format.
    """
    n_candles = len(df)
    if n_candles == 0:
        raise ValueError("DataFrame must contain at least 1 candle.")

    opens = df["Open"].values
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values

    # Determine price range with padding for candle wicks
    price_min = np.min(lows)
    price_max = np.max(highs)
    p_range = max(price_max - price_min, 1e-6)
    y_margin = 0.05 * p_range
    y_min_axis = price_min - y_margin
    y_max_axis = price_max + y_margin
    total_y_span = y_max_axis - y_min_axis

    # Palette configuration
    if style in ["dark", "dark_grid"]:
        bg_color = "#131722"
        up_color = "#089981"
        down_color = "#F23645"
        grid_color = "#2A2E39"
    else:
        bg_color = "#FFFFFF"
        up_color = "#26A69A"
        down_color = "#EF5350"
        grid_color = "#E0E0E0"

    fig_size_inches = image_size / dpi
    fig = Figure(figsize=(fig_size_inches, fig_size_inches), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])  # Occupy full canvas exactly
    ax.set_facecolor(bg_color)
    fig.patch.set_facecolor(bg_color)

    # Plot grid if requested
    if style in ["grid", "dark_grid", "noisy"]:
        ax.grid(True, linestyle="--", linewidth=0.7, color=grid_color, alpha=0.7)
        ax.set_axisbelow(True)
    else:
        ax.set_axis_off()

    # Candle dimensions
    candle_width = 0.72
    wick_width = 1.8

    for i in range(n_candles):
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        is_up = c >= o
        color = up_color if is_up else down_color

        # Plot high-low wick
        ax.plot([i, i], [l, h], color=color, linewidth=wick_width, solid_capstyle="round")

        # Plot real body
        body_bottom = min(o, c)
        body_height = max(abs(c - o), p_range * 0.008)  # minimum visual height for doji
        rect = patches.Rectangle(
            (i - candle_width / 2, body_bottom),
            candle_width,
            body_height,
            facecolor=color,
            edgecolor=color,
            linewidth=1.0,
            zorder=3,
        )
        ax.add_patch(rect)

    # Optional volume bars at bottom
    if (style in ["volume", "noisy"] or volume is not None) and volume is not None:
        v_max = np.max(volume) if np.max(volume) > 0 else 1.0
        v_bottom = y_min_axis
        v_height_scale = 0.20 * total_y_span
        for i in range(n_candles):
            v_h = (volume[i] / v_max) * v_height_scale
            v_color = up_color if closes[i] >= opens[i] else down_color
            v_rect = patches.Rectangle(
                (i - candle_width / 2, v_bottom),
                candle_width,
                v_h,
                facecolor=v_color,
                alpha=0.35,
                linewidth=0,
                zorder=2,
            )
            ax.add_patch(v_rect)

    # Optional noise / watermark
    if style == "noisy":
        ax.text(
            0.5, 0.5, "TRADINGVIEW / BROKER SIMULATION",
            transform=ax.transAxes,
            fontsize=14, color="#888888", alpha=0.18,
            ha="center", va="center", rotation=25, weight="bold"
        )

    # Set exact axis limits
    ax.set_xlim(-0.5, n_candles - 0.5)
    ax.set_ylim(y_min_axis, y_max_axis)
    ax.axis("off")

    # Render canvas to RGB numpy buffer
    canvas.draw()
    rgba_buffer = canvas.buffer_rgba()
    image_rgb = np.asarray(rgba_buffer)[:, :, :3].copy()

    # Compute normalized bounding boxes for ground truth patterns
    bboxes: List[BoundingBox] = []
    if patterns:
        for p in patterns:
            # X dimension
            # x_left corresponds to start_idx - 0.5, x_right to end_idx + 0.5
            x_left = (p.start_idx - 0.5 - (-0.5)) / n_candles
            x_right = (p.end_idx + 0.5 - (-0.5)) / n_candles
            w = max(x_right - x_left, 1e-4)
            x_center = (x_left + x_right) / 2.0

            # Y dimension (Image Y=0 is TOP, Y=1 is BOTTOM)
            pad_price = bbox_padding * (p.price_max - p.price_min)
            p_top = min(p.price_max + pad_price, y_max_axis)
            p_bottom = max(p.price_min - pad_price, y_min_axis)

            y_top_norm = (y_max_axis - p_top) / total_y_span
            y_bottom_norm = (y_max_axis - p_bottom) / total_y_span
            h = max(y_bottom_norm - y_top_norm, 1e-4)
            y_center = (y_top_norm + y_bottom_norm) / 2.0

            # Clamp coordinates to [0, 1]
            x_center = float(np.clip(x_center, 0.0, 1.0))
            y_center = float(np.clip(y_center, 0.0, 1.0))
            w = float(np.clip(w, 0.005, 1.0))
            h = float(np.clip(h, 0.005, 1.0))

            bboxes.append(
                BoundingBox(
                    class_id=p.class_id,
                    x_center=x_center,
                    y_center=y_center,
                    width=w,
                    height=h,
                )
            )

    return image_rgb, bboxes
