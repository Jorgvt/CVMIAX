from __future__ import annotations

import sys
from pathlib import Path

# Add experiments/07_candles to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from data_generator import generate_synthetic_ohlc
from patterns import detect_all_patterns


def generate_broker_mockups(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. TradingView Dark Theme Mockup
    df1 = generate_synthetic_ohlc(n_candles=35, seed=101)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120, facecolor="#131722")
    ax.set_facecolor("#131722")

    # Grid
    ax.grid(True, linestyle="--", linewidth=0.6, color="#2A2E39", alpha=0.8)

    # Candles
    for i in range(len(df1)):
        o, h, l, c = df1.iloc[i][["Open", "High", "Low", "Close"]]
        color = "#089981" if c >= o else "#F23645"
        ax.plot([i, i], [l, h], color=color, linewidth=1.5)
        ax.add_patch(
            patches.Rectangle(
                (i - 0.35, min(o, c)), 0.7, max(abs(c - o), 0.05),
                facecolor=color, edgecolor=color, linewidth=0.8, zorder=3
            )
        )

    # Platform UI elements (Watermark, Ticker text, Time axes)
    ax.text(0.03, 0.92, "AAPL · 1D · NASDAQ", transform=ax.transAxes, color="#D1D4DC", fontsize=11, weight="bold")
    ax.text(0.03, 0.86, f"O: {df1.iloc[-1]['Open']:.2f} H: {df1.iloc[-1]['High']:.2f} L: {df1.iloc[-1]['Low']:.2f} C: {df1.iloc[-1]['Close']:.2f}", transform=ax.transAxes, color="#787B86", fontsize=9)
    ax.text(0.5, 0.5, "TradingView", transform=ax.transAxes, color="#2A2E39", fontsize=28, weight="bold", ha="center", va="center", alpha=0.3)

    ax.set_xlim(-0.5, len(df1) - 0.5)
    ax.set_ylim(df1["Low"].min() * 0.99, df1["High"].max() * 1.01)
    ax.tick_params(colors="#787B86", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#2A2E39")

    plt.tight_layout()
    fig.savefig(output_dir / "tradingview_dark_aapl.png", dpi=120)
    plt.close(fig)

    # 2. Bloomberg / Light Theme Broker Mockup
    df2 = generate_synthetic_ohlc(n_candles=40, seed=202)
    fig, (ax_main, ax_vol) = plt.subplots(2, 1, figsize=(8, 6), dpi=120, facecolor="#F8F9FA", gridspec_kw={"height_ratios": [3, 1]})
    ax_main.set_facecolor("#FFFFFF")
    ax_vol.set_facecolor("#FFFFFF")

    for i in range(len(df2)):
        o, h, l, c = df2.iloc[i][["Open", "High", "Low", "Close"]]
        color = "#26A69A" if c >= o else "#EF5350"
        ax_main.plot([i, i], [l, h], color=color, linewidth=1.4)
        ax_main.add_patch(
            patches.Rectangle(
                (i - 0.35, min(o, c)), 0.7, max(abs(c - o), 0.05),
                facecolor=color, edgecolor=color, linewidth=0.8, zorder=3
            )
        )
        # Volume
        v = df2.iloc[i]["Volume"]
        ax_vol.bar(i, v, color=color, alpha=0.6, width=0.7)

    ax_main.set_title("EURUSD Currency Cross (Daily)", fontsize=11, weight="bold", color="#1E222D", loc="left")
    ax_main.grid(True, linestyle=":", linewidth=0.5, color="#D0D0D0")
    ax_vol.grid(True, linestyle=":", linewidth=0.5, color="#E0E0E0")
    ax_vol.set_ylabel("Vol", fontsize=8, color="#666666")

    ax_main.set_xlim(-0.5, len(df2) - 0.5)
    ax_vol.set_xlim(-0.5, len(df2) - 0.5)

    plt.tight_layout()
    fig.savefig(output_dir / "broker_light_eurusd.png", dpi=120)
    plt.close(fig)

    print(f"Generated sample broker chart mockups in {output_dir}")


if __name__ == "__main__":
    target_dir = Path(__file__).parent / "real_charts"
    generate_broker_mockups(target_dir)
