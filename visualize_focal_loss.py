"""
visualize_focal_loss.py

A standalone, publication-quality visualization suite for Focal Loss vs. Traditional
Cross-Entropy (CE), demonstrating the mathematical mechanism and the impact of the
focusing parameter gamma (γ) and balancing factor alpha (α).

Generates 4 pedagogical figures:
1. `focal_loss_01_curves.png`: Standard and log-scale loss curves comparing CE (γ=0) to Focal Loss (γ > 0).
2. `focal_loss_02_modulating_factor_and_gradients.png`: Modulating factor (1-pt)^γ and gradient dynamics vs. pt.
3. `focal_loss_03_class_imbalance_simulation.png`: Dense object detection simulation (100k background vs 10 foreground).
4. `focal_loss_04_alpha_gamma_landscape.png`: 2D parameter sweep and loss attenuation heatmaps.

Usage:
    uv run python visualize_focal_loss.py
"""

from pathlib import Path
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# ---------------------------------------------------------------------------
# Visual Styling Constants
# ---------------------------------------------------------------------------
COLOR_CE = "#2C3E50"        # Dark Slate Navy (Standard Cross Entropy, γ=0)
COLOR_GAMMA_05 = "#3498DB"  # Bright Blue (γ=0.5)
COLOR_GAMMA_10 = "#1ABC9C"  # Turquoise (γ=1.0)
COLOR_GAMMA_20 = "#E67E22"  # Vivid Orange (γ=2.0, RetinaNet Default)
COLOR_GAMMA_30 = "#E74C3C"  # Crimson / Coral (γ=3.0)
COLOR_GAMMA_50 = "#9B59B6"  # Royal Purple (γ=5.0)

GAMMA_PALETTE = {
    0.0: COLOR_CE,
    0.5: COLOR_GAMMA_05,
    1.0: COLOR_GAMMA_10,
    2.0: COLOR_GAMMA_20,
    3.0: COLOR_GAMMA_30,
    5.0: COLOR_GAMMA_50,
}

FONT_FAMILY = "sans-serif"

plt.rcParams.update({
    "font.family": FONT_FAMILY,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.titlesize": 16,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# ---------------------------------------------------------------------------
# Core Focal Loss Formulas
# ---------------------------------------------------------------------------
def focal_loss(pt: np.ndarray, gamma: float = 2.0, alpha: float = 1.0, eps: float = 1e-12) -> np.ndarray:
    """
    Computes Focal Loss: FL(pt) = -alpha * (1 - pt)^gamma * log(pt).
    When gamma=0 and alpha=1, this is standard Cross Entropy: CE(pt) = -log(pt).
    """
    pt_clipped = np.clip(pt, eps, 1.0 - eps)
    modulating_factor = (1.0 - pt_clipped) ** gamma
    ce = -np.log(pt_clipped)
    return alpha * modulating_factor * ce


def modulating_factor(pt: np.ndarray, gamma: float) -> np.ndarray:
    """Computes the modulating term: (1 - pt)^gamma."""
    return (1.0 - pt) ** gamma


def focal_loss_gradient_wrt_logit(pt: np.ndarray, gamma: float = 2.0, y: int = 1) -> np.ndarray:
    """
    Computes magnitude |dFL/dz| with respect to the pre-sigmoid logit z.
    For y=1 (pt = sigmoid(z)):
      FL = -(1 - pt)^gamma * log(pt)
      dFL/dz = (1 - pt)^gamma * [ gamma * pt * log(pt) + (pt - 1) ]
    For standard CE (gamma=0):
      dCE/dz = pt - 1  (so magnitude is 1 - pt).
    """
    eps = 1e-12
    pt_c = np.clip(pt, eps, 1.0 - eps)
    if gamma == 0:
        return 1.0 - pt_c
    term1 = (1.0 - pt_c) ** gamma
    term2 = gamma * pt_c * np.log(pt_c) + (pt_c - 1.0)
    return np.abs(term1 * term2)


# ---------------------------------------------------------------------------
# Figure 1: Classic Focal Loss vs Cross Entropy Curves
# ---------------------------------------------------------------------------
def plot_focal_loss_curves(output_dir: Path) -> Path:
    """
    Plots the classic Focal Loss comparison across multiple gamma values
    on both linear and logarithmic scales.
    """
    pt = np.linspace(0.001, 1.0, 1000)
    gammas = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))

    # --- Left: Linear Scale ---
    for g in gammas:
        loss = focal_loss(pt, gamma=g)
        color = GAMMA_PALETTE[g]
        label = rf"$\gamma = {g:g}$ (Cross Entropy)" if g == 0 else rf"$\gamma = {g:g}$"
        lw = 3.0 if g in [0.0, 2.0] else 1.8
        ls = "-" if g in [0.0, 2.0] else "--"
        ax1.plot(pt, loss, color=color, linewidth=lw, linestyle=ls, label=label)

    # Highlight RetinaNet reference point
    pt_sample = 0.9
    ce_val = focal_loss(np.array([pt_sample]), gamma=0.0)[0]
    fl_val = focal_loss(np.array([pt_sample]), gamma=2.0)[0]
    ax1.scatter([pt_sample, pt_sample], [ce_val, fl_val], color=[COLOR_CE, COLOR_GAMMA_20], s=60, zorder=5)
    ax1.annotate(
        f"At $p_t = 0.9$ (Easy sample):\nCE loss = {ce_val:.3f}\nFL ($\gamma=2$) = {fl_val:.5f} ({ce_val/fl_val:.0f}x smaller!)",
        xy=(pt_sample, fl_val),
        xytext=(0.48, 1.4),
        arrowprops=dict(arrowstyle="->", color=COLOR_GAMMA_20, lw=1.5, connectionstyle="arc3,rad=-0.2"),
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#FEF9E7", edgecolor=COLOR_GAMMA_20, lw=1.2),
    )

    ax1.axvspan(0.0, 0.5, color="#FDEDEC", alpha=0.4, label="Hard / Misclassified ($p_t < 0.5$)")
    ax1.axvspan(0.5, 1.0, color="#E8F8F5", alpha=0.4, label=r"Easy / Well-Classified ($p_t \geq 0.5$)")

    ax1.set_xlim(0, 1.0)
    ax1.set_ylim(0, 4.5)
    ax1.set_xlabel(r"Ground Truth Class Probability $p_t$", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Loss Value", fontsize=12, fontweight="bold")
    ax1.set_title("Linear Scale: Down-weighting Easy Examples", fontsize=13, fontweight="bold", pad=12)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", framealpha=0.95, fontsize=10)

    # --- Right: Logarithmic Scale ---
    for g in gammas:
        loss = focal_loss(pt, gamma=g)
        color = GAMMA_PALETTE[g]
        label = rf"$\gamma = {g:g}$ (CE)" if g == 0 else rf"$\gamma = {g:g}$"
        lw = 3.0 if g in [0.0, 2.0] else 1.8
        ls = "-" if g in [0.0, 2.0] else "--"
        ax2.plot(pt, loss, color=color, linewidth=lw, linestyle=ls, label=label)

    ax2.set_yscale("log")
    ax2.set_xlim(0, 1.0)
    ax2.set_ylim(1e-6, 1e1)
    ax2.set_xlabel(r"Ground Truth Class Probability $p_t$", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Loss Value (Log Scale)", fontsize=12, fontweight="bold")
    ax2.set_title("Logarithmic Scale: Orders-of-Magnitude Suppression", fontsize=13, fontweight="bold", pad=12)
    ax2.grid(True, which="both", linestyle=":", alpha=0.5)

    # Annotate extreme suppression at pt=0.99
    pt_extreme = 0.99
    ce_ext = focal_loss(np.array([pt_extreme]), gamma=0.0)[0]
    fl_ext = focal_loss(np.array([pt_extreme]), gamma=2.0)[0]
    ratio = ce_ext / fl_ext
    ax2.annotate(
        f"At $p_t = 0.99$:\nSuppression is $\mathbf{{{ratio:,.0f}\\times}}$ with $\gamma=2$",
        xy=(pt_extreme, fl_ext),
        xytext=(0.52, 1e-4),
        arrowprops=dict(arrowstyle="->", color=COLOR_GAMMA_20, lw=1.5),
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#FDEDEC", edgecolor="#E74C3C", lw=1.2),
    )
    ax2.legend(loc="lower left", framealpha=0.95, fontsize=10)

    plt.suptitle(
        r"$\mathbf{Focal\ Loss\ vs.\ Cross\ Entropy:}\ \mathrm{FL}(p_t) = -(1 - p_t)^\gamma \log(p_t)$",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_file = output_dir / "focal_loss_01_curves.png"
    plt.savefig(out_file)
    plt.close()
    return out_file


# ---------------------------------------------------------------------------
# Figure 2: Modulating Factor & Gradient Dynamics
# ---------------------------------------------------------------------------
def plot_modulating_factor_and_gradients(output_dir: Path) -> Path:
    """
    Visualizes the modulating factor (1-pt)^γ and the backpropagated gradient
    magnitude |dFL/dz| with respect to logits across probability spectrum.
    """
    pt = np.linspace(0.0001, 0.9999, 1000)
    gammas = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.2))

    # --- Panel 1: Modulating Factor (1 - pt)^γ ---
    for g in gammas:
        mf = modulating_factor(pt, gamma=g)
        color = GAMMA_PALETTE[g]
        lw = 3.0 if g in [0.0, 2.0] else 1.8
        label = rf"$\gamma = {g:g}$ (Constant 1.0)" if g == 0 else rf"$\gamma = {g:g}$"
        ax1.plot(pt, mf, color=color, lw=lw, label=label)

    ax1.set_xlim(0, 1.0)
    ax1.set_ylim(-0.02, 1.05)
    ax1.set_xlabel(r"Ground Truth Probability $p_t$", fontsize=12, fontweight="bold")
    ax1.set_ylabel(r"Modulating Weight $(1 - p_t)^\gamma$", fontsize=12, fontweight="bold")
    ax1.set_title("Modulating Factor: Dynamic Example Weighting", fontsize=13, fontweight="bold", pad=12)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Shading regions
    ax1.axvspan(0.7, 1.0, color="#E8F8F5", alpha=0.5)
    ax1.text(0.85, 0.65, r"Easy Examples:" + "\n" + r"Weight $\approx 0$" + "\n" + r"for $\gamma \geq 2$",
             ha="center", va="center", fontsize=10,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#16A085", lw=1))
    ax1.legend(loc="upper right", framealpha=0.95, fontsize=10)

    # --- Panel 2: Gradient Magnitude |dFL / dz| ---
    for g in gammas:
        grad = focal_loss_gradient_wrt_logit(pt, gamma=g)
        color = GAMMA_PALETTE[g]
        lw = 3.0 if g in [0.0, 2.0] else 1.8
        label = rf"$\gamma = {g:g}$ (CE)" if g == 0 else rf"$\gamma = {g:g}$"
        ax2.plot(pt, grad, color=color, lw=lw, label=label)

    ax2.set_xlim(0, 1.0)
    ax2.set_ylim(-0.02, 1.05)
    ax2.set_xlabel(r"Ground Truth Probability $p_t$", fontsize=12, fontweight="bold")
    ax2.set_ylabel(r"Gradient Magnitude $|\partial \mathrm{Loss} / \partial z|$", fontsize=12, fontweight="bold")
    ax2.set_title("Gradient Dynamics: Where Learning Signal Originates", fontsize=13, fontweight="bold", pad=12)
    ax2.grid(True, linestyle=":", alpha=0.6)

    # Annotation
    ax2.annotate(
        "Standard CE ($\gamma=0$):\nEasy examples ($p_t=0.95$)\nstill produce linear gradient ($0.05$),\noverwhelming rare positives!",
        xy=(0.95, 0.05),
        xytext=(0.45, 0.40),
        arrowprops=dict(arrowstyle="->", color=COLOR_CE, lw=1.5),
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#EAECEE", edgecolor=COLOR_CE, lw=1.2),
    )
    ax2.legend(loc="upper right", framealpha=0.95, fontsize=10)

    plt.suptitle("Mathematical Mechanics of Focal Loss: Modulating Factor & Gradient Attenuation",
                 fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_file = output_dir / "focal_loss_02_modulating_factor_and_gradients.png"
    plt.savefig(out_file)
    plt.close()
    return out_file


# ---------------------------------------------------------------------------
# Figure 3: Dense Object Detection Imbalance Simulation
# ---------------------------------------------------------------------------
def plot_class_imbalance_simulation(output_dir: Path) -> Path:
    """
    Simulates a realistic one-stage detector scene (RetinaNet):
    - 100,000 background anchor boxes (easy negatives, pt in [0.95, 0.999])
    - 50 foreground object boxes (hard positives, pt in [0.05, 0.5])

    Compares cumulative gradient/loss under CE (γ=0) vs FL (γ=2).
    """
    np.random.seed(42)

    n_bg = 100_000
    n_fg = 50

    # Easy background negatives (well classified, high pt)
    pt_bg = np.random.uniform(0.95, 0.995, size=n_bg)
    # Hard foreground objects (misclassified or low confidence)
    pt_fg = np.random.uniform(0.1, 0.5, size=n_fg)

    # Compute individual losses
    ce_loss_bg = focal_loss(pt_bg, gamma=0.0)
    ce_loss_fg = focal_loss(pt_fg, gamma=0.0)

    fl_loss_bg = focal_loss(pt_bg, gamma=2.0)
    fl_loss_fg = focal_loss(pt_fg, gamma=2.0)

    # Total cumulative losses
    total_ce_bg = np.sum(ce_loss_bg)
    total_ce_fg = np.sum(ce_loss_fg)
    total_ce = total_ce_bg + total_ce_fg

    total_fl_bg = np.sum(fl_loss_bg)
    total_fl_fg = np.sum(fl_loss_fg)
    total_fl = total_fl_bg + total_fl_fg

    pct_ce_bg = (total_ce_bg / total_ce) * 100
    pct_ce_fg = (total_ce_fg / total_ce) * 100

    pct_fl_bg = (total_fl_bg / total_fl) * 100
    pct_fl_fg = (total_fl_fg / total_fl) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.2))

    # --- Left Subplot: Stacked Bar Chart of Cumulative Loss ---
    bars = ["Cross-Entropy\n($\gamma = 0$)", "Focal Loss\n($\gamma = 2$)"]
    y_pos = np.arange(len(bars))
    width = 0.5

    # Normalized percentages for direct visual comparison of gradient dominance
    ax1.bar(y_pos, [pct_ce_bg, pct_fl_bg], width, label="100,000 Easy Backgrounds", color="#E74C3C", alpha=0.85)
    ax1.bar(y_pos, [pct_ce_fg, pct_fl_fg], width, bottom=[pct_ce_bg, pct_fl_bg],
            label="50 Hard Foreground Objects", color="#2ECC71", alpha=0.85)

    # Annotate percentages directly inside bars
    ax1.text(0, pct_ce_bg / 2, f"Background\n{pct_ce_bg:.1f}%\nof Loss", ha="center", va="center", color="white", fontweight="bold", fontsize=11)
    ax1.text(0, pct_ce_bg + pct_ce_fg / 2, f"FG\n{pct_ce_fg:.1f}%", ha="center", va="center", color="#2C3E50", fontweight="bold", fontsize=10)

    ax1.text(1, pct_fl_bg / 2, f"BG\n{pct_fl_bg:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=10)
    ax1.text(1, pct_fl_bg + pct_fl_fg / 2, f"Foreground\n{pct_fl_fg:.1f}%\nof Loss", ha="center", va="center", color="white", fontweight="bold", fontsize=11)

    ax1.set_xticks(y_pos)
    ax1.set_xticklabels(bars, fontsize=11, fontweight="bold")
    ax1.set_ylabel("Share of Total Cumulative Loss / Gradient (%)", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 105)
    ax1.set_title("Gradient Dominance in Dense Anchor Imbalance", fontsize=13, fontweight="bold", pad=12)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, framealpha=0.95, fontsize=10.5)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)

    # --- Right Subplot: Absolute Loss Values (Log Scale) & Summary Cards ---
    categories = ["Easy Background\n(100k anchors)", "Hard Foreground\n(50 objects)"]
    x = np.arange(len(categories))
    bar_width = 0.35

    ax2.bar(x - bar_width/2, [total_ce_bg, total_ce_fg], bar_width, label=r"CE ($\gamma=0$)", color=COLOR_CE, alpha=0.9)
    ax2.bar(x + bar_width/2, [total_fl_bg, total_fl_fg], bar_width, label=r"FL ($\gamma=2$)", color=COLOR_GAMMA_20, alpha=0.9)

    ax2.set_yscale("log")
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=11, fontweight="bold")
    ax2.set_ylabel("Cumulative Loss (Log Scale)", fontsize=11, fontweight="bold")
    ax2.set_title("Absolute Cumulative Loss Comparison", fontsize=13, fontweight="bold", pad=12)
    ax2.grid(True, which="both", linestyle=":", alpha=0.5)
    ax2.legend(loc="upper right", framealpha=0.95, fontsize=10.5)

    # Values annotations above bars
    ax2.text(0 - bar_width/2, total_ce_bg * 1.3, f"{total_ce_bg:,.1f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=COLOR_CE)
    ax2.text(0 + bar_width/2, total_fl_bg * 1.3, f"{total_fl_bg:.2f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=COLOR_GAMMA_20)
    ax2.text(1 - bar_width/2, total_ce_fg * 1.3, f"{total_ce_fg:,.1f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=COLOR_CE)
    ax2.text(1 + bar_width/2, total_fl_fg * 1.3, f"{total_fl_fg:.2f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=COLOR_GAMMA_20)

    # Summary callout box
    callout = (
        r"$\mathbf{Key\ Takeaway:}$" + "\n"
        r"• Under Cross-Entropy, 100k easy negatives swamp the gradient ($>95\%$), drowning out foreground targets." + "\n"
        r"• Focal Loss ($\gamma=2$) cuts easy negative loss by $>99.9\%$, restoring gradient priority to the 50 real objects."
    )
    plt.suptitle("The Single-Stage Object Detection Dilemma: 100,000 Negatives vs. 50 Objects",
                 fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    out_file = output_dir / "focal_loss_03_class_imbalance_simulation.png"
    plt.savefig(out_file)
    plt.close()
    return out_file


# ---------------------------------------------------------------------------
# Figure 4: 2D Parameter Landscape (Gamma vs Probability & Alpha Balancing)
# ---------------------------------------------------------------------------
def plot_alpha_gamma_landscape(output_dir: Path) -> Path:
    """
    Plots a 2D contour map / heatmap showing Focal Loss across the full
    (pt, gamma) plane, plus the role of the alpha class-balance hyperparameter.
    """
    pt_grid = np.linspace(0.01, 0.99, 200)
    gamma_grid = np.linspace(0.0, 5.0, 200)
    PT, GAMMA = np.meshgrid(pt_grid, gamma_grid)

    FL_GRID = -(1.0 - PT) ** GAMMA * np.log(PT)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.2))

    # --- Left: 2D Contour Map of Loss Values ---
    levels = np.logspace(-3, 1, 20)
    cnt = ax1.contourf(PT, GAMMA, FL_GRID, levels=levels, cmap="plasma", norm=plt.matplotlib.colors.LogNorm())
    cbar = fig.colorbar(cnt, ax=ax1)
    cbar.set_label("Focal Loss Value (Log Scale)", fontsize=10.5, fontweight="bold")

    cs = ax1.contour(PT, GAMMA, FL_GRID, levels=[0.001, 0.01, 0.1, 0.5, 1.0, 2.0], colors="white", alpha=0.6, linewidths=1.0)
    ax1.clabel(cs, inline=True, fontsize=8.5, fmt="%g")

    ax1.axhline(2.0, color="#2ECC71", linestyle="--", lw=2, label=r"Standard $\gamma = 2.0$ (RetinaNet)")
    ax1.axhline(0.0, color="black", linestyle=":", lw=2, label=r"Standard CE ($\gamma = 0.0$)")

    ax1.set_xlabel(r"Ground Truth Probability $p_t$", fontsize=12, fontweight="bold")
    ax1.set_ylabel(r"Focusing Parameter $\gamma$", fontsize=12, fontweight="bold")
    ax1.set_title(r"2D Focal Loss Landscape: $\mathrm{FL}(p_t, \gamma)$", fontsize=13, fontweight="bold", pad=12)
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=9.5)

    # --- Right: Alpha-Balanced Focal Loss Effect ---
    # FL = - alpha_t * (1 - pt)^gamma * log(pt)
    # alpha balances positive vs negative importance
    pt_line = np.linspace(0.01, 0.99, 500)
    alphas = [0.25, 0.50, 0.75, 1.00]
    palette_alpha = ["#34495E", "#2980B9", "#E67E22", "#E74C3C"]

    for a, c in zip(alphas, palette_alpha):
        loss_pos = focal_loss(pt_line, gamma=2.0, alpha=a)
        ax2.plot(pt_line, loss_pos, label=rf"$\alpha = {a:.2f}\ (\gamma = 2.0)$", color=c, lw=2.2)

    ax2.set_xlim(0, 1.0)
    ax2.set_ylim(0, 2.5)
    ax2.set_xlabel(r"Ground Truth Probability $p_t$", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Focal Loss Value", fontsize=12, fontweight="bold")
    ax2.set_title(r"Role of $\alpha$: Scaling Class Importance with $\gamma=2$", fontsize=13, fontweight="bold", pad=12)
    ax2.grid(True, linestyle=":", alpha=0.6)

    # Note explaining alpha in RetinaNet
    alpha_note = (
        r"$\mathbf{RetinaNet\ Best\ Practice:}$" + "\n"
        r"• $\gamma = 2.0$ focuses on hard examples." + "\n"
        r"• $\alpha = 0.25$ balances foreground vs. background frequencies."
    )
    ax2.text(0.48, 1.6, alpha_note, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#EBF5FB", edgecolor="#2980B9", lw=1.2))
    ax2.legend(loc="upper right", framealpha=0.95, fontsize=10)

    plt.suptitle(r"$\mathbf{Hyperparameter\ Space:}\ \mathrm{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$",
                 fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_file = output_dir / "focal_loss_04_alpha_gamma_landscape.png"
    plt.savefig(out_file)
    plt.close()
    return out_file


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------
def main():
    output_dir = Path("outputs/focal_loss_visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Generating Publication-Quality Focal Loss Visualizations...")
    print("=" * 65)

    f1 = plot_focal_loss_curves(output_dir)
    print(f"  [✔] Figure 1 (Loss Curves vs. Gamma):              {f1}")

    f2 = plot_modulating_factor_and_gradients(output_dir)
    print(f"  [✔] Figure 2 (Modulating Factor & Gradients):     {f2}")

    f3 = plot_class_imbalance_simulation(output_dir)
    print(f"  [✔] Figure 3 (Class Imbalance Simulation):        {f3}")

    f4 = plot_alpha_gamma_landscape(output_dir)
    print(f"  [✔] Figure 4 (2D Landscape & Alpha Balancing):    {f4}")

    print("=" * 65)
    print(f"All figures saved successfully to: {output_dir.resolve()}")
    print("=" * 65)


if __name__ == "__main__":
    main()
