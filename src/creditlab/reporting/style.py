"""Shared, understated figure style (matplotlib for report figures, Plotly for the dashboard).

Categorical slots follow a fixed order validated for colour-vision-deficiency separation
(first three slots validate for all pairs). Colour always follows the entity, never its rank.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#ffffff"
NEUTRAL = "#9b9a94"
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

# Fixed entity -> colour mapping used across every figure and the dashboard.
ENTITY_COLORS = {
    "dummy": NEUTRAL,
    "logistic": SERIES[0],
    "xgboost": SERIES[1],
    "observed": TEXT_SECONDARY,
    "base": SERIES[0],
    "base_correlated": SERIES[2],
    "mild_stress": SERIES[3],
    "severe_stress": SERIES[7],
    "correlated_downturn": SERIES[6],
    "A_retain_all": NEUTRAL,
    "B_pd_threshold": SERIES[0],
    "C_expected_contribution": SERIES[1],
    "D_random_equal_size": SERIES[2],
    "raw": SERIES[0],
    "sigmoid": SERIES[1],
    "isotonic": SERIES[2],
}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.prop_cycle": matplotlib.cycler(color=SERIES),
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "lines.linewidth": 2.0,
        "font.size": 9.5,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def save(fig, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def caption(ax, text: str) -> None:
    """Small source/caveat note under the axes."""
    ax.annotate(text, xy=(0, -0.2), xycoords="axes fraction", fontsize=7.5,
                color=TEXT_SECONDARY, va="top")
