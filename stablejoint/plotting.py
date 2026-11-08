"""Shared publication-figure style for the AS-MVSPD project."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


PROJECT = Path(__file__).resolve().parents[1]
STYLE_PATH = Path(__file__).with_name("figure_style.json")


def load_style() -> dict:
    return json.loads(STYLE_PATH.read_text(encoding="utf-8"))


STYLE = load_style()
METHOD_COLORS = STYLE["method_colors"]
METHOD_MARKERS = STYLE["method_markers"]
OMIC_COLORS = STYLE["omics_colors"]
TIME_COLORS = STYLE["time_colors"]
NEUTRAL = STYLE["neutral_colors"]


def mm_to_inch(value: float) -> float:
    return float(value) / 25.4


def apply_publication_style() -> None:
    fonts = STYLE["fonts"]
    lines = STYLE["line_width_pt"]
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": fonts["family"],
            "font.size": fonts["body_pt"],
            "axes.labelsize": fonts["axis_label_pt"],
            "axes.titlesize": fonts["axis_label_pt"],
            "xtick.labelsize": fonts["tick_pt"],
            "ytick.labelsize": fonts["tick_pt"],
            "legend.fontsize": fonts["legend_pt"],
            "axes.linewidth": lines["axes"],
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.edgecolor": NEUTRAL["text"],
            "axes.labelcolor": NEUTRAL["text"],
            "xtick.color": NEUTRAL["text"],
            "ytick.color": NEUTRAL["text"],
            "text.color": NEUTRAL["text"],
            "legend.frameon": False,
            "legend.handlelength": 1.6,
            "legend.handletextpad": 0.5,
            "legend.columnspacing": 1.0,
            "lines.linewidth": lines["data"],
            "lines.markersize": 3.5,
            "patch.linewidth": lines["axes"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def panel_label(axis: plt.Axes, label: str, x: float = -0.12, y: float = 1.05) -> None:
    axis.text(
        x,
        y,
        label,
        transform=axis.transAxes,
        fontsize=STYLE["fonts"]["panel_label_pt"],
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )


def format_pvalue(value: float) -> str:
    if value < STYLE["pvalue"]["small_threshold"]:
        return f"{value:.2e}"
    return f"{value:.4f}"


def save_publication_figure(
    fig: plt.Figure,
    output_stem: Path,
    dpi: int | None = None,
) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    target_dpi = dpi or int(STYLE["canvas"]["preview_dpi"])
    fig.savefig(output_stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(output_stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(output_stem.with_suffix(".png"), dpi=target_dpi, facecolor="white")


apply_publication_style()
