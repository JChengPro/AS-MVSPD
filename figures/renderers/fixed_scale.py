from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def render_supplementary9() -> None:
    source = DATA / "fixed_scale_ablation_summary.csv"
    data = pd.read_csv(source, keep_default_na=False)
    methods = ["AS-MVSPD"] + [f"Fixed top-{k}" for k in (1, 2, 5, 10, 20)]
    colors = ["#5E7FA3", "#4F9F89", "#8DBEAE", "#8179AA", "#AAA5C4", "#C98652"]
    markers = ["o", "s", "^", "D", "v", "P"]
    linestyles = ["-", "-", "--", "-", "--", "-"]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(183 / 25.4, 145 / 25.4))
    for row, scenario in enumerate(["stable_sparse", "one_block"]):
        for col, effect in enumerate([1.1, 1.4]):
            ax = axes[row, col]
            for index, method in enumerate(methods):
                part = data.loc[
                    (data.scenario == scenario)
                    & (data.effect == effect)
                    & (data.method == method)
                ].sort_values("n_signal")
                x = np.arange(3) + (index - 2.5) * 0.035
                ax.errorbar(
                    x,
                    part.rate,
                    yerr=np.vstack([part.rate - part.ci_low, part.ci_high - part.rate]),
                    color=colors[index],
                    marker=markers[index],
                    linestyle=linestyles[index],
                    mfc="white",
                    ms=3,
                    lw=0.8,
                    elinewidth=0.6,
                    capsize=1.5,
                )
            ax.set_xticks(range(3), ["1", "5", "20"])
            ax.set_xlim(-0.2, 2.2)
            ax.set_ylim(-0.015, 1.05) if row == 0 else ax.set_ylim(0, 0.12)
            ax.set_xlabel("Signals per active view")
            ax.set_ylabel("Joint power" if row == 0 else "Joint false rejection rate")
            title = "Joint alternative: 2/2 active" if row == 0 else "Joint null: 1/2 active"
            ax.set_title(f"{title}\nEffect = {effect}", loc="left")
            ax.text(
                -0.16,
                1.10,
                "ABCD"[row * 2 + col],
                transform=ax.transAxes,
                weight="bold",
                fontsize=10,
            )
            if row == 1:
                ax.axhline(0.05, color="#777777", ls="--", lw=0.65)
                ax.text(0.30, 0.052, r"$\alpha = 0.05$", ha="left", va="bottom", fontsize=6)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="#E8E8E8", linewidth=0.5)
            ax.set_axisbelow(True)
    handles = [
        Line2D(
            [],
            [],
            color=color,
            marker=marker,
            linestyle=linestyle,
            mfc="white",
            lw=0.8,
            ms=3,
            label=method,
        )
        for method, color, marker, linestyle in zip(methods, colors, markers, linestyles)
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=6,
        fontsize=6.5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.99),
    )
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.14, top=0.87, hspace=0.57, wspace=0.35)
    fig.text(
        0.10,
        0.045,
        "n = 12; two views; 200 features/view; 500 replicates/setting; 95% Wilson intervals.\n"
        "Same 256-element closed subgroup for all methods; markers offset horizontally for visibility.",
        fontsize=6.5,
    )
    stem = OUT_SUPP / "Figure_S9_fixed_scale_ablation"
    emit(fig, stem)
    plt.close(fig)
