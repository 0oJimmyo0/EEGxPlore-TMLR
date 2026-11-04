"""Plot audited paired effects across available backbone--dataset cells.

Circles show native-minus-probe effects (RQ1), while diamonds show
native-minus-token-wise effects (RQ2). The six locked RQ2 cells are the
primary comparison; SEED-V and PhysioNet-MI remain secondary coverage.
"""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "LaBraM" / "analysis" / "tmlr_manuscript" / "manuscript_final_v2"
V4 = ROOT / "LaBraM" / "analysis" / "tmlr_manuscript" / "manuscript_final_v4"
OUT = Path(__file__).resolve().parent
SEEDS = (42, 1024, 3407)

ORDER = [
    ("CBraMod", "FACED"),
    ("CBraMod", "ISRUC"),
    ("CBraMod", "TUEV"),
    ("CBraMod", "SEED-V"),
    ("CBraMod", "PhysioNet-MI"),
    ("LaBraM", "FACED"),
    ("LaBraM", "ISRUC"),
    ("LaBraM", "TUEV"),
    ("LaBraM", "PhysioNet-MI"),
]
PLOT_ORDER = list(reversed(ORDER))


def norm(value: str) -> str:
    return value.lower().replace("_", "-")


def read_summary(path: Path):
    values = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["backbone"], norm(row["dataset"]), row["metric"])
            values[key] = (float(row["effect_mean"]), float(row["effect_sd"]))
    return values


def main():
    rq1 = read_summary(V4 / "rq1_effect_summary.csv")
    rq2 = read_summary(V4 / "rq2_effect_summary.csv")
    supporting_rq1 = read_summary(V2 / "rq1_effect_summary.csv")

    data = {}
    for backbone, dataset in ORDER:
        key = (backbone, norm(dataset))
        data[key] = {}
        for metric in ("balanced_accuracy", "macro_f1"):
            summary_key = (backbone, norm(dataset), metric)
            if summary_key in rq1:
                data[key][(metric, "rq1")] = rq1[summary_key]
            elif summary_key in supporting_rq1:
                data[key][(metric, "rq1")] = supporting_rq1[summary_key]
            if summary_key in rq2:
                data[key][(metric, "rq2")] = rq2[summary_key]

    records = []
    for (backbone, dataset), effects in data.items():
        for (metric, contrast), (effect_mean, effect_sd) in effects.items():
            records.append({
                "backbone": backbone,
                "dataset": dataset,
                "metric": metric,
                "contrast": contrast,
                "effect_mean": f"{effect_mean:.9f}",
                "effect_sd": f"{effect_sd:.9f}",
            })
    with (OUT / "paired_effects_all.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    labels = {"rq1": "Native $-$ probe", "rq2": "Native $-$ token-wise"}
    colors = {"rq1": "#2563eb", "rq2": "#d97706"}
    markers = {"rq1": "o", "rq2": "D"}
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9})

    def draw_panel(ax, metric, xlim, show_ylabels, contrasts=("rq1", "rq2")):
        y_positions = np.arange(len(ORDER))
        for index, key in enumerate(PLOT_ORDER):
            plot_key = (key[0], norm(key[1]))
            for offset, contrast in ((-0.10, "rq1"), (0.10, "rq2")):
                if contrast not in contrasts:
                    continue
                value = data[plot_key].get((metric, contrast))
                if value is None:
                    continue
                effect_mean, effect_sd = value
                ax.errorbar(
                    effect_mean,
                    index + offset,
                    xerr=effect_sd,
                    fmt=markers[contrast],
                    color=colors[contrast],
                    ecolor=colors[contrast],
                    markersize=4.5,
                    capsize=2.5,
                    linewidth=0.9,
                    label=labels[contrast] if index == 0 else None,
                    zorder=3,
                )
        ax.axvline(0, color="#475569", linestyle="--", linewidth=1.0)
        ax.set_xlim(*xlim)
        ax.set_yticks(y_positions)
        if show_ylabels:
            ax.set_yticklabels(
                [f"{backbone}--{dataset}" for backbone, dataset in PLOT_ORDER], fontsize=7.5
            )
        ax.invert_yaxis()
        metric_label = "BA" if metric == "balanced_accuracy" else "Macro-F1"
        ax.set_xlabel(f"Native-minus-reference effect ({metric_label})")
        ax.grid(axis="x", color="#e5e7eb", linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelsize=7.5)
        ax.legend(frameon=False, loc="lower right", fontsize=7)

    def render_combined():
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.9), sharey=True,
                                 constrained_layout=True)
        draw_panel(axes[0], "balanced_accuracy", (-0.14, 0.14), True)
        draw_panel(axes[1], "macro_f1", (-0.15, 0.15), False)
        axes[0].set_title("(a) Balanced accuracy", fontsize=9)
        axes[1].set_title("(b) Macro-F1", fontsize=9)
        fig.savefig(OUT / "paired_effects.pdf", bbox_inches="tight")
        plt.close(fig)

    def render_rq1_only():
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.9), sharey=True,
                                 constrained_layout=True)
        for ax, metric, xlim, title in (
            (axes[0], "balanced_accuracy", (-0.14, 0.14), "(a) Balanced accuracy"),
            (axes[1], "macro_f1", (-0.15, 0.15), "(b) Macro-F1"),
        ):
            draw_panel(ax, metric, xlim, ax is axes[0], contrasts=("rq1",))
            ax.set_title(title, fontsize=9)
        fig.savefig(OUT / "paired_effects_rq1.pdf", bbox_inches="tight")
        plt.close(fig)

    render_combined()
    render_rq1_only()


if __name__ == "__main__":
    main()
