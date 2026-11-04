"""Plot the six primary paired effects as a zero-centred effect matrix."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "LaBraM/analysis/tmlr_manuscript/manuscript_final_v4"
OUT = Path(__file__).resolve().parent
BACKBONES = ("CBraMod", "LaBraM")
DATASETS = ("faced", "isruc", "tuev")
METRICS = (("balanced_accuracy", "Balanced accuracy"), ("macro_f1", "Macro-F1"))


def read_effects(filename: str, field: str) -> dict[tuple[str, str, str], float]:
    values = {}
    with (DATA / filename).open(newline="") as handle:
        for row in csv.DictReader(handle):
            values[(row["backbone"], row["dataset"].lower(), row["metric"])] = float(row[field])
    return values


def main() -> None:
    rq1 = read_effects("rq1_effect_summary.csv", "effect_mean")
    rq2 = read_effects("rq2_effect_summary.csv", "effect_mean")
    effects = {"RQ1": rq1, "RQ2": rq2}

    rows = []
    for contrast, values in effects.items():
        for metric, _ in METRICS:
            for backbone in BACKBONES:
                for dataset in DATASETS:
                    rows.append({
                        "contrast": contrast,
                        "metric": metric,
                        "backbone": backbone,
                        "dataset": dataset,
                        "effect": values[(backbone, dataset, metric)],
                    })
    with (OUT / "effect_matrix.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9})
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.6), constrained_layout=True)
    norm = TwoSlopeNorm(vmin=-10.0, vcenter=0.0, vmax=10.0)
    cmap = plt.get_cmap("RdBu_r")
    image = None
    for row_index, contrast in enumerate(("RQ1", "RQ2")):
        for column_index, (metric, metric_label) in enumerate(METRICS):
            matrix = np.array([
                [effects[contrast][(backbone, dataset, metric)] for dataset in DATASETS]
                for backbone in BACKBONES
            ]) * 100.0
            ax = axes[row_index, column_index]
            image = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
            ax.set_xticks(range(len(DATASETS)), [name.upper() for name in DATASETS])
            ax.set_yticks(range(len(BACKBONES)), BACKBONES)
            ax.tick_params(length=0, labelsize=8)
            ax.set_title(f"({chr(97 + row_index * 2 + column_index)}) {contrast}: {metric_label}")
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    value = matrix[i, j]
                    color = "white" if abs(value) >= 5.5 else "#111827"
                    ax.text(j, i, f"{value:+.1f}", ha="center", va="center", color=color, fontsize=8)
            for spine in ax.spines.values():
                spine.set_color("white")
                spine.set_linewidth(1.4)
    axes[0, 0].set_ylabel("RQ1\nNative − probe", labelpad=9)
    axes[1, 0].set_ylabel("RQ2\nNative − token-wise", labelpad=9)
    cbar = fig.colorbar(image, ax=axes, shrink=0.82, pad=0.025)
    cbar.set_label("Effect (percentage points)\npositive = native better", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.savefig(OUT / "effect_matrix.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
