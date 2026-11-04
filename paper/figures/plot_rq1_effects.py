"""Plot seed-level RQ1 effects for the six primary cells."""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "results/primary/native_minus_token.csv"
OUT = Path(__file__).resolve().parent
CELLS = (
    ("CBraMod", "faced"), ("CBraMod", "isruc"), ("CBraMod", "tuev"),
    ("LaBraM", "faced"), ("LaBraM", "isruc"), ("LaBraM", "tuev"),
)
SEEDS = (42, 1024, 3407)
METRICS = (("balanced_accuracy", "Balanced accuracy", (-0.065, 0.065)),
           ("macro_f1", "Macro-F1", (-0.100, 0.060)))


def load() -> dict[tuple[str, str, str], dict[int, float]]:
    values = {}
    with DATA.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["backbone"], row["dataset"].lower(), row["metric"])
            values.setdefault(key, {})[int(row["seed"])] = float(row["effect_native_minus_axisblind"])
    for key in values:
        if set(values[key]) != set(SEEDS):
            raise ValueError((key, len(values[key])))
    return values


def main() -> None:
    values = load()
    records = []
    for backbone, dataset in CELLS:
        for metric, _, _ in METRICS:
            seed_values = values[(backbone, dataset, metric)]
            for seed in SEEDS:
                value = seed_values[seed]
                records.append({"backbone": backbone, "dataset": dataset,
                                "metric": metric, "seed": seed, "effect": value})
    with (OUT / "rq1_effects.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(6.35, 2.85), sharey=True,
                             constrained_layout=True)
    colors = {"CBraMod": "#2563eb", "LaBraM": "#d97706"}
    y = np.arange(len(CELLS), dtype=float)
    for panel, (metric, label, xlim) in zip(axes, METRICS):
        for index, (backbone, dataset) in enumerate(CELLS):
            seed_values = [values[(backbone, dataset, metric)][seed] for seed in SEEDS]
            extent = [*seed_values, mean(seed_values) - stdev(seed_values),
                      mean(seed_values) + stdev(seed_values)]
            if min(extent) <= xlim[0] or max(extent) >= xlim[1]:
                raise ValueError(f"Plot bounds clip {backbone}/{dataset}/{metric}: {extent}")
            panel.scatter(seed_values, [y[index] - 0.11, y[index], y[index] + 0.11],
                          facecolors="none", edgecolors=colors[backbone],
                          s=21, linewidths=0.8, zorder=3)
            panel.errorbar(mean(seed_values), y[index], xerr=stdev(seed_values),
                           fmt="D", color=colors[backbone], ecolor=colors[backbone],
                           markersize=4.5, capsize=2.2, linewidth=0.9, zorder=4)
        panel.axvline(0, color="#111827", linestyle="--", linewidth=0.9)
        panel.axhline(2.5, color="#cbd5e1", linewidth=0.7)
        panel.set_xlim(*xlim)
        panel.set_title(label)
        panel.set_xlabel("Native − token-wise")
        panel.grid(axis="x", color="#e5e7eb", linewidth=0.5)
        panel.spines["top"].set_visible(False)
        panel.spines["right"].set_visible(False)
        panel.tick_params(axis="x", labelsize=7)
        panel.set_xticks([-0.06, -0.03, 0.0, 0.03, 0.06]
                        if metric == "balanced_accuracy" else [-0.10, -0.05, 0.0, 0.05])
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f"{b}–{d.upper()}" for b, d in CELLS], fontsize=7.5)
    axes[0].invert_yaxis()
    fig.text(0.5, 1.01, "Token-wise better  ←  0  →  Native better",
             ha="center", va="bottom", fontsize=8, color="#475569")
    fig.savefig(OUT / "rq1_effects.pdf", bbox_inches="tight")
    fig.savefig(OUT / "rq1_effects.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
