"""Plot the conceptual relationship between RQ1 and RQ2 effects."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "LaBraM/analysis/tmlr_manuscript/manuscript_final_v5"
OUT = Path(__file__).resolve().parent
CELLS = (
    ("CBraMod", "faced", "C-FACED"),
    ("CBraMod", "isruc", "C-ISRUC"),
    ("CBraMod", "tuev", "C-TUEV"),
    ("LaBraM", "faced", "L-FACED"),
    ("LaBraM", "isruc", "L-ISRUC"),
    ("LaBraM", "tuev", "L-TUEV"),
)


def read(filename: str, field: str) -> dict[tuple[str, str, str], float]:
    values = {}
    with (DATA / filename).open(newline="") as handle:
        for row in csv.DictReader(handle):
            values[(row["backbone"], row["dataset"].lower(), row["metric"])] = float(row[field]) * 100.0
    return values


def main() -> None:
    rq2 = read("rq1_effect_summary.csv", "effect_mean")
    rq1 = read("rq2_effect_summary.csv", "effect_mean")
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.5), constrained_layout=True)
    styles = {"CBraMod": ("#2563eb", "o"), "LaBraM": ("#d97706", "D")}
    offsets = {
        "C-FACED": (4, 7), "C-ISRUC": (6, -7), "C-TUEV": (0, -20),
        "L-FACED": (4, 5), "L-ISRUC": (-38, 5), "L-TUEV": (4, -11),
    }
    for ax, metric, title, xlim, ylim in (
        (axes[0], "balanced_accuracy", "(a) Balanced accuracy", (-2.5, 11.0), (-8.0, 6.0)),
        (axes[1], "macro_f1", "(b) Macro-F1", (-2.5, 11.0), (-8.0, 6.0)),
    ):
        ax.add_patch(Rectangle((0, ylim[0]), xlim[1], -ylim[0],
                               facecolor="#d97706", alpha=0.045,
                               edgecolor="none", zorder=0))
        for backbone, dataset, label in CELLS:
            x = rq2[(backbone, dataset, metric)]
            y = rq1[(backbone, dataset, metric)]
            color, marker = styles[backbone]
            highlight = (backbone, dataset) == ("LaBraM", "tuev")
            ax.scatter(x, y, s=43 if highlight else 27, marker=marker, color=color,
                       edgecolor="#111827" if highlight else "white",
                       linewidth=0.9 if highlight else 0.7, zorder=3)
            display_label = label + r"$^{\dagger}$" if label == "L-ISRUC" else label
            ax.annotate(display_label, (x, y), xytext=offsets[label], textcoords="offset points",
                        ha="center" if label == "C-TUEV" else "left",
                        fontsize=7, color="#111827",
                        fontweight="normal")
        ax.axvline(0, color="#475569", linestyle="--", linewidth=0.8)
        ax.axhline(0, color="#475569", linestyle="--", linewidth=0.8)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(title)
        ax.set_xlabel("RQ2: Native − probe (pp)")
        ax.grid(color="#e5e7eb", linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("RQ1: Native − token-wise (pp)")
    fig.text(0.5, 1.01,
             "Native adapter favored  ↑     |     Token-wise control favored  ↓",
             ha="center", va="bottom", fontsize=7.5, color="#475569")
    fig.savefig(OUT / "rq1_rq2_quadrants.pdf", bbox_inches="tight")
    fig.savefig(OUT / "rq1_rq2_quadrants.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
