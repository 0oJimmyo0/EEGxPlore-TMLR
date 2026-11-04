"""Create the compact scientific-question overview figure."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


OUT = Path(__file__).resolve().parent


def box(ax, x, y, width, height, text, face, edge="#334155", fontsize=9,
        weight="normal", linestyle="-"):
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.1, facecolor=face, edgecolor=edge, linestyle=linestyle,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center",
            fontsize=fontsize, weight=weight, color="#0f172a",
            linespacing=1.15)


def tensor_box(ax, x, y, width, height, text, face="#dbeafe", fontsize=9,
               weight="bold"):
    ax.add_patch(Rectangle(
        (x, y), width, height, linewidth=1.1, facecolor=face,
        edgecolor="#334155",
    ))
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center",
            fontsize=fontsize, weight=weight, color="#0f172a",
            linespacing=1.15)


def arrow(ax, start, end, color="#64748b", lw=1.4):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=11,
        linewidth=lw, color=color, shrinkA=4, shrinkB=4,
    ))


def main():
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.02, 0.96, "Controlled question: does native interaction structure help?",
            fontsize=10.5, weight="bold", color="#0f172a", va="top")
    ax.text(0.02, 0.89, "Question  →  interventions  →  matched comparison  →  RQs",
            fontsize=7.2, color="#64748b", va="top")

    tensor_box(ax, 0.38, 0.70, 0.24, 0.11,
        "Pretrained representation\n$H$", "#dbeafe", fontsize=8.4,
        weight="bold")

    box(ax, 0.08, 0.48, 0.27, 0.13,
        "Native adapter\nMix native $C/S$ axes",
        "#dcfce7", fontsize=7.5, weight="bold")
    box(ax, 0.65, 0.48, 0.27, 0.13,
        "Token-wise residual control",
        "#ffedd5", fontsize=7.5, weight="bold")
    arrow(ax, (0.43, 0.70), (0.25, 0.61))
    arrow(ax, (0.57, 0.70), (0.75, 0.61))

    box(ax, 0.35, 0.32, 0.30, 0.09,
        "Parameter-matched residual interventions", "#f1f5f9",
        edge="#64748b", fontsize=7.4, weight="bold", linestyle="--")
    arrow(ax, (0.25, 0.48), (0.40, 0.41))
    arrow(ax, (0.75, 0.48), (0.60, 0.41))
    arrow(ax, (0.50, 0.32), (0.50, 0.25))

    box(ax, 0.40, 0.21, 0.20, 0.06,
        "Downstream classifier", "#e2e8f0", fontsize=7.1,
        weight="bold")

    ax.plot([0.04, 0.96], [0.14, 0.14], color="#cbd5e1", linewidth=1)
    questions = [
        (0.04, "RQ1", "Native vs probe", "#e0f2fe"),
        (0.36, "RQ2  PRIMARY", "Native vs token-wise control", "#dcfce7"),
        (0.68, "RQ3  SECONDARY", "Full+native vs dense", "#fef3c7"),
    ]
    for x, title, text, face in questions:
        box(ax, x, 0.04, 0.28, 0.07, f"{title}\n{text}", face,
            fontsize=7.0, weight="bold")

    fig.savefig(OUT / "study_overview.pdf", bbox_inches="tight")
    fig.savefig(OUT / "study_overview.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
