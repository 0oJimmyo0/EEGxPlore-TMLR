"""Create the backbone-specific native-adapter architecture figure."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


OUT = Path(__file__).resolve().parent


def rounded(ax, xy, width, height, text, face="#f8fafc", fontsize=7.2,
            weight="normal", edge="#334155"):
    patch = FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.008,rounding_size=0.015",
        linewidth=0.9, facecolor=face, edgecolor=edge,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text,
            ha="center", va="center", fontsize=fontsize, weight=weight,
            color="#0f172a", linespacing=1.05)


def arrow(ax, start, end, color="#64748b", lw=1.0):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=8,
        linewidth=lw, color=color, shrinkA=2, shrinkB=2,
    ))


def panel(ax, x0, title):
    ax.text(x0 + 0.15, 0.94, title, ha="center", va="top",
            fontsize=9, weight="bold", color="#0f172a")


def native_panel(ax, x0, labram=False):
    panel(ax, x0, "(b) LaBraM" if labram else "(a) CBraMod")
    if labram:
        ax.add_patch(Rectangle((x0 + 0.045, 0.76), 0.21, 0.09,
                               facecolor="#dbeafe", edgecolor="#334155",
                               linewidth=0.9))
        ax.text(x0 + 0.15, 0.805,
                "patch tokens\n$H \u2208 \mathbb{R}^{B\u00d7C\u00d7S\u00d7D}$",
                ha="center", va="center", fontsize=6.5, weight="bold",
                color="#0f172a", linespacing=1.0)
        rounded(ax, (x0 + 0.07, 0.60), 0.16, 0.08,
                "reconstruct\n$C \u00d7 S$ grid", face="#eff6ff", fontsize=6.8)
        arrow(ax, (x0 + 0.15, 0.76), (x0 + 0.15, 0.68))
        rounded(ax, (x0 + 0.015, 0.40), 0.13, 0.10,
                "sequence\nover $C$\n(full $D$)", face="#dcfce7", fontsize=6.5)
        rounded(ax, (x0 + 0.185, 0.40), 0.13, 0.10,
                "sequence\nover $S$\n(full $D$)", face="#dcfce7", fontsize=6.5)
        arrow(ax, (x0 + 0.15, 0.60), (x0 + 0.08, 0.50))
        arrow(ax, (x0 + 0.15, 0.60), (x0 + 0.24, 0.50))
    else:
        ax.add_patch(Rectangle((x0 + 0.045, 0.76), 0.21, 0.09,
                               facecolor="#dbeafe", edgecolor="#334155",
                               linewidth=0.9))
        ax.add_patch(Rectangle((x0 + 0.045, 0.76), 0.105, 0.09,
                               facecolor="#bfdbfe", edgecolor="none"))
        ax.text(x0 + 0.15, 0.805, "$H$ grid", ha="center", va="center",
                fontsize=7.2, weight="bold")
        ax.text(x0 + 0.098, 0.735, "$D/2$", ha="center", va="center",
                fontsize=6.5, color="#334155")
        ax.text(x0 + 0.203, 0.735, "$D/2$", ha="center", va="center",
                fontsize=6.5, color="#334155")
        rounded(ax, (x0 + 0.015, 0.40), 0.13, 0.10,
                "channel\nsequence", face="#dcfce7", fontsize=6.8)
        rounded(ax, (x0 + 0.185, 0.40), 0.13, 0.10,
                "patch\nsequence", face="#dcfce7", fontsize=6.8)
        arrow(ax, (x0 + 0.098, 0.76), (x0 + 0.08, 0.50))
        arrow(ax, (x0 + 0.203, 0.76), (x0 + 0.24, 0.50))

    rounded(ax, (x0 + 0.015, 0.22), 0.13, 0.10,
            "Down $\u2192$ MHA($C$)\n$\u2192$ Up", face="#fef3c7", fontsize=6.2)
    rounded(ax, (x0 + 0.185, 0.22), 0.13, 0.10,
            "Down $\u2192$ MHA($S$)\n$\u2192$ Up", face="#fef3c7", fontsize=6.2)
    arrow(ax, (x0 + 0.08, 0.40), (x0 + 0.08, 0.32))
    arrow(ax, (x0 + 0.24, 0.40), (x0 + 0.24, 0.32))
    rounded(ax, (x0 + 0.07, 0.06), 0.16, 0.08,
            "residual add $\u2192 H'$", face="#e2e8f0", fontsize=6.8,
            weight="bold")
    arrow(ax, (x0 + 0.08, 0.22), (x0 + 0.15, 0.14))
    arrow(ax, (x0 + 0.24, 0.22), (x0 + 0.17, 0.14))


def control_panel(ax, x0):
    panel(ax, x0, "(c) Token-wise residual control")
    labels = ["$H$", "LN", "Down", "GELU", "Up", "$+\u00a0H$"]
    faces = ["#dbeafe", "#f8fafc", "#ffedd5", "#ffedd5", "#ffedd5", "#e2e8f0"]
    ys = [0.76, 0.64, 0.52, 0.40, 0.28, 0.14]
    for y, label, face in zip(ys, labels, faces):
        rounded(ax, (x0 + 0.09, y), 0.12, 0.07, label, face=face,
                fontsize=7.0, weight=label in ("$H$", "$+\u00a0H$"))
    for y1, y2 in zip(ys[:-1], ys[1:]):
        arrow(ax, (x0 + 0.15, y1), (x0 + 0.15, y2 + 0.07))
    ax.text(x0 + 0.15, 0.045, "No mixing over $C$ or $S$", ha="center",
            va="center", fontsize=7.0, color="#9a3412", weight="bold")


def main():
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.3),
                             gridspec_kw={"wspace": 0.12})
    for ax in axes:
        ax.set_xlim(0, 0.34)
        ax.set_ylim(0, 1)
        ax.axis("off")
    native_panel(axes[0], 0.0, labram=False)
    native_panel(axes[1], 0.0, labram=True)
    control_panel(axes[2], 0.0)
    fig.savefig(OUT / "adapter_architecture.pdf", bbox_inches="tight")
    fig.savefig(OUT / "adapter_architecture.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
