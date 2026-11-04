"""Plot an annotated 2-by-6 TUEV class-level F1 effect heatmap."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle


OUT = Path(__file__).resolve().parent
DATA = OUT / "tuev_class_effects.csv"
classes = ["SPSW", "GPED", "PLED", "EYEM", "ARTF", "BCKG"]
data = pd.read_csv(DATA)
matrix = data.pivot(index="backbone", columns="class_name", values="delta_f1_mean").loc[["CBraMod", "LaBraM"], classes] * 100
supports = data.drop_duplicates("class_name").set_index("class_name").loc[classes, "support"]

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
fig, ax = plt.subplots(figsize=(6.35, 1.95), constrained_layout=True)
image = ax.imshow(matrix.values, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-20, vcenter=0, vmax=20), aspect="auto")
ax.set_xticks(range(len(classes)), [f"{name}\n({int(supports[name])})" for name in classes])
ax.set_yticks(range(2), ["CBraMod", "LaBraM"])
ax.tick_params(length=0, labelsize=7.5)
for row in range(matrix.shape[0]):
    for column in range(matrix.shape[1]):
        value = matrix.iloc[row, column]
        color = "white" if abs(value) >= 8 else "#111827"
        highlight = row == 1 and classes[column] in {"GPED", "EYEM"}
        if highlight:
            border = Rectangle((column - 0.45, row - 0.45), 0.9, 0.9,
                               fill=False, edgecolor="white", linewidth=0.8)
            ax.add_patch(border)
        ax.text(column, row, f"{value:+.1f}", ha="center", va="center", color=color,
                fontsize=8.5 if highlight else 8,
                fontweight="bold" if highlight else "normal")
for spine in ax.spines.values():
    spine.set_color("white")
    spine.set_linewidth(1.2)
cbar = fig.colorbar(image, ax=ax, shrink=0.82, pad=0.025)
cbar.set_label("Native − token-wise F1 (pp)", fontsize=7.5)
cbar.ax.tick_params(labelsize=7)
ax.set_title("TUEV class-level F1 differences (native − token-wise)", fontsize=9)
fig.savefig(OUT / "tuev_class_effects.pdf", bbox_inches="tight")
fig.savefig(OUT / "tuev_class_effects.png", dpi=240, bbox_inches="tight")
plt.close(fig)
