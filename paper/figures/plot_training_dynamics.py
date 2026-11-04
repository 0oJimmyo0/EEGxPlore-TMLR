"""Plot paired native-minus-token-wise validation trajectories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
LABRAM = ROOT / "LaBraM"
V5 = LABRAM / "analysis/tmlr_manuscript/manuscript_final_v5"
OUT = Path(__file__).resolve().parent
SEEDS = (42, 1024, 3407)
CELLS = (
    ("CBraMod", "faced"),
    ("CBraMod", "isruc"),
    ("CBraMod", "tuev"),
    ("LaBraM", "faced"),
    ("LaBraM", "isruc"),
    ("LaBraM", "tuev"),
)


def primary_rows() -> dict[tuple[str, str, str, int], Path]:
    """Use the same seed pairs as the primary results table."""
    rows = {}
    with (V5 / "rq2_seed_effects.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            backbone, dataset = row["backbone"], row["dataset"].lower()
            root = (ROOT / "CBraMod/results" / dataset
                    if backbone == "CBraMod" else LABRAM / "checkpoints")
            for method, field in (("native", "native_run_id"),
                                  ("axis_blind", "axisblind_run_id")):
                key = (backbone, dataset, method, int(row["seed"]))
                path = root / row[field]
                if key in rows and rows[key] != path:
                    raise ValueError(f"Conflicting primary artifact for {key}")
                rows[key] = path
    return rows


def trajectory(path: Path, backbone: str) -> list[dict[str, float]]:
    config_name = "resolved_config.json" if backbone == "CBraMod" else "run_config.json"
    budget = int(json.loads((path / config_name).read_text())["epochs"])
    if backbone == "CBraMod":
        payloads = [json.loads(line) for line in (path / "metrics_by_epoch.jsonl").open()]
        records = [
            {"epoch": int(item["epoch"]), "ba": float(item["balanced_accuracy"]),
             "kappa": float(item["cohen_kappa"])}
            for item in payloads
        ]
    else:
        records = []
        for line in (path / "log.txt").open():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "epoch" in item and "val_cohen_kappa" in item:
                records.append({
                    "epoch": int(item["epoch"]) + 1,
                    "ba": float(item["val_balanced_accuracy"]),
                    "kappa": float(item["val_cohen_kappa"]),
                })
    epochs = [item["epoch"] for item in records]
    if epochs != list(range(1, budget + 1)):
        raise ValueError(f"Trajectory does not match the {budget}-epoch budget: {path}")
    if not all(np.isfinite(item[metric]) for item in records for metric in ("ba", "kappa")):
        raise ValueError(f"Non-finite validation metric: {path}")
    return records


def collect(paths: dict[tuple[str, str, str, int], Path], backbone: str, dataset: str):
    native = {}
    axis = {}
    for seed in SEEDS:
        native_path = paths[(backbone, dataset, "native", seed)]
        axis_path = paths[(backbone, dataset, "axis_blind", seed)]
        native[seed] = trajectory(native_path, backbone)
        axis[seed] = trajectory(axis_path, backbone)
        if len(native[seed]) != len(axis[seed]):
            raise ValueError((backbone, dataset, seed, len(native[seed]), len(axis[seed])))
        if [x["epoch"] for x in native[seed]] != [x["epoch"] for x in axis[seed]]:
            raise ValueError((backbone, dataset, seed, "epoch mismatch"))
        if [x["epoch"] for x in native[seed]] != [x["epoch"] for x in native[SEEDS[0]]]:
            raise ValueError((backbone, dataset, seed, "seed budget mismatch"))
    return native, axis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outputs', type=Path, help='New OUTPUT_ROOT; otherwise plot the included CSV')
    args = parser.parse_args()
    panels = []
    records = []
    if args.outputs:
        paths = {}
        for backbone, dataset in CELLS:
            for seed in SEEDS:
                for method, condition in (('native', 'native'), ('axis_blind', 'token')):
                    if backbone == 'CBraMod':
                        path = args.outputs / 'cbramod' / dataset / f'{dataset}_{condition}_s{seed}'
                    else:
                        path = args.outputs / 'labram' / f'{dataset.upper()}_{condition}_s{seed}'
                    paths[(backbone, dataset, method, seed)] = path
        for backbone, dataset in CELLS:
            native, axis = collect(paths, backbone, dataset)
            epochs = np.asarray([x['epoch'] for x in native[SEEDS[0]]])
            for seed in SEEDS:
                for i, epoch in enumerate(epochs):
                    records.append(dict(backbone=backbone, dataset=dataset, seed=seed, epoch=int(epoch),
                        delta_ba=native[seed][i]['ba'] - axis[seed][i]['ba'],
                        delta_kappa=native[seed][i]['kappa'] - axis[seed][i]['kappa']))
    else:
        with (OUT / 'training_dynamics.csv').open(newline='') as handle:
            records = list(csv.DictReader(handle))
    for backbone, dataset in CELLS:
        rows = [r for r in records if r['backbone'] == backbone and r['dataset'] == dataset]
        epochs = np.asarray(sorted({int(r['epoch']) for r in rows}))
        lookup = {(int(r['seed']), int(r['epoch'])): r for r in rows}
        deltas = {metric: np.asarray([[float(lookup[(seed, int(epoch))]['delta_' + metric])
                  for epoch in epochs] for seed in SEEDS]) for metric in ('ba', 'kappa')}
        if not all(np.isfinite(values).all() for values in deltas.values()):
            raise ValueError('Non-finite trajectory values')
        panels.append((backbone, dataset, epochs, deltas))
    if args.outputs:
        with (OUT / 'training_dynamics.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 3, figsize=(6.6, 3.4), sharey=True,
                             constrained_layout=True)
    for panel_index, (backbone, dataset, epochs, deltas) in enumerate(panels):
        ax = axes.flat[panel_index]
        values = deltas["kappa"]
        for seed_index in range(len(SEEDS)):
            ax.plot(epochs, values[seed_index], color="#64748b", linewidth=0.7,
                    alpha=0.7, zorder=2)
        ax.plot(epochs, values.mean(axis=0), color="#1d4ed8", linewidth=1.7, zorder=3)
        ax.axhline(0.0, color="#111827", linewidth=0.9, linestyle="--", zorder=1)
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(f"{backbone}–{dataset.upper()}", fontsize=8)
        ax.tick_params(labelsize=6.5)
        if panel_index % 3 == 0:
            ax.set_ylabel("Native − token-wise\nvalidation $\\kappa$", fontsize=7)
        else:
            ax.tick_params(labelleft=False)
        ax.set_xlabel("Epoch", fontsize=7)
        budget = int(epochs[-1])
        ax.set_xlim(1, budget)
        step = 20 if budget >= 80 else 10 if budget >= 30 else 5
        ax.set_xticks(sorted({1, budget, *range(step, budget, step)}))
        ax.set_ylim(-0.13, 0.07)
    fig.savefig(OUT / "training_dynamics.pdf", bbox_inches="tight")
    fig.savefig(OUT / "training_dynamics.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
