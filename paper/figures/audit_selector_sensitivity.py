"""Audit validation-kappa versus validation-balanced-accuracy selectors."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "LaBraM/analysis/tmlr_manuscript/manuscript_final_v2/manuscript_final_manifest.csv"
OUT = Path(__file__).resolve().parent / "selector_sensitivity.csv"


def trajectory(row):
    artifact = Path(row["source_artifact"])
    if row["backbone"] == "CBraMod":
        records = [json.loads(line) for line in (artifact / "metrics_by_epoch.jsonl").open()]
        return [(int(r["epoch"]), float(r["cohen_kappa"]), float(r["balanced_accuracy"])) for r in records]
    records = []
    for line in (artifact / "log.txt").open():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "epoch" in record and "val_cohen_kappa" in record:
            records.append((int(record["epoch"]) + 1, float(record["val_cohen_kappa"]), float(record["val_balanced_accuracy"])))
    return records


def main():
    rows = list(csv.DictReader(MANIFEST.open()))
    cells = [
        ("CBraMod", "faced", "interaction_aligned", "axis_blind", "CBraMod--FACED"),
        ("CBraMod", "tuev", "interaction_aligned", "axis_blind", "CBraMod--TUEV"),
        ("LaBraM", "TUEV", "frozen_native_channel_patch", "axis_blind", "LaBraM--TUEV"),
    ]
    output = []
    for backbone, dataset, native_method, axis_method, label in cells:
        for method, method_label in ((native_method, "Native"), (axis_method, "Axis-blind")):
            selected = [
                row for row in rows
                if row["backbone"] == backbone and row["dataset"] == dataset and row["method"] == method
            ]
            selected.sort(key=lambda row: int(row["seed"]))
            shifts = []
            same = 0
            epoch_pairs = []
            for row in selected:
                values = trajectory(row)
                epochs = np.asarray([x[0] for x in values])
                kappa = np.asarray([x[1] for x in values])
                ba = np.asarray([x[2] for x in values])
                k_epoch = int(epochs[np.argmax(kappa)])
                ba_epoch = int(epochs[np.argmax(ba)])
                shifts.append(abs(k_epoch - ba_epoch))
                same += int(k_epoch == ba_epoch)
                epoch_pairs.append(f"{row['seed']}:{k_epoch}->{ba_epoch}")
            output.append({
                "cell": label,
                "condition": method_label,
                "seeds_same_epoch": f"{same}/3",
                "mean_abs_epoch_shift": f"{np.mean(shifts):.2f}",
                "median_abs_epoch_shift": f"{np.median(shifts):.2f}",
                "seed_epoch_pairs": "; ".join(epoch_pairs),
            })
    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)


if __name__ == "__main__":
    main()
