"""Aggregate selected-checkpoint TUEV class recall from the final manifest."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "LaBraM/analysis/tmlr_manuscript/manuscript_final_v2/manuscript_final_manifest.csv"
OUT = Path(__file__).resolve().parent / "tuev_class_recall_summary.csv"


def read_recall(row):
    artifact = Path(row["source_artifact"])
    source = artifact / "test_metrics.json" if row["backbone"] == "CBraMod" else artifact / "final_test.json"
    payload = json.load(source.open())
    if "primary_test" in payload:
        payload = payload["primary_test"]
    return np.asarray(payload["per_class_recall"], dtype=float)


def main():
    rows = list(csv.DictReader(MANIFEST.open()))
    cells = [
        ("CBraMod", "tuev", "frozen_probe", "Probe"),
        ("CBraMod", "tuev", "interaction_aligned", "Native"),
        ("CBraMod", "tuev", "axis_blind", "Axis-blind"),
        ("LaBraM", "TUEV", "frozen_probe", "Probe"),
        ("LaBraM", "TUEV", "frozen_native_channel_patch", "Native"),
        ("LaBraM", "TUEV", "axis_blind", "Axis-blind"),
    ]
    output = []
    for backbone, dataset, method, label in cells:
        selected = [
            row for row in rows
            if row["backbone"] == backbone and row["dataset"] == dataset and row["method"] == method
        ]
        selected.sort(key=lambda row: int(row["seed"]))
        values = np.asarray([read_recall(row) for row in selected])
        for class_index, (mean, std) in enumerate(zip(values.mean(0), values.std(0, ddof=1)), start=1):
            output.append({
                "backbone": backbone,
                "dataset": dataset,
                "method": label,
                "class": class_index,
                "mean_recall": f"{mean:.6f}",
                "sd_recall": f"{std:.6f}",
            })
    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)


if __name__ == "__main__":
    main()
