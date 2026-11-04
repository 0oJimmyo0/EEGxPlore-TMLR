"""Summarize normalized native-minus-axis-blind TUEV confusion matrices."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "LaBraM/analysis/tmlr_manuscript/manuscript_final_v2/manuscript_final_manifest.csv"
OUT = Path(__file__).resolve().parent / "tuev_confusion_deltas.csv"
CLASS_NAMES = ["SPSW", "GPED", "PLED", "EYEM", "ARTF", "BCKG"]


def matrices(rows, backbone, dataset, method):
    selected = [
        row for row in rows
        if row["backbone"] == backbone and row["dataset"] == dataset and row["method"] == method
    ]
    selected.sort(key=lambda row: int(row["seed"]))
    result = []
    for row in selected:
        artifact = Path(row["source_artifact"])
        source = artifact / "test_metrics.json" if backbone == "CBraMod" else artifact / "final_test.json"
        container = json.load(source.open())
        payload = container.get("primary_test", container)
        matrix = np.asarray(payload["confusion_matrix"], dtype=float)
        result.append(matrix / matrix.sum(axis=1, keepdims=True))
    return np.asarray(result)


def main():
    rows = list(csv.DictReader(MANIFEST.open()))
    cells = [
        ("CBraMod", "tuev", "interaction_aligned", "axis_blind"),
        ("LaBraM", "TUEV", "frozen_native_channel_patch", "axis_blind"),
    ]
    output = []
    for backbone, dataset, native_method, axis_method in cells:
        native = matrices(rows, backbone, dataset, native_method)
        axis = matrices(rows, backbone, dataset, axis_method)
        delta = native - axis
        mean = delta.mean(axis=0)
        sd = delta.std(axis=0, ddof=1)
        for i, source_class in enumerate(CLASS_NAMES):
            for j, target_class in enumerate(CLASS_NAMES):
                output.append({
                    "backbone": backbone,
                    "dataset": dataset,
                    "source_class": source_class,
                    "target_class": target_class,
                    "delta_mean": f"{mean[i, j]:.6f}",
                    "delta_sd": f"{sd[i, j]:.6f}",
                })
    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)


if __name__ == "__main__":
    main()
