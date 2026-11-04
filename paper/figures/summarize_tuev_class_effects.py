"""Aggregate absolute and paired TUEV class metrics from frozen artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "LaBraM/analysis/tmlr_manuscript/manuscript_final_v2/manuscript_final_manifest.csv"
OUT = Path(__file__).resolve().parent / "tuev_class_effects.csv"
CLASS_NAMES = ["SPSW", "GPED", "PLED", "EYEM", "ARTF", "BCKG"]


def metrics(row):
    artifact = Path(row["source_artifact"])
    source = artifact / "test_metrics.json" if row["backbone"] == "CBraMod" else artifact / "final_test.json"
    payload = json.load(source.open())
    return payload.get("primary_test", payload)


def collect(rows, backbone, dataset, method):
    selected = [
        row for row in rows
        if row["backbone"] == backbone and row["dataset"] == dataset and row["method"] == method
    ]
    selected.sort(key=lambda row: int(row["seed"]))
    recalls, f1s, supports = [], [], []
    for row in selected:
        payload = metrics(row)
        recalls.append(payload["per_class_recall"])
        f1s.append(payload["per_class_f1"])
        if "per_class_support" in payload:
            supports.append(payload["per_class_support"])
        else:
            supports.append([payload[f"target_count_class_{i}"] for i in range(6)])
    return np.asarray(recalls), np.asarray(f1s), np.asarray(supports)


def main():
    rows = list(csv.DictReader(MANIFEST.open()))
    cells = [
        ("CBraMod", "tuev", "interaction_aligned", "axis_blind"),
        ("LaBraM", "TUEV", "frozen_native_channel_patch", "axis_blind"),
    ]
    output = []
    for backbone, dataset, native_method, axis_method in cells:
        native_recall, native_f1, native_support = collect(rows, backbone, dataset, native_method)
        axis_recall, axis_f1, _ = collect(rows, backbone, dataset, axis_method)
        for index, class_name in enumerate(CLASS_NAMES):
            delta_recall = native_recall[:, index] - axis_recall[:, index]
            delta_f1 = native_f1[:, index] - axis_f1[:, index]
            output.append({
                "backbone": backbone,
                "dataset": dataset,
                "class_index": index,
                "class_name": class_name,
                "support": int(round(native_support[:, index].mean())),
                "native_recall_mean": f"{native_recall[:, index].mean():.6f}",
                "axis_blind_recall_mean": f"{axis_recall[:, index].mean():.6f}",
                "delta_recall_mean": f"{delta_recall.mean():.6f}",
                "delta_recall_sd": f"{delta_recall.std(ddof=1):.6f}",
                "native_f1_mean": f"{native_f1[:, index].mean():.6f}",
                "axis_blind_f1_mean": f"{axis_f1[:, index].mean():.6f}",
                "delta_f1_mean": f"{delta_f1.mean():.6f}",
                "delta_f1_sd": f"{delta_f1.std(ddof=1):.6f}",
            })
    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)


if __name__ == "__main__":
    main()
