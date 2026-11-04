#!/usr/bin/env python3
"""Build the manuscript-v4 primary package.

The package retains the four existing v2 primary cells and adds the fresh
three-seed LaBraM FACED/ISRUC closure packet.  The older heterogeneous
LaBraM FACED/ISRUC records are deliberately not pooled with these rows.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev


OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parents[2]
V2 = ROOT / "analysis/tmlr_manuscript/manuscript_final_v2"
CHECKPOINTS = ROOT / "checkpoints"
SEEDS = (42, 1024, 3407)
PRIMARY_CELLS = (
    ("CBraMod", "faced"),
    ("CBraMod", "isruc"),
    ("CBraMod", "tuev"),
    ("LaBraM", "faced"),
    ("LaBraM", "isruc"),
    ("LaBraM", "tuev"),
)


def f1_from_test(payload: dict[str, object]) -> float:
    values = payload.get("per_class_f1")
    if not isinstance(values, list) or not values:
        raise ValueError("final_test.json does not contain per-class F1 values")
    return mean(float(value) for value in values)


def load_v2_rows(filename: str) -> list[dict[str, str]]:
    with (V2 / filename).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(filename: str, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {filename}")
    with (OUTPUT / filename).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fresh_name(dataset: str, method: str, seed: int) -> str:
    if dataset == "faced":
        prefix = {
            "dense": "labram_faced_frozen_dense_primary",
            "native": "labram_faced_frozen_native_cp_lowrank_b64",
            "axis": "labram_faced_frozen_axisblind_b213",
        }[method]
        return f"{prefix}_s{seed}_b32_lr7e-4_e80_scale100_tmlr_table2_final"
    prefix = {
        "dense": "labram_isruc_frozen_dense_primary",
        "native": "labram_isruc_frozen_native_cp_lowrank_b64",
        "axis": "labram_isruc_frozen_axisblind_b213",
    }[method]
    return f"{prefix}_s{seed}_b16_lr2e-4_e30_scale1_tmlr_table2_final"


def read_fresh_scores() -> tuple[list[dict[str, object]], list[str]]:
    scores: list[dict[str, object]] = []
    artifacts: list[str] = []
    for dataset in ("faced", "isruc"):
        for seed in SEEDS:
            for method in ("dense", "native", "axis"):
                path = CHECKPOINTS / fresh_name(dataset, method, seed)
                if not path.is_dir():
                    raise FileNotFoundError(path)
                config = json.loads((path / "run_config.json").read_text(encoding="utf-8"))
                final = json.loads((path / "final_test.json").read_text(encoding="utf-8"))
                test = final["primary_test"]
                if final.get("selection_metric") != "cohen_kappa":
                    raise ValueError(f"Unexpected selector in {path}")
                if config.get("backbone_mode") != "frozen" or not config.get("frozen_backbone_eval_mode"):
                    raise ValueError(f"Fresh closure is not frozen in {path}")
                if config.get("num_workers") != 0:
                    raise ValueError(f"Fresh closure uses nonzero workers in {path}")
                expected = {
                    "dense": ("none", "full", 64),
                    "native": ("channel_patch", "low_rank", 64),
                    "axis": ("generic", "full", 213),
                }[method]
                actual = (
                    config.get("adapter_type"),
                    config.get("adapter_variant"),
                    config.get("adapter_bottleneck"),
                )
                if actual != expected:
                    raise ValueError(f"Unexpected adapter configuration in {path}: {actual}")
                row = {
                    "backbone": "LaBraM",
                    "dataset": dataset,
                    "method": {"dense": "frozen_probe", "native": "interaction_aligned", "axis": "axis_blind"}[method],
                    "seed": seed,
                    "run_id": path.name,
                    "source_artifact": str(path.relative_to(ROOT)),
                    "balanced_accuracy": float(test["balanced_accuracy"]),
                    "macro_f1": f1_from_test(test),
                    "cohen_kappa": float(test["cohen_kappa"]),
                    "weighted_f1": float(test["f1_weighted"]),
                    "adapter_parameter_count": int(config.get("adapter_parameter_count", 0)),
                    "selection_metric": final["selection_metric"],
                }
                if not all(math.isfinite(float(row[key])) for key in ("balanced_accuracy", "macro_f1", "cohen_kappa", "weighted_f1")):
                    raise ValueError(f"Non-finite metric in {path}")
                scores.append(row)
                artifacts.append(str(path.relative_to(ROOT)))
    return scores, artifacts


def build_effects(scores: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    score_map = {(row["dataset"], row["method"], int(row["seed"])): row for row in scores}
    rq1: list[dict[str, object]] = []
    rq2: list[dict[str, object]] = []
    for dataset in ("faced", "isruc"):
        for seed in SEEDS:
            native = score_map[(dataset, "interaction_aligned", seed)]
            probe = score_map[(dataset, "frozen_probe", seed)]
            axis = score_map[(dataset, "axis_blind", seed)]
            mismatch = abs(int(native["adapter_parameter_count"]) - int(axis["adapter_parameter_count"])) / int(native["adapter_parameter_count"]) * 100
            for metric in ("balanced_accuracy", "macro_f1"):
                rq1.append({
                    "backbone": "LaBraM", "dataset": dataset, "axis": "channel_patch", "seed": seed,
                    "native_run_id": native["run_id"], "probe_run_id": probe["run_id"], "metric": metric,
                    "native_value": native[metric], "probe_value": probe[metric],
                    "effect_native_minus_probe": float(native[metric]) - float(probe[metric]),
                })
                rq2.append({
                    "backbone": "LaBraM", "dataset": dataset, "axis": "channel_patch", "seed": seed,
                    "native_run_id": native["run_id"], "axisblind_run_id": axis["run_id"],
                    "parameter_match_pct": mismatch, "parameter_matching_basis": "adapter_module_parameters",
                    "pair_status": "VALID", "analysis_scope": "primary",
                    "pair_audit_notes": "fresh three-seed closure packet; shared frozen protocol",
                    "metric": metric, "native_value": native[metric], "axisblind_value": axis[metric],
                    "effect_native_minus_axisblind": float(native[metric]) - float(axis[metric]),
                })

    for filename, target, fields in (
        ("rq1_seed_effects.csv", rq1, ("effect_native_minus_probe",)),
        ("rq2_seed_effects.csv", rq2, ("effect_native_minus_axisblind",)),
    ):
        target.extend(
            row for row in load_v2_rows(filename)
            if (row["backbone"], row["dataset"]) in {
                ("CBraMod", "faced"), ("CBraMod", "isruc"), ("CBraMod", "tuev"), ("LaBraM", "TUEV")
            }
        )
        target.sort(key=lambda row: (row["backbone"], row["dataset"].lower(), int(row["seed"]), row["metric"]))
    return rq1, rq2


def aggregate(rows: list[dict[str, object]], effect_field: str) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        key = (str(row["backbone"]), str(row["dataset"]), str(row["metric"]))
        groups.setdefault(key, []).append(float(row[effect_field]))
    output = []
    for (backbone, dataset, metric), values in sorted(groups.items()):
        if len(values) != 3:
            raise ValueError(f"Expected three seeds for {(backbone, dataset, metric)}")
        output.append({
            "backbone": backbone, "dataset": dataset, "metric": metric, "n": 3,
            "effect_mean": mean(values), "effect_sd": stdev(values),
        })
    return output


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    scores, artifacts = read_fresh_scores()
    rq1, rq2 = build_effects(scores)
    write_csv("rq1_seed_effects.csv", rq1)
    write_csv("rq2_seed_effects.csv", rq2)
    write_csv("rq1_effect_summary.csv", aggregate(rq1, "effect_native_minus_probe"))
    write_csv("rq2_effect_summary.csv", aggregate(rq2, "effect_native_minus_axisblind"))

    method_rows = []
    for row in scores:
        method_rows.append({
            "backbone": row["backbone"], "dataset": row["dataset"], "method": row["method"],
            "seed": row["seed"], "balanced_accuracy": row["balanced_accuracy"],
            "macro_f1": row["macro_f1"], "cohen_kappa": row["cohen_kappa"],
            "weighted_f1": row["weighted_f1"], "run_id": row["run_id"],
            "source_artifact": row["source_artifact"], "selection_metric": row["selection_metric"],
        })
    write_csv("fresh_labram_closure_seed_results.csv", method_rows)

    existing = set()
    for filename in ("rq1_seed_effects.csv", "rq2_seed_effects.csv"):
        for row in load_v2_rows(filename):
            if (row["backbone"], row["dataset"]) in {
                ("CBraMod", "faced"), ("CBraMod", "isruc"), ("CBraMod", "tuev"), ("LaBraM", "TUEV")
            }:
                for name in (row["native_run_id"], row.get("probe_run_id", row.get("axisblind_run_id", ""))):
                    if name:
                        prefix = (f"../CBraMod/results/{row['dataset'].lower()}"
                                  if row["backbone"] == "CBraMod" else "checkpoints")
                        existing.add(f"{prefix}/{name}")
    source_artifacts = sorted(set(artifacts) | existing)
    for relative_path in source_artifacts:
        if not (ROOT / relative_path).is_dir():
            raise FileNotFoundError(f"Missing primary source artifact: {relative_path}")
    contract = {
        "version": "manuscript_final_v4",
        "status": "complete",
        "seeds": list(SEEDS),
        "primary_rq2_cells": [f"{backbone}--{dataset.upper()}" for backbone, dataset in PRIMARY_CELLS],
        "primary_rq2_pair_rows": 18,
        "primary_rq1_cells": [f"{backbone}--{dataset.upper()}" for backbone, dataset in PRIMARY_CELLS],
        "new_closure_packet": "*_tmlr_table2_final",
        "new_closure_artifact_count": len(artifacts),
        "source_artifacts": source_artifacts,
        "test_metrics_used_for_selection": False,
        "selection_metric": "cohen_kappa",
        "held_out_metrics": ["balanced_accuracy", "macro_f1"],
        "parameter_matching": "adapter module parameter count only; classifier excluded",
        "excluded_legacy_records": "Older heterogeneous LaBraM FACED/ISRUC native/probe/control rows are not pooled with the fresh closure packet.",
    }
    (OUTPUT / "analysis_contract.json").write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    sums = []
    for path in sorted(OUTPUT.iterdir()):
        if path.name == "SHA256SUMS" or not path.is_file():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        sums.append(f"{digest}  {path.name}")
    (OUTPUT / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
