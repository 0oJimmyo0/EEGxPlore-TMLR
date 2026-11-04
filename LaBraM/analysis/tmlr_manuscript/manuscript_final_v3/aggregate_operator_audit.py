#!/usr/bin/env python3
"""Aggregate the targeted TUEV branch-local-MLP control audit.

The audit is intentionally separate from ``manuscript_final_v2``.  The
branch-local MLP runs are a secondary follow-up: they preserve the native
branch layout, remove cross-axis interaction, and are not pooled into the
primary native-versus-axis-agnostic estimand.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev


WORKSPACE = Path(__file__).resolve().parents[4]
OUTPUT = Path(__file__).resolve().parent
SEEDS = (42, 1024, 3407)


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def cb_metrics(path: Path) -> dict[str, float | int]:
    payload = read_json(path / "test_metrics.json")
    trainability = read_json(path / "trainability_report.json")
    counts = trainability["component_trainable_parameter_counts"]
    return {
        "balanced_accuracy": float(payload["balanced_accuracy"]),
        "macro_f1": float(payload["macro_f1"]),
        "selected_epoch": int(payload["selected_epoch"]),
        "adapter_params": int(counts["adapter"] + counts.get("adapter_scalar", 0)),
    }


def labram_selected_epoch(path: Path) -> int:
    records = []
    with (path / "log.txt").open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "epoch" in record and "val_cohen_kappa" in record:
                records.append(record)
    if not records:
        raise RuntimeError(f"No validation trajectory found in {path / 'log.txt'}")
    return int(max(records, key=lambda item: float(item["val_cohen_kappa"]))["epoch"])


def labram_metrics(path: Path) -> dict[str, float | int]:
    payload = read_json(path / "final_test.json")["primary_test"]
    per_class_f1 = [float(value) for value in payload["per_class_f1"]]
    run_config = read_json(path / "run_config.json")
    return {
        "balanced_accuracy": float(payload["balanced_accuracy"]),
        "macro_f1": mean(per_class_f1),
        "selected_epoch": labram_selected_epoch(path),
        # Use the realized count recorded by the run, rather than the target
        # budget used to choose the bottleneck.  The branch-local MLP has the
        # same bottleneck as the native LaBraM condition but a slightly
        # different realized parameter count.
        "adapter_params": int(
            run_config.get("adapter_parameter_count", run_config.get("target_adapter_params", 0))
        ),
    }


def metric_loader(backbone: str):
    return cb_metrics if backbone == "CBraMod" else labram_metrics


def source_path(backbone: str, condition: str, seed: int) -> Path:
    if backbone == "CBraMod":
        names = {
            "native": f"tuev_cbramod_frozen_channel_patch_s{seed}_r64_lr1e-4_b8_e20",
            "axis_agnostic": f"tuev_cbramod_frozen_axisblind_s{seed}_r148_lr1e-4_b8_e20",
            "branch_local_mlp": (
                f"tuev_cbramod_frozen_axis_decomposed_mlp_s{seed}_"
                "r147_lr1e-4_h2e-4_a5e-5_b8_e20"
            ),
        }
        return WORKSPACE / "CBraMod/results/tuev" / names[condition]

    names = {
        "native": f"tuev_labram_frozen_channel_patch_lr5e-4_s{seed}_b64_e15_scale100",
        "axis_agnostic": f"tuev_labram_axis_blind_b213_s{seed}_b64_lr5e-4_e15_scale100",
        "branch_local_mlp": (
            f"tuev_labram_frozen_axis_decomposed_mlp_s{seed}_"
            "b106_lr5e-4_b64_e15_scale100"
        ),
    }
    return WORKSPACE / "LaBraM/checkpoints" / names[condition]


def fmt(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:.9f}"


def aggregate(values: list[float]) -> tuple[float, float]:
    return mean(values), stdev(values) if len(values) > 1 else math.nan


def main() -> None:
    rows: list[dict[str, object]] = []
    for backbone in ("CBraMod", "LaBraM"):
        loader = metric_loader(backbone)
        for seed in SEEDS:
            metrics = {}
            paths = {}
            for condition in ("native", "branch_local_mlp", "axis_agnostic"):
                path = source_path(backbone, condition, seed)
                if not path.is_dir():
                    raise FileNotFoundError(path)
                metrics[condition] = loader(path)
                paths[condition] = path

            native = metrics["native"]
            branch = metrics["branch_local_mlp"]
            agnostic = metrics["axis_agnostic"]
            rows.append(
                {
                    "backbone": backbone,
                    "dataset": "TUEV",
                    "seed": seed,
                    "native_ba": native["balanced_accuracy"],
                    "branch_local_mlp_ba": branch["balanced_accuracy"],
                    "native_minus_branch_local_mlp_ba": native["balanced_accuracy"] - branch["balanced_accuracy"],
                    "native_macro_f1": native["macro_f1"],
                    "branch_local_mlp_macro_f1": branch["macro_f1"],
                    "native_minus_branch_local_mlp_macro_f1": native["macro_f1"] - branch["macro_f1"],
                    "axis_agnostic_ba": agnostic["balanced_accuracy"],
                    "branch_local_mlp_minus_axis_agnostic_ba": branch["balanced_accuracy"] - agnostic["balanced_accuracy"],
                    "axis_agnostic_macro_f1": agnostic["macro_f1"],
                    "branch_local_mlp_minus_axis_agnostic_macro_f1": branch["macro_f1"] - agnostic["macro_f1"],
                    "branch_local_mlp_selected_epoch": branch["selected_epoch"],
                    "branch_local_mlp_adapter_params": branch["adapter_params"],
                    "native_artifact": str(paths["native"].relative_to(WORKSPACE)),
                    "branch_local_mlp_artifact": str(paths["branch_local_mlp"].relative_to(WORKSPACE)),
                    "axis_agnostic_artifact": str(paths["axis_agnostic"].relative_to(WORKSPACE)),
                }
            )

    per_seed_fields = list(rows[0].keys())
    with (OUTPUT / "operator_audit_seed_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_seed_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(value) if isinstance(value, (float, int)) else value for key, value in row.items()})

    summary_rows = []
    for backbone in ("CBraMod", "LaBraM"):
        group = [row for row in rows if row["backbone"] == backbone]
        values = {}
        for key in (
            "native_ba",
            "branch_local_mlp_ba",
            "native_minus_branch_local_mlp_ba",
            "native_macro_f1",
            "branch_local_mlp_macro_f1",
            "native_minus_branch_local_mlp_macro_f1",
            "branch_local_mlp_minus_axis_agnostic_ba",
            "branch_local_mlp_minus_axis_agnostic_macro_f1",
        ):
            values[key] = [float(row[key]) for row in group]
        summary = {"backbone": backbone, "dataset": "TUEV", "seeds": len(group)}
        for key, items in values.items():
            avg, spread = aggregate(items)
            summary[f"{key}_mean"] = avg
            summary[f"{key}_sd"] = spread
        summary["branch_local_mlp_better_ba_count"] = sum(value < 0 for value in values["native_minus_branch_local_mlp_ba"])
        summary["branch_local_mlp_better_macro_f1_count"] = sum(value < 0 for value in values["native_minus_branch_local_mlp_macro_f1"])
        summary_rows.append(summary)

    summary_fields = list(summary_rows[0].keys())
    with (OUTPUT / "operator_audit_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({key: fmt(value) if isinstance(value, (float, int)) else value for key, value in row.items()})

    contract = {
        "analysis": "targeted_branch_local_mlp_operator_audit",
        "status": "complete",
        "dataset": "TUEV",
        "seeds": list(SEEDS),
        "primary_selection_metric": "validation_cohen_kappa",
        "test_metrics_used_for_selection": False,
        "primary_estimand_unchanged": True,
        "interpretation": "secondary robustness audit; native attention versus branch-local no-mixing MLP",
        # Record every artifact used by the audit, not only the newly added
        # branch-local controls.  This makes the contract sufficient to
        # reproduce the native, branch-local, and axis-agnostic comparisons.
        "source_artifacts": sorted(
            {
                row[field]
                for row in rows
                for field in (
                    "native_artifact",
                    "branch_local_mlp_artifact",
                    "axis_agnostic_artifact",
                )
            }
        ),
        "outputs": [
            "operator_audit_seed_results.csv",
            "operator_audit_summary.csv",
        ],
    }
    with (OUTPUT / "operator_audit_contract.json").open("w", encoding="utf-8") as handle:
        json.dump(contract, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
