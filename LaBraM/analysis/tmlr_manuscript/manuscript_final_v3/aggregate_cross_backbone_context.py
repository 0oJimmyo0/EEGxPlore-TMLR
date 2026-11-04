#!/usr/bin/env python3
"""Aggregate LaBraM FACED/ISRUC results retained as supporting context.

These rows are intentionally separate from the primary v2 estimands.  They
contain heterogeneous historical and trainability regimes, so this table is
for transparent reporting rather than a cross-dataset causal aggregate.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = Path(__file__).resolve().parent
CHECKPOINTS = ROOT / "LaBraM/checkpoints"
SEEDS = (42, 1024, 3407)


CONDITIONS = {
    "FACED": {
        "frozen_probe": {
            "label": "Frozen probe",
            "regime": "frozen",
            "qualification": "legacy frozen packet",
            "names": [f"faced_reviewer_r2_frozen_dense_seed{s}" for s in SEEDS],
        },
        "native_patch": {
            "label": "Native patch",
            "regime": "frozen",
            "qualification": "legacy full-width packet",
            "names": [f"faced_reviewer_r2_frozen_patch_seed{s}" for s in SEEDS],
        },
        "generic": {
            "label": "Generic bottleneck",
            "regime": "frozen",
            "qualification": "legacy frozen packet",
            "names": [f"faced_reviewer_r2_frozen_generic_seed{s}" for s in SEEDS],
        },
        "lora": {
            "label": "LoRA",
            "regime": "LoRA",
            "qualification": "three-seed context",
            "names": [f"faced_reviewer_r2_lora_qkv_r8_seed{s}" for s in SEEDS],
        },
        "upper2": {
            "label": "Upper-2",
            "regime": "upper-layer",
            "qualification": "three-seed context",
            "names": [f"faced_reviewer_r2_upper2_seed{s}" for s in SEEDS],
        },
    },
    "ISRUC": {
        "frozen_probe": {
            "label": "Frozen probe",
            "regime": "frozen",
            "qualification": "legacy frozen packet",
            "names": [
                "isruc_dense_bls0_lr2e-4_s42_b16_e30",
                "isruc_dense_bls0_s1024_lr2e-4_b16_e30",
                "isruc_dense_bls0_s3407_lr2e-4_b16_e30",
            ],
        },
        "native_patch": {
            "label": "Native patch",
            "regime": "frozen",
            "qualification": "legacy full-width packet",
            "names": [
                "isruc_patch_util_bls0_lr2e-4_s42_b16_e30",
                "isruc_patch_util_bls0_s1024_lr2e-4_b16_e30",
                "isruc_patch_util_bls0_s3407_lr2e-4_b16_e30",
            ],
        },
        "axis_blind_b400": {
            "label": "Axis-blind $b=400$",
            "regime": "frozen",
            "qualification": "strict patch-control",
            "names": [
                f"isruc_labram_axis_blind_b400_s{s}_b16_lr2e-4_e30_retry_seqfix"
                for s in SEEDS
            ],
        },
        "native_channel_patch_rank32": {
            "label": "Native C+P, rank 32",
            "regime": "trainable backbone",
            "qualification": "three-seed context",
            "names": [
                f"isruc_labram_lowrank_a05_channel_patch_s{s}_b16_r32_lr2e-4_e30_test"
                for s in SEEDS
            ],
        },
        "lora": {
            "label": "LoRA",
            "regime": "LoRA",
            "qualification": "three-seed context",
            "names": [
                f"isruc_labram_lora_qkv_r8_s{s}_b16_lr2e-4_e30_test_retry1"
                for s in SEEDS
            ],
        },
        "upper2": {
            "label": "Upper-2",
            "regime": "upper-layer",
            "qualification": "three-seed context",
            "names": [f"isruc_labram_upper2_s{s}_b16_lr2e-4_e30_test" for s in SEEDS],
        },
    },
}


def read_test(path: Path) -> tuple[float, float]:
    payload = json.loads((path / "final_test.json").read_text(encoding="utf-8"))["primary_test"]
    per_class_f1 = payload["per_class_f1"]
    if not isinstance(per_class_f1, list):
        raise TypeError(f"Macro-F1 is not recoverable as per-class values in {path}")
    return float(payload["balanced_accuracy"]), mean(float(value) for value in per_class_f1)


def fmt(value: float | int) -> str:
    return str(value) if isinstance(value, int) else f"{value:.9f}"


def main() -> None:
    seed_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for dataset, conditions in CONDITIONS.items():
        for condition, spec in conditions.items():
            values: list[tuple[float, float]] = []
            for seed, name in zip(SEEDS, spec["names"]):
                path = CHECKPOINTS / name
                if not path.is_dir():
                    raise FileNotFoundError(path)
                ba, f1 = read_test(path)
                values.append((ba, f1))
                seed_rows.append(
                    {
                        "backbone": "LaBraM",
                        "dataset": dataset,
                        "condition": condition,
                        "condition_label": spec["label"],
                        "regime": spec["regime"],
                        "seed": seed,
                        "balanced_accuracy": ba,
                        "macro_f1": f1,
                        "qualification": spec["qualification"],
                        "source_artifact": str(path.relative_to(ROOT)),
                    }
                )
            ba_values = [value[0] for value in values]
            f1_values = [value[1] for value in values]
            summary_rows.append(
                {
                    "backbone": "LaBraM",
                    "dataset": dataset,
                    "condition": condition,
                    "condition_label": spec["label"],
                    "regime": spec["regime"],
                    "seeds": len(values),
                    "balanced_accuracy_mean": mean(ba_values),
                    "balanced_accuracy_sd": stdev(ba_values),
                    "macro_f1_mean": mean(f1_values),
                    "macro_f1_sd": stdev(f1_values),
                    "qualification": spec["qualification"],
                }
            )

    for filename, rows in (
        ("cross_backbone_context_seed_results.csv", seed_rows),
        ("cross_backbone_context_summary.csv", summary_rows),
    ):
        with (OUTPUT / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow({key: fmt(value) if isinstance(value, (float, int)) else value for key, value in row.items()})

    contract = {
        "analysis": "supporting_labram_cross_backbone_context",
        "status": "complete",
        "backbone": "LaBraM",
        "datasets": ["FACED", "ISRUC"],
        "seeds": list(SEEDS),
        "primary_estimand_unchanged": True,
        "interpretation": "supporting context only; heterogeneous historical and trainability regimes",
        "outputs": [
            "cross_backbone_context_seed_results.csv",
            "cross_backbone_context_summary.csv",
        ],
    }
    (OUTPUT / "cross_backbone_context_contract.json").write_text(
        json.dumps(contract, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
