"""Streaming provenance and integrity audit for serialized TUEV."""

from __future__ import annotations

import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Dict

import numpy as np

from datasets.tuev_tmlr_dataset import (
    EXPECTED_RAW_LABELS, EXPECTED_RAW_SHAPE, EXPECTED_SCALE_DIVISOR, EXPECTED_SHAPE,
)
from .provenance import sha256_file


def _file_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        encoded = path.name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big")); digest.update(encoded)
    return digest.hexdigest()


def _scan_split(split_dir: Path) -> Dict[str, Any]:
    paths = sorted(split_dir.glob("*.pkl"), key=lambda path: path.name)
    if not paths: raise ValueError(f"Empty TUEV split: {split_dir}")
    labels = Counter(); shapes = Counter(); schema = Counter(); prefixes = set()
    finite = True; missing = 0; raw_min = float("inf"); raw_max = float("-inf")
    raw_sum = 0.0; raw_sq = 0.0; raw_count = 0
    for path in paths:
        prefixes.add(path.name.split("_", 1)[0])
        try:
            with path.open("rb") as handle: record = pickle.load(handle)
        except Exception as exc:
            raise RuntimeError(f"Could not read TUEV record {path}: {exc}") from exc
        if not isinstance(record, dict): raise ValueError(f"TUEV record {path} is not a dict")
        schema.update([tuple(sorted(record.keys()))])
        if "signal" not in record or "label" not in record: raise ValueError(f"TUEV record {path} lacks signal/label")
        signal = np.asarray(record["signal"]); label = np.asarray(record["label"]).reshape(-1)
        shapes[tuple(signal.shape)] += 1
        if tuple(signal.shape) != EXPECTED_RAW_SHAPE: raise ValueError(f"TUEV shape mismatch at {path}: {signal.shape}")
        if label.size != 1: raise ValueError(f"TUEV non-scalar label at {path}")
        label_value = int(label[0]); labels[label_value] += 1
        if label_value not in EXPECTED_RAW_LABELS: raise ValueError(f"TUEV label {label_value} at {path}")
        if not np.isfinite(signal).all(): finite = False
        if signal.size:
            values = np.asarray(signal, dtype=np.float64)
            raw_min = min(raw_min, float(values.min())); raw_max = max(raw_max, float(values.max()))
            raw_sum += float(values.sum()); raw_sq += float(np.square(values).sum()); raw_count += int(values.size)
        del record, signal, label
    mean = raw_sum / raw_count
    variance = max(raw_sq / raw_count - mean * mean, 0.0)
    return {
        "directory": str(split_dir), "sample_count": len(paths), "filename_sha256": _file_digest(paths),
        "prefix_count": len(prefixes), "prefixes": sorted(prefixes), "schema_keys": [list(item) for item in schema],
        "shape_set": [list(shape) for shape in sorted(shapes)], "label_counts": {str(k): int(v) for k, v in sorted(labels.items())},
        "finite_values": finite, "raw_value_min": raw_min, "raw_value_max": raw_max,
        "raw_value_mean": mean, "raw_value_std": variance ** 0.5, "raw_value_count": raw_count,
        "scaled_shape": list(EXPECTED_SHAPE), "input_scale_divisor": EXPECTED_SCALE_DIVISOR,
        "missing_record_count": missing,
    }


def audit_tuev(root: str | Path, repo_root: str | Path, source_preprocessing: str | Path) -> Dict[str, Any]:
    root = Path(root).expanduser().resolve(); repo_root = Path(repo_root).expanduser().resolve()
    if not root.is_dir(): raise FileNotFoundError(root)
    source = Path(source_preprocessing).expanduser(); source = source if source.is_absolute() else repo_root / source
    split_dirs = {"train": root / "processed_train", "val": root / "processed_eval", "test": root / "processed_test"}
    summaries = {name: _scan_split(path) for name, path in split_dirs.items()}
    prefix_sets = {name: set(summary["prefixes"]) for name, summary in summaries.items()}
    prefix_overlap = {
        "train_val": sorted(prefix_sets["train"] & prefix_sets["val"]),
        "train_test": sorted(prefix_sets["train"] & prefix_sets["test"]),
        "val_test": sorted(prefix_sets["val"] & prefix_sets["test"]),
    }
    if any(prefix_overlap.values()): raise ValueError(f"TUEV serialized prefix overlap: {prefix_overlap}")
    for name, summary in summaries.items():
        if summary["shape_set"] != [list(EXPECTED_RAW_SHAPE)] or not summary["finite_values"]:
            raise ValueError(f"TUEV integrity failure in {name}: {summary}")
        if set(map(int, summary["label_counts"])) != set(EXPECTED_RAW_LABELS):
            raise ValueError(f"TUEV label vocabulary failure in {name}: {summary['label_counts']}")
    return {
        "dataset": "TUEV", "absolute_dataset_path": str(root), "split_summary": summaries,
        "prefix_overlap": prefix_overlap, "stored_shape": list(EXPECTED_RAW_SHAPE),
        "model_shape": list(EXPECTED_SHAPE), "raw_label_vocabulary": list(EXPECTED_RAW_LABELS),
        "model_label_vocabulary": list(range(6)), "input_scale_divisor": EXPECTED_SCALE_DIVISOR,
        "channel_count": 16, "sampling_rate_hz": 200, "epoch_seconds": 5.0,
        "source_preprocessing": str(source.resolve()), "source_preprocessing_sha256": sha256_file(source),
        "preprocessing_contract": {
            "source": "preprocessing/preprocessing_tuev.py",
            "differential_montage_channels": 16, "resampling_hz": 200,
            "high_pass_hz": 0.3, "low_pass_hz": 75.0, "notch_hz": 60.0,
            "event_window_seconds": 5.0, "serialized_label_mapping": "raw 1..6 -> model 0..5",
            "split_rule": "sorted filename prefixes from processed_train; first 80% train, last 20% validation; processed_eval copied to test",
        },
        "test_provenance_note": "The serialized processed_test filenames expose class prefixes (bckg/gped/pled/spsw), not the original subject identifiers; subject-disjointness of test cannot be independently reconstructed from these files.",
        "audit_scope": "CBraMod-only read-only serialized-data audit",
    }
