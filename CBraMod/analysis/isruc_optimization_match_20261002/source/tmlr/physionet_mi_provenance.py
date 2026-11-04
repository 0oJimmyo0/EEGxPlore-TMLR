"""Serialized PhysioNet-MI provenance and integrity audit for CBraMod."""

from __future__ import annotations

import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Dict

import lmdb
import numpy as np

from datasets.physionet_mi_dataset import (
    CHANNELS,
    EXPECTED_LABELS,
    EXPECTED_SCALE_DIVISOR,
    EXPECTED_SHAPE,
    _key_bytes,
    _subject,
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _key_digest(keys: list[Any]) -> str:
    digest = hashlib.sha256()
    for key in keys:
        encoded = _key_bytes(key)
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _record_summary(txn, keys: list[Any]) -> dict[str, Any]:
    labels = Counter()
    subjects: set[str] = set()
    shapes: set[tuple[int, ...]] = set()
    finite = True
    value_min = float("inf")
    value_max = float("-inf")
    total = 0
    sum_value = 0.0
    sum_square = 0.0
    missing = []
    for key in keys:
        raw = txn.get(_key_bytes(key))
        if raw is None:
            missing.append(str(key))
            continue
        record = pickle.loads(raw)
        sample = np.asarray(record.get("sample")) if isinstance(record, dict) else np.asarray([])
        label = np.asarray(record.get("label")).reshape(-1) if isinstance(record, dict) and "label" in record else np.asarray([])
        shapes.add(tuple(sample.shape))
        finite = finite and bool(np.isfinite(sample).all()) and bool(np.isfinite(label).all())
        if sample.size:
            values = np.asarray(sample, dtype=np.float64)
            value_min = min(value_min, float(values.min()))
            value_max = max(value_max, float(values.max()))
            total += int(values.size)
            sum_value += float(values.sum())
            sum_square += float(np.square(values).sum())
        if label.size == 1:
            labels[int(label[0])] += 1
        subjects.add(_subject(key))
    mean = sum_value / total if total else None
    variance = max(sum_square / total - mean * mean, 0.0) if total and mean is not None else None
    return {
        "sample_count": len(keys),
        "missing_record_count": len(missing),
        "missing_records": missing[:10],
        "shape_set": [list(shape) for shape in sorted(shapes)],
        "finite_values": finite,
        "label_counts": {str(label): int(labels[label]) for label in sorted(labels)},
        "subjects": sorted(subjects),
        "subject_count": len(subjects),
        "value_min": value_min,
        "value_max": value_max,
        "value_mean": mean,
        "value_std": variance ** 0.5 if variance is not None else None,
        "value_count": total,
    }


def audit_physionet_mi(
    root: str | Path,
    repo_root: str | Path,
    channel_manifest: str | Path,
    source_preprocessing: str | Path,
    compute_lmdb_hash: bool = True,
) -> Dict[str, Any]:
    root = Path(root).expanduser().resolve()
    repo_root = Path(repo_root).expanduser().resolve()
    manifest_path = Path(channel_manifest).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path
    source_path = Path(source_preprocessing).expanduser()
    if not source_path.is_absolute():
        source_path = repo_root / source_path
    if not root.is_dir():
        raise FileNotFoundError(f"PhysioNet-MI root not found: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("channels") != CHANNELS:
        raise ValueError("PhysioNet-MI manifest does not match the serialized channel row order")
    if manifest.get("stored_shape") != list(EXPECTED_SHAPE):
        raise ValueError("PhysioNet-MI manifest shape mismatch")
    if float(manifest.get("input_scale_divisor")) != EXPECTED_SCALE_DIVISOR:
        raise ValueError("PhysioNet-MI manifest scale mismatch")

    env = lmdb.open(str(root), readonly=True, lock=False, readahead=False, meminit=False, max_readers=512)
    with env.begin(write=False) as txn:
        raw_index = txn.get(b"__keys__")
        if raw_index is None:
            raise KeyError("PhysioNet-MI LMDB is missing __keys__")
        split_index = pickle.loads(raw_index)
        if not isinstance(split_index, dict) or set(split_index) != {"train", "val", "test"}:
            raise ValueError(f"Unexpected PhysioNet-MI split index: {split_index!r}")
        split_keys = {mode: list(split_index[mode]) for mode in ("train", "val", "test")}
        split_summary = {mode: _record_summary(txn, split_keys[mode]) for mode in split_keys}

    all_keys = [key for keys in split_keys.values() for key in keys]
    split_subjects = {mode: set(split_summary[mode]["subjects"]) for mode in split_summary}
    overlaps = {
        "train_val": sorted(split_subjects["train"] & split_subjects["val"]),
        "train_test": sorted(split_subjects["train"] & split_subjects["test"]),
        "val_test": sorted(split_subjects["val"] & split_subjects["test"]),
    }
    duplicate_keys = sorted(str(key) for key, count in Counter(all_keys).items() if count > 1)
    expected_subjects = {
        "train": [f"S{i:03d}" for i in range(1, 71)],
        "val": [f"S{i:03d}" for i in range(71, 90)],
        "test": [f"S{i:03d}" for i in range(90, 110)],
    }
    for mode in split_summary:
        if split_summary[mode]["subjects"] != expected_subjects[mode]:
            raise ValueError(
                f"Unexpected {mode} subjects: {split_summary[mode]['subjects'][:5]} ..."
            )
        if split_summary[mode]["shape_set"] != [list(EXPECTED_SHAPE)]:
            raise ValueError(f"Unexpected {mode} shapes: {split_summary[mode]['shape_set']}")
        if not split_summary[mode]["finite_values"] or split_summary[mode]["missing_record_count"]:
            raise ValueError(f"PhysioNet-MI integrity failure in {mode}: {split_summary[mode]}")
        labels = set(int(label) for label in split_summary[mode]["label_counts"])
        if labels != set(EXPECTED_LABELS):
            raise ValueError(f"Unexpected {mode} labels: {labels}")
    if any(overlaps.values()) or duplicate_keys:
        raise ValueError(f"PhysioNet-MI split overlap: {overlaps}; duplicate keys: {duplicate_keys[:5]}")

    data_file = root / "data.mdb"
    report = {
        "dataset": "PhysioNet-MI",
        "absolute_dataset_path": str(root),
        "lmdb_data_file": str(data_file),
        "lmdb_data_file_size": data_file.stat().st_size,
        "lmdb_sha256": sha256_file(data_file) if compute_lmdb_hash else None,
        "split_index_sha256": _key_digest(all_keys),
        "split_key_sha256": {mode: _key_digest(split_keys[mode]) for mode in split_keys},
        "split_summary": split_summary,
        "subject_overlap": overlaps,
        "duplicate_key_check": not duplicate_keys,
        "stored_shape": list(EXPECTED_SHAPE),
        "model_shape": list(EXPECTED_SHAPE),
        "label_vocabulary": list(EXPECTED_LABELS),
        "channel_order": list(CHANNELS),
        "input_scale_divisor": EXPECTED_SCALE_DIVISOR,
        "sampling_rate_hz": 200,
        "preprocessing": {
            "source_script": str(source_path),
            "source_script_sha256": sha256_file(source_path),
            "selected_runs": ["R04", "R06", "R08", "R10", "R12", "R14"],
            "average_reference": True,
            "high_pass_hz": 0.3,
            "notch_hz": 60.0,
            "resampling_hz": 200,
            "epoch_seconds": 4.0,
            "retained_samples": 800,
            "event_1_excluded": True,
            "subject_split": "sorted subjects 70/19/20",
        },
        "serialized_protocol_limitation": "Raw EDF tree is unavailable; audit validates the serialized preprocessing contract.",
        "finite_value_check": True,
        "subject_overlap_check": not any(overlaps.values()),
        "audit_scope": "CBraMod-only read-only serialized-data audit",
    }
    env.close()
    return report
