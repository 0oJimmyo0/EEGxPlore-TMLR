"""CBraMod-only reader for serialized TUEV preprocessing output."""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import Dataset

EXPECTED_RAW_SHAPE = (16, 1000)
EXPECTED_SHAPE = (16, 5, 200)
EXPECTED_RAW_LABELS = tuple(range(1, 7))
EXPECTED_LABELS = tuple(range(6))
EXPECTED_SCALE_DIVISOR = 100.0
SPLITS = ("train", "val", "test")


def _validate_record(record: Any, path: Path) -> tuple[np.ndarray, int]:
    if not isinstance(record, dict) or "signal" not in record or "label" not in record:
        raise ValueError(f"TUEV record {path} must contain signal and label")
    signal = np.asarray(record["signal"])
    if tuple(signal.shape) != EXPECTED_RAW_SHAPE:
        raise ValueError(f"TUEV record {path} shape {tuple(signal.shape)} != {EXPECTED_RAW_SHAPE}")
    if not np.isfinite(signal).all():
        raise ValueError(f"TUEV record {path} contains non-finite values")
    label_values = np.asarray(record["label"]).reshape(-1)
    if label_values.size != 1:
        raise ValueError(f"TUEV record {path} label is not scalar")
    raw_label = int(label_values[0])
    if raw_label not in EXPECTED_RAW_LABELS:
        raise ValueError(f"TUEV record {path} has raw label {raw_label}, expected 1..6")
    return signal, raw_label - 1


class TUEVDataset(Dataset):
    """Lazy, sorted, validated TUEV split reader."""

    def __init__(self, root: str | Path, mode: str, input_scale_divisor: float = 100.0):
        self.root = Path(root).expanduser().resolve()
        self.mode = str(mode).strip().lower()
        self.input_scale_divisor = float(input_scale_divisor)
        if self.mode not in SPLITS:
            raise ValueError(f"Unknown TUEV split {self.mode!r}")
        if self.input_scale_divisor != EXPECTED_SCALE_DIVISOR:
            raise ValueError("TUEV requires exactly one /100 input scale")
        split_name = {"train": "processed_train", "val": "processed_eval", "test": "processed_test"}[self.mode]
        self.split_dir = self.root / split_name
        if not self.split_dir.is_dir():
            raise FileNotFoundError(f"TUEV split directory not found: {self.split_dir}")
        self.files = sorted(self.split_dir.glob("*.pkl"), key=lambda path: path.name)
        if not self.files:
            raise ValueError(f"TUEV split {self.mode!r} is empty")
        with self.files[0].open("rb") as handle:
            _validate_record(pickle.load(handle), self.files[0])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.files[int(index)]
        with path.open("rb") as handle:
            record = pickle.load(handle)
        signal, label = _validate_record(record, path)
        signal = np.asarray(signal, dtype=np.float32).reshape(EXPECTED_SHAPE) / self.input_scale_divisor
        return torch.from_numpy(signal), torch.tensor(label, dtype=torch.long)

    def split_metadata(self) -> Dict[str, Any]:
        digest = hashlib.sha256()
        for path in self.files:
            encoded = path.name.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big")); digest.update(encoded)
        prefixes = sorted({path.name.split("_", 1)[0] for path in self.files})
        return {
            "mode": self.mode, "directory": str(self.split_dir), "sample_count": len(self.files),
            "filename_sha256": digest.hexdigest(), "sample_shape_raw": list(EXPECTED_RAW_SHAPE),
            "sample_shape_model": list(EXPECTED_SHAPE), "input_scale_divisor": EXPECTED_SCALE_DIVISOR,
            "filename_prefix_count": len(prefixes), "filename_prefixes": prefixes,
        }


def make_tuev_datasets(root: str | Path, input_scale_divisor: float = 100.0) -> dict[str, TUEVDataset]:
    return {mode: TUEVDataset(root, mode, input_scale_divisor=input_scale_divisor) for mode in SPLITS}
