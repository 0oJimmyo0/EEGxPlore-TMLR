"""CBraMod-only PhysioNet-MI LMDB loader.

The serialized records follow the shared EEG preprocessing preprocessing contract and are
stored as raw microvolt tensors ``[64, 4, 200]``.  This loader applies the
single downstream ``/100`` scale and preserves the explicit channel order.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset


EXPECTED_SHAPE = (64, 4, 200)
EXPECTED_LABELS = (0, 1, 2, 3)
EXPECTED_SCALE_DIVISOR = 100.0
CHANNELS = [
    "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "C5", "C3", "C1", "CZ", "C2",
    "C4", "C6", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "FP1", "FPZ", "FP2",
    "AF7", "AF3", "AFZ", "AF4", "AF8", "F7", "F5", "F3", "F1", "FZ", "F2", "F4",
    "F6", "F8", "FT7", "FT8", "T7", "T8", "T9", "T10", "TP7", "TP8", "P7", "P5",
    "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "PO7", "PO3", "POZ", "PO4", "PO8",
    "O1", "OZ", "O2", "IZ",
]


def _key_bytes(key: Any) -> bytes:
    if isinstance(key, bytes):
        return key
    if isinstance(key, str):
        return key.encode("utf-8")
    raise TypeError(f"LMDB key must be str or bytes, got {type(key).__name__}")


def _subject(key: Any) -> str:
    text = key.decode(errors="replace") if isinstance(key, bytes) else str(key)
    return text.split("R", 1)[0]


class PhysioNetMIDataset(Dataset):
    """Lazy, validated split reader for the serialized PhysioNet-MI LMDB."""

    def __init__(self, root: str | Path, mode: str, input_scale_divisor: float = 100.0):
        self.root = Path(root).expanduser().resolve()
        self.mode = str(mode).strip().lower()
        self.input_scale_divisor = float(input_scale_divisor)
        if self.input_scale_divisor != EXPECTED_SCALE_DIVISOR:
            raise ValueError("PhysioNet-MI requires exactly one /100 input scale")
        if self.mode not in {"train", "val", "test"}:
            raise ValueError(f"Unknown PhysioNet-MI split {self.mode!r}")
        self._env = None
        with self._open().begin(write=False) as txn:
            raw_keys = txn.get(b"__keys__")
        if raw_keys is None:
            raise KeyError(f"PhysioNet-MI LMDB is missing __keys__: {self.root}")
        split_index = pickle.loads(raw_keys)
        if not isinstance(split_index, dict) or self.mode not in split_index:
            raise KeyError(f"PhysioNet-MI LMDB has no split {self.mode!r}")
        self.keys = list(split_index[self.mode])
        if not self.keys:
            raise ValueError(f"PhysioNet-MI split {self.mode!r} is empty")
        with self._open().begin(write=False) as txn:
            raw = txn.get(_key_bytes(self.keys[0]))
        if raw is None:
            raise KeyError(f"Missing first PhysioNet-MI record: {self.keys[0]!r}")
        self._validate_record(pickle.loads(raw), self.keys[0])

    def _open(self):
        if self._env is None:
            self._env = lmdb.open(
                str(self.root), readonly=True, lock=False, readahead=False,
                meminit=False, max_readers=512,
            )
        return self._env

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_env"] = None
        return state

    def __len__(self) -> int:
        return len(self.keys)

    def get_ch_names(self) -> list[str]:
        return list(CHANNELS)

    def split_metadata(self) -> Dict[str, Any]:
        digest = hashlib.sha256()
        for key in self.keys:
            encoded = _key_bytes(key)
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        return {
            "mode": self.mode,
            "sample_count": len(self.keys),
            "key_sha256": digest.hexdigest(),
            "sample_shape": list(EXPECTED_SHAPE),
            "subject_count": len({_subject(key) for key in self.keys}),
            "subjects": sorted({_subject(key) for key in self.keys}),
            "channel_names": list(CHANNELS),
            "input_scale_divisor": EXPECTED_SCALE_DIVISOR,
        }

    @staticmethod
    def _validate_record(record: Any, key: Any) -> tuple[int, ...]:
        if not isinstance(record, dict) or "sample" not in record or "label" not in record:
            raise ValueError(f"Record {key!r} must contain sample and label")
        sample = np.asarray(record["sample"])
        if tuple(sample.shape) != EXPECTED_SHAPE:
            raise ValueError(f"Record {key!r} shape {tuple(sample.shape)} != {EXPECTED_SHAPE}")
        if not np.isfinite(sample).all():
            raise ValueError(f"Record {key!r} contains non-finite signal values")
        labels = np.asarray(record["label"]).reshape(-1)
        if labels.size != 1 or int(labels[0]) not in EXPECTED_LABELS:
            raise ValueError(f"Record {key!r} has invalid scalar label {labels.tolist()}")
        return tuple(sample.shape)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        key = self.keys[int(index)]
        with self._open().begin(write=False) as txn:
            raw = txn.get(_key_bytes(key))
        if raw is None:
            raise KeyError(f"Missing PhysioNet-MI record {key!r}")
        record = pickle.loads(raw)
        self._validate_record(record, key)
        signal = np.asarray(record["sample"], dtype=np.float32) / self.input_scale_divisor
        label = np.int64(record["label"])
        return torch.from_numpy(signal), torch.tensor(label, dtype=torch.long)


def make_physionet_mi_datasets(root: str | Path, input_scale_divisor: float = 100.0) -> dict[str, PhysioNetMIDataset]:
    return {
        mode: PhysioNetMIDataset(root, mode, input_scale_divisor=input_scale_divisor)
        for mode in ("train", "val", "test")
    }
