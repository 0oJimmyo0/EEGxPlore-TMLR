"""Resolved configuration for the isolated CBraMod TUEV runner."""

from __future__ import annotations

import ast
import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None

SUPPORTED_METHODS = {
    "full_finetune", "frozen_probe", "interaction_aligned", "native_full_finetune",
    "axis_decomposed_mlp", "generic_bottleneck", "lora", "upper_k_finetune", "axis_blind",
}
ADAPTER_TYPES = {"channel", "patch", "channel_patch"}
SELECTION_METRICS = {"cohen_kappa", "balanced_accuracy", "macro_f1"}
DEFAULTS: Dict[str, Any] = {
    "dataset_name": "TUEV", "dataset_path": "prepared_data/TUEV",
    "audit_path": "results/tuev/audits/tuev_serialized_audit.json",
    "source_preprocessing": "preprocessing/preprocessing_tuev.py",
    "checkpoint": "weights/pretrained_weights.pth",
    "output_root": "results/tuev", "method": "full_finetune", "adapter_type": None,
    "adapter_bottleneck": 64, "adapter_heads": 4, "adapter_dropout": 0.0,
    "adapter_init_alpha": 0.01, "adapter_gamma": 1.0, "adapter_zero_init_output": True,
    "seed": 42, "head_seed": 10042, "loader_seed": 20042, "adapter_seed": 30042,
    "generic_bottleneck": 64, "axis_blind_bottleneck": 148,
    # Two CBraMod branch-wise MLPs at b=147 contain 59,696 parameters,
    # matching the native/control adapter capacities within 0.5%.
    "axis_decomposed_mlp_bottleneck": 147, "upper_k": 2,
    "lora_rank": 8, "lora_alpha": 16.0, "batch_size": 8, "epochs": 20,
    "warmup_epochs": 0,
    "max_train_batches": None, "max_val_batches": None, "max_test_batches": None,
    "num_workers": 0, "lr": 2e-4, "head_lr": 2e-4, "adapter_lr": 5e-5,
    "weight_decay": 0.05, "head_weight_decay": 0.05, "adapter_weight_decay": 0.05,
    "adapter_alpha_weight_decay": 0.0, "adapter_alpha_lr_scale": 1.0,
    "lora_lr": 5e-5, "upper_lr": 2e-4,
    "scheduler_eta_min": 1e-6, "label_smoothing": 0.1, "clip_grad": 1.0,
    "selection_metric": "cohen_kappa", "classifier": "all_patch_reps", "dropout": 0.1,
    "input_scale_divisor": 100.0, "num_classes": 6, "device": "auto", "run_id": None,
    "save_test": True, "audit_only": False, "smoke": False, "overwrite": False,
}


@dataclass
class TUEVTMLRConfig:
    values: Dict[str, Any]
    def __getattr__(self, name: str) -> Any:
        try: return self.values[name]
        except KeyError as exc: raise AttributeError(name) from exc
    def to_dict(self) -> Dict[str, Any]: return copy.deepcopy(self.values)


def _scalar(value: str) -> Any:
    value = value.strip()
    if value.lower() in {"null", "none", "~"}: return None
    if value.lower() in {"true", "false"}: return value.lower() == "true"
    try: return ast.literal_eval(value)
    except (ValueError, SyntaxError): return value


def _load_file(path: Optional[str]) -> Dict[str, Any]:
    if not path: return {}
    path_obj = Path(path)
    if not path_obj.is_file(): raise FileNotFoundError(path_obj)
    with path_obj.open("r", encoding="utf-8") as handle:
        if path_obj.suffix.lower() == ".json": payload = json.load(handle)
        elif yaml is not None: payload = yaml.safe_load(handle)
        else:
            payload = {}
            for number, raw in enumerate(handle.read().splitlines(), 1):
                line = raw.strip()
                if not line or line.startswith("#"): continue
                if ":" not in line: raise ValueError(f"Unsupported YAML line {number}: {raw!r}")
                key, value = line.split(":", 1); payload[key.strip()] = _scalar(value)
    return payload or {}


def validate_config(values: Dict[str, Any]) -> Dict[str, Any]:
    resolved = copy.deepcopy(values)
    if str(resolved.get("dataset_name", "TUEV")).upper().replace("_", "-") != "TUEV":
        raise ValueError("This runner only accepts TUEV")
    resolved["dataset_name"] = "TUEV"
    method = str(resolved["method"]).strip().lower()
    if method not in SUPPORTED_METHODS: raise ValueError(f"Unknown method {method!r}")
    resolved["method"] = method
    if method in {"interaction_aligned", "native_full_finetune", "axis_decomposed_mlp"}:
        if resolved.get("adapter_type") not in ADAPTER_TYPES: raise ValueError("native method requires adapter_type")
    elif resolved.get("adapter_type") is not None:
        raise ValueError(f"method={method} must not request adapter_type")
    if resolved["selection_metric"] not in SELECTION_METRICS: raise ValueError("Invalid selection metric")
    if int(resolved["num_classes"]) != 6: raise ValueError("TUEV requires six classes")
    if float(resolved["input_scale_divisor"]) != 100.0: raise ValueError("TUEV requires /100 scaling")
    if int(resolved["batch_size"]) <= 0 or int(resolved["epochs"]) <= 0: raise ValueError("batch_size/epochs must be positive")
    if int(resolved["warmup_epochs"]) < 0 or int(resolved["warmup_epochs"]) >= int(resolved["epochs"]):
        raise ValueError("warmup_epochs must be nonnegative and smaller than epochs")
    if float(resolved["adapter_alpha_lr_scale"]) <= 0:
        raise ValueError("adapter_alpha_lr_scale must be positive")
    return resolved


def build_config(config_path: Optional[str], overrides: Optional[Dict[str, Any]] = None) -> TUEVTMLRConfig:
    values = copy.deepcopy(DEFAULTS); values.update(_load_file(config_path))
    if overrides: values.update({key: value for key, value in overrides.items() if value is not None})
    return TUEVTMLRConfig(validate_config(values))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CBraMod TUEV TMLR runner")
    parser.add_argument("--config", default=None)
    for field in ("dataset_path", "audit_path", "source_preprocessing", "checkpoint", "output_root", "method", "adapter_type", "device", "run_id", "selection_metric"):
        parser.add_argument(f"--{field.replace('_','-')}", dest=field, default=None)
    for field in ("adapter_bottleneck", "adapter_heads", "generic_bottleneck", "axis_blind_bottleneck", "axis_decomposed_mlp_bottleneck", "upper_k", "lora_rank", "seed", "head_seed", "loader_seed", "adapter_seed", "batch_size", "epochs", "warmup_epochs", "max_train_batches", "max_val_batches", "max_test_batches", "num_workers"):
        parser.add_argument(f"--{field.replace('_','-')}", dest=field, default=None, type=int)
    for field in ("adapter_dropout", "adapter_init_alpha", "adapter_gamma", "lr", "head_lr", "adapter_lr", "weight_decay", "head_weight_decay", "adapter_weight_decay", "adapter_alpha_weight_decay", "adapter_alpha_lr_scale", "lora_lr", "upper_lr", "lora_alpha", "scheduler_eta_min", "dropout", "input_scale_divisor", "label_smoothing", "clip_grad"):
        parser.add_argument(f"--{field.replace('_','-')}", dest=field, default=None, type=float)
    for field in ("adapter_zero_init_output", "save_test", "overwrite"):
        parser.add_argument(f"--{field.replace('_','-')}", dest=field, default=None, action=argparse.BooleanOptionalAction)
    parser.add_argument("--audit-only", default=None, action="store_true")
    parser.add_argument("--smoke", default=None, action="store_true")
    return parser


def parse_config(argv: Optional[Iterable[str]] = None) -> TUEVTMLRConfig:
    parser = build_parser(); args = parser.parse_args(argv)
    overrides = vars(args).copy(); config_path = overrides.pop("config")
    return build_config(config_path, overrides)
