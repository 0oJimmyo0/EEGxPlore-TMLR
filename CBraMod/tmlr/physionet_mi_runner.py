"""Auditable CBraMod-only PhysioNet-MI TMLR runner."""

from __future__ import annotations

import copy
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.physionet_mi_dataset import CHANNELS, EXPECTED_SCALE_DIVISOR, EXPECTED_SHAPE, make_physionet_mi_datasets
from models.lora import lora_diagnostics
from models.model_for_physionet_mi_tmlr import Model

from .artifact_writer import ArtifactWriter
from .eager_optimizer import EagerAdamW, EagerCosineAnnealing
from .faced_runner import _gradient_report, _snapshot, _update_report, resolve_device, set_seed
from .metrics import classification_metrics, selection_value
from .physionet_mi_config import PhysioNetMITMLRConfig
from .physionet_mi_provenance import audit_physionet_mi, sha256_file
from .provenance import collect_environment
from .trainability import apply_trainability_contract, configure_training_modes


def _make_params(config: PhysioNetMITMLRConfig) -> SimpleNamespace:
    values = config.to_dict()
    values.update({
        "foundation_dir": values["checkpoint"],
        "use_pretrained_weights": True,
        "num_of_classes": values["num_classes"],
    })
    return SimpleNamespace(**values)


def _evaluate(model, loader, criterion, device, max_batches=None) -> Dict[str, Any]:
    model.eval()
    targets, predictions, losses = [], [], []
    with torch.no_grad():
        for batch_index, (inputs, labels) in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long().reshape(-1)
            logits = model(inputs)
            losses.append(float(criterion(logits, labels).detach().cpu()))
            targets.extend(labels.cpu().tolist())
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
    return classification_metrics(targets, predictions, losses, num_classes=4)


def _runtime_geometry(model: nn.Module, sample: torch.Tensor, device: torch.device) -> Dict[str, Any]:
    records: Dict[str, Any] = {"input_shape": list(sample.shape)}
    hooks = [
        model.backbone.patch_embedding.register_forward_hook(
            lambda _module, _args, output: records.update({"patch_embedding_shape": list(output.shape)})
        ),
        model.backbone.encoder.register_forward_hook(
            lambda _module, _args, output: records.update({"encoder_output_shape": list(output.shape)})
        ),
    ]
    model.eval()
    with torch.no_grad():
        output = model(sample.to(device))
    for hook in hooks:
        hook.remove()
    if list(sample.shape[1:]) != list(EXPECTED_SHAPE):
        raise RuntimeError(f"PhysioNet-MI input geometry mismatch: {list(sample.shape)}")
    if records.get("encoder_output_shape", [])[1:] != list(EXPECTED_SHAPE):
        raise RuntimeError(f"PhysioNet-MI CBraMod grid mismatch: {records.get('encoder_output_shape')}")
    if list(output.shape) != [sample.shape[0], 4]:
        raise RuntimeError(f"PhysioNet-MI classifier output mismatch: {list(output.shape)}")
    adapter = getattr(model.backbone, "native_axis_adapter", None)
    adapter_type = getattr(model.backbone, "adapter_type", "none")
    actual = {name for name in ("channel_branch", "patch_branch") if adapter is not None and hasattr(adapter, name)}
    expected = {
        "none": set(), "channel": {"channel_branch"}, "patch": {"patch_branch"},
        "channel_patch": {"channel_branch", "patch_branch"},
    }.get(adapter_type, set())
    if adapter_type in {"channel", "patch", "channel_patch"} and actual != expected:
        raise RuntimeError(f"PhysioNet-MI native branch mismatch: actual={actual} expected={expected}")
    records.update({
        "dataset": "PHYSIONET-MI",
        "channels": 64,
        "patches": 4,
        "embedding_dim": 200,
        "channel_attention_sequence_length": 64 if "channel_branch" in actual else 0,
        "patch_attention_sequence_length": 4 if "patch_branch" in actual else 0,
        "channel_spatial_interactions_active": "channel_branch" in actual,
        "patch_temporal_interactions_active": "patch_branch" in actual,
        "channel_names": list(CHANNELS),
        "input_scale_divisor": EXPECTED_SCALE_DIVISOR,
        "requested_adapter_type": adapter_type,
    })
    return records


def _resolve_paths(config: PhysioNetMITMLRConfig, repo_root: Path) -> None:
    for field in ("dataset_path", "checkpoint", "source_preprocessing"):
        path = Path(getattr(config, field)).expanduser()
        if not path.is_absolute():
            path = repo_root / path
        config.values[field] = str(path.resolve())
    for field in ("channel_manifest", "audit_path", "output_root"):
        path = Path(getattr(config, field)).expanduser()
        if not path.is_absolute():
            path = repo_root / path
        config.values[field] = str(path.resolve())


def _load_audit(config: PhysioNetMITMLRConfig) -> Dict[str, Any]:
    path = Path(config.audit_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing PhysioNet-MI provenance audit {path}; run scripts/audit_physionet_mi.py first"
        )
    audit = json.loads(path.read_text(encoding="utf-8"))
    if audit.get("absolute_dataset_path") != str(Path(config.dataset_path).resolve()):
        raise ValueError("PhysioNet-MI audit dataset path does not match the run config")
    if audit.get("stored_shape") != list(EXPECTED_SHAPE) or audit.get("model_shape") != list(EXPECTED_SHAPE):
        raise ValueError("PhysioNet-MI audit geometry does not match [64,4,200]")
    if float(audit.get("input_scale_divisor")) != EXPECTED_SCALE_DIVISOR:
        raise ValueError("PhysioNet-MI audit scale does not match /100")
    if audit.get("channel_order") != CHANNELS:
        raise ValueError("PhysioNet-MI audit channel order mismatch")
    if any(audit.get("subject_overlap", {}).values()) or not audit.get("duplicate_key_check"):
        raise ValueError("PhysioNet-MI audit reports split overlap or duplicate keys")
    return audit


def run(config: PhysioNetMITMLRConfig) -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    _resolve_paths(config, repo_root)
    if config.audit_only:
        report = audit_physionet_mi(
            config.dataset_path, repo_root, config.channel_manifest,
            config.source_preprocessing, compute_lmdb_hash=True,
        )
        audit_path = Path(config.audit_path)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {"status": "audit_only", "audit_path": str(audit_path)}

    audit = _load_audit(config)
    manifest = json.loads(Path(config.channel_manifest).read_text(encoding="utf-8"))
    if manifest.get("channels") != CHANNELS:
        raise ValueError("PhysioNet-MI channel manifest mismatch")
    run_id = config.run_id or f"physionet_mi_cbramod_{config.method}_s{config.seed}_lr{config.lr:g}_b{config.batch_size}_e{config.epochs}"
    writer = ArtifactWriter(config.output_root, run_id, overwrite=config.overwrite)
    writer.write_json("resolved_config.json", config.to_dict())
    writer.write_json("provenance.json", collect_environment(repo_root, config.device))
    writer.write_json("environment.json", collect_environment(repo_root, config.device))
    writer.write_json("dataset_audit.json", audit)

    set_seed(config.seed)
    device = resolve_device(config.device)
    datasets = make_physionet_mi_datasets(config.dataset_path, config.input_scale_divisor)
    generator = torch.Generator().manual_seed(int(config.loader_seed))
    loaders = {
        mode: DataLoader(
            dataset, batch_size=int(config.batch_size), shuffle=mode == "train",
            generator=generator if mode == "train" else None,
            num_workers=int(config.num_workers), pin_memory=device.type == "cuda",
        )
        for mode, dataset in datasets.items()
    }
    model = Model(_make_params(config)).to(device)
    checkpoint_report = dict(model.checkpoint_report)
    checkpoint_report["checkpoint_sha256"] = sha256_file(config.checkpoint)
    writer.write_json("checkpoint_load_report.json", checkpoint_report)
    if not checkpoint_report.get("strict_loading_status"):
        raise RuntimeError("PhysioNet-MI checkpoint loading did not pass strictly")

    sample_inputs, _ = next(iter(loaders["val"]))
    geometry = _runtime_geometry(model, sample_inputs[: min(2, len(sample_inputs))], device)
    writer.write_json("structure_spec.json", geometry)
    groups, trainability = apply_trainability_contract(
        model, config.method, config.adapter_type,
        lr=config.lr, head_lr=config.head_lr, adapter_lr=config.adapter_lr,
        lora_lr=config.lora_lr, upper_lr=config.upper_lr,
        weight_decay=config.weight_decay, head_weight_decay=config.head_weight_decay,
        adapter_weight_decay=config.adapter_weight_decay,
        adapter_alpha_weight_decay=config.adapter_alpha_weight_decay,
    )
    optimizer = EagerAdamW(groups)
    scheduler = EagerCosineAnnealing(
        optimizer, t_max=int(config.epochs) * len(loaders["train"]), eta_min=float(config.scheduler_eta_min)
    )
    writer.write_json("trainability_report.json", trainability)
    writer.write_json("optimizer_groups.json", {
        "contract": "physionet_mi_cbramod",
        "implementation": "eager_adamw_equivalent",
        "groups": trainability["optimizer_groups"],
    })
    writer.write_json("training_mode_report.json", configure_training_modes(model, config.method, upper_k=config.upper_k))

    criterion = nn.CrossEntropyLoss(label_smoothing=float(config.label_smoothing)).to(device)
    best_score, best_epoch, best_record = float("-inf"), 0, None
    best_state = copy.deepcopy(model.state_dict())
    started = time.time()
    for epoch in range(1, int(config.epochs) + 1):
        epoch_start = time.time()
        configure_training_modes(model, config.method, upper_k=config.upper_k)
        before = _snapshot(model)
        model.train()
        configure_training_modes(model, config.method, upper_k=config.upper_k)
        losses, last_grad = [], {}
        for batch_index, (inputs, labels) in enumerate(loaders["train"]):
            if config.max_train_batches is not None and batch_index >= int(config.max_train_batches):
                break
            inputs, labels = inputs.to(device), labels.to(device).long().reshape(-1)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), labels)
            loss.backward()
            last_grad = _gradient_report(model)
            if config.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], config.clip_grad)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))
        validation = _evaluate(model, loaders["val"], criterion, device, config.max_val_batches)
        validation.update({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "elapsed_seconds": float(time.time() - epoch_start),
            "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
        })
        writer.append_jsonl("metrics_by_epoch.jsonl", validation)
        diagnostics = model.backbone.get_adapter_diagnostics()
        diagnostics.update(last_grad)
        diagnostics.update(_update_report(model, before))
        diagnostics.update(lora_diagnostics(model.backbone))
        diagnostics["epoch"] = epoch
        writer.append_jsonl("adapter_diagnostics_by_epoch.jsonl", diagnostics)
        score = selection_value(validation, config.selection_metric)
        if score > best_score:
            best_score, best_epoch, best_record = score, epoch, copy.deepcopy(validation)
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state, strict=True)
    writer.write_json("best_validation_metrics.json", {
        "selection_metric": config.selection_metric,
        "selection_value": best_score,
        "best_epoch": best_epoch,
        "metrics": best_record,
    })
    writer.save_model(model)
    test = _evaluate(model, loaders["test"], criterion, device, config.max_test_batches)
    test.update({"selection_metric": config.selection_metric, "selected_epoch": best_epoch})
    writer.write_json("test_metrics.json", test)
    writer.write_json("per_class_metrics.json", {
        "precision": test["per_class_precision"], "recall": test["per_class_recall"],
        "f1": test["per_class_f1"], "support": test["per_class_support"],
    })
    writer.write_json("confusion_matrix.json", {"confusion_matrix": test["confusion_matrix"]})
    epoch_records = [json.loads(line) for line in (writer.run_dir / "metrics_by_epoch.jsonl").read_text().splitlines()]
    peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    writer.write_json("timing.json", {
        "training_wall_seconds": float(time.time() - started),
        "best_epoch": best_epoch,
        "time_per_epoch_seconds": [record["elapsed_seconds"] for record in epoch_records],
        "peak_gpu_memory_bytes": peak_memory,
    })
    writer.write_json("memory.json", {"peak_gpu_memory_bytes": peak_memory, "cuda_available": bool(torch.cuda.is_available())})
    summary = {
        "status": "completed", "run_id": run_id, "artifact_dir": str(writer.run_dir),
        "dataset": "PHYSIONET-MI", "method": config.method, "adapter_type": config.adapter_type,
        "seed": config.seed, "trainable_parameter_count": trainability["trainable_parameter_count"],
        "best_epoch": best_epoch, "best_validation_selection_value": best_score,
        "test_evaluation_status": "completed",
    }
    writer.write_json("summary.json", summary)
    return summary
