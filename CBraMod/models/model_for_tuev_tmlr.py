"""CBraMod TUEV model: strict pretrained encoder plus six-class head."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from .cbramod import CBraMod
from .lora import inject_qkv_lora


def _load_checkpoint(path: str | Path) -> dict[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and isinstance(state.get("model"), dict):
        state = state["model"]
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint must be a nonempty state dict")
    return state


class Model(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.backbone = CBraMod(
            in_dim=200, out_dim=200, d_model=200, dim_feedforward=800,
            seq_len=5, n_layer=12, nhead=8,
        )
        checkpoint_path = Path(config.checkpoint).expanduser().resolve()
        state = _load_checkpoint(checkpoint_path)
        report = self.backbone.load_state_dict(state, strict=True)
        if report.missing_keys or report.unexpected_keys:
            raise RuntimeError(f"strict checkpoint mismatch: {report}")
        self.checkpoint_report = {
            "checkpoint_path": str(checkpoint_path), "strict_loading_status": True,
            "missing_keys": [], "unexpected_keys": [], "checkpoint_state_key_count": len(state),
        }
        self.backbone.proj_out = nn.Identity()
        method = str(config.method).strip().lower()
        if method in {"interaction_aligned", "native_full_finetune"}:
            self.backbone.enable_interaction_adapter(
                adapter_type=config.adapter_type, bottleneck=int(config.adapter_bottleneck),
                num_heads=int(config.adapter_heads), dropout=float(config.adapter_dropout),
                init_alpha=float(config.adapter_init_alpha), gamma=float(config.adapter_gamma),
                zero_init_output=bool(config.adapter_zero_init_output), seed=int(config.adapter_seed),
            )
        elif method == "axis_decomposed_mlp":
            self.backbone.enable_axis_decomposed_mlp_adapter(
                adapter_type=config.adapter_type,
                bottleneck=int(config.axis_decomposed_mlp_bottleneck),
                dropout=float(config.adapter_dropout),
                init_alpha=float(config.adapter_init_alpha),
                gamma=float(config.adapter_gamma),
                zero_init_output=bool(config.adapter_zero_init_output),
                seed=int(config.adapter_seed),
            )
        elif method == "generic_bottleneck":
            self.backbone.enable_generic_adapter(
                bottleneck=int(config.generic_bottleneck), init_alpha=float(config.adapter_init_alpha),
                gamma=float(config.adapter_gamma), zero_init_output=bool(config.adapter_zero_init_output),
                seed=int(config.adapter_seed), adapter_type="generic_bottleneck",
            )
        elif method == "axis_blind":
            self.backbone.enable_generic_adapter(
                bottleneck=int(config.axis_blind_bottleneck), init_alpha=float(config.adapter_init_alpha),
                gamma=float(config.adapter_gamma), zero_init_output=bool(config.adapter_zero_init_output),
                seed=int(config.adapter_seed), adapter_type="axis_blind",
            )
        elif method == "lora":
            inject_qkv_lora(self.backbone, rank=int(config.lora_rank), alpha=float(config.lora_alpha))
        elif method not in {"full_finetune", "frozen_probe", "upper_k_finetune"}:
            raise ValueError(f"unsupported method {method}")
        if config.head_seed is not None:
            torch.manual_seed(int(config.head_seed))
        self.classifier = nn.Sequential(
            Rearrange("b c s d -> b (c s d)"),
            nn.Linear(16 * 5 * 200, 5 * 200), nn.ELU(), nn.Dropout(float(config.dropout)),
            nn.Linear(5 * 200, 200), nn.ELU(), nn.Dropout(float(config.dropout)),
            nn.Linear(200, int(config.num_classes)),
        )
        self.checkpoint_report["model_parameter_count_after_attachment"] = int(sum(p.numel() for p in self.parameters()))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or tuple(inputs.shape[1:]) != (16, 5, 200):
            raise ValueError(f"TUEV model expects [B,16,5,200], got {tuple(inputs.shape)}")
        return self.classifier(self.backbone(inputs))
