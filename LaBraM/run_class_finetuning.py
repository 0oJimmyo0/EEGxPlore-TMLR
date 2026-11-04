# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BEiT-v2, timm, DeiT, and DINO code bases
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# ---------------------------------------------------------

import argparse
import datetime
import hashlib
import random
from pyexpat import model
import numpy as np
import time
import warnings
import torch
import torch.backends.cudnn as cudnn
import json
import os
import subprocess

from pathlib import Path
from collections import OrderedDict
from timm.data.mixup import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
from optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner

from engine_for_finetuning import train_one_epoch, evaluate
from utils import NativeScalerWithGradNormCount as NativeScaler
import utils
from scipy import interpolate
import modeling_finetune


def configure_labram_trainability(model, backbone_mode, upper_k=2):
    """Configure and summarize native-adapter, LoRA, head, and backbone trainability.

    LoRA is an independent generic PEFT control.  Keep its parameters separate
    from ``native_axis_adapter`` so mode validation cannot mistake a valid LoRA
    run for a native-adapter run.
    """
    if backbone_mode not in {"trainable", "frozen", "upper_k", "lora"}:
        raise ValueError(f"Unsupported backbone_mode={backbone_mode!r}")
    if backbone_mode == "upper_k":
        upper_k = int(upper_k)
        if upper_k <= 0 or upper_k > len(model.blocks):
            raise ValueError(f"upper_k must be in [1, {len(model.blocks)}], got {upper_k}")

    summary = {
        "backbone": {"total": 0, "trainable": 0},
        "head": {"total": 0, "trainable": 0},
        "adapter": {"total": 0, "trainable": 0},
        "lora": {"total": 0, "trainable": 0},
    }
    for name, parameter in model.named_parameters():
        if name.startswith("native_axis_adapter."):
            component = "adapter"
        elif ".lora_A" in name or ".lora_B" in name:
            component = "lora"
        elif name.startswith("head."):
            component = "head"
        else:
            component = "backbone"
        if backbone_mode == "trainable":
            should_train = True
        elif backbone_mode == "frozen":
            should_train = component in {"head", "adapter"}
        elif backbone_mode == "lora":
            should_train = component in {"head", "lora"}
        else:
            block_index = None
            if name.startswith("blocks."):
                try:
                    block_index = int(name.split('.')[1])
                except (IndexError, ValueError):
                    block_index = None
            should_train = (
                component == "head"
                or (block_index is not None and block_index >= len(model.blocks) - upper_k)
                or name.startswith(("norm.", "fc_norm.", "sequence_encoder."))
            )
        parameter.requires_grad_(should_train)
        summary[component]["total"] += parameter.numel()
        if parameter.requires_grad:
            summary[component]["trainable"] += parameter.numel()

    if backbone_mode in {"frozen", "lora"}:
        if summary["backbone"]["trainable"] != 0:
            raise RuntimeError("Frozen-backbone mode left trainable backbone parameters.")
        if summary["head"]["trainable"] == 0:
            raise RuntimeError("Frozen-backbone mode accidentally froze the classifier head.")
    if backbone_mode == "frozen" and summary["lora"]["trainable"] != 0:
        raise RuntimeError("Native frozen-adapter mode unexpectedly left LoRA parameters trainable.")
    if backbone_mode == "lora":
        if summary["lora"]["trainable"] == 0:
            raise RuntimeError("LoRA mode did not leave any LoRA parameters trainable.")
        if summary["adapter"]["trainable"] != 0:
            raise RuntimeError("LoRA mode unexpectedly left native adapter parameters trainable.")
    return summary


def assert_optimizer_matches_trainability(model, optimizer):
    """Ensure optimizer membership exactly matches requires_grad flags."""
    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    errors = []
    for name, parameter in model.named_parameters():
        in_optimizer = id(parameter) in optimizer_parameter_ids
        if parameter.requires_grad and not in_optimizer:
            errors.append(f"Trainable parameter missing from optimizer: {name}")
        if not parameter.requires_grad and in_optimizer:
            errors.append(f"Frozen parameter present in optimizer: {name}")
    if errors:
        raise RuntimeError("\n".join(errors))

def get_args():
    parser = argparse.ArgumentParser('LaBraM fine-tuning and evaluation script for EEG classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=5, type=int)

    # robust evaluation
    parser.add_argument('--robust_test', default=None, type=str,
                        help='robust evaluation dataset')
    
    # Model parameters
    parser.add_argument('--model', default='labram_base_patch200_200', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--qkv_bias', action='store_true')
    parser.add_argument('--disable_qkv_bias', action='store_false', dest='qkv_bias')
    parser.set_defaults(qkv_bias=True)
    parser.add_argument('--rel_pos_bias', action='store_true')
    parser.add_argument('--disable_rel_pos_bias', action='store_false', dest='rel_pos_bias')
    parser.set_defaults(rel_pos_bias=True)
    parser.add_argument('--abs_pos_emb', action='store_true')
    parser.set_defaults(abs_pos_emb=False)
    parser.add_argument('--layer_scale_init_value', default=0.1, type=float, 
                        help="0.1 for base, 1e-5 for large. set 0 to disable layer scale")

    parser.add_argument('--labram_adapter_type', default='none',
                        choices=['none', 'generic', 'channel', 'patch', 'channel_patch'],
                        help='LaBraM-native structured residual adapter branch.')
    parser.add_argument('--labram_adapter_bottleneck', default=64, type=int,
                        help='Bottleneck width for the optional token MLP or singleton patch residual.')
    parser.add_argument('--labram_adapter_heads', default=4, type=int,
                        help='Attention heads for channel-axis and patch-axis adapter mixers.')
    parser.add_argument('--labram_adapter_dropout', default=0.0, type=float,
                        help='Dropout inside attention weights and the optional token MLP.')
    parser.add_argument('--labram_adapter_variant', default='full',
                        choices=['full', 'output_dropout', 'bottleneck', 'low_rank'],
                        help='Native-axis branch variant: full-width attention, output dropout, singleton bottleneck, or low-rank Down-Mixer-Up.')
    parser.add_argument('--labram_adapter_operator', default='attention',
                        choices=['attention', 'mlp'],
                        help='Operator used inside native channel/patch branches.')
    parser.add_argument('--labram_adapter_patch_output_dropout', default=0.0, type=float,
                        help='Dropout applied to the patch branch output after attention.')
    parser.add_argument('--labram_adapter_init_alpha', default=0.01, type=float,
                        help='Initial scalar for each enabled adapter branch.')
    parser.add_argument('--labram_adapter_gamma', default=1.0, type=float,
                        help='Scalar multiplier on the residual adapter correction.')
    parser.add_argument('--labram_adapter_zero_init_output', action='store_true', default=False,
                        help='Zero-initialize native adapter output projections so the initial branch is identity-like.')
    parser.add_argument('--labram_adapter_lr_scale', default=1.0, type=float,
                        help='Multiplier on the effective LR for native_axis_adapter parameters.')
    parser.add_argument('--labram_adapter_alpha_lr_scale', default=None, type=float,
                        help='Multiplier on alpha_* LR; defaults to the adapter LR scale.')
    parser.add_argument('--backbone_lr_scale', default=1.0, type=float,
                        help='Multiplier on the effective LR for pretrained non-head parameters.')
    parser.add_argument('--head_lr_scale', default=1.0, type=float,
                        help='Multiplier on the effective LR for the classifier head.')
    parser.add_argument('--head_weight_decay', default=None, type=float,
                        help='Override weight decay for classifier weights; bias remains unregularized.')
    parser.add_argument('--labram_backbone_mode', default='trainable',
                        choices=['trainable', 'frozen', 'upper_k', 'lora'],
                        help='trainable=full FT, frozen=head+adapter, upper_k=last blocks+head, lora=LoRA+head.')
    parser.add_argument('--labram_upper_k', default=2, type=int,
                        help='Number of final Transformer blocks to train in upper_k mode.')
    parser.add_argument('--labram_lora_rank', default=8, type=int,
                        help='Rank for the separate generic LoRA control.')
    parser.add_argument('--labram_lora_alpha', default=16.0, type=float,
                        help='Scaling alpha for the separate generic LoRA control.')
    parser.add_argument('--labram_lora_dropout', default=0.0, type=float,
                        help='Dropout on the LoRA input branch.')
    parser.add_argument('--labram_lora_target', default='qkv', type=str,
                        help='Comma-separated attention projections for LoRA: qkv, proj.')
    parser.add_argument('--frozen_backbone_eval_mode', action='store_true', default=False,
                        help='Keep frozen LaBraM modules in eval mode during head/adapter training.')
    parser.add_argument('--labram_adapter_fixed_alpha', default=None, type=float,
                        help='Keep every enabled adapter alpha fixed at this value and exclude it from optimization.')
    parser.add_argument('--labram_adapter_weight_decay', default=None, type=float,
                        help='Weight decay for native_axis_adapter matrices; default inherits global weight decay.')
    parser.add_argument('--labram_adapter_seed', default=12345, type=int,
                        help='Independent seed for adapter-only parameter initialization.')
    parser.add_argument('--labram_adapter_use_token_mlp', action='store_true', default=False,
                        help='Enable optional token-wise MLP adapter branch. Disabled by default.')
    parser.add_argument('--labram_adapter_depth_mode', default='none',
                        choices=['none', 'lastk_delta'],
                        help='Optional depth-aware adapter gating mode. Disabled by default.')
    parser.add_argument('--labram_adapter_depth_k', default=4, type=int,
                        help='Number of final blocks summarized for --labram_adapter_depth_mode lastk_delta.')
    parser.add_argument('--labram_adapter_gamma_zero_skip_branch', action='store_true', default=False,
                        help='If gamma is zero, return dense features before computing the adapter branch.')
    parser.add_argument('--experiment_method', default='', type=str,
                        help='Registry method label recorded in run_config (e.g. axis_blind).')
    parser.add_argument('--target_adapter_params', default=None, type=int,
                        help='Capacity target used by a parameter-matched control.')

    parser.add_argument('--input_size', default=200, type=int,
                        help='EEG input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Attention dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--isruc_sequence_dropout', type=float, default=0.1,
                        help='Dropout in the ISRUC sequence-context Transformer (default: 0.1).')

    parser.add_argument('--disable_eval_during_finetuning', action='store_true', default=False)
    parser.add_argument('--skip_final_test', action='store_true', default=False,
                        help='Skip final test evaluation for validation-only development sweeps.')

    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--requested_lr_string', default=None,
                        help='Original LR string from the launcher, retained for run provenance.')
    parser.add_argument('--layer_decay', type=float, default=0.9)

    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--model_filter_name', default='gzp', type=str)
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')
    parser.add_argument('--disable_weight_decay_on_rel_pos_bias', action='store_true', default=False)

    # Dataset parameters
    parser.add_argument('--nb_classes', default=0, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=1024, type=int)
    parser.add_argument('--deterministic', action='store_true', default=False,
                        help='Enable deterministic CUDA/cuDNN algorithms for reproducibility checks.')
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--eval_selected_checkpoints', action='store_true',
                        help='Evaluate checkpoint-best.pth and checkpoint-best-ba.pth in an existing output directory without retraining.')
    parser.add_argument('--eval_all_checkpoints', action='store_true',
                        help='Evaluate every numeric checkpoint-<epoch>.pth in an existing output directory without retraining.')
    parser.add_argument('--all_checkpoint_eval_output', default='', type=str,
                        help='JSONL destination for --eval_all_checkpoints; defaults to output_dir/all_epoch_test_metrics.jsonl.')
    parser.add_argument('--eval_validation_only', action='store_true', default=False,
                        help='Evaluate the validation split only, without test evaluation or training.')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--enable_deepspeed', action='store_true', default=False)
    parser.add_argument('--dataset', default='TUAB', type=str,
                        help='dataset: TUAB | TUEV | SEED-V | FACED | ISRUC | PhysioNet-MI')
    parser.add_argument('--data_path', default='',
                        help='path to the preprocessed TUAB/TUEV dataset root')
    parser.add_argument('--seedv_channel_manifest', default='',
                        help='validated SEED-V channel manifest JSON path')
    parser.add_argument('--input_scale_divisor', default=100.0, type=float,
                        help='Divide stored SEED-V/FACED samples by this value before LaBraM. Use 1 to preserve raw scale.')
    parser.add_argument('--strict_checkpoint_load', action='store_true', default=False,
                        help='Fail if checkpoint loading has missing/unexpected keys outside the documented task-head/adapter allowlist.')

    known_args, _ = parser.parse_known_args()

    if known_args.enable_deepspeed:
        try:
            import deepspeed
            from deepspeed import DeepSpeedConfig
            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except:
            print("Please 'pip install deepspeed==0.4.0'")
            exit(0)
    else:
        ds_init = None

    return parser.parse_args(), ds_init

def get_models(args):
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        drop_block_rate=None,
        use_mean_pooling=args.use_mean_pooling,
        init_scale=args.init_scale,
        use_rel_pos_bias=args.rel_pos_bias,
        use_abs_pos_emb=args.abs_pos_emb,
        init_values=args.layer_scale_init_value,
        qkv_bias=args.qkv_bias,
        adapter_type=args.labram_adapter_type,
        adapter_bottleneck=args.labram_adapter_bottleneck,
        adapter_num_heads=args.labram_adapter_heads,
        adapter_dropout=args.labram_adapter_dropout,
        adapter_variant=args.labram_adapter_variant,
        adapter_operator=args.labram_adapter_operator,
        adapter_patch_output_dropout=args.labram_adapter_patch_output_dropout,
        adapter_init_alpha=args.labram_adapter_init_alpha,
        adapter_gamma=args.labram_adapter_gamma,
        adapter_zero_init_output=args.labram_adapter_zero_init_output,
        adapter_seed=args.labram_adapter_seed,
        adapter_use_token_mlp=args.labram_adapter_use_token_mlp,
        adapter_depth_mode=args.labram_adapter_depth_mode,
        adapter_depth_k=args.labram_adapter_depth_k,
        adapter_gamma_zero_skip_branch=args.labram_adapter_gamma_zero_skip_branch,
        adapter_fixed_alpha=args.labram_adapter_fixed_alpha,
        isruc_sequence=str(args.dataset).upper().replace('_', '-') == 'ISRUC',
        isruc_sequence_length=20,
        isruc_sequence_dropout=args.isruc_sequence_dropout,
    )

    if args.labram_backbone_mode == 'lora':
        modeling_finetune.inject_lora(
            model,
            rank=args.labram_lora_rank,
            alpha=args.labram_lora_alpha,
            dropout=args.labram_lora_dropout,
            target=args.labram_lora_target,
        )

    return model


def shared_parameter_checksum(model):
    """Hash common model state, excluding adapter-only parameters."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if name.startswith('native_axis_adapter.'):
            continue
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode('utf-8'))
        digest.update(str(value.dtype).encode('ascii'))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _sha256_file(path):
    if not path or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def get_dataset(args):
    if not args.data_path:
        raise ValueError("--data_path must point to the preprocessed dataset root")
    dataset_name = str(args.dataset).upper().replace("_", "-")
    if dataset_name == 'TUAB':
        train_dataset, test_dataset, val_dataset = utils.prepare_TUAB_dataset(args.data_path)
        ch_names = ['EEG FP1', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
        ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]
        args.nb_classes = 1
        metrics = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"]
    elif dataset_name == 'TUEV':
        train_dataset, test_dataset, val_dataset = utils.prepare_TUEV_dataset(args.data_path)
        # The stored rows are bipolar derivations. The pretrained LaBraM
        # checkpoint has positional slots only for the first 128 standard
        # positions, while the final eight bipolar names are not valid
        # checkpoint indices. Preserve the serialized row order and use the
        # model's deterministic sequential slots 1..16 instead of silently
        # mapping bipolar names to unrelated scalp positions.
        ch_names = None
        args.nb_classes = 6
        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]
    elif dataset_name in {'SEED-V', 'SEEDV'}:
        train_dataset, test_dataset, val_dataset = utils.prepare_SEEDV_dataset(
            args.data_path,
            channel_manifest=args.seedv_channel_manifest or None,
        )
        ch_names = getattr(train_dataset, "get_ch_names", lambda: None)()
        if ch_names is None:
            raise RuntimeError(
                "SEED-V LaBraM runs require a validated channel manifest matching the stored tensor. "
                "Provide --seedv_channel_manifest or place channel_names.json beside the LMDB."
            )
        if len(ch_names) != 62:
            raise RuntimeError(
                f"The retained SEED-V protocol expects exactly 62 channels, found {len(ch_names)}."
            )
        input_chans = utils.get_input_chans(ch_names)
        if input_chans[0] != 0 or len(input_chans[1:]) != 62:
            raise RuntimeError("SEED-V channel mapping must contain CLS slot 0 plus 62 electrodes.")
        if len(set(input_chans[1:])) != 62:
            raise RuntimeError("SEED-V channel mapping contains duplicate LaBraM positions.")
        else:
            print(f"Loaded SEED-V channel manifest with {len(ch_names)} channels.")
        args.nb_classes = 5
        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]
    elif dataset_name == 'FACED':
        train_dataset, test_dataset, val_dataset = utils.prepare_FACED_dataset(args.data_path)
        ch_names = getattr(train_dataset, "get_ch_names", lambda: None)()
        if ch_names is None:
            raise RuntimeError(
                "FACED LaBraM runs require a validated 32-channel manifest."
            )
        if len(ch_names) != 32:
            raise RuntimeError(
                "FACED LaBraM runs require a validated 32-channel manifest."
            )
        else:
            print(f"Loaded FACED channel manifest with {len(ch_names)} channels.")
        args.nb_classes = 9
        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]
    elif dataset_name == 'ISRUC':
        train_dataset, test_dataset, val_dataset = utils.prepare_ISRUC_dataset(args.data_path)
        ch_names = list(utils.ISRUC_LABRAM_CH)
        args.nb_classes = 5
        args.input_size = 6000
        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]
    elif dataset_name in {'PHYSIONET-MI', 'PHYSIONETMI'}:
        train_dataset, test_dataset, val_dataset = utils.prepare_PhysioNet_MI_dataset(args.data_path)
        ch_names = list(utils.PHYSIONET_MI_LABRAM_CH)
        if len(ch_names) != 64:
            raise RuntimeError(
                f"PhysioNet-MI LaBraM protocol requires 64 channels, found {len(ch_names)}."
            )
        input_chans = utils.get_input_chans(ch_names)
        if input_chans[0] != 0 or len(input_chans[1:]) != 64 or len(set(input_chans[1:])) != 64:
            raise RuntimeError("PhysioNet-MI channel mapping must contain CLS slot 0 plus 64 unique electrodes.")
        print(f"Loaded PhysioNet-MI channel manifest with {len(ch_names)} channels.")
        args.nb_classes = 4
        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]
    else:
        raise ValueError(
            f"Unsupported dataset '{args.dataset}'. Expected one of: "
            "TUAB, TUEV, SEED-V, FACED, ISRUC, PhysioNet-MI"
        )
    return train_dataset, test_dataset, val_dataset, ch_names, metrics


def main(args, ds_init):
    utils.init_distributed_mode(args)

    if ds_init is not None:
        utils.create_ds_config(args)

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cudnn.benchmark = not args.deterministic
    cudnn.deterministic = args.deterministic
    torch.use_deterministic_algorithms(args.deterministic)

    # dataset_train, dataset_test, dataset_val: follows the standard format of torch.utils.data.Dataset.
    # ch_names: list of strings, channel names of the dataset. It should be in capital letters.
    # metrics: list of strings, the metrics you want to use. We utilize PyHealth to implement it.
    dataset_train, dataset_test, dataset_val, ch_names, metrics = get_dataset(args)

    if args.disable_eval_during_finetuning:
        dataset_val = None
        dataset_test = None

    global_rank = utils.get_rank()

    if args.distributed:
        num_tasks = utils.get_world_size()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if dataset_val is not None:
            if args.dist_eval:
                if len(dataset_val) % num_tasks != 0:
                    print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                          'This will slightly alter validation results as extra duplicate entries are added to achieve '
                          'equal num of samples per-process.')
                sampler_val = torch.utils.data.DistributedSampler(
                    dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
                if type(dataset_test) == list:
                    sampler_test = [torch.utils.data.DistributedSampler(
                        dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False) for dataset in dataset_test]
                else:
                    sampler_test = torch.utils.data.DistributedSampler(
                        dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False)
            else:
                sampler_val = torch.utils.data.SequentialSampler(dataset_val)
                sampler_test = torch.utils.data.SequentialSampler(dataset_test)
        else:
            sampler_val = None
            sampler_test = None
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val) if dataset_val is not None else None
        sampler_test = torch.utils.data.SequentialSampler(dataset_test) if dataset_test is not None else None

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
        if type(dataset_test) == list:
            data_loader_test = [torch.utils.data.DataLoader(
                dataset, sampler=sampler,
                batch_size=int(1.5 * args.batch_size),
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False
            ) for dataset, sampler in zip(dataset_test, sampler_test)]
        else:
            data_loader_test = torch.utils.data.DataLoader(
                dataset_test, sampler=sampler_test,
                batch_size=int(1.5 * args.batch_size),
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False
            )
    else:
        data_loader_val = None
        data_loader_test = None

    model = get_models(args)

    patch_size = model.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (1, args.input_size // patch_size)
    args.patch_size = patch_size

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            # LaBraM checkpoints contain NumPy scalars; newer PyTorch defaults to
            # weights_only=True, which rejects those objects unless we opt out.
            checkpoint = torch.load(args.finetune, map_location='cpu', weights_only=False)

        print("Load ckpt from %s" % args.finetune)
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        if (checkpoint_model is not None) and (args.model_filter_name != ''):
            all_keys = list(checkpoint_model.keys())
            if any(key.startswith('student.') for key in all_keys):
                new_dict = OrderedDict()
                for key in all_keys:
                    if key.startswith('student.'):
                        new_dict[key[8:]] = checkpoint_model[key]
                checkpoint_model = new_dict

        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        all_keys = list(checkpoint_model.keys())
        for key in all_keys:
            if "relative_position_index" in key:
                checkpoint_model.pop(key)

        # LoRA wraps the original projection as ``*.base``.  Remap only the
        # frozen checkpoint weights; fresh lora_A/B parameters retain their
        # exact-zero update initialization.
        if args.labram_backbone_mode == 'lora':
            remapped = OrderedDict()
            model_keys = model.state_dict()
            for key, value in checkpoint_model.items():
                key_parts = key.rsplit('.', 1)
                wrapped_key = (
                    key_parts[0] + '.base.' + key_parts[1]
                    if len(key_parts) == 2 else key
                )
                remapped[wrapped_key if wrapped_key in model_keys else key] = value
            checkpoint_model = remapped

        checkpoint_load_report = utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)

        # These are the only intentionally absent keys for the ISRUC LaBraM
        # fine-tuning contract: the downstream pooling/head/sequence module,
        # plus freshly initialized PEFT/native-adapter parameters.  A missing
        # encoder block or projection is a hard failure when strict loading is
        # requested.
        allowed_missing_prefixes = ('fc_norm.', 'head.', 'sequence_encoder.')
        if args.labram_backbone_mode == 'lora':
            allowed_missing_prefixes += ('.lora_A', '.lora_B')
        if args.labram_adapter_type != 'none':
            allowed_missing_prefixes += ('native_axis_adapter.',)
        allowed_unexpected_keys = {
            'mask_token', 'lm_head.weight', 'lm_head.bias', 'norm.weight', 'norm.bias',
        }
        missing_violations = [
            key for key in checkpoint_load_report['missing_keys']
            if not any(key.startswith(prefix) or prefix in key for prefix in allowed_missing_prefixes)
        ]
        unexpected_violations = [
            key for key in checkpoint_load_report['unexpected_keys']
            if key not in allowed_unexpected_keys
        ]
        checkpoint_load_report.update({
            'checkpoint_path': os.path.abspath(args.finetune) if args.finetune else '',
            'checkpoint_key_count': int(len(checkpoint_model)),
            'model_key_count': int(len(model.state_dict())),
            'allowed_missing_prefixes': list(allowed_missing_prefixes),
            'allowed_unexpected_keys': sorted(allowed_unexpected_keys),
            'missing_key_violations': missing_violations,
            'unexpected_key_violations': unexpected_violations,
            'strict': bool(args.strict_checkpoint_load),
            'strict_pass': not missing_violations and not unexpected_violations and not checkpoint_load_report['error_messages'],
        })
        print('Checkpoint load report: ' + json.dumps(checkpoint_load_report, sort_keys=True), flush=True)
        if args.strict_checkpoint_load and not checkpoint_load_report['strict_pass']:
            raise RuntimeError(
                'Strict checkpoint load failed: '
                + json.dumps({
                    'missing_key_violations': missing_violations,
                    'unexpected_key_violations': unexpected_violations,
                    'error_messages': checkpoint_load_report['error_messages'],
                }, sort_keys=True)
            )
    else:
        checkpoint_load_report = {
            'checkpoint_path': '',
            'strict': bool(args.strict_checkpoint_load),
            'strict_pass': True,
            'note': 'No pretrained checkpoint supplied.',
        }

    trainability_summary = configure_labram_trainability(
        model,
        backbone_mode=args.labram_backbone_mode,
        upper_k=args.labram_upper_k,
    )
    print(
        "Trainability summary: "
        + json.dumps(trainability_summary, indent=2, sort_keys=True),
        flush=True,
    )
    dataset_key = str(args.dataset).upper().replace('_', '-')
    if dataset_key in {'SEED-V', 'SEEDV'} and args.labram_backbone_mode in {'frozen', 'lora'}:
        adapter_trainable = trainability_summary['adapter']['trainable']
        assert trainability_summary['backbone']['trainable'] == 0
        assert trainability_summary['head']['trainable'] > 0
        if args.labram_adapter_type == 'none':
            assert adapter_trainable == 0
        else:
            assert adapter_trainable > 0
    if args.frozen_backbone_eval_mode and args.labram_backbone_mode != 'frozen':
        raise ValueError('--frozen_backbone_eval_mode requires --labram_backbone_mode frozen')
    if args.experiment_method == 'axis_blind':
        if args.labram_backbone_mode != 'frozen' or args.labram_adapter_type != 'generic':
            raise ValueError('axis_blind requires frozen backbone plus generic adapter')
        if args.target_adapter_params is None:
            raise ValueError('axis_blind requires --target_adapter_params')
        adapter_module = model.native_axis_adapter
        actual_adapter_params = sum(p.numel() for p in adapter_module.parameters())
        relative_error = abs(actual_adapter_params - args.target_adapter_params) / max(
            1, args.target_adapter_params
        )
        if relative_error > 0.05:
            raise ValueError(
                f'axis_blind adapter count mismatch: actual={actual_adapter_params} '
                f'target={args.target_adapter_params} relative_error={relative_error:.6f}'
            )
        if hasattr(adapter_module, 'channel_attn') or hasattr(adapter_module, 'patch_attn'):
            raise ValueError('axis_blind must not instantiate channel or patch attention')
        if not hasattr(adapter_module, 'token_mlp'):
            raise ValueError('axis_blind requires the token MLP residual branch')
    if args.experiment_method == 'axis_decomposed_mlp':
        if args.labram_backbone_mode != 'frozen' or args.labram_adapter_type != 'channel_patch':
            raise ValueError('axis_decomposed_mlp requires frozen backbone plus channel_patch adapter')
        if args.labram_adapter_operator != 'mlp':
            raise ValueError('axis_decomposed_mlp requires --labram_adapter_operator mlp')
        if args.target_adapter_params is None:
            raise ValueError('axis_decomposed_mlp requires --target_adapter_params')
        adapter_module = model.native_axis_adapter
        actual_adapter_params = sum(p.numel() for p in adapter_module.parameters())
        relative_error = abs(actual_adapter_params - args.target_adapter_params) / max(
            1, args.target_adapter_params
        )
        if relative_error > 0.05:
            raise ValueError(
                f'axis_decomposed_mlp adapter count mismatch: actual={actual_adapter_params} '
                f'target={args.target_adapter_params} relative_error={relative_error:.6f}'
            )
        if hasattr(adapter_module, 'channel_attn') or hasattr(adapter_module, 'patch_attn'):
            raise ValueError('axis_decomposed_mlp must not instantiate channel or patch attention')
        if not hasattr(adapter_module, 'channel_down') or not hasattr(adapter_module, 'patch_down'):
            raise ValueError('axis_decomposed_mlp requires both native axis MLP branches')

    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    adapter_parameter_count = sum(
        p.numel() for name, p in model_without_ddp.named_parameters()
        if name.startswith('native_axis_adapter.')
    )
    lora_parameter_count = sum(
        p.numel() for name, p in model_without_ddp.named_parameters()
        if '.lora_A' in name or '.lora_B' in name
    )

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)
    print('adapter params:', adapter_parameter_count)
    print('lora params:', lora_parameter_count)

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    if num_training_steps_per_epoch == 0:
        raise ValueError(
            f"num_training_steps_per_epoch is 0: len(dataset_train)={len(dataset_train)}, "
            f"total_batch_size={total_batch_size}. Reduce batch size / update_freq or use more data."
        )
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

    num_layers = model_without_ddp.get_num_layers()
    if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    skip_weight_decay_list = model.no_weight_decay()
    if args.disable_weight_decay_on_rel_pos_bias:
        for i in range(num_layers):
            skip_weight_decay_list.add("blocks.%d.attn.relative_position_bias_table" % i)

    if args.enable_deepspeed:
        loss_scaler = None
        optimizer_params = get_parameter_groups(
            model, args.weight_decay, skip_weight_decay_list,
            assigner.get_layer_id if assigner is not None else None,
            assigner.get_scale if assigner is not None else None,
            adapter_name_prefix='native_axis_adapter.',
            adapter_lr_scale=args.labram_adapter_lr_scale,
            adapter_alpha_lr_scale=args.labram_adapter_alpha_lr_scale,
            adapter_alpha_name_prefix='native_axis_adapter.alpha_',
            adapter_weight_decay=args.labram_adapter_weight_decay,
            backbone_lr_scale=args.backbone_lr_scale,
            head_lr_scale=args.head_lr_scale,
            head_weight_decay=args.head_weight_decay)
        model, optimizer, _, _ = ds_init(
            args=args, model=model, model_parameters=optimizer_params, dist_init_required=not args.distributed,
        )

        print("model.gradient_accumulation_steps() = %d" % model.gradient_accumulation_steps())
        assert model.gradient_accumulation_steps() == args.update_freq
    else:
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            model_without_ddp = model.module

        optimizer = create_optimizer(
            args, model_without_ddp, skip_list=skip_weight_decay_list,
            get_num_layer=assigner.get_layer_id if assigner is not None else None, 
            get_layer_scale=assigner.get_scale if assigner is not None else None,
            adapter_name_prefix='native_axis_adapter.',
            adapter_lr_scale=args.labram_adapter_lr_scale,
            adapter_alpha_lr_scale=args.labram_adapter_alpha_lr_scale,
            adapter_alpha_name_prefix='native_axis_adapter.alpha_',
            adapter_weight_decay=args.labram_adapter_weight_decay,
            backbone_lr_scale=args.backbone_lr_scale,
            head_lr_scale=args.head_lr_scale,
            head_weight_decay=args.head_weight_decay)
        loss_scaler = NativeScaler()

    if not args.enable_deepspeed:
        assert_optimizer_matches_trainability(model_without_ddp, optimizer)

    print("Use step level LR scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    max_scheduled_lr = float(max(lr_schedule_values))
    lr_tolerance = max(1e-12, abs(float(args.lr)) * 1e-6)
    if not args.lr > 0:
        raise ValueError(f"Parsed learning rate must be positive, got {args.lr!r}")
    if not max_scheduled_lr > 0:
        raise ValueError(f"Scheduled learning rate must be positive, got {max_scheduled_lr!r}")
    if abs(max_scheduled_lr - float(args.lr)) >= lr_tolerance:
        raise ValueError(
            f"Maximum scheduled LR {max_scheduled_lr:.12g} does not match parsed LR "
            f"{args.lr:.12g} within tolerance {lr_tolerance:.3g}"
        )
    assert args.lr > 0
    assert max_scheduled_lr > 0
    assert abs(max_scheduled_lr - float(args.lr)) < lr_tolerance

    try:
        git_commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=Path(__file__).resolve().parent,
            text=True,
        ).strip()
        git_status = subprocess.check_output(
            ['git', 'status', '--short'],
            cwd=Path(__file__).resolve().parent,
            text=True,
        ).splitlines()
        git_diff_sha256 = hashlib.sha256(
            subprocess.check_output(
                ['git', 'diff', '--binary'],
                cwd=Path(__file__).resolve().parent,
            )
        ).hexdigest()
    except (OSError, subprocess.CalledProcessError):
        git_commit = 'unknown'
        git_status = []
        git_diff_sha256 = None
    dataset_channel_names = (
        list(utils.TUEV_BIPOLAR_16_CH)
        if str(args.dataset).upper().replace('_', '-') == 'TUEV'
        else ch_names
    )
    channel_manifest_sha256 = None
    channel_manifest_file_sha256 = _sha256_file(args.seedv_channel_manifest)
    checkpoint_sha256 = _sha256_file(args.finetune)
    input_chans = None
    if ch_names is not None:
        channel_manifest_sha256 = hashlib.sha256(
            json.dumps(ch_names, ensure_ascii=True, separators=(',', ':')).encode('utf-8')
        ).hexdigest()
        input_chans = utils.get_input_chans(ch_names)
    elif dataset_channel_names is not None:
        channel_manifest_sha256 = hashlib.sha256(
            json.dumps(dataset_channel_names, ensure_ascii=True, separators=(',', ':')).encode('utf-8')
        ).hexdigest()
    split_metadata = {}
    if str(args.dataset).upper().replace('_', '-') in {'SEED-V', 'SEEDV'}:
        for split_name, dataset in (('train', dataset_train), ('val', dataset_val), ('test', dataset_test)):
            if hasattr(dataset, 'split_metadata'):
                split_metadata[split_name] = dataset.split_metadata()
    if str(args.dataset).upper().replace('_', '-') == 'ISRUC':
        for split_name, dataset in (('train', dataset_train), ('val', dataset_val), ('test', dataset_test)):
            if hasattr(dataset, 'split_metadata'):
                split_metadata[split_name] = dataset.split_metadata()
    if str(args.dataset).upper().replace('_', '-') in {'PHYSIONET-MI', 'PHYSIONETMI'}:
        for split_name, dataset in (('train', dataset_train), ('val', dataset_val), ('test', dataset_test)):
            if hasattr(dataset, 'split_metadata'):
                split_metadata[split_name] = dataset.split_metadata()
    if str(args.dataset).upper().replace('_', '-') == 'TUEV':
        for split_name, dataset in (('train', dataset_train), ('val', dataset_val), ('test', dataset_test)):
            if hasattr(dataset, 'split_metadata'):
                split_metadata[split_name] = dataset.split_metadata()
    run_config = {
        'dataset': str(args.dataset),
        'model': str(args.model),
        'experiment_method': str(args.experiment_method),
        'target_adapter_params': (
            None if args.target_adapter_params is None else int(args.target_adapter_params)
        ),
        'data_path': os.path.abspath(args.data_path) if args.data_path else '',
        'input_scale_divisor': float(args.input_scale_divisor),
        'strict_checkpoint_load': bool(args.strict_checkpoint_load),
        'checkpoint_load_report': checkpoint_load_report,
        'seedv_channel_manifest': os.path.abspath(args.seedv_channel_manifest) if args.seedv_channel_manifest else '',
        'channel_names': dataset_channel_names,
        'input_chans': input_chans,
        'input_position_policy': (
            'sequential_pos_embed_slots_1_to_16'
            if str(args.dataset).upper().replace('_', '-') == 'TUEV'
            else 'manifest_lookup'
        ),
        'channel_manifest_file_sha256': channel_manifest_file_sha256,
        'pretrained_checkpoint': os.path.abspath(args.finetune) if args.finetune else '',
        'pretrained_checkpoint_sha256': checkpoint_sha256,
        'git_commit': git_commit,
        'git_dirty': bool(git_status),
        'git_status': git_status,
        'git_diff_sha256': git_diff_sha256,
        'dataset_split_metadata': split_metadata,
        'isruc_bipolar_channels': list(utils.ISRUC_BIPOLAR_CH) if str(args.dataset).upper().replace('_', '-') == 'ISRUC' else None,
        'requested_lr_string': args.requested_lr_string or str(args.lr),
        'parsed_args_lr': float(args.lr),
        'max_scheduled_lr': max_scheduled_lr,
        'optimizer': str(args.opt),
        'batch_size': int(args.batch_size),
        'epochs': int(args.epochs),
        'update_freq': int(args.update_freq),
        'num_workers': int(args.num_workers),
        'clip_grad': None if args.clip_grad is None else float(args.clip_grad),
        'warmup_epochs': int(args.warmup_epochs),
        'save_ckpt_freq': int(args.save_ckpt_freq),
        'seed': int(args.seed),
        'backbone_mode': args.labram_backbone_mode,
        'backbone_frozen': bool(args.labram_backbone_mode in {'frozen', 'lora'}),
        'upper_k': int(args.labram_upper_k),
        'lora_rank': int(args.labram_lora_rank),
        'lora_alpha': float(args.labram_lora_alpha),
        'lora_dropout': float(args.labram_lora_dropout),
        'lora_target': args.labram_lora_target,
        'frozen_backbone_eval_mode': bool(args.frozen_backbone_eval_mode),
        'trainability_summary': trainability_summary,
        'total_parameter_count': int(sum(p.numel() for p in model_without_ddp.parameters())),
        'trainable_parameter_count': int(sum(
            p.numel() for p in model_without_ddp.parameters() if p.requires_grad
        )),
        'trainable_parameter_fraction': float(
            sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
            / max(1, sum(p.numel() for p in model_without_ddp.parameters()))
        ),
        'trainable_backbone_parameters': int(trainability_summary['backbone']['trainable']),
        'trainable_head_parameters': int(trainability_summary['head']['trainable']),
        'trainable_adapter_parameters': int(trainability_summary['adapter']['trainable']),
        'trainable_lora_parameters': int(trainability_summary['lora']['trainable']),
        'smoothing': float(args.smoothing),
        'drop_path': float(args.drop_path),
        'drop': float(args.drop),
        'attn_drop_rate': float(args.attn_drop_rate),
        'weight_decay': float(args.weight_decay),
        'weight_decay_end': float(args.weight_decay_end),
        'model_ema': bool(args.model_ema),
        'sequence_head_dropout': (
            float(args.isruc_sequence_dropout)
            if str(args.dataset).upper().replace('_', '-') == 'ISRUC' else None
        ),
        'adapter_type': args.labram_adapter_type,
        'adapter_variant': args.labram_adapter_variant,
        'adapter_operator': args.labram_adapter_operator,
        'adapter_patch_output_dropout': float(args.labram_adapter_patch_output_dropout),
        'adapter_bottleneck': int(args.labram_adapter_bottleneck),
        'adapter_heads': int(args.labram_adapter_heads),
        'adapter_dropout': float(args.labram_adapter_dropout),
        'adapter_gamma': float(args.labram_adapter_gamma),
        'adapter_zero_init_output': bool(args.labram_adapter_zero_init_output),
        'adapter_init_alpha': float(args.labram_adapter_init_alpha),
        'adapter_parameter_count': int(adapter_parameter_count),
        'lora_parameter_count': int(lora_parameter_count),
        'adapter_lr_scale': float(args.labram_adapter_lr_scale),
        'adapter_core_lr_scale': float(args.labram_adapter_lr_scale),
        'adapter_alpha_lr_scale': float(
            args.labram_adapter_lr_scale
            if args.labram_adapter_alpha_lr_scale is None
            else args.labram_adapter_alpha_lr_scale
        ),
        'adapter_weight_decay': (
            None if args.labram_adapter_weight_decay is None
            else float(args.labram_adapter_weight_decay)
        ),
        'backbone_lr_scale': float(args.backbone_lr_scale),
        'head_lr_scale': float(args.head_lr_scale),
        'head_weight_decay': (
            None if args.head_weight_decay is None else float(args.head_weight_decay)
        ),
        'adapter_fixed_alpha': (
            None if args.labram_adapter_fixed_alpha is None
            else float(args.labram_adapter_fixed_alpha)
        ),
        'adapter_seed': int(args.labram_adapter_seed),
        'channel_count': None if ch_names is None else len(ch_names),
        'channel_manifest_sha256': channel_manifest_sha256,
        'git_commit': git_commit,
        'output_dir': os.path.abspath(args.output_dir) if args.output_dir else '',
    }
    print('Run contract: ' + json.dumps(run_config, sort_keys=True), flush=True)
    if args.output_dir and utils.is_main_process() and not (args.eval_selected_checkpoints or args.eval_all_checkpoints):
        with open(os.path.join(args.output_dir, 'run_config.json'), 'w', encoding='utf-8') as f:
            json.dump(run_config, f, indent=2)

    if args.nb_classes == 1:
        criterion = torch.nn.BCEWithLogitsLoss()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

    if args.eval_all_checkpoints:
        if data_loader_test is None or data_loader_val is None:
            raise ValueError('All-checkpoint evaluation requires validation and test dataloaders.')

        val_loaders = data_loader_val if isinstance(data_loader_val, list) else [data_loader_val]
        test_loaders = data_loader_test if isinstance(data_loader_test, list) else [data_loader_test]

        def _mean_all_epoch_stat(stats_list, key):
            values = [item[key] for item in stats_list if key in item]
            if not values:
                raise KeyError(f'Missing evaluation statistic: {key}')
            first = values[0]
            if isinstance(first, (list, tuple, np.ndarray)):
                return np.asarray(values, dtype=np.float64).mean(axis=0).tolist()
            return float(np.mean(values))

        checkpoint_paths = []
        for checkpoint_path in Path(args.output_dir).glob('checkpoint-*.pth'):
            suffix = checkpoint_path.stem.rsplit('-', 1)[-1]
            if suffix.isdigit():
                checkpoint_paths.append((int(suffix), checkpoint_path))
        checkpoint_paths.sort(key=lambda item: item[0])
        if not checkpoint_paths:
            raise FileNotFoundError(
                f'No numeric checkpoint-<epoch>.pth files found in {args.output_dir}'
            )

        output_path = args.all_checkpoint_eval_output or os.path.join(
            args.output_dir, 'all_epoch_test_metrics.jsonl')
        output_path = os.path.abspath(output_path)
        if utils.is_main_process():
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as output_file:
                for epoch, checkpoint_path in checkpoint_paths:
                    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                    model_without_ddp.load_state_dict(checkpoint['model'])
                    val_stats_list = [
                        evaluate(
                            loader, model, device, header=f'epoch {epoch} val:',
                            ch_names=ch_names, metrics=metrics,
                            is_binary=args.nb_classes == 1,
                            input_scale_divisor=args.input_scale_divisor,
                        )
                        for loader in val_loaders
                    ]
                    test_stats_list = [
                        evaluate(
                            loader, model, device, header=f'epoch {epoch} test:',
                            ch_names=ch_names, metrics=metrics,
                            is_binary=args.nb_classes == 1,
                            input_scale_divisor=args.input_scale_divisor,
                        )
                        for loader in test_loaders
                    ]
                    record = {
                        'epoch': epoch,
                        'checkpoint': checkpoint_path.name,
                        'validation': {
                            key: _mean_all_epoch_stat(val_stats_list, key)
                            for key in val_stats_list[0]
                        },
                        'test': {
                            key: _mean_all_epoch_stat(test_stats_list, key)
                            for key in test_stats_list[0]
                        },
                    }
                    output_file.write(json.dumps(record) + '\n')
                    output_file.flush()
                    print(
                        f'All-checkpoint epoch {epoch}: '
                        f"val_kappa={record['validation'].get('cohen_kappa', float('nan')):.5f}, "
                        f"test_ba={record['test'].get('balanced_accuracy', float('nan')):.5f}, "
                        f"test_kappa={record['test'].get('cohen_kappa', float('nan')):.5f}",
                        flush=True,
                    )
            print(f'Wrote all-checkpoint evaluation to {output_path}', flush=True)
        exit(0)

    if args.eval_selected_checkpoints:
        if data_loader_test is None or data_loader_val is None:
            raise ValueError('Selected-checkpoint evaluation requires validation and test dataloaders.')

        eval_loaders = data_loader_test if isinstance(data_loader_test, list) else [data_loader_test]

        def _mean_eval_stat(stats_list, key):
            values = [item[key] for item in stats_list if key in item]
            if not values:
                raise KeyError(f'Missing evaluation statistic: {key}')
            first = values[0]
            if isinstance(first, (list, tuple, np.ndarray)):
                return np.asarray(values, dtype=np.float64).mean(axis=0).tolist()
            return float(np.mean(values))

        def _eval_scalar(stats, primary, fallback=None):
            if primary in stats and stats[primary] is not None:
                return float(stats[primary])
            if fallback is not None and fallback in stats and stats[fallback] is not None:
                return float(stats[fallback])
            return float('nan')

        selected = {}
        for filename, label, validation_metric in (
            ('checkpoint-best.pth', 'primary_test', 'cohen_kappa'),
            ('checkpoint-best-ba.pth', 'sensitivity_test', 'balanced_accuracy'),
        ):
            checkpoint_path = os.path.join(args.output_dir, filename)
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(f'Selected checkpoint not found: {checkpoint_path}')
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            model_without_ddp.load_state_dict(checkpoint['model'])
            stats_list = [
                evaluate(
                    loader, model, device, header=f'{label}:', ch_names=ch_names,
                    metrics=metrics, is_binary=args.nb_classes == 1,
                    input_scale_divisor=args.input_scale_divisor,
                )
                for loader in eval_loaders
            ]
            selected[label] = {
                key: _mean_eval_stat(stats_list, key)
                for key in stats_list[0]
            }
            selected[label]['checkpoint'] = filename
            selected[label]['validation_metric'] = validation_metric
            print(
                f'Checkpoint-only {label} ({filename}): '
                f"accuracy={_eval_scalar(selected[label], 'accuracy'):.5f}, "
                f"balanced_accuracy={_eval_scalar(selected[label], 'balanced_accuracy', 'accuracy'):.5f}, "
                f"cohen_kappa={_eval_scalar(selected[label], 'cohen_kappa'):.5f}, "
                f"f1_weighted={_eval_scalar(selected[label], 'f1_weighted'):.5f}",
                flush=True,
            )
        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, 'final_test.json'), 'w', encoding='utf-8') as f:
                json.dump({
                    'selection_metric': 'cohen_kappa',
                    'primary_checkpoint': 'checkpoint-best.pth',
                    'primary_validation_metric': 'cohen_kappa',
                    'primary_test': selected['primary_test'],
                    'sensitivity_checkpoint': 'checkpoint-best-ba.pth',
                    'sensitivity_validation_metric': 'balanced_accuracy',
                    'sensitivity_test': selected['sensitivity_test'],
                    'evaluation_only': True,
                }, f, indent=2)
        exit(0)
            
    if args.eval:
        if data_loader_test is None:
            raise ValueError("Evaluation requested but no test dataloader is available.")
        balanced_accuracy = []
        accuracy = []
        eval_loaders = data_loader_test if isinstance(data_loader_test, list) else [data_loader_test]
        for data_loader in eval_loaders:
            test_stats = evaluate(data_loader, model, device, header='Test:', ch_names=ch_names, metrics=metrics, is_binary=(args.nb_classes == 1), input_scale_divisor=args.input_scale_divisor)
            accuracy.append(test_stats['accuracy'])
            balanced_accuracy.append(test_stats['balanced_accuracy'])
        print(f"======Accuracy: {np.mean(accuracy)} {np.std(accuracy)}, balanced accuracy: {np.mean(balanced_accuracy)} {np.std(balanced_accuracy)}")
        exit(0)

    if args.eval_validation_only:
        if data_loader_val is None:
            raise ValueError("Validation-only evaluation requested but no validation dataloader is available.")
        val_stats = evaluate(
            data_loader_val, model, device, header='Validation-only:', ch_names=ch_names,
            metrics=metrics, is_binary=(args.nb_classes == 1), input_scale_divisor=args.input_scale_divisor)
        def _eval_stat_value(stats, primary, fallback=None):
            if primary in stats and stats[primary] is not None:
                return float(stats[primary])
            if fallback is not None and fallback in stats and stats[fallback] is not None:
                return float(stats[fallback])
            return float('nan')
        print(
            "Validation-only metrics: "
            f"accuracy={_eval_stat_value(val_stats, 'accuracy'):.5f}, "
            f"balanced_accuracy={_eval_stat_value(val_stats, 'balanced_accuracy', 'accuracy'):.5f}, "
            f"cohen_kappa={_eval_stat_value(val_stats, 'cohen_kappa'):.5f}, "
            f"f1_weighted={_eval_stat_value(val_stats, 'f1_weighted'):.5f}",
            flush=True,
        )
        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, 'validation_only.json'), 'w', encoding='utf-8') as f:
                json.dump(val_stats, f, indent=2)
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_primary_score = float('-inf')
    max_val_ba = float('-inf')

    def _stat_value(stats, primary, fallback=None):
        if primary in stats and stats[primary] is not None:
            return float(stats[primary])
        if fallback is not None and fallback in stats and stats[fallback] is not None:
            return float(stats[fallback])
        return float("nan")

    selection_metric = "cohen_kappa" if "cohen_kappa" in metrics else "accuracy"
    sensitivity_metric = "balanced_accuracy" if "balanced_accuracy" in metrics else "accuracy"

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer,
            device, epoch, loss_scaler, args.clip_grad, model_ema,
            log_writer=log_writer, start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq, 
            ch_names=ch_names, is_binary=args.nb_classes == 1,
            input_scale_divisor=args.input_scale_divisor,
            frozen_backbone_eval_mode=args.frozen_backbone_eval_mode,
        )
        
        if data_loader_val is not None:
            val_stats = evaluate(data_loader_val, model, device, header='Val:', ch_names=ch_names, metrics=metrics, is_binary=args.nb_classes == 1, input_scale_divisor=args.input_scale_divisor)
            current_val_score = _stat_value(val_stats, selection_metric, "accuracy")
            current_val_ba = _stat_value(val_stats, sensitivity_metric, "accuracy")

            if max_primary_score < current_val_score:
                max_primary_score = current_val_score
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)

            if max_val_ba < current_val_ba:
                max_val_ba = current_val_ba
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best-ba", model_ema=model_ema)

            if args.output_dir and args.save_ckpt:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema,
                    save_ckpt_freq=args.save_ckpt_freq)

            val_acc = _stat_value(val_stats, "balanced_accuracy", "accuracy")
            val_kappa = _stat_value(val_stats, "cohen_kappa")
            val_f1 = _stat_value(val_stats, "f1_weighted")

            print(
                "Epoch {} : Training Loss: {:.5f}, val acc: {:.5f}, val kappa: {:.5f}, val f1: {:.5f}, "
                "best val kappa: {:.5f}, best val BA: {:.5f}, Time elapsed {:.2f} mins".format(
                    epoch + 1,
                    float(train_stats['loss']),
                    val_acc,
                    val_kappa,
                    val_f1,
                    float(max_primary_score),
                    float(max_val_ba),
                    (time.time() - start_time) / 60.0,
                )
            )
            if log_writer is not None:
                for key, value in val_stats.items():
                    if key == 'accuracy':
                        log_writer.update(accuracy=value, head="val", step=epoch)
                    elif key == 'balanced_accuracy':
                        log_writer.update(balanced_accuracy=value, head="val", step=epoch)
                    elif key == 'f1_weighted':
                        log_writer.update(f1_weighted=value, head="val", step=epoch)
                    elif key == 'pr_auc':
                        log_writer.update(pr_auc=value, head="val", step=epoch)
                    elif key == 'roc_auc':
                        log_writer.update(roc_auc=value, head="val", step=epoch)
                    elif key == 'cohen_kappa':
                        log_writer.update(cohen_kappa=value, head="val", step=epoch)
                    elif key == 'loss':
                        log_writer.update(loss=value, head="val", step=epoch)
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'val_{k}': v for k, v in val_stats.items()},
                         'selection_metric': selection_metric,
                         'val_selection_score': current_val_score,
                         'best_val_selection_score': max_primary_score,
                         'best_val_balanced_accuracy': max_val_ba,
                         'epoch': epoch,
                         'n_parameters': n_parameters}
        else:
            if args.output_dir and args.save_ckpt:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema,
                    save_ckpt_freq=args.save_ckpt_freq)
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

        if args.deterministic:
            log_stats['shared_parameter_checksum'] = shared_parameter_checksum(model_without_ddp)

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    # Test is evaluated only after model selection is complete. The primary
    # checkpoint is selected by validation kappa; validation BA is retained as
    # a sensitivity checkpoint without using test results for selection.
    if not args.skip_final_test and data_loader_test is not None and data_loader_val is not None:
        eval_loaders = data_loader_test if isinstance(data_loader_test, list) else [data_loader_test]
        def _mean_test_stat(stats_list, key):
            """Average scalar test stats while preserving vector/matrix diagnostics."""
            values = [item[key] for item in stats_list if key in item]
            if not values:
                raise KeyError(f"Missing test statistic: {key}")
            first = values[0]
            if isinstance(first, (list, tuple, np.ndarray)):
                return np.asarray(values, dtype=np.float64).mean(axis=0).tolist()
            return float(np.mean(values))

        def _evaluate_selected_checkpoint(filename, label):
            checkpoint_path = os.path.join(args.output_dir, filename)
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(f"Selected checkpoint not found: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model_without_ddp.load_state_dict(checkpoint["model"])
            stats_list = [
                evaluate(loader, model, device, header=f"{label} test:", ch_names=ch_names,
                         metrics=metrics, is_binary=args.nb_classes == 1,
                         input_scale_divisor=args.input_scale_divisor)
                for loader in eval_loaders
            ]
            stats = {
                key: _mean_test_stat(stats_list, key)
                for key in stats_list[0]
            }
            print(
                f"Final {label} test ({filename}): accuracy={_stat_value(stats, 'accuracy'):.5f}, "
                f"balanced_accuracy={_stat_value(stats, 'balanced_accuracy', 'accuracy'):.5f}, "
                f"cohen_kappa={_stat_value(stats, 'cohen_kappa'):.5f}, "
                f"f1_weighted={_stat_value(stats, 'f1_weighted'):.5f}"
            )
            return stats

        primary_test_stats = _evaluate_selected_checkpoint("checkpoint-best.pth", "validation-kappa")
        ba_test_stats = _evaluate_selected_checkpoint("checkpoint-best-ba.pth", "validation-BA")
        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, "final_test.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "selection_metric": selection_metric,
                    "primary_checkpoint": "checkpoint-best.pth",
                    "primary_validation_metric": selection_metric,
                    "primary_test": primary_test_stats,
                    "sensitivity_checkpoint": "checkpoint-best-ba.pth",
                    "sensitivity_validation_metric": sensitivity_metric,
                    "sensitivity_test": ba_test_stats,
                }, f, indent=2)
    elif args.skip_final_test:
        print('Skipping final test evaluation (--skip_final_test).')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    opts, ds_init = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts, ds_init)
