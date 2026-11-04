# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BEiT-v2, timm, DeiT, and DINO code bases
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# ---------------------------------------------------------
import torch
from torch import optim as optim

from timm.optim.adafactor import Adafactor
from timm.optim.adahessian import Adahessian
from timm.optim.adamp import AdamP
from timm.optim.lookahead import Lookahead
from timm.optim.nadam import Nadam
from timm.optim.nvnovograd import NvNovoGrad
from timm.optim.radam import RAdam
from timm.optim.rmsprop_tf import RMSpropTF
from timm.optim.sgdp import SGDP

import json

try:
    from apex.optimizers import FusedNovoGrad, FusedAdam, FusedLAMB, FusedSGD
    has_apex = True
except ImportError:
    has_apex = False


def get_num_layer_for_vit(var_name, num_max_layer):
    if var_name in ("cls_token", "mask_token", "pos_embed"):
        return 0
    elif var_name.startswith("patch_embed"):
        return 0
    elif var_name.startswith("rel_pos_bias"):
        return num_max_layer - 1
    elif var_name.startswith("blocks"):
        layer_id = int(var_name.split('.')[1])
        return layer_id + 1
    else:
        return num_max_layer - 1


class LayerDecayValueAssigner(object):
    def __init__(self, values):
        self.values = values

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        return get_num_layer_for_vit(var_name, len(self.values))


def get_parameter_groups(model, weight_decay=1e-5, skip_list=(), get_num_layer=None,
                         get_layer_scale=None, adapter_name_prefix=None,
                         adapter_lr_scale=1.0, adapter_alpha_lr_scale=None,
                         adapter_alpha_name_prefix=None, adapter_weight_decay=None,
                         backbone_lr_scale=1.0, head_lr_scale=1.0,
                         head_weight_decay=None,
                         head_name_prefix='head.', **kwargs):
    if adapter_lr_scale < 0:
        raise ValueError(f"adapter_lr_scale must be non-negative, got {adapter_lr_scale!r}")
    if adapter_alpha_lr_scale is not None and adapter_alpha_lr_scale < 0:
        raise ValueError(
            f"adapter_alpha_lr_scale must be non-negative, got {adapter_alpha_lr_scale!r}"
        )
    if adapter_alpha_lr_scale is None:
        adapter_alpha_lr_scale = adapter_lr_scale
    if backbone_lr_scale < 0:
        raise ValueError(f"backbone_lr_scale must be non-negative, got {backbone_lr_scale!r}")
    if head_lr_scale < 0:
        raise ValueError(f"head_lr_scale must be non-negative, got {head_lr_scale!r}")

    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(kwargs.get('filter_name', [])) > 0:
            flag = False
            for filter_n in kwargs.get('filter_name', []):
                if filter_n in name:
                    print(f"filter {name} because of the pattern {filter_n}")
                    flag = True
            if flag:
                continue
        # LoRA is a separate generic control, but its trainable A/B matrices
        # still need the adapter LR/decay bucket when the lora mode is used.
        is_adapter = bool(
            (adapter_name_prefix and name.startswith(adapter_name_prefix))
            or ".lora_A" in name or ".lora_B" in name
        )
        is_adapter_alpha = bool(
            is_adapter and adapter_alpha_name_prefix and name.startswith(adapter_alpha_name_prefix)
        )
        is_head = bool(head_name_prefix and name.startswith(head_name_prefix))
        is_head_weight = is_head and param.ndim > 1 and not name.endswith(".bias") and name not in skip_list
        if param.ndim <= 1 or name.endswith(".bias") or name in skip_list: # param.ndim <= 1 len(param.shape) == 1
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
            if is_adapter and adapter_weight_decay is not None:
                this_weight_decay = adapter_weight_decay
            if is_head and head_weight_decay is not None:
                this_weight_decay = head_weight_decay
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if is_adapter_alpha:
            group_name = "adapter_alpha_%s" % group_name
        elif is_adapter:
            group_name = "adapter_%s" % group_name
        elif is_head:
            group_name = "head_%s" % group_name
        else:
            group_name = "backbone_%s" % group_name

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.
            if is_adapter_alpha:
                scale *= adapter_alpha_lr_scale
            elif is_adapter:
                scale *= adapter_lr_scale
            elif is_head:
                scale *= head_lr_scale
            else:
                scale *= backbone_lr_scale

            parameter_group_names[group_name] = {
                "group_name": group_name,
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale,
                "is_adapter": is_adapter,
                "is_adapter_alpha": is_adapter_alpha,
                "is_head": is_head,
                "is_backbone": not is_adapter and not is_head,
                "adapter_weight_decay_fixed": is_adapter and adapter_weight_decay is not None,
                "head_weight_decay_fixed": is_head_weight and head_weight_decay is not None,
            }
            parameter_group_vars[group_name] = {
                "group_name": group_name,
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale,
                "is_adapter": is_adapter,
                "is_adapter_alpha": is_adapter_alpha,
                "is_head": is_head,
                "is_backbone": not is_adapter and not is_head,
                "adapter_weight_decay_fixed": is_adapter and adapter_weight_decay is not None,
                "head_weight_decay_fixed": is_head_weight and head_weight_decay is not None,
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


def create_optimizer(args, model, get_num_layer=None, get_layer_scale=None, filter_bias_and_bn=True, skip_list=None, **kwargs):
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay
    if (weight_decay or kwargs.get('adapter_weight_decay') is not None) and filter_bias_and_bn:
        skip = {}
        if skip_list is not None:
            skip = skip_list
        elif hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        print(f"Skip weight decay name marked in model: {skip}")
        parameters = get_parameter_groups(model, weight_decay, skip, get_num_layer, get_layer_scale, **kwargs)
        weight_decay = 0.
    else:
        parameters = model.parameters()

    if 'fused' in opt_lower:
        assert has_apex and torch.cuda.is_available(), 'APEX and CUDA required for fused optimizers'

    opt_args = dict(lr=args.lr, weight_decay=weight_decay)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['betas'] = args.opt_betas
    
    print('Optimizer config:', opt_args)
    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'momentum':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == 'nadam':
        optimizer = Nadam(parameters, **opt_args)
    elif opt_lower == 'radam':
        optimizer = RAdam(parameters, **opt_args)
    elif opt_lower == 'adamp':
        optimizer = AdamP(parameters, wd_ratio=0.01, nesterov=True, **opt_args)
    elif opt_lower == 'sgdp':
        optimizer = SGDP(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'adadelta':
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == 'adafactor':
        if not args.lr:
            opt_args['lr'] = None
        optimizer = Adafactor(parameters, **opt_args)
    elif opt_lower == 'adahessian':
        optimizer = Adahessian(parameters, **opt_args)
    elif opt_lower == 'rmsprop':
        optimizer = optim.RMSprop(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == 'rmsproptf':
        optimizer = RMSpropTF(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == 'nvnovograd':
        optimizer = NvNovoGrad(parameters, **opt_args)
    elif opt_lower == 'fusedsgd':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'fusedmomentum':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'fusedadam':
        optimizer = FusedAdam(parameters, adam_w_mode=False, **opt_args)
    elif opt_lower == 'fusedadamw':
        optimizer = FusedAdam(parameters, adam_w_mode=True, **opt_args)
    elif opt_lower == 'fusedlamb':
        optimizer = FusedLAMB(parameters, **opt_args)
    elif opt_lower == 'fusednovograd':
        opt_args.setdefault('betas', (0.95, 0.98))
        optimizer = FusedNovoGrad(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"
        raise ValueError

    if len(opt_split) > 1:
        if opt_split[0] == 'lookahead':
            optimizer = Lookahead(optimizer)

    return optimizer
