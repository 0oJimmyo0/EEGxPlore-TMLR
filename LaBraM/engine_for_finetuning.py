# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BEiT-v2, timm, DeiT, and DINO code bases
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# ---------------------------------------------------------
import math
import sys
from typing import Iterable, Optional
import numpy as np
import torch
from timm.utils import ModelEma
import utils
from einops import rearrange

def ensure_patch_tensor(samples, patch_size=200):
    if samples.ndim == 5:
        if samples.shape[-1] != patch_size or samples.shape[-2] <= 0:
            raise ValueError(f"Expected sequence EEG [B,L,C,S,{patch_size}], got shape={tuple(samples.shape)}")
        return samples
    if samples.ndim == 4:
        return samples
    if samples.ndim == 3:
        if samples.shape[-1] % patch_size != 0:
            raise ValueError(
                f"Expected flattened EEG sample with trailing dimension divisible by {patch_size}, "
                f"got shape={tuple(samples.shape)}"
            )
        return rearrange(samples, 'B N (A T) -> B N A T', T=patch_size)
    raise ValueError(f"Expected EEG batch with 3 or 4 dimensions, got shape={tuple(samples.shape)}")

def train_class_batch(model, samples, target, criterion, ch_names):
    outputs = model(samples, ch_names)
    if outputs.ndim == 3 and target.ndim == 2:
        loss = criterion(outputs.reshape(-1, outputs.shape[-1]), target.reshape(-1))
    else:
        loss = criterion(outputs, target)
    return loss, outputs


def _snapshot_update_parameters(model):
    core_model = model.module if hasattr(model, 'module') else model
    last_block_prefix = f"blocks.{len(core_model.blocks) - 1}."
    snapshot = {}
    for name, parameter in core_model.named_parameters():
        if not parameter.requires_grad:
            continue
        if '.lora_A' in name or '.lora_B' in name:
            category = 'lora'
        elif name.startswith('native_axis_adapter.alpha_'):
            category = 'alpha'
        elif name.startswith('native_axis_adapter.'):
            category = 'adapter_core'
        elif name.startswith(last_block_prefix):
            category = 'last_block'
        elif name.startswith('head.'):
            category = 'classifier'
        else:
            continue
        snapshot[name] = (parameter.detach().clone(), category)
    return snapshot


def _parameter_update_norms(model, snapshot):
    core_model = model.module if hasattr(model, 'module') else model
    current = dict(core_model.named_parameters())
    stats = {
        category: {'update_sq': 0.0, 'parameter_sq': 0.0}
        for category in ('adapter_core', 'alpha', 'lora', 'last_block', 'classifier')
    }
    for name, (before, category) in snapshot.items():
        if name not in current:
            continue
        parameter = current[name].detach().float()
        update_sq = float((parameter - before).pow(2).sum().cpu())
        parameter_sq = float(before.float().pow(2).sum().cpu())
        stats[category]['update_sq'] += update_sq
        stats[category]['parameter_sq'] += parameter_sq
    return {
        category: (
            values['update_sq'] ** 0.5,
            values['update_sq'] ** 0.5 / (values['parameter_sq'] ** 0.5 + 1e-12),
        )
        for category, values in stats.items()
    }


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale


def set_frozen_labram_eval_mode(model: torch.nn.Module) -> None:
    """Keep the frozen LaBraM representation deterministic during adaptation."""
    core = model.module if hasattr(model, "module") else model
    core.eval()
    if hasattr(core, "head"):
        core.head.train()
    adapter = getattr(core, "native_axis_adapter", None)
    if adapter is not None:
        adapter.train()


def frozen_repeat_features(core_model: torch.nn.Module, samples: torch.Tensor,
                           input_chans=None) -> torch.Tensor:
    """Extract deterministic representations for the frozen-backbone check.

    ISRUC loaders return subject sequences with shape ``[B, L, C, S, T]``.
    The model's public ``forward`` handles that sequence wrapper, but
    ``forward_features`` intentionally accepts only one epoch at a time.  The
    frozen-repeat diagnostic must therefore flatten the sequence, extract
    epoch features, and apply the frozen sequence encoder before comparing the
    two passes.  Ordinary four-dimensional datasets keep the original path.
    """
    if samples.ndim != 5:
        return core_model.forward_features(samples, input_chans=input_chans)

    batch_size, sequence_length, channels, patches, patch_size = samples.shape
    epoch_features = core_model.forward_features(
        samples.reshape(batch_size * sequence_length, channels, patches, patch_size),
        input_chans=input_chans,
    )
    sequence_encoder = getattr(core_model, "sequence_encoder", None)
    if sequence_encoder is None:
        return epoch_features.reshape(batch_size, sequence_length, -1)
    return sequence_encoder(
        epoch_features.reshape(batch_size, sequence_length, epoch_features.shape[-1])
    )


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, ch_names=None,
                    is_binary=True, input_scale_divisor=100.0,
                    frozen_backbone_eval_mode=False):
    input_chans = None
    if ch_names is not None:
        input_chans = utils.get_input_chans(ch_names)
    model.train(True)
    if frozen_backbone_eval_mode:
        set_frozen_labram_eval_mode(model)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('backbone_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('head_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('adapter_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('adapter_core_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('alpha_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    adapter_step_snapshot = None
    geometry_reported = False
    frozen_feature_repeat_diff = None
    train_confusion = None
    train_logit_sum = 0.0
    train_logit_sq_sum = 0.0
    train_logit_count = 0

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group.get("lr_scale", 1.0)
                if (wd_schedule_values is not None and param_group["weight_decay"] > 0
                        and not param_group.get("adapter_weight_decay_fixed", False)
                        and not param_group.get("head_weight_decay_fixed", False)):
                    param_group["weight_decay"] = wd_schedule_values[it]

        if loss_scaler is not None and data_iter_step % update_freq == 0:
            adapter_step_snapshot = _snapshot_update_parameters(model)

        samples = samples.float().to(device, non_blocking=True) / input_scale_divisor
        samples = ensure_patch_tensor(samples, patch_size=200)
        input_time_window = samples.shape[3] if samples.ndim == 5 else (samples.shape[2] if samples.shape[-1] == 200 else samples.shape[-1])

        if frozen_backbone_eval_mode and frozen_feature_repeat_diff is None:
            core_model = model.module if hasattr(model, 'module') else model
            # Check the frozen representation in full eval mode, then restore
            # train mode only for the explicitly trainable head/adapter.
            core_model.eval()
            with torch.no_grad():
                features_a = frozen_repeat_features(core_model, samples, input_chans=input_chans)
                features_b = frozen_repeat_features(core_model, samples, input_chans=input_chans)
            frozen_feature_repeat_diff = float((features_a - features_b).abs().max().cpu())
            if not torch.equal(features_a, features_b):
                raise RuntimeError(
                    "Frozen LaBraM feature extraction is nondeterministic; "
                    f"maximum repeat difference={frozen_feature_repeat_diff:.6g}"
                )
            set_frozen_labram_eval_mode(model)
            if not core_model.head.training:
                raise RuntimeError("Frozen-backbone eval mode left the classifier head in eval mode")
            adapter = getattr(core_model, "native_axis_adapter", None)
            if adapter is not None and not adapter.training:
                raise RuntimeError("Frozen-backbone eval mode left the trainable adapter in eval mode")
        
        targets = targets.to(device, non_blocking=True)
        if is_binary:
            targets = targets.float().unsqueeze(-1)

        if loss_scaler is None:
            samples = samples.half()
            loss, output = train_class_batch(
                model, samples, targets, criterion, input_chans)
        else:
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss, output = train_class_batch(
                    model, samples, targets, criterion, input_chans)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        adapter_diagnostics = {
            'input_time_window': float(input_time_window),
            'input_finite': float(torch.isfinite(samples).all()),
            'input_nonfinite_count': float((~torch.isfinite(samples)).sum()),
            'input_max_abs': float(samples.detach().abs().amax().cpu()),
            'output_finite': float(torch.isfinite(output).all()),
            'output_nonfinite_count': float((~torch.isfinite(output)).sum()),
            'output_max_abs': float(output.detach().abs().amax().cpu()),
        }
        core_model = model.module if hasattr(model, 'module') else model

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            adapter_diagnostics.update({
                'amp_grad_finite': float(loss_scaler.last_grad_finite),
                'amp_step_skipped': float(loss_scaler.last_step_skipped),
                'amp_nonfinite_param_count': float(loss_scaler.last_nonfinite_param_count),
                'amp_scale_before': float(loss_scaler.last_scale_before),
                'amp_scale_after': float(loss_scaler.last_scale_after),
            })
            if loss_scaler.last_nonfinite_group_names:
                print(
                    "AMP nonfinite gradient groups: "
                    + ",".join(loss_scaler.last_nonfinite_group_names),
                    flush=True,
                )
            classifier_grad_sq = 0.0
            for parameter in core_model.head.parameters():
                if parameter.grad is not None:
                    classifier_grad_sq += float(parameter.grad.detach().float().pow(2).sum().cpu())
            adapter_diagnostics['classifier_grad_norm'] = classifier_grad_sq ** 0.5
            if hasattr(core_model, 'get_adapter_diagnostics'):
                # NativeScaler performs backward, unscaling, and optimizer.step,
                # but it does not clear gradients. Capture them before zero_grad.
                adapter_diagnostics.update(core_model.get_adapter_diagnostics())
                if (
                    not geometry_reported
                    and utils.is_main_process()
                    and 'patch_attention_sequence_length' in adapter_diagnostics
                ):
                    print(
                        "[LaBraM adapter geometry] "
                        f"input_shape={tuple(samples.shape)} "
                        f"token_grid=[B,{int(adapter_diagnostics['adapter_channel_count'])},"
                        f"{int(adapter_diagnostics['adapter_patch_count'])},"
                        f"{int(adapter_diagnostics['adapter_embed_dim'])}] "
                        f"patch_attention_sequence_length="
                        f"{int(adapter_diagnostics['patch_attention_sequence_length'])} "
                        f"temporal_interactions="
                        f"{int(adapter_diagnostics['patch_temporal_interactions_active'])}",
                        flush=True,
                    )
                    geometry_reported = True
            if (data_iter_step + 1) % update_freq == 0 and adapter_step_snapshot is not None:
                update_norms = _parameter_update_norms(model, adapter_step_snapshot)
                for category, (absolute_norm, relative_norm) in update_norms.items():
                    adapter_diagnostics[f'{category}_update_norm'] = absolute_norm
                    adapter_diagnostics[f'{category}_relative_update_norm'] = relative_norm
                adapter_step_snapshot = None
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        if is_binary:
            class_acc = utils.get_metrics(torch.sigmoid(output).detach().cpu().numpy(), targets.detach().cpu().numpy(), ["accuracy"], is_binary)["accuracy"]
        else:
            flat_output = output.detach().float().reshape(-1, output.shape[-1])
            flat_targets = targets.detach().long().reshape(-1)
            flat_predictions = flat_output.argmax(dim=-1)
            num_classes = flat_output.shape[-1]
            batch_confusion = torch.bincount(
                flat_targets * num_classes + flat_predictions,
                minlength=num_classes * num_classes,
            ).reshape(num_classes, num_classes).to(dtype=torch.float64)
            if train_confusion is None:
                train_confusion = batch_confusion
            else:
                train_confusion += batch_confusion
            # Accumulate diagnostics in float64 so large logits do not make
            # the reported variance overflow while the actual metrics remain
            # finite.
            flat_output64 = flat_output.double()
            train_logit_sum += float(flat_output64.sum().cpu())
            train_logit_sq_sum += float(flat_output64.square().sum().cpu())
            train_logit_count += int(flat_output.numel())
            class_acc = (flat_predictions == flat_targets).float().mean()
            
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        backbone_max_lr = 0.
        head_max_lr = 0.
        adapter_max_lr = 0.
        adapter_core_max_lr = 0.
        alpha_max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
            if group.get("is_adapter", False):
                adapter_max_lr = max(adapter_max_lr, group["lr"])
                if group.get("is_adapter_alpha", False):
                    alpha_max_lr = max(alpha_max_lr, group["lr"])
                else:
                    adapter_core_max_lr = max(adapter_core_max_lr, group["lr"])
            elif group.get("is_head", False):
                head_max_lr = max(head_max_lr, group["lr"])
            else:
                backbone_max_lr = max(backbone_max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        metric_logger.update(backbone_lr=backbone_max_lr)
        metric_logger.update(head_lr=head_max_lr)
        metric_logger.update(adapter_lr=adapter_max_lr)
        metric_logger.update(adapter_core_lr=adapter_core_max_lr)
        metric_logger.update(alpha_lr=alpha_max_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        for name, value in adapter_diagnostics.items():
            metric_logger.update(**{name: value})

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(backbone_lr=backbone_max_lr, head="opt")
            log_writer.update(head_lr=head_max_lr, head="opt")
            log_writer.update(adapter_lr=adapter_max_lr, head="opt")
            log_writer.update(adapter_core_lr=adapter_core_max_lr, head="opt")
            log_writer.update(alpha_lr=alpha_max_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            for name, value in adapter_diagnostics.items():
                log_writer.update(**{name: value}, head="adapter")

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if frozen_feature_repeat_diff is not None:
        stats['frozen_feature_repeat_diff'] = frozen_feature_repeat_diff
    if not is_binary and train_confusion is not None:
        utils.all_reduce(train_confusion)
        reduction = torch.tensor(
            [train_logit_sum, train_logit_sq_sum, float(train_logit_count)],
            dtype=torch.float64,
            device=train_confusion.device,
        )
        utils.all_reduce(reduction)
        stats.update(utils.classification_diagnostics_from_confusion(
            train_confusion,
            logit_sum=float(reduction[0].cpu()),
            logit_sq_sum=float(reduction[1].cpu()),
            logit_count=int(reduction[2].cpu()),
        ))
    return stats


@torch.no_grad()
def evaluate(data_loader, model, device, header='Test:', ch_names=None, metrics=['acc'],
             is_binary=True, input_scale_divisor=100.0):
    input_chans = None
    if ch_names is not None:
        input_chans = utils.get_input_chans(ch_names)
    if is_binary:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")

    # switch to evaluation mode
    model.eval()
    pred = []
    true = []
    print_freq = max(len(data_loader), 1)
    for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        EEG = batch[0]
        target = batch[-1]
        EEG = EEG.float().to(device, non_blocking=True) / input_scale_divisor
        EEG = ensure_patch_tensor(EEG, patch_size=200)
        target = target.to(device, non_blocking=True)
        if is_binary:
            target = target.float().unsqueeze(-1)
        
        # compute output
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            output = model(EEG, input_chans=input_chans)
            if output.ndim == 3 and target.ndim == 2:
                loss = criterion(output.reshape(-1, output.shape[-1]), target.reshape(-1))
            else:
                loss = criterion(output, target)
        
        if is_binary:
            output = torch.sigmoid(output).cpu()
        else:
            output = output.cpu()
        target = target.cpu()
        pred.append(output.reshape(-1, output.shape[-1]) if output.ndim == 3 else output)
        true.append(target.reshape(-1) if target.ndim == 2 else target)

        batch_size = EEG.shape[0]
        metric_logger.update(loss=loss.item())
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* loss {losses.global_avg:.3f}'
          .format(losses=metric_logger.loss))
    
    pred = torch.cat(pred, dim=0).numpy()
    true = torch.cat(true, dim=0).numpy()

    ret = utils.get_metrics(pred, true, metrics, is_binary, 0.5)
    ret['loss'] = metric_logger.loss.global_avg
    if not is_binary:
        prediction = np.argmax(pred, axis=-1).reshape(-1)
        target = true.reshape(-1).astype(np.int64)
        num_classes = pred.shape[-1]
        confusion = np.bincount(
            target * num_classes + prediction,
            minlength=num_classes * num_classes,
        ).reshape(num_classes, num_classes)
        ret.update(utils.classification_diagnostics_from_confusion(
            confusion,
            logit_sum=float(np.asarray(pred, dtype=np.float64).sum()),
            logit_sq_sum=float(np.square(np.asarray(pred, dtype=np.float64)).sum()),
            logit_count=int(pred.size),
        ))
    return ret
