#!/usr/bin/env bash
# Source this file from Bash. These functions call the existing training programs.
TMLR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TMLR_ROOT

run_cbramod() (
  set -euo pipefail
  local dataset="${1:?dataset: FACED, ISRUC, TUEV, SEED-V, PHYSIONET-MI}"
  local condition="${2:?condition: probe, native, token, full, native_full, branch, generic, lora, upper}"
  local seed="${3:?training seed}"
  shift 3
  : "${DATA_ROOT:?Set DATA_ROOT to the prepared-data directory}"
  : "${WEIGHTS_ROOT:?Set WEIGHTS_ROOT to the checkpoint directory}"
  : "${OUTPUT_ROOT:?Set OUTPUT_ROOT to a new output directory}"
  local entry config data branch=channel_patch
  local -a options=() method=()
  case "$dataset" in
    FACED)
      entry=faced; config=faced_tmlr_locked; data=FACED
      options+=(--lr 1e-4) ;;
    ISRUC)
      entry=isruc; config=isruc_tmlr_locked; data=ISRUC
      options+=(--head-seed 10042 --loader-seed "$((20000 + seed))"
        --adapter-alpha-weight-decay 0.05) ;;
    TUEV)
      entry=tuev; config=tuev_tmlr_locked; data=TUEV
      options+=(--lr 1e-4 --head-seed "$((10000 + seed))"
        --loader-seed "$((20000 + seed))"
        --audit-path "$OUTPUT_ROOT/data_checks/tuev.json") ;;
    SEED-V)
      entry=seedv; config=seedv_tmlr_locked; data=SEED-V; branch=channel ;;
    PHYSIONET-MI)
      entry=physionet_mi; config=physionet_mi_tmlr_locked; data=PHYSIO_MI
      options+=(--head-seed "$((10000 + seed))"
        --loader-seed "$((20000 + seed))"
        --audit-path "$OUTPUT_ROOT/data_checks/physionet_mi.json") ;;
    *) printf 'Unknown dataset: %s\n' "$dataset" >&2; return 2 ;;
  esac
  case "$condition" in
    probe) method+=(--method frozen_probe) ;;
    native) method+=(--method interaction_aligned --adapter-type "$branch") ;;
    token) method+=(--method axis_blind --axis-blind-bottleneck 148) ;;
    generic) method+=(--method generic_bottleneck --generic-bottleneck 64) ;;
    lora) method+=(--method lora --lora-rank 8 --lora-alpha 16) ;;
    upper) method+=(--method upper_k_finetune --upper-k 2) ;;
    full) method+=(--method full_finetune) ;;
    native_full) method+=(--method native_full_finetune --adapter-type "$branch") ;;
    branch)
      [[ "$dataset" == TUEV ]] || { echo 'CBraMod branch control is TUEV only' >&2; return 2; }
      method+=(--method axis_decomposed_mlp --adapter-type channel_patch --axis-decomposed-mlp-bottleneck 147) ;;
    *) printf 'Unknown condition: %s\n' "$condition" >&2; return 2 ;;
  esac
  cd "$TMLR_ROOT/CBraMod"
  python "run_${entry}_tmlr.py" --config "configs/${config}.yaml" \
    --dataset-path "$DATA_ROOT/$data" \
    --checkpoint "$WEIGHTS_ROOT/pretrained_weights.pth" \
    --output-root "$OUTPUT_ROOT/cbramod/$entry" \
    --run-id "${entry}_${condition}_s${seed}" --device cuda \
    --seed "$seed" --adapter-seed "$((30000 + seed))" \
    "${options[@]}" "${method[@]}" "$@"
)

run_labram() (
  set -euo pipefail
  local dataset="${1:?dataset: FACED, ISRUC, TUEV, PhysioNet-MI}"
  local condition="${2:?condition: probe, native, token, branch, full, native_full, generic, lora, upper, trainable32}"
  local seed="${3:?training seed}"
  shift 3
  : "${DATA_ROOT:?Set DATA_ROOT to the prepared-data directory}"
  : "${WEIGHTS_ROOT:?Set WEIGHTS_ROOT to the checkpoint directory}"
  : "${OUTPUT_ROOT:?Set OUTPUT_ROOT to a new output directory}"
  local data epochs batch lr warmup scale drop_path
  local -a adapter=() training=(--labram_backbone_mode frozen --frozen_backbone_eval_mode --backbone_lr_scale 0) extra=()
  if [[ "$condition" == full || "$condition" == native_full ]]; then
    [[ "$dataset" == PhysioNet-MI ]] || { echo 'LaBraM full fine-tuning is documented for PhysioNet-MI only' >&2; return 2; }
    training=(--labram_backbone_mode trainable --backbone_lr_scale 1)
  fi
  case "$condition" in
    lora)
      local lora_backbone_scale=1
      [[ "$dataset" != TUEV ]] || lora_backbone_scale=0
      training=(--labram_backbone_mode lora --backbone_lr_scale "$lora_backbone_scale") ;;
    upper) training=(--labram_backbone_mode upper_k --backbone_lr_scale 1) ;;
    trainable32)
      [[ "$dataset" == ISRUC ]] || { echo 'trainable32 is the ISRUC context comparison' >&2; return 2; }
      training=(--labram_backbone_mode trainable --backbone_lr_scale 1) ;;
  esac
  case "$dataset" in
    FACED) data=FACED; epochs=80; batch=32; lr=7e-4; warmup=10; scale=100; drop_path=0.1 ;;
    ISRUC) data=ISRUC; epochs=30; batch=16; lr=2e-4; warmup=1; scale=1; drop_path=0.1 ;;
    TUEV) data=TUEV; epochs=15; batch=64; lr=5e-4; warmup=5; scale=100; drop_path=0.2 ;;
    PhysioNet-MI) data=PHYSIO_MI; epochs=40; batch=64; lr=5e-4; warmup=5; scale=100; drop_path=0.1 ;;
    *) printf 'Unknown dataset: %s\n' "$dataset" >&2; return 2 ;;
  esac
  [[ "$dataset" != PhysioNet-MI ]] || extra+=(--clip_grad 1)
  if [[ "$dataset" == ISRUC && "$condition" =~ ^(lora|upper|trainable32)$ ]]; then
    extra+=(--labram_adapter_lr_scale 0.1 --labram_adapter_alpha_lr_scale 0.1)
  fi
  case "$condition" in
    probe|full) adapter+=(--labram_adapter_type none) ;;
    generic) adapter+=(--labram_adapter_type generic --labram_adapter_variant full
      --labram_adapter_bottleneck 64 --experiment_method generic_bottleneck) ;;
    lora) adapter+=(--labram_adapter_type none --labram_lora_rank 8
      --labram_lora_alpha 16 --labram_lora_target qkv) ;;
    upper) adapter+=(--labram_adapter_type none --labram_upper_k 2) ;;
    trainable32) adapter+=(--labram_adapter_type channel_patch
      --labram_adapter_variant low_rank --labram_adapter_bottleneck 32) ;;
    native|native_full) adapter+=(--labram_adapter_type channel_patch
      --labram_adapter_variant low_rank --labram_adapter_bottleneck 64
      --experiment_method interaction_aligned) ;;
    token) adapter+=(--labram_adapter_type generic --labram_adapter_variant full
      --labram_adapter_bottleneck 213 --target_adapter_params 85810
      --experiment_method axis_blind) ;;
    branch) adapter+=(--labram_adapter_type channel_patch --labram_adapter_variant low_rank
      --labram_adapter_operator mlp --labram_adapter_bottleneck 106
      --experiment_method axis_decomposed_mlp) ;;
    *) printf 'Unknown condition: %s\n' "$condition" >&2; return 2 ;;
  esac
  cd "${LABRAM_SOURCE_DIR:-$TMLR_ROOT/LaBraM}"
  OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc_per_node=1 \
    run_class_finetuning.py --model labram_base_patch200_200 \
    --finetune "$WEIGHTS_ROOT/labram-base.pth" \
    --dataset "$dataset" --data_path "$DATA_ROOT/$data" \
    --output_dir "$OUTPUT_ROOT/labram/${dataset}_${condition}_s${seed}" \
    --log_dir "$OUTPUT_ROOT/tensorboard/${dataset}_${condition}_s${seed}" \
    --seed "$seed" --batch_size "$batch" --epochs "$epochs" \
    --num_workers 0 --update_freq 1 --lr "$lr" --requested_lr_string "$lr" \
    --warmup_epochs "$warmup" --min_lr 1e-6 --weight_decay 0.05 \
    --layer_decay 0.65 --smoothing 0.1 --drop 0 --drop_path "$drop_path" \
    --input_scale_divisor "$scale" --disable_rel_pos_bias --abs_pos_emb \
    --disable_qkv_bias --use_mean_pooling --save_ckpt_freq 1 \
    --strict_checkpoint_load --no_auto_resume \
    "${training[@]}" --head_lr_scale 1 \
    --labram_adapter_heads 4 --labram_adapter_dropout 0 \
    --labram_adapter_init_alpha 0.01 --labram_adapter_gamma 1 \
    --labram_adapter_seed 12345 --labram_adapter_depth_mode none \
    --labram_adapter_lr_scale 1 --labram_adapter_alpha_lr_scale 1 \
    --isruc_sequence_dropout 0.1 "${extra[@]}" "${adapter[@]}" "$@"
)
