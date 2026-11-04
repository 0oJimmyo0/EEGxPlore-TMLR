# Backbone-Informed Adaptation after EEG Pretraining

Code and reproduction guide for **Backbone-Informed Adaptation after EEG
Pretraining: A Capacity-Controlled Study**.

Mingyang Jiang, Yamin Li, Fan Ma, Daniel Moyer, Hua Xu, and Catie Chang.

Department of Computer Science, Vanderbilt University (Jiang, Li, Moyer,
Chang); Department of Biomedical Informatics and Data Science, Yale School
of Medicine (Ma, Xu).

The paper compares a native-axis attention adapter with a parameter-matched,
token-wise MLP and a no-adapter classifier on frozen **CBraMod** and **LaBraM**
features. The six primary settings are the two backbones on FACED, ISRUC,
and TUEV. PhysioNet-MI, SEED-V, full fine-tuning, and the branch/sequence
controls provide additional comparisons.

## Start here

Clone the complete code release, which includes both backbones:

```bash
git clone https://github.com/0oJimmyo0/EEGxPlore-TMLR.git
cd EEGxPlore-TMLR
```

1. [Install the environment](#1-environment).
2. [Download the pretrained weights](#2-pretrained-weights).
3. [Obtain and preprocess the datasets](#3-datasets).
4. [Check the prepared data](#4-check-the-inputs).
5. [Run the primary comparisons](#5-primary-results-rq1-and-rq2).
6. [Run additional experiments](#6-additional-experiments).
7. [Read and aggregate the results](#7-results-and-paper-tables).

Use the modified `CBraMod/` and `LaBraM/` directories supplied with this study.
The original upstream repositories do not contain all the adapter controls
used here. Backbone pretraining and tokenizer training are unnecessary: the
experiments start from released checkpoints.

```text
README.md                   This guide
reproduce/commands.sh       Portable wrappers around the existing training entry points
reproduce/requirements.txt Downstream environment versions
reproduce/PREPROCESSING.md Dataset acquisition, script configuration, and storage formats
CBraMod/                    CBraMod models, dataset readers, configs, and training
LaBraM/                     LaBraM models, dataset readers, and training
paper/                      Figure scripts and numerical inputs
```

## 1. Environment

The experiments used Linux, Python 3.11.5, and NVIDIA GPUs, including RTX A6000
48 GB cards. Each command below uses **one GPU**; repeat seeds sequentially
or assign separate GPUs. The original cluster `.slurm` files contain local
scheduler settings; the commands in this guide do not require Slurm.

Create a Python 3.11 environment, then run from the directory containing this
README:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r reproduce/requirements-lock.txt
python -c 'import torch, timm, lmdb, mne; print(torch.__version__); print(torch.cuda.is_available())'
```

CUDA availability must print `True` for the training commands. Install a
PyTorch build compatible with your NVIDIA driver if the default wheel does
not work; use the [official installation instructions](https://pytorch.org/get-started/locally/).

`reproduce/requirements-lock.txt` contains the complete public-package
Linux environment checked with Python 3.11.5. `requirements.txt` lists the
direct dependencies. The paper used the same principal package versions in
cluster builds: PyTorch 2.12.1, torchvision 0.27.1, NumPy 2.4.2, SciPy 1.17.0,
scikit-learn 1.8.0, MNE 1.12.1, timm 0.4.12, and LMDB 1.7.5.
The default public PyTorch wheel uses CUDA 13.0; select a compatible build
using the installation instructions above if your driver requires it.
DeepSpeed and PyHealth are unnecessary for these downstream commands.

## 2. Pretrained weights

Download the original **CBraMod** checkpoint from its
[released model page](https://huggingface.co/weighting666/CBraMod), and
**LaBraM-base** from the checkpoint resources in the
[original LaBraM repository](https://github.com/935963004/LaBraM/tree/main/checkpoints).
Do not substitute LaBraM-large or a downstream-fine-tuned checkpoint.

Place them in a directory of your choice:

```text
weights/
  pretrained_weights.pth   CBraMod
  labram-base.pth          LaBraM-base
```

The files used for the paper have these SHA-256 digests:

| File | SHA-256 |
| --- | --- |
| `pretrained_weights.pth` | `0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178` |
| `labram-base.pth` | `7c50583826afac76c4ab18f43d958df40496c8229accc09ed6a227c9bb57c37c` |

```bash
sha256sum /absolute/path/to/weights/pretrained_weights.pth
sha256sum /absolute/path/to/weights/labram-base.pth
```

## 3. Datasets

Download each dataset from its provider and follow its access terms. Raw EEG
and pretrained weights are obtained separately from the code. Both backbones
read the **same prepared examples and splits**.

| Dataset | Obtain | Prepared format | Train / validation / test examples |
| --- | --- | --- | --- |
| FACED | [Dataset publication and data-access instructions](https://doi.org/10.1038/s41597-023-02650-w); use `Processed_data` | LMDB, `(32,10,200)` | 6,720 / 1,680 / 1,932 |
| ISRUC | [ISRUC-Sleep](https://sleeptight.isr.uc.pt/), subgroup I, 100 subjects | Paired signal/label `.npy` sequences, `(20,6,6000)` | 3,559 / 468 / 435 |
| TUEV | [TUH EEG corpus access](https://isip.piconepress.com/projects/tuh_eeg/), select the **Events** corpus | Event pickle files, `(16,1000)` | 68,712 / 15,220 / 29,421 |
| PhysioNet-MI | [EEG Motor Movement/Imagery v1.0.0](https://physionet.org/content/eegmmidb/1.0.0/) | LMDB, `(64,4,200)` | 6,300 / 1,734 / 1,803 |
| SEED-V | [SEED dataset application](https://bcmi.sjtu.edu.cn/ApplicationForm/apply_form/), select **SEED-V** and obtain EEG `.cnt` files | LMDB, `(62,1,200)` | 34,432 / 42,960 / 40,352 |

Shapes exclude the batch dimension. ISRUC has 20 consecutive 30-second epochs
per example; its loaders reshape each epoch into 30 one-second patches.
FACED, ISRUC, and PhysioNet-MI use subject-disjoint splits. SEED-V uses
within-subject trial splits. TUEV uses record/prefix splits.

Follow [the preprocessing instructions](reproduce/PREPROCESSING.md) before
training. They identify the existing scripts, the path variables to change,
the channel metadata, and the expected directory layout. The scripts have
fixed paths rather than a common command-line interface.

TUEV release versions can produce different counts. The count row above
identifies the examples used in this paper; it does not establish the original
corpus version. The exact paper split filenames are included in
`reproduce/splits/TUEV/`; verify them with `reproduce/check_splits.py`. The released CBraMod checkpoint does not identify
whether TUEV was excluded from pretraining, as discussed in the paper.

## 4. Check the inputs

Use absolute paths. The launcher functions change into each backbone directory.
Set these variables once in the Bash shell where you will run experiments:

```bash
export DATA_ROOT=/absolute/path/to/prepared_data
export WEIGHTS_ROOT=/absolute/path/to/weights
export OUTPUT_ROOT=/absolute/path/to/new_experiment_outputs
source reproduce/commands.sh
```

`DATA_ROOT` must contain `FACED/`, `ISRUC/`, `TUEV/`, `PHYSIO_MI/`, and
`SEED-V/` for the corresponding experiments. Prepare only the datasets you
intend to evaluate. Keep each experiment's output directory separate from
previous runs.

Check that the entry points import and display their configuration options:

```bash
(cd "$TMLR_ROOT/CBraMod" && python run_faced_tmlr.py --help)
(cd "$TMLR_ROOT/LaBraM" && python -c 'from run_class_finetuning import get_args; print(get_args()[0])')
```

Before the TUEV runs, verify split membership, then create the data check
required by the CBraMod reader:

```bash
python "$TMLR_ROOT/reproduce/check_splits.py" "$DATA_ROOT/TUEV"
run_cbramod TUEV probe 42 --audit-only
```

The output is `$OUTPUT_ROOT/data_checks/tuev.json`. Regenerate it if you
move or rebuild the prepared TUEV directory. FACED and ISRUC can also be
checked without training:

```bash
run_cbramod FACED probe 42 --audit-only
run_cbramod ISRUC probe 42 --audit-only
```

For a short end-to-end CBraMod check, use a separate run name:

```bash
run_cbramod FACED probe 42 --smoke --run-id faced_probe_smoke
```

A smoke run is a software check and is not included in the paper results.

## 5. Primary results: RQ1 and RQ2

Run all three conditions at training seeds **42, 1024, and 3407**.

| Name used below | CBraMod method | LaBraM construction | Comparison |
| --- | --- | --- | --- |
| `probe` | `frozen_probe` | No adapter, frozen backbone | RQ2 reference |
| `native` | `interaction_aligned`, channel + patch | Channel + patch attention, low-rank width 64 | RQ1/RQ2 native adapter |
| `token` | `axis_blind`, width 148 | `generic` token-wise MLP, width 213 | RQ1 control |

Adapter counts exclude the shared downstream classifier: CBraMod native /
token **59,610 / 59,949**; LaBraM native / token **85,810 / 86,014**.

| Backbone | Dataset | Epochs | Batch | Adapter LR | Head LR | Warmup epochs |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CBraMod | FACED | 50 | 64 | 5e-4 | 5e-4 | 0 |
| CBraMod | ISRUC | 20 | 8 | 2e-5 | 3.536e-4 | 0 |
| CBraMod | TUEV | 20 | 8 | 5e-5 | 2e-4 | 0 |
| LaBraM | FACED | 80 | 32 | 7e-4 | 7e-4 | 10 |
| LaBraM | ISRUC | 30 | 16 | 2e-4 | 2e-4 | 1 |
| LaBraM | TUEV | 15 | 64 | 5e-4 | 5e-4 | 5 |

The helpers set these values using the existing YAML files and explicit
arguments. Shared settings are AdamW, weight decay 0.05, label smoothing
0.1, cosine schedules, adapter dropout 0, residual multiplier 1, and initial
residual scale 0.01. CBraMod zero-initializes adapter output projections;
LaBraM does not. Frozen pretrained encoders remain in evaluation mode during
training. CBraMod--ISRUC trains its downstream sequence encoder; the primary
LaBraM--ISRUC comparison freezes its newly initialized sequence encoder.

### CBraMod

A single run, using the shell setup above:

```bash
run_cbramod FACED native 42
```

All 27 CBraMod primary runs:

```bash
for dataset in FACED ISRUC TUEV; do
  for seed in 42 1024 3407; do
    for condition in probe native token; do
      run_cbramod "$dataset" "$condition" "$seed" || exit 1
    done
  done
done
```

The underlying entry points are `CBraMod/run_faced_tmlr.py`,
`run_isruc_tmlr.py`, and `run_tuev_tmlr.py`. `finetune_main.py` is the
upstream interface, not the entry point for these paper comparisons.

### LaBraM

A single run:

```bash
run_labram FACED native 42
```

All 27 LaBraM primary runs:

```bash
for dataset in FACED ISRUC TUEV; do
  for seed in 42 1024 3407; do
    for condition in probe native token; do
      run_labram "$dataset" "$condition" "$seed" || exit 1
    done
  done
done
```

These commands launch `LaBraM/run_class_finetuning.py` through single-process
`torchrun`. The flags include the original mean-pooling head, absolute
positional embeddings, and the appropriate channel mapping. Do not pre-divide
the stored signals: the readers/training engine apply `/100` exactly once for
FACED, TUEV, PhysioNet-MI, and SEED-V, and no division for ISRUC.

## 6. Additional experiments

Use new output directories for follow-ups so their results cannot be confused
with the primary frozen-backbone runs. Extra flags after the seed override
helper defaults.

### Branch-preserving MLP

LaBraM retains the native branches and replaces attention with independent
MLPs of width 106 (86,214 adapter parameters):

```bash
run_labram FACED branch 42
run_labram ISRUC branch 42
run_labram TUEV branch 42 --labram_adapter_seed 30042
run_cbramod TUEV branch 42
```

Repeat at all three training seeds. FACED/ISRUC keep adapter seed 12345;
for the historical TUEV branch control use `30000 + training_seed`.
The ordinary LaBraM native/token references keep adapter seed 12345.
The CBraMod TUEV branch control uses width 147 (59,696 parameters).

### LaBraM ISRUC with a trainable sequence encoder

This follow-up uses the separate `source/` implementation below, which adds
`--isruc_train_sequence_encoder`. That flag is absent from the primary
`LaBraM/run_class_finetuning.py`.

```bash
export LABRAM_SOURCE_DIR="$TMLR_ROOT/LaBraM/analysis/isruc_sequence_trainable_20261005/source"
export OUTPUT_ROOT=/absolute/path/to/new_sequence_followup_outputs
for seed in 42 1024 3407; do
  for condition in probe native token branch; do
    run_labram ISRUC "$condition" "$seed" --isruc_train_sequence_encoder || exit 1
  done
done
unset LABRAM_SOURCE_DIR
```

The pretrained epoch encoder stays frozen. The sequence encoder and classifier
train in all four conditions, including `probe`. They contribute 483,605
trainable parameters, in addition to any adapter.

### PhysioNet-MI and SEED-V probe comparisons

Set `OUTPUT_ROOT` to the destination for these experiments. Check PhysioNet-MI
first, then compare native with probe:

```bash
run_cbramod PHYSIONET-MI probe 42 --audit-only
run_cbramod PHYSIONET-MI probe 42
run_cbramod PHYSIONET-MI native 42
run_labram PhysioNet-MI probe 42
run_labram PhysioNet-MI native 42
run_cbramod SEED-V probe 42
run_cbramod SEED-V native 42
```

Repeat at seeds 1024 and 3407. SEED-V has one temporal patch; its CBraMod
native condition uses the **channel branch only**. These results extend RQ2;
they are not additional capacity-matched primary RQ1 settings.

### Full fine-tuning (RQ3)

Compare full fine-tuning with and without the native adapter at all three seeds.
The four paper settings are CBraMod on FACED, TUEV, and SEED-V, and LaBraM on
PhysioNet-MI. Use a separate `OUTPUT_ROOT` from the frozen-backbone runs.

```bash
run_cbramod FACED full 42
run_cbramod FACED native_full 42
run_cbramod TUEV full 42 --adapter-lr 2e-4
run_cbramod TUEV native_full 42 --adapter-lr 2e-4
run_cbramod SEED-V full 42
run_cbramod SEED-V native_full 42
run_labram PhysioNet-MI full 42
run_labram PhysioNet-MI native_full 42
```

The `full` and `native_full` conditions train the backbone and classifier,
with the native adapter added in `native_full`. RQ3 has no capacity-matched
token-wise condition. Commands for the contextual comparisons are given below.

### Contextual adapter and tuning baselines

These are the broader reference methods reported in Appendix C. Repeat each
command at seeds 42, 1024, and 3407, using a separate output root:

```bash
for dataset in FACED TUEV; do
  for seed in 42 1024 3407; do
    for condition in generic lora upper; do
      run_cbramod "$dataset" "$condition" "$seed" || exit 1
    done
  done
done
for dataset in FACED ISRUC TUEV; do
  for seed in 42 1024 3407; do
    for condition in lora upper; do
      run_labram "$dataset" "$condition" "$seed" || exit 1
    done
  done
done
for seed in 42 1024 3407; do
  run_labram TUEV generic "$seed" || exit 1
  run_labram ISRUC trainable32 "$seed" || exit 1
done
```

`generic` uses token-wise bottleneck width 64. `lora` uses QKV rank 8 and
scale 16; `upper` trains the last two backbone blocks. The `trainable32`
ISRUC context trains the backbone with a rank-32 native adapter. The helpers
retain the corresponding dataset budgets and learning-rate settings from the
saved experiments; these contextual methods are not capacity-matched RQ1
controls.

## 7. Results and paper tables

Both programs select the primary checkpoint using **validation Cohen's kappa**,
then report its held-out test performance. Validation-balanced-accuracy
selection is a separate sensitivity analysis. Read the kappa-selected results
when reproducing the primary tables.

| Backbone | Per-run output | Primary metrics |
| --- | --- | --- |
| CBraMod | `$OUTPUT_ROOT/cbramod/<dataset>/<dataset>_<condition>_s<seed>/` | `test_metrics.json`; selected weights `best_model.pt` |
| LaBraM | `$OUTPUT_ROOT/labram/<dataset>_<condition>_s<seed>/` | `final_test.json` → `primary_test`; selected weights `checkpoint-best.pth` |

CBraMod also writes `resolved_config.json` and `trainability_report.json`;
LaBraM writes `run_config.json`. Check the adapter counts against Section 5
and keep the same data and classifier within each paired comparison.
LaBraM's `sensitivity_test` entry is not the primary score.

For each seed, compute **native − token** for RQ1, **native − probe** for
RQ2, and **full fine-tuning with native − full fine-tuning without native**
for RQ3. Report the mean and **sample standard deviation** (`ddof=1`) of the
three paired differences, plus the number of positive differences. Do not
subtract the standard deviations of the two conditions.

Aggregate newly trained runs from the package root:

```bash
python reproduce/summarize_results.py "$OUTPUT_ROOT" --comparison rq1 --out summaries/rq1
python reproduce/summarize_results.py "$OUTPUT_ROOT" --comparison rq2 --out summaries/rq2
# For a separate full-fine-tuning output root:
# python reproduce/summarize_results.py "$OUTPUT_ROOT" --comparison rq3 --out summaries/rq3
```

The script writes individual scores, paired seed effects, and mean/sample-SD
summaries. It requires all three seeds for each included comparison, rejects
duplicate runs, and reads the kappa-selected test results. For LaBraM outputs
without a macro-F1 field, it averages per-class F1 (or reconstructs it from
the confusion matrix); weighted F1 is not substituted for macro-F1.

The paper's primary seed effects are in `results/primary/`:

- `native_minus_token.csv`: RQ1.
- `native_minus_probe.csv`: RQ2.
- `branch_control.csv`: attention, branch-preserving MLP, and token-wise controls.

The ISRUC sequence-follow-up summaries are in
`LaBraM/analysis/isruc_sequence_trainable_20261005/results_review/`.
See [ANALYSIS.md](ANALYSIS.md) for figure/table commands. New training runs
write to your `OUTPUT_ROOT`; aggregate those results separately from the
included paper summaries.

The LaBraM--TUEV probe uses a common reevaluated reference in the paper;
three saved checkpoints produced identical scores and do not establish three
independent training replications. New independently trained probes should be
reported as new reproductions, rather than forced to match that reference.

## Troubleshooting

- **Missing channel metadata:** add FACED's validated `channel_manifest.json`
  beside its LMDB, as described in the preprocessing guide. Keep physical
  channel order unchanged.
- **Missing TUEV/PhysioNet data check:** run the corresponding `--audit-only`
  command using the same `DATA_ROOT` and `OUTPUT_ROOT` as training.
- **Missing sequence flag:** use the dedicated follow-up source directory;
  do not pass it to the primary LaBraM entry point.
- **Nonempty output directory:** choose a new output directory or run name.
  Avoid resuming a previous experiment when reproducing a fresh seed.
- **Scores differ:** first compare checkpoint hashes, dataset splits, signal
  scaling, adapter counts, sequence-encoder trainability, and checkpoint
  selection. CUDA and package builds can also affect numerical results.

## Package contents

This repository includes downstream training, preprocessing, the sequence and
matched-optimization follow-up implementations, and selected analysis scripts
with numerical summaries. Third-party license notices are retained. Raw EEG,
model weights, scheduler logs, and local environment files are not included.
Obtain datasets and pretrained checkpoints through the links above.

See [ANALYSIS.md](ANALYSIS.md) for figure/table commands and the requirements
of scripts that read per-run outputs. The primary training instructions above
apply to this repository. Set your own absolute data, weight, and output paths.

## Upstream code and licenses

The study builds on [CBraMod](https://github.com/wjq-learning/CBraMod) and
[LaBraM](https://github.com/935963004/LaBraM). Retain their license and
attribution files (`CBraMod/LICENSE`, `LaBraM/LICENSE`). Dataset access is
governed by each provider's terms.
