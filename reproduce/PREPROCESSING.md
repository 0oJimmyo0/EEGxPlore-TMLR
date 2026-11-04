# Dataset preparation

Use the same prepared files for CBraMod and LaBraM. Paths below are relative
to the directory containing the top-level README. The scripts implement the
CBraMod preprocessing protocol; set their input and output paths before
running them. Do not apply an additional normalization step to stored data.

## Expected directory layout

```text
$DATA_ROOT/
  FACED/                       LMDB: data.mdb, lock.mdb
    channel_manifest.json
  ISRUC/
    seq/ISRUC-group1-1/ISRUC-group1-1-0.npy
    labels/ISRUC-group1-1/ISRUC-group1-1-0.npy
    ...                        subjects 1 through 100
  TUEV/
    processed_train/*.pkl       training
    processed_eval/*.pkl        validation
    processed_test/*.pkl        held-out test
  PHYSIO_MI/                   LMDB
  SEED-V/                      LMDB
```

Each LMDB contains a pickled `__keys__` dictionary with `train`, `val`, and
`test` lists. Each example is a pickled dictionary with `sample` and `label`.
The readers use these saved lists, not a new random split.

Reference split lists for all five datasets are in `reproduce/splits/`.
LMDB lists preserve the stored key order. To compare a prepared dataset, use
`python reproduce/check_splits.py <prepared-directory> --dataset <name>`;
accepted names are FACED, ISRUC, TUEV, PHYSIONET-MI, and SEED-V.

## FACED

Obtain `Processed_data` using the [FACED data-access instructions](https://doi.org/10.1038/s41597-023-02650-w).
The script reads subject-level pickle files, not raw BDF recordings or
precomputed differential-entropy features.

1. In `CBraMod/preprocessing/preprocessing_faced.py`, set `root_dir` to the
   downloaded `Processed_data` directory.
2. Change the output passed to `lmdb.open(...)` to `$DATA_ROOT/FACED`, using
   the actual absolute path. Create its parent directory first.
3. Run:

   ```bash
   python CBraMod/preprocessing/preprocessing_faced.py
   ```

The script sorts subject files, uses the first 80 / next 20 / remaining 23
subjects for training / validation / test, resamples each trial to 6,000
samples, and divides each trial into three nonoverlapping ten-second
segments. Each stored example is `(32,10,200)`, with labels `0..8`.

LaBraM also needs the stored channel order. Verify that your source files
use the order in `CBraMod/configs/faced_channel_manifest.json`, then copy it
beside the LMDB:

```bash
cp CBraMod/configs/faced_channel_manifest.json "$DATA_ROOT/FACED/channel_manifest.json"
```

This file documents the existing order; copying it does not reorder signals.
Both training pipelines divide FACED amplitudes by 100 during loading/training.
Do not divide them while writing the LMDB.

## ISRUC-Sleep subgroup I

Download subgroup I (100 subjects) from [ISRUC-Sleep](https://sleeptight.isr.uc.pt/).
Use the first scorer's labels (`<subject>_1.txt`) and each subject's recording
(`<subject>.rec`). The input layout expected by the script is:

```text
raw_isruc/group1/1/1.rec
raw_isruc/group1/1/1_1.txt
...
raw_isruc/group1/100/100.rec
raw_isruc/group1/100/100_1.txt
```

In `CBraMod/preprocessing/ISRUC/prepare_ISRUC_1.py`, set:

- `dir_path`: the extracted `group1` directory;
- `seq_dir`: `$DATA_ROOT/ISRUC/seq`;
- `label_dir`: `$DATA_ROOT/ISRUC/labels`.

Use literal absolute paths in the Python file. Run it from its own directory
so that its bundled EDF reader can be imported:

```bash
(cd CBraMod/preprocessing/ISRUC && python prepare_ISRUC_1.py)
```

The script filters at 0.3–35 Hz with a 50 Hz notch, selects six EEG channels,
segments the signal into 30-second epochs, and groups 20 consecutive epochs
per sequence. It maps sleep-stage labels `0,1,2,3,5` to `0,1,2,3,4` and drops
incomplete sequences. The saved signal shape is `(20,6,6000)`, paired with
20 integer labels.

The source script assumes 200 Hz input; its `raw.resample(...)` line is
commented out. Check the recording sample rate before using its fixed
6,000-sample epoch construction. It selects columns corresponding to
`F3-A2, C3-A2, O1-A2, F4-A1, C4-A1, O2-A1`; confirm this order in the
recording returned by the bundled reader before preprocessing a different
release. LaBraM maps these six channels to its electrode embeddings without
changing the stored signals.

No LMDB conversion is needed. Both training readers load the `.npy` files
and split subjects **1–80 / 81–90 / 91–100**. They reshape each epoch to
`(6,30,200)` and use amplitude divisor **1**.

## TUEV

Request the **TUH EEG Events Corpus** through the
[official corpus-access page](https://isip.piconepress.com/projects/tuh_eeg/).
Preserve the original `edf/train` and `edf/eval` trees and their annotations.

In `CBraMod/preprocessing/preprocessing_tuev.py`, set:

- the first `root` (the raw EDF directory) to the extracted EDF root;
- `target` to an intermediate output directory;
- the second `root`, under `#transfer to train, eval, and test`, to that same
  intermediate output directory.

Run:

```bash
python CBraMod/preprocessing/preprocessing_tuev.py
```

The script creates event pickle files, then writes the final split directories
under `<target>/processed/`. Use that final directory as `$DATA_ROOT/TUEV`
(or point a symlink named `TUEV` to it). The final names have the following
meaning:

| Directory | Role |
| --- | --- |
| `processed_train` | Training portion of original training records |
| `processed_eval` | Validation portion of original training records |
| `processed_test` | Original evaluation records, used as held-out test |

The splitter sorts record prefixes and assigns 80% / 20% to training /
validation. The event processing selects 16 bipolar channels, resamples to
200 Hz, and extracts five-second events. Pickles contain `signal` shaped
`(16,1000)` and labels `1..6`. The loaders convert labels to `0..5`, reshape
the signals to `(16,5,200)`, and divide amplitudes by 100 exactly once.

The paper uses **68,712 / 15,220 / 29,421** serialized examples. Exact
filenames are included in `reproduce/splits/TUEV/{train,val,test}.txt`.
The lists match the filename digests saved with the original experiments.
After preprocessing, run from the package root:

```bash
python reproduce/check_splits.py "$DATA_ROOT/TUEV"
```

This checks split membership, including missing and extra files. It does not
verify signal contents; the data audit below checks shapes and labels.
Corpus versions can change the resulting event files. The original upstream
corpus version is not established by these serialized examples.

Run the TUEV `--audit-only` command from the top-level README before training.
It checks shapes, labels, and split-prefix overlap. It does not establish
subject disjointness.

## PhysioNet-MI

Download the EDF files from the [EEG Motor Movement/Imagery Dataset
v1.0.0](https://physionet.org/content/eegmmidb/1.0.0/), retaining subject
directories such as `S001/`.

In `CBraMod/preprocessing/preprocessing_physio.py`, set `root_dir` to the
EDF subject-directory root and the `lmdb.open(...)` output to
`$DATA_ROOT/PHYSIO_MI`. Run:

```bash
python CBraMod/preprocessing/preprocessing_physio.py
```

The script uses imagery runs **04, 06, 08, 10, 12, 14**, orders the 64
channels explicitly, applies average reference, a 0.3 Hz high-pass and 60 Hz
notch, resamples to 200 Hz, and extracts four-second imagery epochs. It
excludes rest events and creates four classes. Stored examples have shape
`(64,4,200)` and labels `0..3`.

Sorted subjects are split **70 / 19 / 20**. The dataset directory must
contain subject directories only: extra download files can change an
unfiltered `os.listdir` split. Both training pipelines divide amplitudes by
100; do not pre-divide the LMDB signals.

## SEED-V

Apply for **SEED-V** through the [SEED dataset application
form](https://bcmi.sjtu.edu.cn/ApplicationForm/apply_form/). Obtain EEG `.cnt`
recordings, not only extracted feature matrices.

In `CBraMod/preprocessing/preprocessing_SEEDV.py`, set `root_dir` to the CNT
directory and the `lmdb.open(...)` output to `$DATA_ROOT/SEED-V`. Run:

```bash
python CBraMod/preprocessing/preprocessing_SEEDV.py
```

The script removes M1, M2, VEO, and HEO, resamples to 200 Hz, filters at
0.3–75 Hz, and extracts one-second segments using the session trial timings
and labels encoded in the script. Trials **0–4 / 5–9 / 10–14** form training /
validation / test within subjects. Each stored example is `(62,1,200)` with
labels `0..4`. The CBraMod reader divides amplitudes by 100.

The paper's SEED-V probe and native comparisons use CBraMod. Its native
adapter mixes channels only because the temporal-patch count is one.
