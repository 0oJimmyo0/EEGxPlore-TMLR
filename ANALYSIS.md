# Analyses and figures

Start with the training instructions in [README.md](README.md). This folder
also contains the paper's numerical summaries and the scripts used to analyze
them. The selected training sources include the primary runs, branch controls,
full fine-tuning, sequence-encoder follow-up, and matched-optimization follow-up.

## Included numerical results

- `results/primary/native_minus_token.csv`: RQ1, native minus token-wise.
- `results/primary/native_minus_probe.csv`: RQ2, native minus probe.
- `results/primary/branch_control.csv`: branch-preserving MLP comparisons.
- `LaBraM/analysis/isruc_sequence_trainable_20261005/results_review/`: condition,
  paired-seed, and protocol-shift summaries for the sequence follow-up.
- `CBraMod/analysis/isruc_optimization_match_20261002/`: matched-optimization
  summaries and the implementation used for those runs.
- `paper/figures/*.csv`: numerical figure inputs, including training dynamics,
  class-level effects, and checkpoint-selector sensitivity.
- `LaBraM/analysis/tmlr_manuscript/`: additional summaries for primary,
  supporting, full-fine-tuning, and operator comparisons.

Use the contrast column names when reading historical CSVs. Some older
filenames and script variables use the earlier RQ numbering; the mapping
above follows the submitted paper.

## Plotting from included summaries

Run these commands from the package root after installing the environment:

```bash
python paper/figures/plot_rq1_effects.py
python paper/figures/plot_rq1_rq2_quadrants.py
python paper/figures/plot_tuev_class_effects.py
python paper/figures/plot_training_dynamics.py
python LaBraM/analysis/tmlr_manuscript/manuscript_final_v6/generate_tables.py
```

Figures are written beside their plotting scripts. The table exporter writes
to `paper/tables/`. These commands use included numerical summaries and do
not rerun model training. For new trajectories, pass `--outputs "$OUTPUT_ROOT"`
to `plot_training_dynamics.py` after completing the primary runs.

## Analyses that read experiment outputs

`plot_training_dynamics.py --outputs "$OUTPUT_ROOT"`, the TUEV class/confusion and selector summarizers,
and the residual-measurement scripts read per-run metrics, logs, or selected
weights. To rerun them, first train the corresponding conditions and point
the run lookups at your output directories. The original relative run names
are retained in the included CSVs; weights and full historical run directories
are not distributed in this package.

The `integrate_*.py`, historical aggregation scripts, and
`isruc_sequence_trainable_20261005/review_results.py` describe the analyses
used to build the paper summaries. They also reference historical completion
records and run directories, which are not included in this repository. Use
the included CSVs for the plotting commands above, or adapt their input paths
to newly reproduced runs for a fresh analysis.

Raw EEG and trained weights are not included. Download the datasets and
pretrained checkpoints through the links in the main guide. Exact TUEV split filenames and a membership checker are included in
`reproduce/splits/TUEV/` and `reproduce/check_splits.py`. Use
`reproduce/summarize_results.py` to aggregate newly trained runs.
