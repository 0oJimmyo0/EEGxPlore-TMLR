"""Summarize new CBraMod/LaBraM runs and paired test-score differences."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

METRICS = ('balanced_accuracy', 'macro_f1')
CONTRASTS = {'rq1': ('native', 'token'), 'rq2': ('native', 'probe'),
             'rq3': ('native_full', 'full'), 'branch': ('native', 'branch'), 'branch_token': ('branch', 'token')}


def read_run(path, root):
    directory = path.parent
    if path.name == 'test_metrics.json':
        cfg_path = directory / 'resolved_config.json'
        if not cfg_path.is_file():
            return None
        cfg = json.loads(cfg_path.read_text())
        condition = {'frozen_probe': 'probe', 'interaction_aligned': 'native',
                     'axis_blind': 'token', 'axis_decomposed_mlp': 'branch',
                     'full_finetune': 'full', 'native_full_finetune': 'native_full',
                     'generic_bottleneck': 'generic', 'lora': 'lora', 'upper_k_finetune': 'upper'}.get(cfg['method'])
        backbone, scores = 'CBraMod', json.loads(path.read_text())
        protocol = 'standard'
    else:
        cfg_path = directory / 'run_config.json'
        if not cfg_path.is_file():
            return None
        cfg = json.loads(cfg_path.read_text())
        result = json.loads(path.read_text())
        if result.get('primary_validation_metric') != 'cohen_kappa':
            raise ValueError(f'Unexpected primary selector: {directory.name}')
        scores = result['primary_test']
        adapter, mode = cfg['adapter_type'], cfg['backbone_mode']
        if mode == 'trainable':
            condition = 'full' if adapter == 'none' else 'native_full' if adapter == 'channel_patch' else None
        elif mode == 'frozen':
            condition = ('probe' if adapter == 'none' else ('token' if int(cfg.get('adapter_bottleneck', 0)) == 213 else 'generic') if adapter == 'generic'
                         else 'branch' if cfg.get('adapter_operator') == 'mlp' else 'native')
        elif mode == 'lora':
            condition = 'lora'
        elif mode == 'upper_k':
            condition = 'upper'
        else:
            condition = None
        backbone = 'LaBraM'
        # Keep the sequence-trained follow-up separate even if output roots are merged.
        protocol = ('sequence_trained' if cfg.get('sequence_encoder_trainable', cfg.get('isruc_sequence_encoder_trainable',
                    cfg.get('isruc_train_sequence_encoder', False))) else 'standard')
    if condition is None:
        return None
    if 'macro_f1' not in scores:
        if 'per_class_f1' in scores:
            scores['macro_f1'] = mean(scores['per_class_f1'])
        else:
            cm = scores['confusion_matrix']
            per_class = []
            for i in range(len(cm)):
                denominator = sum(cm[i]) + sum(row[i] for row in cm)
                per_class.append(2 * cm[i][i] / denominator if denominator else 0.0)
            scores['macro_f1'] = mean(per_class)
    dataset = str(cfg.get('dataset_name', cfg.get('dataset'))).upper().replace('_', '-')
    if dataset in {'PHYSIONETMI', 'PHYSIO-MI'}:
        dataset = 'PHYSIONET-MI'
    return dict(backbone=backbone, dataset=dataset, protocol=protocol,
                condition=condition, seed=int(cfg['seed']),
                **{metric: float(scores[metric]) for metric in METRICS},
                run=str(directory.relative_to(root)))


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('outputs', type=Path, help='OUTPUT_ROOT used by commands.sh')
    parser.add_argument('--out', type=Path, required=True, help='Destination for summary CSVs')
    parser.add_argument('--comparison', choices=[*CONTRASTS, 'scores'], required=True)
    args = parser.parse_args()
    root = args.outputs.resolve()
    runs = []
    for filename in ('test_metrics.json', 'final_test.json'):
        for path in sorted(root.rglob(filename)):
            row = read_run(path, root)
            if row is not None:
                runs.append(row)
    if not runs:
        raise SystemExit('No completed runs found.')
    lookup = {}
    for row in runs:
        key = tuple(row[k] for k in ('backbone', 'dataset', 'protocol', 'condition', 'seed'))
        if key in lookup:
            raise SystemExit(f'Duplicate run for {key}; choose a single experiment output root.')
        lookup[key] = row
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / 'scores.csv', runs)
    score_groups = defaultdict(list)
    for row in runs:
        key = tuple(row[k] for k in ('backbone', 'dataset', 'protocol', 'condition'))
        score_groups[key].append(row)
    condition_summary = []
    for group, rows in sorted(score_groups.items()):
        for metric in METRICS:
            values = [r[metric] for r in rows]
            condition_summary.append(dict(zip(('backbone', 'dataset', 'protocol', 'condition'), group),
                metric=metric, n=len(values), mean=mean(values),
                sample_sd=stdev(values) if len(values) > 1 else ''))
    write_csv(args.out / 'condition_summary.csv', condition_summary)
    if args.comparison == 'scores':
        print(f'Wrote {len(runs)} run scores and {len(condition_summary)} condition summaries.')
        return
    condition_a, condition_b = CONTRASTS[args.comparison]
    groups = sorted({key[:3] for key in lookup})
    effects, summaries = [], []
    for group in groups:
        a_seeds = {k[-1] for k in lookup if k[:3] == group and k[3] == condition_a}
        b_seeds = {k[-1] for k in lookup if k[:3] == group and k[3] == condition_b}
        if not a_seeds and not b_seeds:
            continue
        if a_seeds != {42, 1024, 3407} or b_seeds != a_seeds:
            raise SystemExit(f'Incomplete paired seeds for {group}: {condition_a}={sorted(a_seeds)}, {condition_b}={sorted(b_seeds)}')
        for metric in METRICS:
            values = []
            for seed in sorted(a_seeds):
                a, b = lookup[(*group, condition_a, seed)], lookup[(*group, condition_b, seed)]
                effect = a[metric] - b[metric]
                effects.append(dict(zip(('backbone', 'dataset', 'protocol'), group),
                    contrast=f'{condition_a}-{condition_b}', metric=metric, seed=seed,
                    a_score=a[metric], b_score=b[metric], effect=effect))
                values.append(effect)
            summaries.append(dict(zip(('backbone', 'dataset', 'protocol'), group),
                contrast=f'{condition_a}-{condition_b}', metric=metric, n=len(values),
                mean=mean(values), sample_sd=stdev(values), positive_seeds=sum(x > 0 for x in values)))
    if not effects:
        raise SystemExit('No runs for the requested comparison.')
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / 'scores.csv', runs)
    write_csv(args.out / 'paired_seeds.csv', effects)
    write_csv(args.out / 'paired_summary.csv', summaries)
    print(f'Wrote {len(effects)} paired effects and {len(summaries)} summaries.')


if __name__ == '__main__':
    main()
