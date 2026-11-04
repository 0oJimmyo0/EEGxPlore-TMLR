"""Inference-only validation residual measurements at primary selected checkpoints."""
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from models.cbramod import CBraMod
from datasets.faced_dataset import CustomDataset
from datasets.isruc_dataset import ISRUCSequenceDataset
from datasets.tuev_tmlr_dataset import TUEVDataset

OUT = Path(__file__).resolve().parent

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    torch.set_num_threads(1)
    assert torch.cuda.is_available(), 'This audit requires a GPU for the full validation passes'
    registry = REPO.parent / 'LaBraM/analysis/tmlr_manuscript/manuscript_final_v6/rq2_seed_effects.csv'
    rows = list(csv.DictReader(registry.open()))
    runs = {(r['dataset'].upper(), r['seed'], r['axisblind_run_id']) for r in rows if r['backbone'] == 'CBraMod'}
    output = []
    for dataset, seed, run_id in sorted(runs):
        start = time.time()
        folder = REPO / 'results' / dataset.lower() / run_id
        cfg = json.loads((folder / 'resolved_config.json').read_text())
        model = CBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800,
                        seq_len=5 if dataset == 'TUEV' else 30, n_layer=12, nhead=8)
        model.proj_out = nn.Identity()
        model.enable_generic_adapter(bottleneck=cfg['axis_blind_bottleneck'],
            init_alpha=cfg['adapter_init_alpha'], gamma=cfg['adapter_gamma'],
            zero_init_output=cfg['adapter_zero_init_output'], seed=cfg['adapter_seed'],
            adapter_type='axis_blind')
        checkpoint = folder / 'best_model.pt'
        state = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
        model.load_state_dict({k.removeprefix('backbone.'): v for k, v in state.items()
                               if k.startswith('backbone.')}, strict=True)
        del state
        model.to('cuda').eval()
        if dataset == 'FACED':
            data = CustomDataset(cfg['dataset_path'], mode='val')
        elif dataset == 'ISRUC':
            data = ISRUCSequenceDataset(cfg['dataset_path'], 'val', cfg['input_scale_divisor'])
        else:
            data = TUEVDataset(cfg['dataset_path'], 'val', cfg['input_scale_divisor'])
        ratios, norm_h_sq, norm_delta_sq = [], [], []
        def capture(module, args, delta):
            h = args[0].detach().float().flatten(1)
            d = delta.detach().float().flatten(1)
            h2, d2 = h.square().sum(1), d.square().sum(1)
            assert torch.all(h2 > 0)
            ratios.extend(torch.sqrt(d2 / h2).cpu().tolist())
            norm_h_sq.extend(h2.cpu().tolist())
            norm_delta_sq.extend(d2.cpu().tolist())
        hook = model.generic_adapter.register_forward_hook(capture)
        with torch.inference_mode():
            for x, _ in DataLoader(data, batch_size=cfg['batch_size'], shuffle=False, num_workers=0):
                x = x.float().to('cuda')
                if dataset == 'ISRUC':
                    x = x.reshape(-1, *x.shape[2:])
                model(x)
        hook.remove()
        adapter = model.generic_adapter
        ratio = np.asarray(ratios)
        record = dict(dataset=dataset, seed=int(seed), run_id=run_id,
            checkpoint_sha256=sha256(checkpoint), config_sha256=sha256(folder / 'resolved_config.json'),
            split='validation', sequence_count=len(data), evaluated_feature_tensors=len(ratio),
            alpha_tokenwise=float(adapter.alpha.detach().cpu()), gamma=cfg['adapter_gamma'],
            delta_ratio_mean=float(ratio.mean()), delta_ratio_sd=float(ratio.std(ddof=1)),
            delta_ratio_median=float(np.median(ratio)),
            delta_ratio_global=float(np.sqrt(np.sum(norm_delta_sq) / np.sum(norm_h_sq))),
            elapsed_seconds=time.time()-start)
        output.append(record)
        (OUT / 'results.json').write_text(json.dumps(dict(
            protocol='Full validation split, eval mode, float32, selected primary checkpoint. '
                     'Delta is the scaled adapter output including gamma and learned alpha. '
                     'Per-example Frobenius norm ratio; ISRUC measures individual epochs.',
            source_script_sha256=sha256(Path(__file__)), results=output), indent=2)+'\n')
        print(json.dumps(record), flush=True)
        del model, data
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
