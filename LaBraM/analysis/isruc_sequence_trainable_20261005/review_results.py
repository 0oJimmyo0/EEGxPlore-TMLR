"""Summarize audited primary-selector scores; do not modify the sealed packet."""
import csv
import hashlib
import json
import statistics as stats
import subprocess
from pathlib import Path

PACKET = Path(__file__).resolve().parent
ROOT = PACKET.parents[2]
SEEDS = (42, 1024, 3407)
METHODS = ('probe', 'native', 'token', 'branch')
CONTRASTS = (('native','token'), ('native','branch'), ('branch','token'),
             ('native','probe'), ('token','probe'), ('branch','probe'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def read_metrics(path):
    final = json.loads(path.read_text())
    assert final['primary_validation_metric'] == 'cohen_kappa'
    t = final['primary_test']
    cm = t['confusion_matrix']
    supports = [sum(r) for r in cm]
    predicted = [sum(r[i] for r in cm) for i in range(len(cm))]
    recalls = [cm[i][i] / supports[i] for i in range(len(cm))]
    f1 = [2*cm[i][i]/(supports[i]+predicted[i]) for i in range(len(cm))]
    ba, macro = stats.mean(recalls), stats.mean(f1)
    assert abs(ba-t['balanced_accuracy']) < 1e-10
    assert all(abs(x-y) < 1e-10 for x,y in zip(f1,t['per_class_f1']))
    assert sum(supports) == 8700
    return ba, macro, supports


def main():
    seal = json.loads((PACKET/'packet_seal.json').read_text())
    assert all(sha(PACKET/p) == h for p,h in seal.items())
    completion = json.loads((PACKET/'completion_report.json').read_text())
    assert len(completion) == 12 and all(x['status']=='PASS' for x in completion)
    submissions = json.loads((PACKET/'submission.json').read_text())
    submit = {x['run_id']: x for x in submissions}
    op = list(csv.DictReader((ROOT/'LaBraM/analysis/tmlr_manuscript/manuscript_final_v6/operator_audit_seed_results.csv').open()))
    branch_refs = {int(x['seed']):ROOT/x['branch_local_mlp_artifact'] for x in op
                   if x['backbone']=='LaBraM' and x['dataset']=='ISRUC'}
    rows = []
    supports_expected = None
    for c in completion:
        sub = submit[c['run_id']]
        cfg = json.loads(Path(sub['config']).read_text())
        assert sha(sub['config']) == sub['config_sha256']
        for regime,out in [('trainable_sequence', Path(sub['output_dir'])),
                           ('frozen_sequence', branch_refs[c['seed']] if c['method']=='branch'
                            else Path(cfg['historical_reference']))]:
            f = out/'final_test.json'
            ba,macro,supports = read_metrics(f)
            if supports_expected is None: supports_expected = supports
            assert supports == supports_expected, 'Test supports differ'
            if regime=='trainable_sequence':
                assert sha(f)==c['final_test_sha256']
                assert sha(out/'checkpoint-best.pth')==c['checkpoint_sha256']
                assert abs(macro-c['macro_f1'])<1e-10 and abs(ba-c['balanced_accuracy'])<1e-10
                logs = [json.loads(x) for x in (out/'log.txt').read_text().splitlines() if x.strip()]
                assert len(logs)==30
                assert all(__import__('math').isfinite(x['train_loss']) for x in logs)
                assert all(x['train_input_nonfinite_count']==0 and x['train_output_nonfinite_count']==0 for x in logs)
            rows.append(dict(regime=regime,method=c['method'],seed=c['seed'],
                balanced_accuracy=ba,macro_f1=macro,
                selected_epoch_zero_based=c['selected_epoch_zero_based'] if regime=='trainable_sequence' else '',
                output_dir=str(out),final_test_sha256=sha(f),
                checkpoint_sha256=c['checkpoint_sha256'] if regime=='trainable_sequence' else '',
                job_id=sub['job_id'] if regime=='trainable_sequence' else 'historical'))
    assert len(rows)==24 and len({(r['regime'],r['method'],r['seed']) for r in rows})==24
    summary=[]; effects=[]; seed_effects=[]; cross=[]
    for regime in ('frozen_sequence','trainable_sequence'):
        lookup={(r['method'],r['seed']):r for r in rows if r['regime']==regime}
        for method in METHODS:
            for metric in ('balanced_accuracy','macro_f1'):
                v=[lookup[(method,s)][metric] for s in SEEDS]
                summary.append(dict(regime=regime,method=method,metric=metric,n=3,mean=stats.mean(v),sample_sd=stats.stdev(v)))
        for a,b in CONTRASTS:
            for metric in ('balanced_accuracy','macro_f1'):
                v=[lookup[(a,s)][metric]-lookup[(b,s)][metric] for s in SEEDS]
                effects.append(dict(regime=regime,contrast=a+'-'+b,metric=metric,n=3,mean=stats.mean(v),sample_sd=stats.stdev(v),positive_seeds=sum(x>0 for x in v)))
                seed_effects.extend(dict(regime=regime,contrast=a+'-'+b,metric=metric,seed=s,effect=x) for s,x in zip(SEEDS,v))
    for method in METHODS:
        for metric in ('balanced_accuracy','macro_f1'):
            v=[next(r for r in rows if r['regime']=='trainable_sequence' and r['method']==method and r['seed']==s)[metric]-next(r for r in rows if r['regime']=='frozen_sequence' and r['method']==method and r['seed']==s)[metric] for s in SEEDS]
            cross.append(dict(method=method,metric=metric,n=3,mean_trainable_minus_frozen=stats.mean(v),sample_sd=stats.stdev(v),positive_seeds=sum(x>0 for x in v)))
    dest=PACKET/'results_review'; dest.mkdir(exist_ok=True)
    for name,data in [('seed_results',rows),('condition_summary',summary),('paired_effect_summary',effects),('paired_seed_effects',seed_effects),('protocol_shift_summary',cross)]:
        write_csv(dest/(name+'.csv'),data)
    scheduler=subprocess.run(['sacct','-j',','.join(x['job_id'] for x in submissions),'--format=JobIDRaw,State,ExitCode,Elapsed','--parsable2','-n','-X'],capture_output=True,text=True,check=True).stdout
    assert all('COMPLETED|0:0|' in x for x in scheduler.splitlines() if x.strip())
    assert len(scheduler.strip().splitlines())==12
    (dest/'scheduler_completion.txt').write_text(scheduler)
    (dest/'review_contract.json').write_text(json.dumps(dict(status='PASS',runs=12,epochs_per_run=30,test_sequences=435,test_predictions=8700,class_supports=supports_expected,primary_selector='validation_cohen_kappa',macro_f1_rule='mean per-class F1 reconstructed from confusion matrix, not f1_weighted',sealed_protocol_unchanged=True,all_conditions_all_seeds_included=True,completion_report_sha256=sha(PACKET/'completion_report.json'),interpretation='Separate prospectively configured sequence-encoder-training sensitivity; no pooling or substitution into historical primary cells; descriptive paired-seed results, no population significance or equivalence claim.'),indent=2)+'\n')
    print(json.dumps(dict(status='PASS',condition_summary=summary,paired_effect_summary=effects,protocol_shift_summary=cross),indent=2))


if __name__=='__main__':
    main()
