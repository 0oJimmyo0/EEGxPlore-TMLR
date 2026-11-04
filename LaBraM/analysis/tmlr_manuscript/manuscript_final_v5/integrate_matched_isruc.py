"""Versioned integration of protocol-selected optimization-matched ISRUC controls."""
import csv,hashlib,json,shutil
from pathlib import Path
from statistics import mean,stdev
P=Path(__file__).resolve().parent;ROOT=P.parents[3];V4=P.parent/'manuscript_final_v4';REPAIR=ROOT/'CBraMod/analysis/isruc_optimization_match_20261002'
def readrows(p):
 with p.open(newline='') as f:return list(csv.DictReader(f))
def write(name,rows):
 with (P/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 assert json.loads((REPAIR/'completion_report.json').read_text())['status']=='PASS'
 for name in ['rq1_seed_effects.csv','rq1_effect_summary.csv','fresh_labram_closure_seed_results.csv']:
  shutil.copy2(V4/name,P/name)
 matched={(int(r['seed']),r['metric']):r for r in readrows(REPAIR/'matched_seed_effects.csv')}
 rq2=readrows(V4/'rq2_seed_effects.csv')
 for row in rq2:
  if row['backbone']=='CBraMod' and row['dataset'].lower()=='isruc':
   new=matched[int(row['seed']),row['metric']]
   row.update(axisblind_run_id=new['tokenwise_run_id'],native_value=new['native_value'],axisblind_value=new['tokenwise_value'],effect_native_minus_axisblind=new['effect_native_minus_tokenwise'],pair_audit_notes='optimization-matched scalar WD 0.05; locked protocol; completion audit PASS')
 write('rq2_seed_effects.csv',rq2)
 old=readrows(V4/'rq2_effect_summary.csv')
 for row in old:
  vals=[float(r['effect_native_minus_axisblind']) for r in rq2 if all(r[k]==row[k] for k in ['backbone','dataset','metric'])]
  if row['backbone']=='CBraMod' and row['dataset'].lower()=='isruc':row.update(effect_mean=mean(vals),effect_sd=stdev(vals))
 write('rq2_effect_summary.csv',old)
 contract=json.loads((V4/'analysis_contract.json').read_text());contract.update(version='manuscript_final_v5',inherited_package='manuscript_final_v4',matched_isruc_packet=str(REPAIR.relative_to(ROOT)),new_isruc_training_runs=2,isruc_scalar_weight_decay=0.05)
 paths=set()
 for name in ['rq1_seed_effects.csv','rq2_seed_effects.csv']:
  for row in readrows(P/name):
   for field in ['native_run_id','probe_run_id','axisblind_run_id']:
    if field in row:
     paths.add(('../CBraMod/results/'+row['dataset'].lower()+'/' if row['backbone']=='CBraMod' else 'checkpoints/')+row[field])
 assert len(paths)==54 and all((ROOT/'LaBraM'/x).is_dir() for x in paths)
 contract['source_artifacts']=sorted(paths)
 contract['integration_policy']='Replace all three ISRUC token-wise controls with scalar-WD=0.05 counterparts; reuse existing seed42; inherit all native/probe runs; retain v4 as historical sensitivity evidence.'
 (P/'analysis_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
 (P/'README.md').write_text('# Manuscript-final v5\n\nVersioned successor to v4 after optimization-matched CBraMod ISRUC completion. RQ1 is unchanged. RQ2 replaces only the three ISRUC token-wise controls, using the existing matched seed42 run and two newly trained controls. Protocol choice was locked before new test scores. Source paths are relative to the LaBraM repository. The prior v4 package remains intact as historical evidence.\n\nReproduce with `python integrate_matched_isruc.py`; the matched packet must pass its completion audit first.\n')
 print('Integrated six ISRUC metric rows; preserved RQ1 and all other primary cells.')
if __name__=='__main__':main()
