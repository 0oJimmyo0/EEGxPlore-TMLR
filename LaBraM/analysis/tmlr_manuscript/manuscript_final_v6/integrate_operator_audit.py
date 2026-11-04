"""Add six completed operator controls; inherit the primary matrix unchanged."""
import csv,hashlib,json,shutil
from pathlib import Path
from statistics import mean,stdev
P=Path(__file__).resolve().parent;ROOT=P.parents[3];V5=P.parent/'manuscript_final_v5';V3=P.parent/'manuscript_final_v3';PACKET=ROOT/'LaBraM/analysis/operator_audit_20261002'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):return list(csv.DictReader(Path(p).open(newline='')))
def write(name,records):
 with (P/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator='\n');w.writeheader();w.writerows(records)
def metrics(p):
 d=Path(p);filename='final_test.json' if (d/'final_test.json').exists() else 'test_metrics.json'
 payload=json.loads((d/filename).read_text());payload=payload['primary_test'] if filename=='final_test.json' else payload
 return {'ba':payload['balanced_accuracy'],'macro_f1':mean(payload['per_class_f1'])},sha(d/filename)
def adapter_seed(cfg):
 model=cfg.get('model',{})
 return cfg.get('adapter_seed',model.get('adapter_seed') if isinstance(model,dict) else None)
def main():
 completed=json.loads((PACKET/'completion_report.json').read_text());assert len(completed)==6 and all(x['status']=='PASS' for x in completed)
 for name in ['rq1_seed_effects.csv','rq1_effect_summary.csv','rq2_seed_effects.csv','rq2_effect_summary.csv','fresh_labram_closure_seed_results.csv']:
  shutil.copy2(V5/name,P/name);assert sha(P/name)==sha(V5/name)
 primary=rows(P/'rq2_seed_effects.csv');records=[]
 entries=[]
 for row in json.loads((PACKET/'manifest.json').read_text())['runs']:
  cfg=json.loads(Path(row['config']).read_text())
  entries.append(('LaBraM',row['dataset'].upper(),row['seed'],Path(cfg['native_reference']),Path(cfg['args']['output_dir']),Path(cfg['token_reference']),'new_completion_audit_PASS'))
 for old in rows(V3/'operator_audit_seed_results.csv'):
  bb,ds,seed=old['backbone'],old['dataset'],int(old['seed'])
  pair=next(x for x in primary if x['backbone']==bb and x['dataset'].upper()==ds and int(x['seed'])==seed)
  def path(run):return ROOT/('LaBraM/checkpoints/'+run if bb=='LaBraM' else 'CBraMod/results/'+ds.lower()+'/'+run)
  entries.append((bb,ds,seed,path(pair['native_run_id']),ROOT/old['branch_local_mlp_artifact'],path(pair['axisblind_run_id']),'inherited_TUEV_v3'))
 for bb,ds,seed,a,b,c,status in sorted(entries,key=lambda x:(x[0],x[1],x[2])):
  native,hs1=metrics(a);branch,hs2=metrics(b);token,hs3=metrics(c)
  config_name='run_config.json' if bb=='LaBraM' else 'resolved_config.json'
  ncfg=json.loads((a/config_name).read_text());bcfg=json.loads((b/config_name).read_text());tcfg=json.loads((c/config_name).read_text())
  row={'backbone':bb,'dataset':ds,'seed':seed}
  for metric in ['ba','macro_f1']:
   row.update({f'native_{metric}':native[metric],f'branch_local_mlp_{metric}':branch[metric],f'tokenwise_{metric}':token[metric],
               f'native_minus_branch_local_mlp_{metric}':native[metric]-branch[metric],f'branch_local_mlp_minus_tokenwise_{metric}':branch[metric]-token[metric]})
  row.update(native_artifact=str(a.relative_to(ROOT)),branch_local_mlp_artifact=str(b.relative_to(ROOT)),tokenwise_artifact=str(c.relative_to(ROOT)),
             native_evaluation_sha256=hs1,branch_evaluation_sha256=hs2,tokenwise_evaluation_sha256=hs3,
             native_adapter_seed=adapter_seed(ncfg),
             branch_adapter_seed=adapter_seed(bcfg),evidence_status=status)
  records.append(row)
 write('operator_audit_seed_results.csv',records)
 summary=[]
 for bb,ds in sorted({(x['backbone'],x['dataset']) for x in records}):
  selected=[x for x in records if x['backbone']==bb and x['dataset']==ds];assert len(selected)==3
  row={'backbone':bb,'dataset':ds,'seeds':3}
  for metric in ['ba','macro_f1']:
   for kind in ['native','branch_local_mlp','tokenwise','native_minus_branch_local_mlp','branch_local_mlp_minus_tokenwise']:
    values=[x[f'{kind}_{metric}'] for x in selected];row[f'{kind}_{metric}_mean']=mean(values);row[f'{kind}_{metric}_sd']=stdev(values)
   row[f'native_gt_branch_{metric}_count']=sum(x[f'native_minus_branch_local_mlp_{metric}']>0 for x in selected)
  summary.append(row)
 write('operator_audit_summary.csv',summary)
 contract=json.loads((V5/'analysis_contract.json').read_text());contract.update(version='manuscript_final_v6',inherited_package='manuscript_final_v5',
  operator_audit_packet=str(PACKET.relative_to(ROOT)),new_operator_runs=6,operator_audit_cells=['CBraMod--TUEV','LaBraM--FACED','LaBraM--ISRUC','LaBraM--TUEV'],
  operator_audit_policy='Secondary three-way audit; retain primary RQ1/RQ2 unchanged; new FACED/ISRUC controls use matched adapter seed 12345; legacy LaBraM TUEV branch controls use seed-specific adapter seeds.')
 (P/'analysis_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
 (P/'SHA256SUMS').write_text(''.join(f'{sha(f)}  {f.name}\n' for f in sorted(P.iterdir()) if f.is_file() and f.name!='SHA256SUMS'))
 print('Integrated six new controls and six historical TUEV controls; primary files byte-identical to v5.')
if __name__=='__main__':main()
