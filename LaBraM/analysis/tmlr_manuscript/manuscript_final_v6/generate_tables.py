"""Export source-linked LaTeX tables for the extended operator audit."""
import csv
from pathlib import Path
P=Path(__file__).resolve().parent
PAPER=P.parents[3]/'paper'
OUT=PAPER/'tables';OUT.mkdir(parents=True, exist_ok=True)
summary=list(csv.DictReader((P/'operator_audit_summary.csv').open()))
seeds=list(csv.DictReader((P/'operator_audit_seed_results.csv').open()))
def fmt(value,signed=False):
 s=format(float(value), ('+.5f' if 0 < abs(float(value)) < 0.00005 else '+.4f') if signed else '.4f')
 return s.replace('0.','.')
def pair(row,key,signed=False):return '$'+fmt(row[key+'_mean'],signed)+r'\pm'+fmt(row[key+'_sd'])+'$'
lines=[r'\begin{table}[!ht]',r'\centering',r'\caption{LaBraM operator audit at validation-$\kappa$-selected checkpoints (mean $\pm$ SD, three paired seeds). Attn: native-axis attention; Branch: branch-preserving MLP; Token: token-wise MLP. BA: balanced accuracy; F1: macro-F1. $\dagger$ TUEV Branch uses seed-specific adapter initialization; all other conditions use adapter seed 12345. ISRUC uses a newly initialized frozen sequence encoder (Table~\ref{tab:primary-configs}).}',r'\label{tab:labram-operator-audit}',r'\scriptsize',r'\setlength{\tabcolsep}{3pt}',r'\resizebox{\linewidth}{!}{%',r'\begin{tabular}{@{}llccccc@{}}',r'\toprule',r'Dataset & Metric & Attn & Branch & Token & Attn $-$ Branch & Branch $-$ Token \\',r'\midrule']
for ds in ('FACED','ISRUC','TUEV'):
 row=next(x for x in summary if x['backbone']=='LaBraM' and x['dataset']==ds)
 for metric,label in [('ba','BA'),('macro_f1','F1')]:
  name=ds+(r'$^{\dagger}$' if ds=='TUEV' else '')
  cells=[name,label]+[pair(row,k+'_'+metric,k.startswith(('native_minus','branch_local_mlp_minus'))) for k in ['native','branch_local_mlp','tokenwise','native_minus_branch_local_mlp','branch_local_mlp_minus_tokenwise']]
  lines.append(' & '.join(cells)+r' \\')
lines += [r'\bottomrule',r'\end{tabular}%',r'}',r'\end{table}']
(OUT/'labram_operator_audit.tex').write_text('\n'.join(lines)+'\n')
lines=[r'\begin{table}[!ht]',r'\centering',r'\caption{New branch-preserving MLP controls on LaBraM FACED/ISRUC. B denotes Branch, A denotes native attention, and T denotes the ordinary token-wise MLP. Each row uses its validation-$\kappa$-selected checkpoint.}',r'\label{tab:operator-new-seeds}',r'\small',r'\setlength{\tabcolsep}{3pt}',r'\begin{tabular}{@{}llrrrrrr@{}}',r'\toprule',r'Dataset & Seed & B BA & A $-$ B BA & B $-$ T BA & B F1 & A $-$ B F1 & B $-$ T F1 \\',r'\midrule']
for row in seeds:
 if row['backbone']!='LaBraM' or row['dataset']=='TUEV':continue
 keys=[k+'_'+m for m in ['ba','macro_f1'] for k in ['branch_local_mlp','native_minus_branch_local_mlp','branch_local_mlp_minus_tokenwise']]
 cells=[row['dataset'],row['seed']]+['$'+fmt(row[k],k.startswith(('native_minus','branch_local_mlp_minus')))+'$' for k in keys]
 lines.append(' & '.join(cells)+r' \\')
lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
(OUT/'operator_new_seeds.tex').write_text('\n'.join(lines)+'\n')
print('Exported main three-way audit and six new seed-level rows.')
