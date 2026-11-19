"""Validate refined references while retaining the original importance draws."""
from pathlib import Path
import sys,json
import numpy as np
root=Path(__file__).resolve().parents[3];folder=root/'docs/studies/event_uncertainty';sys.path.insert(0,str(folder))
import posterior_convergence as run
cat=dict(np.load(folder/'convergence_20260910/ensemble/catalogue_1000_perfect_49.npz'))
new=root/'docs/studies/paper_completion/reference_refinement_20260911'
summaries=[];pooled=[]
for scramble in [0,1]:
 old=np.load(folder/f'convergence_20260910/ensemble/reference_n1000_p18_{scramble}.npz');q=np.load(folder/f'convergence_20260910/ensemble/proposal_n1000_p18_{scramble}.npz')['logq']
 fresh=np.load(new/f'ensemble1000_case49_p18_{scramble}.npz')
 phi=np.vstack([old['phi'],fresh['phi']]);rates=np.vstack([old['rates'],fresh['rates']]);logq=np.r_[q,fresh['logq']]
 exact=run.event_target(rates,cat['coeff'],float(cat['duration']))
 sm=run.base.Experiment.summarize(phi,rates,float(cat['duration']),len(cat['coeff']),run.core.TRUTH,exact-logq)
 pooled.append(sm)
fresh_summaries=json.loads((new/'reference_checks.json').read_text())[-1]['summaries']
result={'summaries':pooled,'check':run.base.precision(*pooled),'agreement_with_new_only':[run.base.precision(a,b) for a,b in zip(pooled,fresh_summaries)]}
print(result['check'])
(new/'pooled_reference_check.json').write_text(json.dumps(result,indent=2)+'\n')
