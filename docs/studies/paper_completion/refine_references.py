"""Bounded reference-only checks, preserving all original campaign records."""
from pathlib import Path
import sys,json,time,logging
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'event_uncertainty'))
import essential_runs as base
import posterior_convergence as target

def main():
 out=HERE/'reference_refinement_20260911';out.mkdir(exist_ok=True)
 loader=base.Experiment(out/f'population_{time.time_ns()}','paper_reference_only',False)
 logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
 root=HERE.parent/'event_uncertainty';results=[]
 sweep=root/'essential_20260910/sweep';cat=dict(np.load(sweep/'catalogue_500_8_scale2.0.npz'))
 coeff=cat['varied_coeff'];duration=500/loader.fiducial_total
 for power in [16,17,18]:
  refs=[]
  for scramble in [0,1]:
   folder=root/'convergence_20260910/ensemble';rf=folder/f'reference_n500_p{power}_{scramble}.npz'
   if rf.exists():
    d=np.load(rf);phi,rates=d['phi'],d['rates'];q=np.load(folder/f'proposal_n500_p{power}_{scramble}.npz')['logq']
   else:
    fisher=base.core.fisher_matrix(lambda t:loader.fast(*t),base.core.TRUTH,base.core.WIDTH,duration)
    phi,q=base.reference_design(base.core.TRUTH,fisher,202609225+scramble,power)
    rates=np.array([loader.fast(p[0],p[1],.012,p[2]).ravel()[loader.active] for p in phi])
    np.savez(out/f'width_reference_p{power}_{scramble}.npz',phi=phi,rates=rates,logq=q)
   exact=target.event_target(rates,coeff,duration)
   summary=base.Experiment.summarize(phi,rates,duration,len(coeff),base.core.TRUTH,exact-q)
   refs.append((phi,q,rates,exact,summary))
  ok,check=base.precision(refs[0][-1],refs[1][-1]);results.append(dict(study='double_width',power=power,numerical_pass=ok,check=check,summaries=[r[-1] for r in refs]))
  base.write_json(out/'reference_checks.json',results)
  if ok:break
 # Refit unchanged saved final training points, then score using mean-only prediction.
 records=[]
 for strategy in ['bo','random']:
  f=sweep/f'exposure500_case8_scale2.0_{strategy}_seed17/checkpoint_738.npz';d=np.load(f);x,y=d['x'],d['lnl']
  scaler=base.core.AdaptiveRobustScaler(compression='sqrt',soft_clipping=False,lower_clip_value=float(y.min()),max_scale=1e9);scaler.initialize_with_data(y)
  model=base.core.fit_exact_gp(base.core.JaxTrainingData(x,-scaler.transform(y)[:,None]),
    np.array([base.core.LOW[base.core.SHAPE],(base.core.LOW+base.core.WIDTH)[base.core.SHAPE]]),config=base.core.JaxGPConfig(optimisation_steps=250),seed=17)
  summaries=[];metrics=[]
  for phi,q,rates,exact,ds in refs:
   pred=scaler.inverse_transform(-target.reference_means(model,phi));gs=base.Experiment.summarize(phi,rates,duration,len(coeff),base.core.TRUTH,pred-q)
   summaries.append(gs);metrics.append({**base.core.compare(ds,gs),**{k:v for k,v in base.core.posterior_metrics(exact-q,pred-q,phi).items() if k.startswith('kl_')}})
  gpok,gpcheck=base.precision(*summaries)
  records.append(dict(exposure=500,case=8,scale=2.,strategy=strategy,points=738,summaries=summaries,metrics=metrics,numerical_pass=bool(ok and gpok),accuracy_pass=base.accuracy(metrics),numerical_check=gpcheck))
 base.write_json(out/'width_records.json',records)
 # The ensemble's n1000 case49 already failed its frozen maximum power18.
 # Check a genuinely independent power18 design; do not change its original outcome.
 cat=dict(np.load(root/'convergence_20260910/ensemble/catalogue_1000_perfect_49.npz'))
 duration=float(cat['duration']);fisher=base.core.fisher_matrix(lambda t:loader.fast(*t),base.core.TRUTH,base.core.WIDTH,duration)
 summaries=[]
 for scramble in [0,1]:
  phi,q=base.reference_design(base.core.TRUTH,fisher,202609227+scramble,18)
  f=out/f'ensemble1000_case49_p18_{scramble}.npz'
  if f.exists():rates=np.load(f)['rates']
  else:
   rates=np.array([loader.fast(p[0],p[1],.012,p[2]).ravel()[loader.active] for p in phi]);np.savez(f,phi=phi,rates=rates,logq=q)
  exact=target.event_target(rates,cat['coeff'],duration)
  summaries.append(base.Experiment.summarize(phi,rates,duration,len(cat['coeff']),base.core.TRUTH,exact-q))
 ok,check=base.precision(*summaries)
 results.append(dict(study='ensemble1000_case49',power=18,numerical_pass=ok,check=check,summaries=summaries,scope='Independent follow-up; original unresolved outcome retained, no new training'))
 base.write_json(out/'reference_checks.json',results);base.write_json(out/'status.json',dict(state='complete'))
if __name__=='__main__':main()
