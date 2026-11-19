"""Complete the one previously untrained catalogue after reference-only refinement."""
from pathlib import Path
import sys,json,logging,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'event_uncertainty'))
import posterior_convergence as run

def main():
 out=HERE/'refined_case49_20260911'
 obj=run.ConvergenceExperiment(out,1,4322,False,[(1000,'perfect')],False)
 source=HERE.parent/'event_uncertainty/convergence_20260910/ensemble/catalogue_1000_perfect_49.npz'
 cat=dict(np.load(source));coeff=cat['coeff'];duration=float(cat['duration'])
 (out/source.name).write_bytes(source.read_bytes())
 validation=[];hashes={}
 for scramble in [0,1]:
  p=HERE/f'reference_refinement_20260911/ensemble1000_case49_p18_{scramble}.npz';hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
  d=np.load(p);phi,rates,q=d['phi'],d['rates'],d['logq'];exact=run.event_target(rates,coeff,duration)
  summary=obj.summarize(phi,rates,duration,len(coeff),run.core.TRUTH,exact-q)
  validation.append((phi,q,rates,exact,summary))
 passed,check=run.base.precision(validation[0][-1],validation[1][-1]);assert passed
 manifest=json.loads((out/'manifest.json').read_text());manifest.update(cases=[49],catalogue_source=str(source),reference_source_sha256=hashes,
 scope='Reference-only follow-up of previously unresolved case49; unchanged acquisition settings; raw original outcome preserved')
 run.base.write_json(out/'manifest.json',manifest)
 (out/'source_snapshot/complete_refined_case.py').write_bytes(Path(__file__).read_bytes())
 logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
 for strategy in ['bo','random']:
  obj.train(coeff,duration,validation,True,dict(exposure=1000,mode='perfect',case=49,events=len(coeff)),strategy)
 obj.progress('complete')
if __name__=='__main__':main()
