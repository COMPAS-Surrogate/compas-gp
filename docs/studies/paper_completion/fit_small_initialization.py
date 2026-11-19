"""Fit exactly the first 25 or 50 initialization labels using the frozen campaign runtime."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

REPO=Path(__file__).resolve().parents[3]
ROOT=REPO/'docs/studies/operational_stopping/gpry_compas/count_history_20260912'
COUNT=int(sys.argv[sys.argv.index('--points')+1])
assert COUNT in (25,50)
OUT=ROOT.parent/f'count_history_analysis_20260913/early{COUNT}'


def main():
    if '--worker' not in sys.argv:
        env=dict(os.environ, PYTHONPATH=str(ROOT/'frozen_runtime/src'))
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker','--points',str(COUNT)],env=env,check=True)
        return
    sys.path.insert(0,str(ROOT/'frozen_runtime/docs/studies/operational_stopping/gpry_compas'))
    import run_experiment as base
    import numpy as np
    for n in (100,1000):
        for mode in ('perfect','uncertain'):
            name=f't0_n{n}_{mode}_c200_bo'
            source=ROOT/'training'/name/'checkpoint_58.npz'
            with np.load(source) as d: unit,lnl=d['unit'][:COUNT],d['lnl'][:COUNT]
            folder=OUT/name;folder.mkdir(parents=True,exist_ok=True)
            state=base.fit(unit,lnl,COUNT)
            # A truncated initialization diagnostic; the original campaign is unchanged.
            parameters={f'param_{k}':np.asarray(v) for k,v in state.items() if k not in ['x','mask','inverse','alpha']}
            np.savez(folder/f'checkpoint_{COUNT}.npz',unit=unit,lnl=lnl,capacity=COUNT,**parameters)
            (folder/'fit.json').write_text(json.dumps(dict(name=name,points=COUNT,seed=17,optimisation_steps=250,
                provenance=f'First {COUNT} initialization labels, selected by stored order before looking at likelihoods; 25 has no anchors, 50 has 32 Sobol plus 18 anchors; diagnostic only',
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                runtime=str(ROOT/'frozen_runtime'),checkpoint_sha256=hashlib.sha256((folder/f'checkpoint_{COUNT}.npz').read_bytes()).hexdigest()),indent=2)+'\n')
            print(name,'fit complete',flush=True)


if __name__=='__main__':main()
