"""Fit exactly the first 100 acquired labels using the frozen campaign runtime."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

REPO=Path(__file__).resolve().parents[3]
ROOT=REPO/'docs/studies/operational_stopping/gpry_compas/count_history_20260912'
OUT=ROOT.parent/'count_history_analysis_20260913/early100'


def main():
    if '--worker' not in sys.argv:
        env=dict(os.environ, PYTHONPATH=str(ROOT/'frozen_runtime/src'))
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker'],env=env,check=True)
        return
    sys.path.insert(0,str(ROOT/'frozen_runtime/docs/studies/operational_stopping/gpry_compas'))
    import run_experiment as base
    import numpy as np
    for n in (100,1000):
        for mode in ('perfect','uncertain'):
            name=f't0_n{n}_{mode}_c200_bo'
            source=ROOT/'training'/name/'checkpoint_118.npz'
            with np.load(source) as d: unit,lnl=d['unit'][:100],d['lnl'][:100]
            batches=json.loads((source.parent/'batches.json').read_text())
            observed=np.concatenate([np.asarray(b['unit']) for b in batches[:14]])
            values=np.concatenate([np.asarray(b['actual']) for b in batches[:14]])
            np.testing.assert_array_equal(unit[58:],observed)
            np.testing.assert_array_equal(lnl[58:],values)
            folder=OUT/name;folder.mkdir(parents=True,exist_ok=True)
            state=base.fit(unit,lnl,100)
            # This artifact is a fresh hyperparameter refit at 100, rather than
            # the original trajectory's 88-point fit conditioned on 12 labels.
            parameters={f'param_{k}':np.asarray(v) for k,v in state.items() if k not in ['x','mask','inverse','alpha']}
            np.savez(folder/'checkpoint_100.npz',unit=unit,lnl=lnl,capacity=100,**parameters)
            (folder/'fit.json').write_text(json.dumps(dict(name=name,points=100,seed=17,optimisation_steps=250,
                provenance='58 initialization labels plus first 14 acquired batches; extra refit at 100',
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                runtime=str(ROOT/'frozen_runtime'),checkpoint_sha256=hashlib.sha256((folder/'checkpoint_100.npz').read_bytes()).hexdigest()),indent=2)+'\n')
            print(name,'fit complete',flush=True)


if __name__=='__main__':main()
