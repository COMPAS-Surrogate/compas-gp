"""Direct, paired COMPAS-size sensitivity versus observed catalogue size."""
from __future__ import annotations
import copy
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
import numpy as np
from scipy.sparse import load_npz
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
sys.path.insert(0,str(HERE.parent/'event_uncertainty'))
import essential_runs as base
import posterior_convergence as target

def population_model(template,blocks:list,count:int):
    """Change unnormalized population contributions and normalization together."""
    model=copy.copy(template);model.ci=copy.copy(template.ci);model.ci.compas=copy.copy(template.ci.compas)
    model.ci.compas.nSystems=count*32000000
    model.operator=sum(blocks[1:count],blocks[0].copy())
    return model

def shifts(reference:dict,other:dict)->dict:
    sd=np.array(reference['sd']);delta=(np.array(other['mean'])-reference['mean'])/sd
    q=np.array(other['quantiles']);r=np.array(reference['quantiles'])
    ratio=(q[4]-q[2])/(r[4]-r[2])
    return dict(mean_shift_sd=delta.tolist(),width68_ratio=ratio.tolist(),
                maximum_mean_shift_sd=float(np.max(abs(delta))),maximum_width_change=float(np.max(abs(ratio-1))))

def main()->None:
    out=HERE/'population_size_20260911';out.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    manifest=dict(seed=202609214,expected=[50,300,1000],cases=3,sizes=[32,128,256,512],measurement='nominal continuous',
        truth=base.core.TRUTH.tolist(),scope='Nested-population sensitivity relative to 512M; not safety or absolute bias',
        sources={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),HERE/'PLAN.md',*sorted((REPO/'src/cosmic_integration').rglob('*.py'))]})
    if (out/'manifest.json').exists():assert json.loads((out/'manifest.json').read_text())==manifest
    else:base.write_json(out/'manifest.json',manifest)
    loader=base.Experiment(out/f'population_{time.time_ns()}','population_size_direct',False)
    blocks=[load_npz(HERE.parent/f'population_stability/block_results/block_{i:02d}.npz') for i in range(16)]
    models={32*k:population_model(loader.fast,blocks,k) for k in [1,4,8,16]}
    full=models[512];active=np.flatnonzero(full.operator.getnnz(axis=1))
    mass,z,_=base.core.configurations()['mass10'];mi,zi=np.unravel_index(active,full.shape);ze=z*.1
    lower=np.column_stack([mass[mi],ze[zi]]);upper=np.column_stack([mass[mi+1],ze[zi+1]])
    assert np.isfinite(upper).all()
    rate=full(*base.core.TRUTH).ravel()[active];cats=[]
    for n in manifest['expected']:
        for case in range(3):
            rng=np.random.default_rng(np.random.SeedSequence([manifest['seed'],n,case]));number=rng.poisson(n)
            bins=rng.choice(len(rate),number,p=rate/rate.sum());latent=rng.uniform(lower[bins],upper[bins])
            rho=12*rng.random(number)**(-1/3);scales=np.array([.03,.3])[None,:]*12/rho[:,None]
            y=np.log(latent)+scales*rng.normal(size=(number,2))
            cat=dict(n=n,case=case,number=int(number),latent=latent,scales=scales,y=y,duration=n/rate.sum())
            _,cat['coeff'],_=base.measurement(cat,1.,lower,upper);cats.append(cat)
            np.savez(out/f'catalogue_{n}_{case}.npz',**cat)
    # One broad full-support proposal shared by all sizes and catalogue counts.
    fisher=base.core.fisher_matrix(lambda t:full(*t),base.core.TRUTH,base.core.WIDTH,1000/rate.sum())
    summaries={}; comparisons=[]
    for power in [15,17]:
        for scramble in [0,1]:
            phi,logq=base.reference_design(base.core.TRUTH,fisher,202609215+scramble,power)
            for size,model in models.items():
                path=out/f'rates_{size}_{power}_{scramble}.npy'
                if not path.exists():
                    arr=np.lib.format.open_memmap(path,mode='w+',dtype='float64',shape=(len(phi),len(active)))
                    for first in range(0,len(phi),512):
                        arr[first:first+512]=[model(p[0],p[1],.012,p[2]).ravel()[active] for p in phi[first:first+512]]
                    arr.flush();del arr
                rates=np.load(path,mmap_mode='r')
                for cat in cats:
                    lw=target.event_target(rates,cat['coeff'],cat['duration'])-logq
                    summary=base.Experiment.summarize(phi,rates,cat['duration'],cat['number'],base.core.TRUTH,lw)
                    summaries[(cat['n'],cat['case'],size,scramble)]=summary
                logging.info('Population %sM power %s scramble %s done',size,power,scramble)
        comparisons=[];all_ok=True
        for cat in cats:
            ref=[summaries[(cat['n'],cat['case'],512,s)] for s in [0,1]]
            refok,refcheck=base.precision(*ref)
            for size in models:
                ss=[summaries[(cat['n'],cat['case'],size,s)] for s in [0,1]];ok,check=base.precision(*ss)
                valid=bool(ok and refok);all_ok &= valid
                metrics=[shifts(a,b) for a,b in zip(ref,ss)]
                comparisons.append(dict(expected=cat['n'],case=cat['case'],events=cat['number'],size=size,power=power,numerical_pass=valid,
                    numerical_check=check,reference_check=refcheck,metrics=metrics,summaries=ss,reference_summaries=ref))
        base.write_json(out/f'records_p{power}.json',comparisons)
        base.write_json(out/'records.json',comparisons)
        if all_ok:break
    base.write_json(out/'status.json',dict(state='complete',comparisons=len(comparisons),numerically_resolved=sum(r['numerical_pass'] for r in comparisons),power=power))
if __name__=='__main__':main()
