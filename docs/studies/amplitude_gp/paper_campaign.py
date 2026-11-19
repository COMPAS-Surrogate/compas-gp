"""Four-input paper campaign: prepare immutable inputs, train, then score.

This fixed-budget campaign records checkpoints for later stopping-policy replay.
Training never loads reference designs. Completed training is not acceptance.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import logging
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import load_npz, save_npz
from scipy.stats import qmc

import small_batch_followup as study
from cosmic_integration.ratesSampler.ratesSampler import CosmicIntegration
from cosmic_integration.observation.binned_response import correlated_bin_response

LOW, WIDTH = study.pilot.LOW, study.pilot.WIDTH
write = study.pilot.base.write_json


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class RateEngine:
    """Portable exact fixed-population contraction; no likelihood interpolation."""
    def __init__(self, folder):
        meta = json.loads((folder/'rates.json').read_text())
        self.operator = load_npz(folder/'operator.npz')
        self.redshifts = np.load(folder/'redshifts.npy')
        self.zmin, self.zmax = meta['zmin'], meta['zmax']
        self.normalization = meta['mass_per_binary']*meta['systems']
        self.pdraw = meta['pdraw']
        self.shape = tuple(meta['shape'])
        self.active = np.asarray(meta['active'])

    def __call__(self, p):
        density, _, _ = CosmicIntegration.FindZdistribution(self, self.redshifts,
            np.log(self.zmin), np.log(self.zmax), p_Alpha=p[0], p_Sigma=p[1])
        formed = CosmicIntegration.CalculateSFR(self, self.redshifts, p[2], p[3])/self.normalization
        return (self.operator @ (formed[:, None]*density/self.pdraw).ravel())[self.active]


def evaluate(engine, cat, unit):
    physical = LOW+WIDTH*np.asarray(unit)
    rates = np.array([engine([p[0], p[1], .012, p[3]]) for p in physical])
    return study.pilot.joint_amplitude_log_likelihood(rates@cat['coeff'].T,
        rates.sum(axis=1), physical[:, 2], float(cat['duration']))


def prepare(folder):
    folder.mkdir(parents=True, exist_ok=False)
    (folder/'catalogues').mkdir()
    loader = study.pilot.base.data_source.Experiment(folder/'preparation', 'paper_four_input', False)
    fast = loader.fast
    save_npz(folder/'operator.npz', fast.operator)
    np.save(folder/'redshifts.npy', fast.ci.redshifts)
    write(folder/'rates.json', dict(zmin=fast.ci.compas.zamsZmin,zmax=fast.ci.compas.zamsZmax,
        mass_per_binary=fast.ci.compas.massEvolvedPerBinary,systems=fast.ci.compas.nSystems,
        pdraw=fast.p_draw,shape=fast.shape,active=loader.active.tolist()))
    engine = RateEngine(folder)
    points = LOW+WIDTH*qmc.Sobol(4,scramble=True,seed=202609151).random_base2(5)
    expected = np.array([fast(*p).ravel()[loader.active] for p in points])
    actual = np.array([engine(p) for p in points])
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.savez_compressed(folder/'rate_parity.npz',points=points,expected=expected)
    jobs=[]; sources={}
    def add(name,cat,group,strategies=('bo',),seeds=(17,),ti=0,n=100,source=None):
        if source is not None: sources[str(source.relative_to(study.pilot.REPO))]=digest(source)
        path=folder/'catalogues'/f'{name}.npz'
        np.savez_compressed(path,**cat)
        for seed,strategy in itertools.product(seeds,strategies):
            jobs.append(dict(name=f'{name}_s{seed}_{strategy}',catalogue=path.name,
                group=group,seed=seed,strategy=strategy,truth_index=ti,reference_count=100 if n<=200 else 1000,
                catalogue_sha256=digest(path)))
    studies=study.HERE.parent
    # Main figures: preserve the exact 35 latent catalogues and paired views.
    for case,n,mode in itertools.product(range(200,205),(10,50,100,200,500,700,1000),('perfect','uncertain')):
        src=studies/'operational_stopping/gpry_compas/count_history_20260912/training'/f't0_n{n}_{mode}_c{case}_bo/catalogue.npz'
        add(f'count_n{n}_{mode}_c{case}',dict(np.load(src)),'count_history',('bo','random'),n=n,source=src)
    for n,mode in itertools.product((30,1200),('perfect','uncertain')):
        src=studies/'event_uncertainty/paired_20260912/ensemble'/f'catalogue_{n}_{mode}_0.npz'
        add(f'predictive_n{n}_{mode}',dict(np.load(src)),'predictive',n=n,source=src)
    # Appendix robustness: retain original catalogues and three paired seeds.
    for ti,n,case in itertools.product(range(3),(50,500),range(2)):
        src=studies/'event_uncertainty/essential_20260910/robustness'/f'catalogue_t{ti}_{n}_{case}.npz'
        add(f'robust_t{ti}_n{n}_c{case}',dict(np.load(src)),'robustness',('bo','random','sobol'),(17,29,43),ti,n,src)
    for n,case,scale in itertools.product((50,500),range(10),(0.,.5,1.,2.)):
        src=studies/'event_uncertainty/results'/f'catalogue_{n}_{case}.npz'
        cat=dict(np.load(src))
        if scale==0: coeff=np.eye(len(loader.active))[cat['bins']]
        elif scale==1: coeff=cat['coeff']
        else: _,coeff,_=study.pilot.base.data_source.measurement(cat,scale,loader.lower,loader.upper)
        cat.update(coeff=coeff,duration=n/loader.fiducial_total,truth=study.pilot.base.core.TRUTH)
        add(f'width_n{n}_c{case}_x{scale:g}',cat,'width_sweep',('bo','random'),n=n,source=src)
    src=studies/'event_count_bo/results/catalogues.npz'; saved=np.load(src)
    for i,(counts,n,duration) in enumerate(zip(saved['counts'],saved['expected_events'],saved['durations'])):
        counts=counts[loader.active].astype(int)
        cat=dict(coeff=np.repeat(np.eye(len(loader.active)),counts,axis=0),duration=duration,truth=saved['truth'])
        add(f'space_n{int(n)}_c{i%10}',cat,'space_filling',('bo','random','sobol'),n=int(n),source=src)
    src=studies/'joint_pilot/uncertain_100_results/catalogues.npz'; saved=np.load(src)
    response=correlated_bin_response(fast.shape).toarray()[:,loader.active]
    for i,(truth,counts) in enumerate(zip(saved['truths'],saved['observed'])):
        cat=dict(coeff=np.repeat(response,counts.ravel().astype(int),axis=0),duration=.1,truth=truth)
        add(f'categorical_c{i}',cat,'categorical',n=int(counts.sum()),source=src)
    assert len(jobs)==632 and len({j['name'] for j in jobs})==632
    # Independent reference rates are stored separately and accessed only by score().
    refs=folder/'references';refs.mkdir()
    for ti,n in itertools.product(range(3),(100,1000)):
        provenance={}
        for s,(phi,rates,logq) in enumerate(study.pilot.reference.cached_references(ti,n,provenance)):
            np.savez_compressed(refs/f't{ti}_n{n}_{s}.npz',phi=phi,rates=rates,logq=logq)
        sources.update(provenance)
    write(folder/'manifest.json',dict(jobs=jobs,input_sources=sources,maximum=2994,initial=144,
        checkpoints='Every 30 added labels; three-point batches; refit every 30',
        inputs=['alpha','sigma','SFRa','SFRd'],reference_access_during_training=False,
        direct_only=['population likelihood loss','pixel effective counts','physical rate and event-weight illustrations'],
        downstream=['Reference-free stopping replay','Conditional amplitude checks','Posterior predictive figures'],
        scope='Four-input retraining on existing paper catalogues. Fixed budget is not an accepted stop.'))
    files=[p for p in folder.rglob('*') if p.is_file() and 'preparation' not in p.parts]
    write(folder/'input_hashes.json',{str(p.relative_to(folder)):digest(p) for p in files})
    logging.info('Prepared %d jobs',len(jobs))


def fit(unit,values,capacity):
    state=study.pilot.fit(unit,values)
    template={k:v for k,v in state.items() if k not in ('x','mask','alpha','inverse')}
    template.update(x=jnp.zeros((capacity,4)),mask=jnp.zeros(capacity))
    return study.condition_on_data(template,unit,values)


def save(path,state,unit,values,rng):
    params={f'param_{k}':np.asarray(v) for k,v in state.items() if k not in ('x','mask','alpha','inverse')}
    temp=path.with_suffix('.tmp.npz')
    np.savez_compressed(temp,unit=unit,lnl=values,rng=json.dumps(rng.bit_generator.state),**params)
    temp.replace(path)


def restore(path,capacity):
    with np.load(path) as f:
        unit,values=f['unit'],f['lnl']
        state={k[6:]:jnp.asarray(f[k]) for k in f.files if k.startswith('param_')}
        rng=np.random.default_rng();rng.bit_generator.state=json.loads(str(f['rng']))
    state.update(x=jnp.zeros((capacity,4)),mask=jnp.zeros(capacity))
    return study.condition_on_data(state,unit,values),unit,values,rng


def train(inputs,output,index,maximum):
    if maximum<144 or (maximum-144)%30: raise ValueError('Budget must be 144 plus a multiple of 30')
    job=json.loads((inputs/'manifest.json').read_text())['jobs'][index]
    folder=output/job['name'];folder.mkdir(parents=True,exist_ok=True)
    identity=dict(job=job,maximum=maximum,input_manifest_sha256=digest(inputs/'manifest.json'),
                  runner_sha256=digest(__file__))
    frozen=study.pilot.REPO/'paper_source_manifest.json'
    if frozen.exists():
        for relative,expected in json.loads(frozen.read_text()).items():
            if digest(study.pilot.REPO/relative)!=expected:
                raise ValueError(f'Frozen source changed: {relative}')
        identity['source_manifest_sha256']=digest(frozen)
    if (folder/'run.json').exists():
        if json.loads((folder/'run.json').read_text())!=identity:
            raise ValueError('Run identity changed; use a new output directory')
    else:write(folder/'run.json',identity)
    if (folder/'complete.json').exists():return
    catpath=inputs/'catalogues'/job['catalogue'];assert digest(catpath)==job['catalogue_sha256']
    cat=dict(np.load(catpath));engine=RateEngine(inputs)
    capacity=32*int(np.ceil(maximum/32))
    rng=np.random.default_rng(job['seed']+2026091500)
    resume=sorted(folder.glob('checkpoint_*.npz'),key=lambda p:int(p.stem.split('_')[-1]))
    if resume:state,unit,values,rng=restore(resume[-1],capacity)
    else:
        boundary=np.array(list(itertools.product([0.,.5,1.],repeat=4)))
        boundary=boundary[np.any((boundary==0)|(boundary==1),axis=1)]
        unit=np.vstack([qmc.Sobol(4,scramble=True,seed=job['seed']).random_base2(6),boundary])
        values=evaluate(engine,cat,unit)
        if not np.isfinite(values).all():raise ValueError('Nonfinite initialization labels')
        state=fit(unit,values,capacity)
        save(folder/'checkpoint_144.npz',state,unit,values,rng)
    if not np.isfinite(values).all():raise ValueError('Nonfinite labels; support failure is not floored')
    sequence=qmc.Sobol(4,scramble=True,seed=job['seed']+1).random_base2(12)
    started=time.perf_counter()
    while len(unit)<maximum:
        count=len(unit)
        if job['strategy']=='bo':selected,diagnostics=study.acquire_small_batch(state,unit,values,rng)
        elif job['strategy']=='random':selected,diagnostics=rng.random((3,4)),[]
        else:selected,diagnostics=sequence[count-144:count-141],[]
        prediction=study.pilot.base.predict(state,selected)
        actual=evaluate(engine,cat,selected)
        if not np.isfinite(actual).all():raise ValueError('Nonfinite acquired likelihood')
        if len(np.unique(np.vstack([unit,selected]),axis=0))!=count+3:raise ValueError('Duplicate acquisition')
        write(folder/f'batch_{count}.json',dict(unit=selected.tolist(),predicted=prediction.tolist(),actual=actual.tolist(),acquisition=diagnostics))
        unit,values=np.vstack([unit,selected]),np.r_[values,actual]
        if (len(unit)-144)%30==0:
            state=fit(unit,values,capacity)
            save(folder/f'checkpoint_{len(unit)}.npz',state,unit,values,rng)
            write(folder/'status.json',dict(state='training',labels=len(unit),maximum=maximum,session_seconds=time.perf_counter()-started))
        else:state=study.condition_on_data(state,unit,values)
    write(folder/'complete.json',dict(job=job,labels=len(unit),checkpoint_sha256=digest(folder/f'checkpoint_{maximum}.npz'),
        state='training_complete_not_validated',reference_accessed=False))


def score(inputs,output,index,maximum):
    job=json.loads((inputs/'manifest.json').read_text())['jobs'][index];folder=output/job['name']
    if not (folder/'complete.json').exists():raise ValueError('Training must complete before reference access')
    cat=dict(np.load(inputs/'catalogues'/job['catalogue']))
    references=[]
    for s in (0,1):
        with np.load(inputs/'references'/f't{job["truth_index"]}_n{job["reference_count"]}_{s}.npz') as f:
            references.append((f['phi'],f['rates'],f['logq']))
    checked=study.pilot.reference.direct_checks(references,cat,cat['truth'])
    refs=[]
    for s,(phi,rates,logq) in enumerate(references):
        constant=np.concatenate([np.log(float(cat['duration'])*(r@cat['coeff'].T)).sum(axis=1) for r in np.array_split(rates,int(np.ceil(len(rates)/512)))])
        analytic=dict(checked[2][s]);analytic['quantiles']=np.asarray(analytic['quantiles'])[[0,2,3,4,6]].tolist()
        refs.append((phi,logq,float(cat['duration'])*rates.sum(axis=1)/.012,constant,analytic))
    metadata=dict(analytic_reference=checked[-1],events=len(cat['coeff']))
    state,_,_,_=restore(folder/f'checkpoint_{maximum}.npz',32*int(np.ceil(maximum/32)))
    attempts=[]
    for replicas in (16,64):
        result=study.score_once(state,refs,metadata,replicas,2026091510+index)
        attempts.append(result);write(folder/'score.json',dict(job=job,labels=maximum,attempts=attempts,outcome=result['outcome']))
        if result['outcome']!='unresolved':break
    write(folder/'status.json',dict(state='scored',labels=maximum,outcome=result['outcome']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','pilot','run','score'])
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--index',type=int,default=0)
    parser.add_argument('--maximum',type=int,default=2994)
    args=parser.parse_args();logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    if args.stage=='prepare':prepare(args.inputs.resolve());return
    if args.output is None:parser.error('--output required')
    if args.stage=='pilot':
        for relative,expected in json.loads((args.inputs/'input_hashes.json').read_text()).items():
            if digest(args.inputs/relative)!=expected:raise ValueError(f'Input hash mismatch: {relative}')
        engine=RateEngine(args.inputs);parity=np.load(args.inputs/'rate_parity.npz')
        np.testing.assert_allclose([engine(p) for p in parity['points']],parity['expected'],rtol=1e-11,atol=1e-11)
        train(args.inputs,args.output,args.index,174)
    elif args.stage=='run':
        train(args.inputs,args.output,args.index,args.maximum)
        score(args.inputs,args.output,args.index,args.maximum)
    else:score(args.inputs,args.output,args.index,args.maximum)


if __name__=='__main__':main()
