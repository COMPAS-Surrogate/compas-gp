"""Bounded full-trajectory four-input validation; separate from the paper campaign."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import qmc

HERE=Path(__file__).resolve().parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def build_tasks(source_jobs: list[dict]) -> list[dict]:
    """Eight saved-design checks followed by 32 matched fresh trajectories."""
    by_name={j['name']:(i,j) for i,j in enumerate(source_jobs)}
    tasks=[]
    for n,mode,strategy in itertools.product((100,500),('perfect','uncertain'),('bo','random')):
        i,_=by_name[f'count_n{n}_{mode}_c200_s17_{strategy}']
        tasks.append(dict(kind='saved_design',source_index=i))
    for case,n,mode,strategy in itertools.product((200,201,202),(50,500),('perfect','uncertain'),('bo','random')):
        _,source=by_name[f'count_n{n}_{mode}_c{case}_s17_{strategy}']
        tasks.append(dict(kind='trajectory',job=dict(source)))
    for n,mode,strategy in itertools.product((50,500),('perfect','uncertain'),('bo','random')):
        _,source=by_name[f'count_n{n}_{mode}_c200_s17_{strategy}']
        job=dict(source,seed=29,name=source['name'].replace('_s17_','_s29_'))
        tasks.append(dict(kind='trajectory',job=job))
    return tasks


def target_template(previous: dict, values: np.ndarray, core) -> dict:
    """Refresh only transform metadata before warm optimization on enlarged data."""
    scaler=core.AdaptiveRobustScaler(compression='sqrt',soft_clipping=False,
        lower_clip_value=float(values.min()),max_scale=1e9)
    scaler.initialize_with_data(values)
    return {**previous,'reference':jnp.asarray(scaler.reference_value),
            'scale':jnp.asarray(scaler.scale),'floor':jnp.asarray(scaler.lower_clip_value)}


def train(campaign, scorer, root: Path, inputs: Path, job: dict, maximum: int, identity: dict, stop_after: int | None = None):
    folder=root/'runs'/job['name'];folder.mkdir(parents=True,exist_ok=True)
    marker=folder/'run.json'
    if marker.exists() and json.loads(marker.read_text())!=identity:raise ValueError('Run identity changed')
    write(marker,identity)
    if (folder/'complete.json').exists():
        complete=json.loads((folder/'complete.json').read_text())
        if digest(folder/f'checkpoint_{maximum}.npz')!=complete['checkpoint_sha256']:raise ValueError('Completed checkpoint changed')
        return folder
    path=inputs/'catalogues'/job['catalogue']
    if digest(path)!=job['catalogue_sha256']:raise ValueError('Catalogue changed')
    cat=dict(np.load(path));engine=campaign.RateEngine(inputs)
    capacity=32*int(np.ceil(maximum/32));timing={k:0. for k in ('labels','acquisition','fit','conditioning')}
    previous_elapsed=0.;start=time.perf_counter()
    def timed(key,fn):
        tick=time.perf_counter();result=fn();jax.block_until_ready(result)
        timing[key]+=time.perf_counter()-tick;return result
    checkpoints=sorted(folder.glob('checkpoint_*.npz'),key=lambda p:int(p.stem.split('_')[-1]))
    if checkpoints:
        latest=checkpoints[-1];state,unit,values,rng=campaign.restore(latest,capacity)
        if len(unit)>maximum:raise ValueError('Checkpoint exceeds requested budget')
        saved=json.loads((folder/f'timing_{len(unit)}.json').read_text())
        timing=saved['components'];previous_elapsed=saved['elapsed_training_seconds']
    else:
        rng=np.random.default_rng(job['seed']+2026091500)
        boundary=np.array(list(itertools.product([0.,.5,1.],repeat=4)))
        boundary=boundary[np.any((boundary==0)|(boundary==1),axis=1)]
        unit=np.vstack([qmc.Sobol(4,scramble=True,seed=job['seed']).random_base2(6),boundary])
        values=timed('labels',lambda:campaign.evaluate(engine,cat,unit))
        if not np.isfinite(values).all():raise ValueError('Nonfinite initialization labels')
        state=timed('fit',lambda:campaign.fit(unit,values,capacity))
    def save():
        n=len(unit)
        write(folder/f'timing_{n}.json',dict(components=dict(timing),elapsed_training_seconds=previous_elapsed+time.perf_counter()-start))
        campaign.save(folder/f'checkpoint_{n}.npz',state,unit,values,rng)
        write(folder/'status.json',dict(state='training',labels=n,maximum=maximum,components=dict(timing)))
    if not checkpoints:save()
    endpoint=maximum if stop_after is None else min(maximum,stop_after)
    while len(unit)<endpoint:
        n=len(unit)
        if job['strategy']=='bo':
            selected,diagnostics=timed('acquisition',lambda:campaign.study.acquire_small_batch(state,unit,values,rng))
        elif job['strategy']=='random':selected,diagnostics=rng.random((3,4)),[]
        else:raise ValueError('Unsupported strategy')
        actual=timed('labels',lambda:campaign.evaluate(engine,cat,selected))
        if not np.isfinite(actual).all():raise ValueError('Nonfinite acquired labels')
        if len(np.unique(np.vstack([unit,selected]),axis=0))!=n+3:raise ValueError('Duplicate acquisition')
        write(folder/f'batch_{n}.json',dict(unit=selected.tolist(),actual=actual.tolist(),acquisition=diagnostics))
        unit,values=np.vstack([unit,selected]),np.r_[values,actual]
        if (len(unit)-144)%30==0:
            current=target_template(state,values,campaign.study.pilot.base.core)
            updated,info=timed('fit',lambda:scorer.refit_saved(campaign,current,state,unit,values,100))
            if not np.isfinite(info['final_loss']) or info['final_loss']>info['initial_loss']+1e-6:
                raise ValueError('Optimizer returned a nonfinite or worsened objective')
            template={k:v for k,v in updated.items() if k not in ('x','mask','alpha','inverse')}
            template.update(x=jnp.zeros((capacity,4)),mask=jnp.zeros(capacity))
            state=timed('conditioning',lambda:campaign.study.condition_on_data(template,unit,values))
            if not np.isfinite(np.asarray(state['alpha'])).all():raise ValueError('Nonfinite conditioned GP')
            write(folder/f'optimizer_{len(unit)}.json',info);save()
        else:state=timed('conditioning',lambda:campaign.study.condition_on_data(state,unit,values))
    if len(unit)<maximum:return folder
    write(folder/'complete.json',dict(labels=len(unit),checkpoint_sha256=digest(folder/f'checkpoint_{maximum}.npz'),
        state='training_complete_not_validated',reference_accessed=False,elapsed_training_seconds=previous_elapsed+time.perf_counter()-start,components=timing))
    return folder


def score(campaign, scorer, folder: Path, inputs: Path, job: dict, checkpoints: list[int]):
    cat=dict(np.load(inputs/'catalogues'/job['catalogue']));references=[]
    ref_hashes={}
    for s in (0,1):
        path=inputs/'references'/f"t{job['truth_index']}_n{job['reference_count']}_{s}.npz"
        ref_hashes[path.name]=digest(path)
        with np.load(path) as f:references.append((f['phi'],f['rates'],f['logq']))
    checked=campaign.study.pilot.reference.direct_checks(references,cat,cat['truth'])
    refs=[]
    for s,(phi,rates,logq) in enumerate(references):
        constant=np.concatenate([np.log(float(cat['duration'])*(r@cat['coeff'].T)).sum(axis=1)
            for r in np.array_split(rates,int(np.ceil(len(rates)/512)))])
        analytic=dict(checked[2][s]);analytic['quantiles']=np.asarray(analytic['quantiles'])[[0,2,3,4,6]].tolist()
        refs.append((phi,logq,float(cat['duration'])*rates.sum(axis=1)/.012,constant,analytic))
    metadata=dict(analytic_reference=checked[-1],events=len(cat['coeff']))
    seed=2026091600+int(hashlib.sha256(job['catalogue'].encode()).hexdigest()[:6],16)
    write(folder/'reference.json',dict(seed=seed,source_sha256=ref_hashes,metadata=metadata))
    rows=[]
    for n in checkpoints:
        path=folder/f'checkpoint_{n}.npz';sha=digest(path);out=folder/f'score_{n}.json'
        if out.exists():
            row=json.loads(out.read_text())
            if row['checkpoint_sha256']!=sha:raise ValueError('Scored checkpoint changed')
        else:
            tick=time.perf_counter();state,_,_,_=campaign.restore(path,n);attempts=[]
            for replicas in (16,64):
                write(folder/'status.json',dict(state='scoring',labels=n,replicas=replicas))
                result=campaign.study.score_once(state,refs,metadata,replicas,seed);attempts.append(result)
                if result['outcome']!='unresolved':break
            row=dict(labels=n,checkpoint_sha256=sha,outcome=attempts[-1]['outcome'],attempts=attempts,
                seconds=time.perf_counter()-tick,maximum_directed_kl=max(max(s['directed_kl']) for s in attempts[-1]['scores']))
            write(out,row)
        rows.append(row)
        write(folder/'history.json',dict(rows=rows,first_confirmed_tested=scorer.first_confirmed(rows)))
    write(folder/'status.json',dict(state='complete',scored_checkpoints=len(rows),first_confirmed_tested=scorer.first_confirmed(rows),final_outcome=rows[-1]['outcome']))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['pilot','run'])
    p.add_argument('--root',type=Path,default=HERE);p.add_argument('--task-index',type=int,default=0)
    args=p.parse_args();root=args.root.resolve()
    for relative,expected in json.loads((root/'checksums.json').read_text()).items():
        if digest(root/relative)!=expected:raise ValueError(f'Validation source changed: {relative}')
    manifest=json.loads((root/'manifest.json').read_text());original=Path(manifest['source_campaign']);inputs=original/'inputs'
    for relative,expected in json.loads((original/'code/paper_source_manifest.json').read_text()).items():
        if digest(original/'code'/relative)!=expected:raise ValueError(f'Frozen source changed: {relative}')
    sys.path.insert(0,str(original/'code/docs/studies/amplitude_gp'));import paper_campaign as campaign
    spec=importlib.util.spec_from_file_location('validation_scoring',root/'score_checkpoint_history.py')
    scorer=importlib.util.module_from_spec(spec);spec.loader.exec_module(scorer)
    if args.stage=='pilot':
        job=next(t['job'] for t in manifest['tasks'] if t['kind']=='trajectory');maximum=204;output=root/'pilot'
    else:
        task=manifest['tasks'][args.task_index]
        if task['kind']=='saved_design':
            subprocess.run([sys.executable,str(root/'score_checkpoint_history.py'),'--campaign',str(original),
                '--output',str(root/'lower_budget'),'--indices',str(task['source_index']),
                '--checkpoints','504','534','744','774','--optimizer-iterations','100'],check=True)
            return
        job=task['job'];maximum=manifest['maximum'];output=root/'trajectories'
    identity=dict(job=job,maximum=maximum,manifest_sha256=digest(root/'manifest.json'),
        checksums_sha256=digest(root/'checksums.json'),input_manifest_sha256=digest(inputs/'manifest.json'),
        source_manifest_sha256=digest(original/'code/paper_source_manifest.json'))
    if args.stage=='pilot':
        train(campaign,scorer,output,inputs,job,maximum,identity,stop_after=174)
    folder=train(campaign,scorer,output,inputs,job,maximum,identity)
    if args.stage=='pilot':
        # Verify restoration at padded and active capacities; replay a single
        # acquisition from the same checkpoint/RNG twice to test resume state.
        path=folder/'checkpoint_204.npz';a,x,y,rng=campaign.restore(path,224);b,_,_,_=campaign.restore(path,204)
        query=np.random.default_rng(1).random((64,4))
        np.testing.assert_allclose(campaign.study.pilot.base.predict(a,query),campaign.study.pilot.base.predict(b,query),rtol=1e-8,atol=1e-6)
        choices,_=campaign.study.acquire_small_batch(a,x,y,rng)
        restored,xx,yy,rr=campaign.restore(path,224);repeated,_=campaign.study.acquire_small_batch(restored,xx,yy,rr)
        np.testing.assert_allclose(choices,repeated,rtol=0,atol=1e-10)
        train(campaign,scorer,output,inputs,job,maximum,identity)
        write(root/'pilot_success.json',dict(labels=204,resumed_from=174,restore_parity=True,rng_acquisition_replay=True,
            checkpoint_sha256=digest(path),scope='Implementation smoke test; production-sized refit timings were measured separately. No scientific acceptance claimed.'))
    else:score(campaign,scorer,folder,inputs,job,manifest['score_checkpoints'])


if __name__=='__main__':main()
