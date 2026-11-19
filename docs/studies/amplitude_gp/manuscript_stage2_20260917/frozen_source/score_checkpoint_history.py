"""Offline four-input checkpoint scoring against frozen independent references.

Never acquires likelihood training labels or changes campaign outputs. A coarse
screen reports only the first confirmed TESTED pair; it is not an exhaustive
minimum-budget estimate. Confirmation requires adjacent 30-label checkpoints.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


def first_confirmed(rows: list[dict]) -> int | None:
    """Find the first tested accurate checkpoint confirmed exactly 30 labels later."""
    by_count = {r['labels']:r['outcome'] for r in rows}
    return next((n for n in sorted(by_count)
                 if by_count[n]=='accurate' and by_count.get(n+30)=='accurate'), None)


def write(path: Path, value: dict) -> None:
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n');temp.replace(path)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--indices',type=int,nargs='+',required=True)
    p.add_argument('--optimizer-iterations',type=int,default=0,help='Opt-in fixed-design warm L-BFGS-B diagnostic; changes the GP')
    p.add_argument('--checkpoints',type=int,nargs='+',default=[414,444,744,774,1254,1284,2964,2994])
    args=p.parse_args()
    if any(n<144 or (n-144)%30 for n in args.checkpoints):p.error('Checkpoints must be 144 + 30k')
    root=args.campaign;code=root/'code';inputs=root/'inputs'
    sys.path.insert(0,str(code/'docs/studies/amplitude_gp'))
    import paper_campaign as campaign
    import jax
    for rel,expected in json.loads((code/'paper_source_manifest.json').read_text()).items():
        if digest(code/rel)!=expected:raise ValueError(f'Frozen source changed: {rel}')
    jobs=json.loads((inputs/'manifest.json').read_text())['jobs']
    args.output.mkdir(parents=True,exist_ok=True)
    cached={}
    for index in args.indices:
        job=jobs[index]; folder=root/'runs'/job['name']; out=args.output/job['name']
        out.mkdir(exist_ok=True)
        catpath=inputs/'catalogues'/job['catalogue']
        if digest(catpath)!=job['catalogue_sha256']:raise ValueError('Catalogue changed')
        # Same amplitude designs for all checkpoints AND methods of a catalogue.
        seed=2026091600+int(hashlib.sha256(job['catalogue'].encode()).hexdigest()[:6],16)
        refpaths=[inputs/'references'/f"t{job['truth_index']}_n{job['reference_count']}_{s}.npz" for s in (0,1)]
        identity=dict(job=job,checkpoints=sorted(set(args.checkpoints)),seed=seed,
            optimizer_iterations=args.optimizer_iterations,runner_sha256=digest(Path(__file__)),input_manifest_sha256=digest(inputs/'manifest.json'),
            source_manifest_sha256=digest(code/'paper_source_manifest.json'),
            reference_sha256={p.name:digest(p) for p in refpaths})
        if (out/'identity.json').exists() and json.loads((out/'identity.json').read_text())!=identity:
            raise ValueError('History-scoring identity changed')
        write(out/'identity.json',identity)
        if job['catalogue'] not in cached:
            cat=dict(np.load(catpath)); refs=[]
            for path in refpaths:
                with np.load(path) as f:refs.append((f['phi'],f['rates'],f['logq']))
            checked=campaign.study.pilot.reference.direct_checks(refs,cat,cat['truth'])
            prepared=[]
            for s,(phi,rates,logq) in enumerate(refs):
                constant=np.concatenate([np.log(float(cat['duration'])*(r@cat['coeff'].T)).sum(axis=1)
                    for r in np.array_split(rates,int(np.ceil(len(rates)/512)))])
                analytic=dict(checked[2][s]);analytic['quantiles']=np.asarray(analytic['quantiles'])[[0,2,3,4,6]].tolist()
                prepared.append((phi,logq,float(cat['duration'])*rates.sum(axis=1)/.012,constant,analytic))
            cached[job['catalogue']]=(prepared,dict(analytic_reference=checked[-1],events=len(cat['coeff'])))
        refs,metadata=cached[job['catalogue']]
        rows=[]
        for n in identity['checkpoints']:
            path=folder/f'checkpoint_{n}.npz'
            if not path.exists():raise FileNotFoundError(path)
            sha=digest(path); resultpath=out/f'score_{n}.json'
            if resultpath.exists():
                row=json.loads(resultpath.read_text())
                if row['checkpoint_sha256']!=sha:raise ValueError('Checkpoint changed')
            else:
                start=time.perf_counter()
                # Algebraically inactive padding may be removed for offline scoring.
                # This does not fit hyperparameters or change training labels.
                state,unit,values,rng=campaign.restore(path,n)
                if len(unit)!=n:raise ValueError('Wrong checkpoint length')
                jax.block_until_ready(state)
                optimizer=None
                if args.optimizer_iterations:
                    previous_path=folder/f'checkpoint_{n-30}.npz'
                    previous,_,_,_=campaign.restore(previous_path,n)
                    state,optimizer=refit_saved(campaign,state,previous,unit,values,args.optimizer_iterations)
                    candidate=out/f'candidate_{n}.npz'
                    campaign.save(candidate,state,unit,values,rng)
                    optimizer.update(previous_checkpoint_sha256=digest(previous_path),candidate_checkpoint_sha256=digest(candidate))
                    write(out/f'optimizer_{n}.json',optimizer)
                attempts=[]
                for replicas in (16,64):
                    write(out/'status.json',dict(state='scoring',labels=n,replicas=replicas))
                    result=campaign.study.score_once(state,refs,metadata,replicas,seed)
                    attempts.append(result)
                    if result['outcome']!='unresolved':break
                row=dict(labels=n,checkpoint_sha256=sha,optimizer=optimizer,attempts=attempts,outcome=attempts[-1]['outcome'],
                         seconds=time.perf_counter()-start,
                         maximum_directed_kl=max(max(s['directed_kl']) for s in attempts[-1]['scores']))
                write(resultpath,row)
            rows.append(row)
            write(out/'history.json',dict(job=job,rows=rows,first_confirmed_tested=first_confirmed(rows),
                scope='First confirmed tested checkpoint, not an exhaustive minimum or operational stopping policy.'))
            print(json.dumps(dict(name=job['name'],labels=n,outcome=row['outcome'],
                kl=row['maximum_directed_kl'],seconds=row['seconds'])),flush=True)
        write(out/'status.json',dict(state='complete',checkpoints=len(rows),first_confirmed_tested=first_confirmed(rows)))


def refit_saved(campaign, current, previous, unit, values, iterations):
    """Warm-fit fixed labels; return changed GP plus optimizer diagnostics."""
    import equinox as eqx
    import gpjax as gpx
    import jax
    import jax.numpy as jnp
    import paramax
    from jax.flatten_util import ravel_pytree
    from scipy.optimize import minimize
    from cosmic_integration.lnl_surrogate.jax_active_learner import FittedJaxGP
    from cosmic_integration.lnl_surrogate.gpry_rules import state_from_gpjax
    core=campaign.study.pilot.base.core
    scaler=core.AdaptiveRobustScaler(compression='sqrt',soft_clipping=False,
        lower_clip_value=float(values.min()),max_scale=1e9)
    scaler.initialize_with_data(values)
    data=gpx.Dataset(X=jnp.asarray(unit),y=jnp.asarray(-scaler.transform(values))[:,None])
    np.testing.assert_allclose(data.y[:,0],jnp.sqrt(jnp.maximum(current['reference']-jnp.maximum(values,current['floor']),0))/current['scale'],rtol=1e-12,atol=1e-12)
    ratio=previous['scale']/current['scale']
    prior=gpx.gps.Prior(kernel=gpx.kernels.Matern52(lengthscale=previous['lengthscale'],variance=previous['amplitude']*ratio**2),
        mean_function=gpx.mean_functions.Constant(constant=previous['constant']*ratio))
    model=prior*gpx.likelihoods.Gaussian(num_datapoints=len(unit),obs_stddev=jnp.sqrt(current['noise']-current['jitter']))
    assert float(model.jitter)==float(current['jitter'])
    model=eqx.tree_at(lambda x:x.likelihood.obs_stddev,model,replace_fn=lambda x:paramax.NonTrainable(x.unwrap()))
    params,static=eqx.partition(model,eqx.is_array);initial,unravel=ravel_pytree(params)
    @jax.jit
    def objective(vector):
        return jax.value_and_grad(lambda x:-gpx.objectives.conjugate_mll(paramax.unwrap(eqx.combine(unravel(x),static)),data))(vector)
    start=time.perf_counter()
    initial_loss=float(objective(initial)[0])
    result=minimize(objective,np.asarray(initial),jac=True,method='L-BFGS-B',options={'maxiter':iterations,'maxls':10,'ftol':1e-9,'gtol':1e-5})
    fitted=FittedJaxGP(eqx.combine(unravel(jnp.asarray(result.x)),static),core.JaxTrainingData(unit,np.asarray(data.y)),np.zeros(1))
    state=state_from_gpjax(fitted,scaler,np.zeros(4),np.ones(4),len(unit))
    jax.block_until_ready(state)
    return state,dict(seconds=time.perf_counter()-start,iterations=int(result.nit),evaluations=int(result.nfev),
        success=bool(result.success),message=str(result.message),initial_loss=initial_loss,final_loss=float(result.fun),
        transform_reference_change=float(current['reference']-previous['reference']),
        scope='Optimized fixed saved design, not a new warm-start BO trajectory.')


def summarize_retrieved(root: Path) -> dict:
    """Verify retrieved evidence and summarize every tested checkpoint, including failures."""
    rows=[]
    for history in sorted((root/'retrieved').glob('*/*/history.json')):
        record=json.loads(history.read_text());identity=json.loads((history.parent/'identity.json').read_text())
        if record['job']!=identity['job']:raise ValueError('Job identity mismatch')
        for row in record['rows']:
            attempt=row['attempts'][-1]
            if row['outcome']!=attempt['outcome']:raise ValueError('Outcome mismatch')
            if row['outcome']=='accurate':
                assert all(r['accuracy_pass'] for r in attempt['scores'])
                assert attempt['direct_precision']['passed'] and attempt['gp_precision']['passed'] and attempt['kl_precision']
                assert all(p['passed'] for p in attempt['parity'])
            if row.get('optimizer'):
                candidate=history.parent/f"candidate_{row['labels']}.npz"
                if digest(candidate)!=row['optimizer']['candidate_checkpoint_sha256']:raise ValueError('Candidate checksum mismatch')
            rows.append(dict(variant=history.parent.parent.name,name=record['job']['name'],
                catalogue=record['job']['catalogue'],strategy=record['job']['strategy'],labels=row['labels'],
                outcome=row['outcome'],maximum_directed_kl=row['maximum_directed_kl'],
                reference_seed=identity['seed'],checkpoint_sha256=row['checkpoint_sha256'],
                optimizer=row.get('optimizer'),scoring_seconds=row['seconds']))
    seeds={}
    for row in rows:seeds.setdefault(row['catalogue'],set()).add(row['reference_seed'])
    if any(len(v)!=1 for v in seeds.values()):raise ValueError('Comparison used different amplitude designs')
    result=dict(rows=rows,scope='One 100-event-expected latent catalogue, two paired measurement views, one training seed. Fixed-design refits are not new BO trajectories. First tested confirmation is not the minimum needed.')
    write(root/'summary.json',result)
    return result


if __name__=='__main__':main()
