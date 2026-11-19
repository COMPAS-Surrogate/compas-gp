"""Blind operational stopping followed by separately executed reference scoring.

The learner receives only a counted evaluation callable, bounds and a seed.
Reference quadratures and exact values are constructed after its decision file
has been written. GP posterior quadrature uses independent, truth-free designs.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import qmc
from scipy.special import logsumexp

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
# Import existing, regression-tested exact mean prediction and likelihood moments.
sys.path.insert(0,str(HERE.parent/'event_uncertainty'))
from posterior_convergence import reference_means
from essential_runs import core
from cosmic_integration.lnl_surrogate.stopping import (
    StoppingTolerances, prediction_check, posterior_summary, summaries_agree,
    summary_change, symmetric_kl, audit_check, next_streak)
from toys import make_target, FAMILIES

TOLERANCES=StoppingTolerances()
CONFIG={'initial_sobol':32,'boundary_points':16,'batch':32,'candidate_points':2048,
        'gp_quadrature_power':14,'audit_posterior':64,'audit_prior':16,
        'maximum_calls':800,'fit_steps':250,'consecutive':2}


def write_json(path: Path, value) -> None:
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


class Evaluations:
    """Account for every expensive value, including rejected audit batches."""
    def __init__(self, evaluator):
        self._evaluator=evaluator
        self.calls=0

    def __call__(self, x):
        self.calls+=len(x)
        result=np.asarray(self._evaluator(x))
        if result.shape!=(len(x),) or not np.isfinite(result).all():
            raise ValueError('Toy evaluation must return one finite value per point')
        return result


def fit(x, y):
    scaler=core.AdaptiveRobustScaler(compression='sqrt',soft_clipping=False,
                                    lower_clip_value=float(y.min()),max_scale=1e9)
    scaler.initialize_with_data(y)
    model=core.fit_exact_gp(core.JaxTrainingData(x,-scaler.transform(y)[:,None]),
                           np.vstack([np.zeros(x.shape[1]),np.ones(x.shape[1])]),
                           config=core.JaxGPConfig(optimisation_steps=CONFIG['fit_steps']),seed=17)
    return model,scaler


def predict(model, scaler, x):
    return scaler.inverse_transform(-reference_means(model,x))


def operational_run(evaluate: Evaluations, seed: int, folder: Path,
                    dimensions: int = 2) -> dict:
    """Make the stopping decision without a target family or reference argument."""
    rng=np.random.default_rng(seed)
    boundary=np.array(np.meshgrid(*([np.linspace(0,1,5)]*dimensions),indexing='ij')).reshape(dimensions,-1).T
    boundary=boundary[np.any((boundary==0)|(boundary==1),axis=1)]
    x=np.vstack([qmc.Sobol(dimensions,scramble=True,seed=seed).random_base2(int(np.log2(CONFIG['initial_sobol']))),boundary])
    y=evaluate(x)
    candidates=qmc.Sobol(dimensions,scramble=True,seed=seed+11).random_base2(int(np.log2(CONFIG['candidate_points'])))
    available=np.ones(len(candidates),dtype=bool)
    quadrature=[qmc.Sobol(dimensions,scramble=True,seed=seed+100+s).random_base2(CONFIG['gp_quadrature_power']) for s in [0,1]]
    previous=previous_summaries=None
    prequential={'passed':False,'reason':'initialization'}
    streak=0;records=[];audits=[];state='resource_limit';started=time.perf_counter()
    while True:
        model,scaler=fit(x,y)
        predictions=[predict(model,scaler,q) for q in quadrature]
        summaries=[posterior_summary(q,p) for q,p in zip(quadrature,predictions)]
        numerical,precision=summaries_agree(*summaries,TOLERANCES)
        change=None;stable=False
        if previous is not None:
            change=max(symmetric_kl(a,b) for a,b in zip(previous,predictions))
            stable=change<TOLERANCES.stability_kl and all(
                summaries_agree(a,b,TOLERANCES)[0] for a,b in zip(previous_summaries,summaries))
        streak=next_streak(streak,prequential['passed'],stable,numerical)
        checkpoint=len(records)
        np.savez(folder/f'checkpoint_{checkpoint:03d}.npz',x=x,lnl=y)
        row={'checkpoint':checkpoint,'training_points':len(x),'calls':evaluate.calls,
             'prediction':prequential,'stability_kl':change,'stable':bool(stable),
             'numerical_pass':numerical,'precision':precision,'summaries':summaries,'streak':streak}
        records.append(row)
        write_json(folder/'records.json',records)
        logging.info('%s points=%s calls=%s prediction=%s stable=%s numerical=%s streak=%s',
                     folder.name,len(x),evaluate.calls,prequential['passed'],stable,numerical,streak)
        if streak>=TOLERANCES.consecutive:
            size=CONFIG['audit_posterior']+CONFIG['audit_prior']
            if evaluate.calls+size>CONFIG['maximum_calls']:
                break
            # Coordinates are sampled from GP-only quadrature, not a reference.
            q=quadrature[len(audits)%2];p=predictions[len(audits)%2]
            weights=np.exp(p-logsumexp(p))
            ix=rng.choice(len(q),size=CONFIG['audit_posterior'],replace=False,p=weights)
            audit_x=np.vstack([q[ix],rng.uniform(size=(CONFIG['audit_prior'],dimensions))])
            before=predict(model,scaler,audit_x)
            actual=evaluate(audit_x)
            check=audit_check(before,actual,CONFIG['audit_posterior'],float(y.max()),TOLERANCES)
            np.savez(folder/f'audit_{len(audits):03d}.npz',x=audit_x,predicted=before,actual=actual)
            audits.append({'checkpoint':checkpoint,'calls':evaluate.calls,**check})
            write_json(folder/'audits.json',audits)
            if check['passed']:
                state='operational_stop'
                break
            # Failed audits become training data only after the frozen scoring.
            prequential=prediction_check(before,actual,TOLERANCES)
            x=np.vstack([x,audit_x]);y=np.r_[y,actual]
            previous,previous_summaries=predictions,summaries
            streak=0
            continue
        if evaluate.calls+CONFIG['batch']>CONFIG['maximum_calls']:
            break
        mu,var=model.predict_f(candidates)
        first=core.sqrt_log_expected_likelihood(mu[:,0],var[:,0],scaler)
        second=core.sqrt_log_expected_likelihood(mu[:,0],var[:,0],scaler,power=2.)
        logvar=second+np.log(np.maximum(-np.expm1(np.minimum(2*first-second,0)),1e-300))
        ids=np.flatnonzero(available)
        probability=np.exp(logvar[ids]-logsumexp(logvar[ids]))*.9+.1/len(ids)
        selected=rng.choice(ids,CONFIG['batch'],replace=False,p=probability/probability.sum())
        available[selected]=False
        new_x=candidates[selected]
        before=predict(model,scaler,new_x)
        actual=evaluate(new_x)
        prequential=prediction_check(before,actual,TOLERANCES)
        np.savez(folder/f'acquisition_{checkpoint:03d}.npz',x=new_x,predicted=before,actual=actual)
        x=np.vstack([x,new_x]);y=np.r_[y,actual]
        previous,previous_summaries=predictions,summaries
    result={'state':state,'calls':evaluate.calls,'training_points':len(x),
            'audits':len(audits),'checkpoint':len(records)-1,'seconds':time.perf_counter()-started,
            'reference_accessed':False}
    write_json(folder/'decision.json',result)
    return result


def reference_score(target, folder: Path, seed: int) -> dict:
    """Reveal the reference only after a durable stopping decision exists."""
    decision_bytes=(folder/'decision.json').read_bytes()
    decision=json.loads(decision_bytes)
    assert not decision['reference_accessed']
    points=[qmc.Sobol(2,scramble=True,seed=seed+900+s).random_base2(17) for s in [0,1]]
    exact=[target(q) for q in points]
    reference_summaries=[posterior_summary(q,p) for q,p in zip(points,exact)]
    reference_precision=summaries_agree(*reference_summaries,TOLERANCES)[0]
    scores=[]
    for record in json.loads((folder/'records.json').read_text()):
        data=np.load(folder/f"checkpoint_{record['checkpoint']:03d}.npz")
        model,scaler=fit(data['x'],data['lnl'])
        predicted=[predict(model,scaler,q) for q in points]
        gs=[posterior_summary(q,p) for q,p in zip(points,predicted)]
        kl=[symmetric_kl(a,b) for a,b in zip(exact,predicted)]
        marginal=[summary_change(a,b) for a,b in zip(reference_summaries,gs)]
        numerical=reference_precision and summaries_agree(*gs,TOLERANCES)[0]
        accurate=max(kl)<.01 and all(m['mean_sd']<.2 and m['endpoint_sd']<.2 and m['width_fraction']<.1 for m in marginal)
        scores.append({'checkpoint':record['checkpoint'],'calls':record['calls'],
                       'reference_kl':kl,'numerical_pass':bool(numerical),
                       'accuracy_pass':bool(accurate),'marginal_errors':marginal})
    result={'decision_sha256':hashlib.sha256(decision_bytes).hexdigest(),
            'reference_power':17,'reference_precision_pass':reference_precision,
            'reference_evaluations':sum(map(len,points)),'scores':scores,
            'premature_stop':bool(decision['state']=='operational_stop' and not scores[-1]['accuracy_pass']),
            'final_numerical_pass':scores[-1]['numerical_pass']}
    assert decision_bytes==(folder/'decision.json').read_bytes()
    write_json(folder/'reference_score.json',result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--outdir',type=Path,required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=[2000,2001,2002,2003,2004])
    parser.add_argument('--families',nargs='+',choices=FAMILIES,default=FAMILIES)
    args=parser.parse_args()
    args.outdir.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__),HERE/'toys.py',HERE/'PROTOCOL.md',REPO/'src/cosmic_integration/lnl_surrogate/stopping.py']
    manifest={'configuration':CONFIG,'tolerances':asdict(TOLERANCES),'families':args.families,'seeds':args.seeds,
              'created_unix':time.time(),'source_sha256':{str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    write_json(args.outdir/'manifest.json',manifest)
    for p in sources:
        dest=args.outdir/'source_snapshot'/p.relative_to(REPO);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    outcomes=[]
    try:
        for family in args.families:
            for seed in args.seeds:
                folder=args.outdir/f'{family}_{seed}';folder.mkdir()
                write_json(args.outdir/'status.json',{'state':'running','current':folder.name,'completed':len(outcomes),'phase':'blind_learning'})
                target=make_target(family,seed)
                decision=operational_run(Evaluations(target),seed+10000,folder)
                write_json(args.outdir/'status.json',{'state':'running','current':folder.name,'completed':len(outcomes),'phase':'post_decision_reference'})
                scored=reference_score(target,folder,seed+10000)
                outcomes.append({'family':family,'seed':seed,**decision,'premature_stop':scored['premature_stop'],
                                 'reference_numerical_pass':scored['final_numerical_pass'],
                                 'final_reference_kl':max(scored['scores'][-1]['reference_kl'])})
                write_json(args.outdir/'outcomes.json',outcomes)
        write_json(args.outdir/'status.json',{'state':'complete','completed':len(outcomes)})
    except Exception as exc:
        write_json(args.outdir/'status.json',{'state':'failed','completed':len(outcomes),'error':repr(exc)})
        raise


if __name__=='__main__':
    main()
