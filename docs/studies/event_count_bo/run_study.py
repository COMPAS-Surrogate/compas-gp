"""Frozen event-count/BO experiment with internal importance-quadrature references.

One interior truth, four exposures, ten catalogues/exposure, matched BO/Sobol
training budgets 226,354,482,610,738. Truth-informed proposals ONLY validate;
the acquisition has no access to injection, reference rates or Fisher matrices.
"""
from __future__ import annotations
import hashlib, json, logging, pickle, sys, time
from pathlib import Path
import numpy as np
from scipy.sparse import load_npz
from scipy.special import logsumexp, xlogy, ndtri
from scipy.stats import qmc, multivariate_t, chi2

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent/'joint_pilot'))
from bo_uncertainty_curve import (configurations, FixedPopulationRates, BinnedCosmicIntegrator,
    AdaptiveRobustScaler, JaxTrainingData, JaxGPConfig, fit_exact_gp,
    vector_marginal, sqrt_log_expected_likelihood, posterior_metrics)
from check_ensemble_quadrature import direct_summary
from confirm_482 import compare

LOW = np.array([-.5,.1,.005,4.2])
WIDTH = np.array([.499,.5,.01,1.])
TRUTH = np.array([-.3,.35,.01,4.7])
SHAPE = [0,1,3]


def poisson_expected_gap(expected: np.ndarray, candidate: np.ndarray) -> float:
    """Expected log likelihood minus its value at the generating intensity."""
    expected, candidate = np.asarray(expected), np.asarray(candidate)
    if expected.shape != candidate.shape or np.any(expected < 0) or np.any(candidate < 0):
        raise ValueError('Matching nonnegative intensities required')
    if np.any((expected > 0)&(candidate == 0)):
        return -np.inf
    occupied = expected > 0
    return float(np.sum(xlogy(expected[occupied], candidate[occupied]/expected[occupied]))
                 - np.sum(candidate-expected))


def fisher_matrix(rate_function, truth: np.ndarray, widths: np.ndarray,
                  duration: float, step: float = 1e-4) -> np.ndarray:
    """Poisson Fisher matrix in unit-prior coordinates; all four parameters free."""
    r = rate_function(truth).ravel()
    derivatives = []
    for j in range(4):
        shift = np.eye(4)[j]*widths*step
        derivatives.append((rate_function(truth+shift).ravel()-rate_function(truth-shift).ravel())/(2*step))
    derivative = np.array(derivatives)[:,r>0]
    return duration*(derivative/r[r>0]) @ derivative.T


def reference_design(expected_n: int, fisher_unit: np.ndarray, seed: int, power: int = 14):
    """Full-support prior/Student-t mixture; use exact proposal-density weights.

    Outside-prior points have zero target mass. Filtering them changes only a
    common normalization, which cancels in posterior ratios. No posterior domain
    is restricted. The injection is used ONLY to make independent checks cheap.
    """
    covariance = np.linalg.inv(fisher_unit)[np.ix_(SHAPE,SHAPE)]
    eigen, vectors = np.linalg.eigh(4*covariance)
    covariance = (vectors*np.clip(eigen, 1e-6, .25)) @ vectors.T
    center = ((TRUTH-LOW)/WIDTH)[SHAPE]
    proposal = multivariate_t(loc=center, shape=covariance, df=5)
    u = qmc.Sobol(4, scramble=True, seed=seed+1000).random_base2(power)[:3*2**(power-2)]
    normal = ndtri(u[:, :3]) @ np.linalg.cholesky(covariance).T
    local = center + normal/np.sqrt(chi2.ppf(u[:, 3], df=5)/5)[:, None]
    unit = np.vstack([qmc.Sobol(3, scramble=True, seed=seed).random_base2(power-2), local])
    inside = np.all((unit >= 0)&(unit <= 1), axis=1)
    unit = unit[inside]
    logq = np.logaddexp(np.log(.25), np.log(.75)+proposal.logpdf(unit))
    return LOW[SHAPE]+unit*WIDTH[SHAPE], logq, covariance


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    out = HERE/'results'
    out.mkdir(exist_ok=False)
    start = time.perf_counter()
    manifest = {'truth':TRUTH.tolist(), 'expected_events':[30,100,300,1000],
        'catalogues_per_exposure':10,'catalogue_seed':20261110,
        'training_seed':17,'budgets':[226,354,482,610,738],
        'strategies':['bo','sobol'],'reference_points_generated_per_scramble':16384,
        'reference_sampling':'scrambled Sobol with inverse-normal and chi-square transforms',
        'reference_proposal':'0.25 uniform-prior + 0.75 Student-t(df5), injection/Fisher centered ONLY for validation',
        'population':'blocks0-7, same for generation and inference','measurement':'perfect binned detections',
        'scope':'Post-processing parameter acquisition, not new binary-evolution simulations; offline reference cost is separate',
        'source_sha256':{str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest()
         for p in [Path(__file__),HERE.parent/'joint_pilot/bo_uncertainty_curve.py',HERE.parent/'joint_pilot/check_ensemble_quadrature.py',
                   HERE.parent/'population_stability/bootstrap_blocks.py',*sorted((REPO/'src/cosmic_integration').rglob('*.py'))]}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    ci = BinnedCosmicIntegrator.from_compas_fpath(str(REPO/'tests/large_test_data/h5out_32M_reduced.h5'))
    mass,z,_=configurations()['mass10'];fast=FixedPopulationRates(ci,mass,z)
    block=HERE.parent/'population_stability/block_results'
    meta=json.loads((block/'population_meta.json').read_text())
    ci.compas.nSystems=256000000;ci.compas.zamsZmin=meta['zmin'];ci.compas.zamsZmax=meta['zmax']
    fast.p_draw=1/np.log(meta['zmax']/meta['zmin'])
    fast.operator=sum((load_npz(block/f'block_{i:02d}.npz') for i in range(1,8)),load_npz(block/'block_00.npz'))
    rate=lambda t:fast(*t)
    r0=rate(TRUTH);fisher=fisher_matrix(rate,TRUTH,WIDTH,1.)
    half=fisher_matrix(rate,TRUTH,WIDTH,1.,step=5e-5)
    doubled=TRUTH.copy();doubled[2]*=2
    probes=np.random.default_rng(20261109).uniform(LOW,LOW+WIDTH,(64,4))
    gaps=[poisson_expected_gap(r0,rate(t)) for t in probes]
    physics={'rate_per_year':float(r0.sum()),'fisher_unit_per_year':fisher.tolist(),
        'fisher_eigenvalues':np.linalg.eigvalsh(fisher).tolist(),
        'fisher_step_relative_norm_change':float(np.linalg.norm(fisher-half)/np.linalg.norm(fisher)),
        'amplitude_scaling_max_relative_error':float(np.max(abs(rate(doubled)-2*r0))/np.max(2*r0)),
        'expected_gap_at_truth':poisson_expected_gap(r0,r0),'max_expected_gap_random_probes':max(gaps)}
    np.savez(out/'physics_maps.npz',truth_rate=r0,mass_edges=mass,z_edges=z*.1)
    rng=np.random.default_rng(manifest['catalogue_seed']);catalogues=[]
    for n in manifest['expected_events']:
        duration=n/r0.sum()
        for case in range(10):
            catalogues.append({'expected_events':n,'case':case,'duration':duration,'counts':rng.poisson(duration*r0).ravel()})
    counts=np.array([c['counts'] for c in catalogues]);occupied=np.any(counts>0,axis=0)
    np.savez(out/'catalogues.npz',counts=counts,expected_events=[c['expected_events'] for c in catalogues],
             durations=[c['duration'] for c in catalogues],truth=TRUTH)
    def cache(points):
        result=[]
        for t in points:
            r=fast(t[0],t[1],.012,t[2]).ravel();result.append(np.r_[r[occupied],r.sum()])
        return np.array(result)
    lo,width=LOW[SHAPE],WIDTH[SHAPE]
    initial=lo+qmc.Sobol(3,scramble=True,seed=17).random_base2(7)*width
    unit=np.array(np.meshgrid(*([np.linspace(0,1,5)]*3),indexing='ij')).reshape(3,-1).T
    initial=np.vstack([initial,lo+unit[np.any((unit==0)|(unit==1),axis=1)]*width])
    candidates=lo+qmc.Sobol(3,scramble=True,seed=117).random_base2(12)*width
    ir,cr=cache(initial),cache(candidates)
    records=[];references=[]
    for n in manifest['expected_events']:
        duration=n/r0.sum();fm=fisher*duration
        physics[str(n)]={'duration':duration,'fisher_sd':(WIDTH*np.sqrt(np.diag(np.linalg.inv(fm)))).tolist()}
        reference=[]
        for scramble in [0,1]:
            seed=20261120+2*manifest['expected_events'].index(n)+scramble
            phi,logq,cov=reference_design(n,fm,seed)
            rr=cache(phi)
            np.savez(out/f'reference_{n}_{scramble}.npz',phi=phi,logq=logq,rates=rr,proposal_covariance=cov)
            reference.append((phi,logq,rr))
            logging.info('N=%d reference %d: %d supported points',n,scramble,len(phi))
        for catalogue in [c for c in catalogues if c['expected_events']==n]:
            c=catalogue['counts'][occupied];mask=c>0;select=np.r_[np.flatnonzero(mask),len(c)];cnt=c[mask]
            # Existing helper assumes 0.1yr: scaling rates gives the exact target
            # for this exposure, including its conditional amplitude rate.
            factor=duration/.1
            initial_rates=ir[:,select]*factor;candidate_rates=cr[:,select]*factor
            validation=[(p,q,r[:,select]*factor) for p,q,r in reference]
            direct=[]
            for p,q,r in validation:
                exact=vector_marginal(r,cnt)[0]
                direct.append((exact,direct_summary(r,cnt,p,TRUTH,log_target=exact-q)))
            refcheck=compare(direct[0][1],direct[1][1])
            refpass=bool(refcheck['max_mean_shift_sd']<.05 and refcheck['max_endpoint_shift_sd']<.05
                         and np.max(abs(np.array(refcheck['width68_ratio'])-1))<.03)
            references.append({'expected_events':n,'case':catalogue['case'],'events':int(cnt.sum()),
                               'summaries':[d[1] for d in direct],'numerical_check':refcheck,'numerical_pass':refpass})
            for strategy in manifest['strategies']:
                x=initial.copy();y=vector_marginal(initial_rates,cnt)[0]
                available=np.ones(len(candidates),bool);acq_rng=np.random.default_rng(17)
                folder=out/f'n{n}_case{catalogue["case"]}_{strategy}';folder.mkdir()
                tick=time.perf_counter()
                for batch in range(5):
                    try:
                        if not np.isfinite(y).all():raise ValueError('Nonfinite labels')
                        scaler=AdaptiveRobustScaler(compression='sqrt',soft_clipping=False,lower_clip_value=float(y.min()),max_scale=1e9)
                        scaler.initialize_with_data(y)
                        model=fit_exact_gp(JaxTrainingData(x,-scaler.transform(y)[:,None]),np.array([lo,lo+width]),
                                           config=JaxGPConfig(optimisation_steps=250),seed=17)
                        summaries=[];metrics=[]
                        for (phi,logq,r),(exact,ds) in zip(validation,direct):
                            pred=scaler.inverse_transform(-model.predict_f(phi)[0][:,0])
                            gs=direct_summary(r,cnt,phi,TRUTH,log_target=pred-logq)
                            metric={**compare(ds,gs),**{k:v for k,v in posterior_metrics(exact-logq,pred-logq,phi).items() if k.startswith('kl_')}}
                            summaries.append(gs);metrics.append(metric)
                        gpcheck=compare(summaries[0],summaries[1])
                        numerical=bool(refpass and gpcheck['max_mean_shift_sd']<.05 and gpcheck['max_endpoint_shift_sd']<.05
                                       and np.max(abs(np.array(gpcheck['width68_ratio'])-1))<.03)
                        passes=all(m['max_mean_shift_sd']<.2 and m['max_endpoint_shift_sd']<.2
                                   and np.max(abs(np.array(m['width68_ratio'])-1))<.1
                                   and np.max(abs(np.array(m['width95_ratio'])-1))<.1
                                   and m['kl_direct_to_gp']<.01 and m['kl_gp_to_direct']<.01 for m in metrics)
                        records.append({'expected_events':n,'events':int(cnt.sum()),'case':catalogue['case'],'strategy':strategy,
                            'points':len(x),'status':'evaluated','summaries':summaries,'metrics':metrics,
                            'numerical_check':gpcheck,'numerical_pass':numerical,'accuracy_pass':bool(passes),
                            'validated_pass':bool(numerical and passes),'elapsed_fit_acquire_score_seconds':time.perf_counter()-tick})
                        np.savez(folder/f'checkpoint_{len(x)}.npz',x=x,lnl=y)
                        if batch==4:
                            with (folder/'model.pkl').open('wb') as stream:pickle.dump(model,stream)
                            scaler.save(str(folder))
                            break
                        if strategy=='sobol':chosen=np.arange(batch*128,(batch+1)*128)
                        else:
                            mu,var=model.predict_f(candidates)
                            first=sqrt_log_expected_likelihood(mu[:,0],var[:,0],scaler)
                            second=sqrt_log_expected_likelihood(mu[:,0],var[:,0],scaler,power=2.)
                            with np.errstate(divide='ignore'):score=second+np.log(-np.expm1(np.minimum(2*first-second,0.)))
                            idx=np.flatnonzero(available);s=score[idx]
                            prob=.9*np.exp(s-logsumexp(s))+.1/len(idx) if np.isfinite(s).any() else np.full(len(idx),1/len(idx))
                            chosen=acq_rng.choice(idx,128,replace=False,p=prob)
                        available[chosen]=False
                        x=np.vstack([x,candidates[chosen]])
                        y=np.r_[y,vector_marginal(candidate_rates[chosen],cnt)[0]]
                    except (ValueError,FloatingPointError) as exc:
                        records.append({'expected_events':n,'case':catalogue['case'],'strategy':strategy,'points':len(x),
                                        'status':'failed','reason':str(exc),'validated_pass':False})
                        break
                logging.info('N=%d case=%d %s final accuracy=%s numerical=%s',n,catalogue['case'],strategy,
                             records[-1].get('accuracy_pass'),records[-1].get('numerical_pass'))
                (out/'records.json').write_text(json.dumps(records,indent=2)+'\n')
                (out/'references.json').write_text(json.dumps(references,indent=2)+'\n')
        (out/'physics.json').write_text(json.dumps(physics,indent=2)+'\n')
    (out/'runtime.json').write_text(json.dumps({'seconds':time.perf_counter()-start},indent=2)+'\n')
    logging.info('Complete in %.1fs',time.perf_counter()-start)

if __name__=='__main__':main()
