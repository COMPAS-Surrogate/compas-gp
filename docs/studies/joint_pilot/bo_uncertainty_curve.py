"""Matched mean versus expected-likelihood KL curves on three simulated catalogues.

Frozen mass10 grid; 128 Sobol + 98 boundary initial points. Each acquisition batch
adds 64 expected-improvement and 64 variance-weighted candidates, each mixed with
10% uniform exploration. All points retained; no oracle acquisition. This is
batched BO/active learning, not the previous 512+128 training protocol.
"""
import argparse,json,sys,time,logging,hashlib
import numpy as np
from scipy.special import log_ndtr,logsumexp,ndtr
from scipy.stats import qmc
from scipy.sparse import load_npz
from run_pilot import HERE,REPO,configurations
from cosmic_integration.ratesSampler import BinnedCosmicIntegrator
from cosmic_integration.ratesSampler.fixed_population import FixedPopulationRates
from cosmic_integration.lnl_surrogate.adaptive_robust_scalar import AdaptiveRobustScaler
from cosmic_integration.lnl_surrogate.jax_active_learner import JaxGPConfig,JaxTrainingData,fit_exact_gp
sys.path.insert(0,str(HERE.parent/'population_stability'))
from bootstrap_blocks import vector_marginal
from compare_populations import weighted_quantile


def sqrt_log_expected_likelihood(mean,variance,scaler,*,power=1.):
    """Exact log E[exp(power*(reference-max(scale*F,0)^2))] for Gaussian F.

    This normalizes the expected unnormalized likelihood, not an ensemble of
    individually normalized random posteriors. Latent variance only, no noise.
    """
    if scaler.compression!='sqrt' or scaler.soft_clipping:
        raise ValueError('Requires sqrt compression with no soft clipping')
    if power<=0:raise ValueError('power must be positive')
    m=np.asarray(mean,dtype=float)*scaler.scale
    v=np.asarray(variance,dtype=float)*scaler.scale**2
    if np.any(v<0) or not np.isfinite(v).all():raise ValueError('Invalid variance')
    safe=np.maximum(v,1e-300);s=np.sqrt(safe);den=1+2*power*safe
    value=np.logaddexp(log_ndtr(-m/s),-.5*np.log(den)-power*m*m/den+log_ndtr(m/(s*np.sqrt(den))))
    value=np.where(v==0,-power*np.maximum(m,0)**2,value)
    return power*(scaler.reference_value+scaler.median)+value


def posterior_metrics(exact,approx,phi):
    """Common full-prior quadrature; KL in nats and shape-parameter constraints."""
    lp=exact-logsumexp(exact);lq=approx-logsumexp(approx)
    p=np.exp(lp);q=np.exp(lq)
    if not np.isfinite(lp).all() or not np.isfinite(lq).all():raise ValueError('Nonfinite KL target')
    pm=p@phi;qm=q@phi;sd=np.sqrt(p@((phi-pm)**2))
    pq=np.array([weighted_quantile(phi[:,i],p,[.16,.84]) for i in range(3)])
    qq=np.array([weighted_quantile(phi[:,i],q,[.16,.84]) for i in range(3)])
    return {'kl_direct_to_gp':float(p@(lp-lq)),'kl_gp_to_direct':float(q@(lq-lp)),
            'direct_ess':float(1/(p@p)),'gp_ess':float(1/(q@q)),
            'max_shape_mean_shift_sd':float(np.max(abs(qm-pm)/sd)),
            'shape_width68_ratio':((qq[:,1]-qq[:,0])/(pq[:,1]-pq[:,0])).tolist()}


def main():
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strategy',choices=['bo','sobol','posterior_variance'],default='bo')
    args=parser.parse_args()
    start=time.perf_counter();out=HERE/f'{args.strategy}_uncertainty_results';out.mkdir(exist_ok=True)
    data=np.load(HERE/'reduced_calibration_100_results/catalogues.npz')
    counts=data['counts'][:3];truths=data['truths'][:3]
    ci=BinnedCosmicIntegrator.from_compas_fpath(str(REPO/'tests/large_test_data/h5out_32M_reduced.h5'))
    mass,z,_=configurations()['mass10'];fast=FixedPopulationRates(ci,mass,z)
    meta=json.loads((HERE.parent/'population_stability/block_results/population_meta.json').read_text())
    ci.compas.nSystems=256000000;ci.compas.zamsZmin=meta['zmin'];ci.compas.zamsZmax=meta['zmax'];fast.p_draw=1/np.log(meta['zmax']/meta['zmin'])
    fast.operator=sum((load_npz(HERE.parent/f'population_stability/block_results/block_{i:02d}.npz') for i in range(9,16)),load_npz(HERE.parent/'population_stability/block_results/block_08.npz'))
    lo=np.array([-.5,.1,4.2]);width=np.array([.499,.5,1.])
    unit=np.array(np.meshgrid(*([np.linspace(0,1,5)]*3),indexing='ij')).reshape(3,-1).T
    anchors=lo+unit[np.any((unit==0)|(unit==1),axis=1)]*width
    validation=[lo+qmc.Sobol(3,scramble=True,seed=s).random_base2(13)*width for s in [20261010,20261011]]
    designs={seed:(np.vstack([lo+qmc.Sobol(3,scramble=True,seed=seed).random_base2(7)*width,anchors]),
                   lo+qmc.Sobol(3,scramble=True,seed=seed+100).random_base2(12)*width) for seed in [17,29,43]}
    occupied=np.any(counts>0,axis=0);obs=counts[:,occupied]
    def rates(points):
        result=[]
        for t in points:
            r=fast(t[0],t[1],.012,t[2]).ravel();result.append(np.r_[r[occupied],r.sum()])
        return np.array(result)
    vr=[rates(x) for x in validation]
    cached={}
    for seed,(initial,candidates) in designs.items():
        cached[seed]=(rates(initial),rates(candidates))
        logging.info('Rate cache prepared for seed %d',seed)
    cache_seconds=time.perf_counter()-start;records=[]
    for case,c in enumerate(obs):
        mask=c>0;selected=np.r_[np.flatnonzero(mask),len(c)];cnt=c[mask]
        direct=[vector_marginal(r[:,selected],cnt)[0] for r in vr]
        for seed,(initial,candidates) in designs.items():
            tick=time.perf_counter();rng=np.random.default_rng(seed);x=initial.copy()
            y=vector_marginal(cached[seed][0][:,selected],cnt)[0]
            candidate_y=vector_marginal(cached[seed][1][:,selected],cnt)[0]
            available=np.ones(len(candidates),dtype=bool);acquired=[]
            for batch in range(5):
                scaler=AdaptiveRobustScaler(compression='sqrt',soft_clipping=False,lower_clip_value=float(y.min()),max_scale=1e9)
                scaler.initialize_with_data(y)
                model=fit_exact_gp(JaxTrainingData(x,-scaler.transform(y)[:,None]),np.array([lo,lo+width]),config=JaxGPConfig(optimisation_steps=250),seed=seed)
                for scramble,(phi,exact) in enumerate(zip(validation,direct)):
                    mu,var=model.predict_f(phi);mu=mu[:,0];var=var[:,0]
                    mean=scaler.inverse_transform(-mu);unc=sqrt_log_expected_likelihood(mu,var,scaler)
                    # Calibration is checked in the space where GP Gaussianity is assumed.
                    truth_f=-scaler.transform(exact);std=np.sqrt(var)
                    w=np.exp(exact-logsumexp(exact))
                    for target,pred in [('mean',mean),('expected_likelihood',unc)]:
                        row={'case':case,'events':int(cnt.sum()),'seed':seed,'points':len(x),'scramble':scramble,'target':target,
                             **posterior_metrics(exact,pred,phi),
                             'latent_1sd_coverage_direct_weighted':float(w@(abs(truth_f-mu)<=std)),
                             'direct_mass_above_training_reference':float(w@(exact>scaler.reference_value)),
                             'fit_acquisition_scoring_seconds':time.perf_counter()-tick}
                        records.append(row)
                folder=out/f'case_{case}_seed_{seed}';folder.mkdir(exist_ok=True)
                np.savez(folder/f'checkpoint_{len(x)}.npz',training_x=x,training_lnl=y,acquired_indices=np.array(acquired),
                         reference=scaler.reference_value,scale=scaler.scale)
                (out/'curves.json').write_text(json.dumps(records,indent=2)+'\n')
                logging.info('Case %d seed %d: %d points scored',case,seed,len(x))
                if batch==4:break
                if args.strategy=='sobol':
                    # Nested prefixes of the same scrambled Sobol candidate pool;
                    # independent of labels and posterior diagnostics.
                    chosen=list(range(batch*128,(batch+1)*128))
                    acquired.extend(chosen);x=np.vstack([x,candidates[chosen]]);y=np.r_[y,candidate_y[chosen]]
                    continue
                mu,var=model.predict_f(candidates)
                if args.strategy=='posterior_variance':
                    first=sqrt_log_expected_likelihood(mu[:,0],var[:,0],scaler)
                    second=sqrt_log_expected_likelihood(mu[:,0],var[:,0],scaler,power=2.)
                    # Var(L) = E[L^2] - E[L]^2; constants cancel in sampling.
                    delta=np.minimum(2*first-second,0.)
                    with np.errstate(divide='ignore'):
                        logvar=second+np.log(-np.expm1(delta))
                    idx=np.flatnonzero(available);scores=logvar[idx]
                    prob=.9*np.exp(scores-logsumexp(scores))+.1/len(idx) if np.isfinite(scores).any() else np.full(len(idx),1/len(idx))
                    chosen=rng.choice(idx,128,replace=False,p=prob).tolist();available[chosen]=False
                    acquired.extend(chosen);x=np.vstack([x,candidates[chosen]]);y=np.r_[y,candidate_y[chosen]]
                    continue
                mu=mu[:,0];std=np.sqrt(var[:,0]);best=float(np.min(-scaler.transform(y)))
                improvement=best-mu;zscore=improvement/np.maximum(std,1e-15)
                ei=np.maximum(improvement*ndtr(zscore)+std*np.exp(-.5*zscore*zscore)/np.sqrt(2*np.pi),0)
                chosen=[]
                for score in [ei,var[:,0]]:
                    idx=np.flatnonzero(available);s=np.maximum(score[idx],0)
                    probability=.9*s/s.sum()+.1/len(idx) if s.sum()>0 else np.full(len(idx),1/len(idx))
                    take=rng.choice(idx,64,replace=False,p=probability);available[take]=False;chosen.extend(take.tolist())
                acquired.extend(chosen);x=np.vstack([x,candidates[chosen]]);y=np.r_[y,candidate_y[chosen]]
    summary={'strategy':args.strategy,'catalogues':3,'cases':[0,1,2],'training_seeds':[17,29,43],'checkpoints':[226,354,482,610,738],
        'seconds':time.perf_counter()-start,'cache_seconds':cache_seconds,
        'source_sha256':hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest(),
        'scope':'Paired targets at identical training data. Initial 128 Sobol plus 98 anchors; four 128-point acquisition batches; strategy field distinguishes EI/variance BO, fixed Sobol prefixes, and likelihood-variance acquisition. Latent GP uncertainty through exact sqrt inverse. Three shape parameters; full-joint KL equal only when both use same exact conditional amplitude. Quadrature KL, not an exact integral. Cached candidate labels never enter acquisition; cache cost is not per-trajectory direct-call cost.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');logging.info('Complete: %.1f seconds',summary['seconds'])

if __name__=='__main__':main()
