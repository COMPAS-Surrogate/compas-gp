"""Matched reduced-target noise repair; four-chain and fresh-location validation."""
from pathlib import Path
import json
import time
import pickle
import logging
import hashlib
import numpy as np
import jax
import jax.numpy as jnp
import paramax
from scipy.stats import gamma
from numpyro.infer import MCMC,NUTS
from run_pilot import HERE,REPO,configurations
from cosmic_integration.lnl_surrogate.jax_active_learner import JaxTrainingData,JaxGPConfig,fit_exact_gp
from cosmic_integration.lnl_surrogate.adaptive_robust_scalar import AdaptiveRobustScaler
from cosmic_integration.lnl_surrogate.nuts_sampler import make_log_likelihood,_numpyro_model,_unconstrained_starts,chain_diagnostics
from cosmic_integration.ratesSampler import BinnedCosmicIntegrator
from cosmic_integration.ratesSampler.fixed_population import FixedPopulationRates
from cosmic_integration.observation.grid_likelihood import amplitude_marginal_log_likelihood


def main():
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    start=time.perf_counter();out=HERE/'repaired_reduced_results';out.mkdir(exist_ok=True)
    data=np.load(HERE/'reduced_gp_results/fit_and_validation.npz')
    x,raw=data['training_x'],data['training_lnl']
    bounds=np.array([[-.5,.1,4.2],[-.001,.6,5.2]])
    scaler=AdaptiveRobustScaler(compression='sqrt',soft_clipping=False,lower_clip_value=float(raw.min()),max_scale=1e9)
    scaler.initialize_with_data(raw)
    ci=BinnedCosmicIntegrator.from_compas_fpath(str(REPO/'tests/large_test_data/h5out_32M_reduced.h5'))
    mass,z,_=configurations()['mass10'];fast=FixedPopulationRates(ci,mass,z)
    counts=np.load(HERE.parent/'grid_resolution/results/pilot_catalogue.npz')['counts_mass10']
    def direct(t):return amplitude_marginal_log_likelihood(fast(t[0],t[1],.012,t[2]),counts,.1)
    ref=np.load(HERE/'direct_reference_results/posterior_chains.npy').reshape(-1,4)
    rng=np.random.default_rng(20260923)
    excluded={tuple(t) for t in np.vstack([x,data['test_x']])}
    fresh=[]
    for t in ref[rng.permutation(len(ref))][:,[0,1,3]]:
        if tuple(t) not in excluded:
            fresh.append(t);excluded.add(tuple(t))
            if len(fresh)==256:break
    fresh=np.array(fresh)
    checks=np.vstack([data['test_x'],fresh])
    truths=np.array([direct(t)[0] for t in checks])
    results={}
    for steps in [250,2000]:
        folder=out/f'steps_{steps}';folder.mkdir(exist_ok=True)
        model=fit_exact_gp(JaxTrainingData(x,-scaler.transform(raw)[:,None]),bounds,config=JaxGPConfig(optimisation_steps=steps),seed=17)
        with (folder/'model.pkl').open('wb') as f:pickle.dump(model,f)
        scaler.save(str(folder))
        target=make_log_likelihood(model,scaler,target='mean')
        prediction=np.asarray(jax.jit(jax.vmap(target))(jnp.asarray(checks)))
        summary={}
        for name,sl in [('matched',slice(0,256)),('fresh_direct',slice(256,512))]:
            residual=prediction[sl]-truths[sl];residual-=residual.mean()
            summary[name]={'rmse':float(np.std(residual)),'max_error':float(np.max(abs(residual)))}
        logging.info('Steps %d likelihood: %s',steps,summary)
        # Generic three-dimensional model; production run_nuts currently fixes 4D bounds.
        mcmc=MCMC(NUTS(_numpyro_model,target_accept_prob=.9),num_warmup=1000,num_samples=2000,num_chains=4,chain_method='sequential',progress_bar=False)
        low,high=jnp.asarray(bounds[0]),jnp.asarray(bounds[1])
        starts=np.random.default_rng(42).uniform(bounds[0]+.1*np.ptp(bounds,axis=0),bounds[1]-.1*np.ptp(bounds,axis=0),(4,3))
        mcmc.run(jax.random.key(42),target,low,high,init_params=_unconstrained_starts(target,low,high,starts))
        shape_chains=np.asarray(mcmc.get_samples(group_by_chain=True)['theta'])
        phi=shape_chains.reshape(-1,3)
        beta=np.array([direct(t)[1] for t in phi])
        shape=counts.sum()+1
        amp_rng=np.random.default_rng(43)
        lo=gamma.cdf(.005,shape,scale=1/beta);hi=gamma.cdf(.015,shape,scale=1/beta)
        a=gamma.ppf(lo+amp_rng.random(len(phi))*(hi-lo),shape,scale=1/beta)
        draws=np.column_stack([phi[:,:2],a,phi[:,2]])
        assert np.isfinite(draws).all()
        chains=draws.reshape(4,2000,4);np.save(folder/'posterior_chains.npy',chains)
        rhat,bulk,tail=chain_diagnostics(chains)
        div=int(np.sum(np.asarray(mcmc.get_extra_fields()['diverging'])))
        q=np.quantile(draws,[.025,.16,.5,.84,.975],axis=0);rq=np.quantile(ref,[.025,.16,.5,.84,.975],axis=0)
        summary.update({'rhat':rhat.tolist(),'bulk_ess':bulk.tolist(),'tail_ess':tail.tolist(),'divergences':div,
            'sampler_pass':bool(np.all(rhat<1.01) and np.all(bulk>400) and np.all(tail>400) and div==0),
            'mean_shift_sd':((draws.mean(0)-ref.mean(0))/ref.std(0)).tolist(),
            'width68_ratio':((q[3]-q[1])/(rq[3]-rq[1])).tolist(),'width95_ratio':((q[4]-q[0])/(rq[4]-rq[0])).tolist(),
            'max_endpoint_shift_sd':np.max(abs(q-rq)/ref.std(0),axis=0).tolist(),
            'max_correlation_difference':float(np.max(abs(np.corrcoef(draws.T)-np.corrcoef(ref.T)))),
            'actual_noise_stddev':float(paramax.unwrap(model.posterior).likelihood.obs_stddev)})
        gpcheck=phi[np.random.default_rng(44).choice(len(phi),256,replace=False)]
        assert not any(tuple(t) in {tuple(v) for v in x} for t in gpcheck)
        gp_truth=np.array([direct(t)[0] for t in gpcheck])
        gp_pred=np.asarray(jax.jit(jax.vmap(target))(jnp.asarray(gpcheck)))
        residual=gp_pred-gp_truth;residual-=residual.mean()
        summary['fresh_gp']={'rmse':float(np.std(residual)),'max_error':float(np.max(abs(residual)))}
        np.savez(folder/'validation.npz',theta=checks,direct_lnl=truths,gp_lnl=prediction,gp_theta=gpcheck,gp_direct=gp_truth,gp_prediction=gp_pred)
        results[str(steps)]=summary
        (folder/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        logging.info('Steps %d complete %s',steps,summary)
    sources=sorted((REPO/'src/cosmic_integration').rglob('*.py'))+[Path(__file__)]
    results['metadata']={'seconds':time.perf_counter()-start,'training_points':len(x),
        'source_sha256':hashlib.sha256(b''.join(p.read_bytes() for p in sources)).hexdigest(),
        'training_file_sha256':hashlib.sha256((HERE/'reduced_gp_results/fit_and_validation.npz').read_bytes()).hexdigest(),
        'scope':'Matched repair test with reference-informed legacy training. Exact direct rates used for conditional amplitude draws; not an end-to-end surrogate speed claim.'}
    (out/'summary.json').write_text(json.dumps(results,indent=2)+'\n')

if __name__=='__main__':main()
