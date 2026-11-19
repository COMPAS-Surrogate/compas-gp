"""Four-chain check of the 512M anchored reduced GP, including amplitude."""
import json,pickle,time
import numpy as np
import jax,jax.numpy as jnp
from scipy.stats import gamma
from scipy.sparse import load_npz
from numpyro.infer import MCMC,NUTS
from run_pilot import HERE,REPO,configurations
from cosmic_integration.ratesSampler import BinnedCosmicIntegrator
from cosmic_integration.ratesSampler.fixed_population import FixedPopulationRates
from cosmic_integration.lnl_surrogate.adaptive_robust_scalar import AdaptiveRobustScaler
from cosmic_integration.lnl_surrogate.nuts_sampler import make_log_likelihood,_numpyro_model,_unconstrained_starts,chain_diagnostics


def main():
    start=time.perf_counter();root=HERE/'reduced_seed_results_512M_anchored/seed_17'
    with (root/'model.pkl').open('rb') as f:model=pickle.load(f)
    scaler=AdaptiveRobustScaler.load(str(root));target=make_log_likelihood(model,scaler,target='mean')
    low=jnp.array([-.5,.1,4.2]);high=jnp.array([-.001,.6,5.2])
    starts=np.random.default_rng(45).uniform(np.array(low)+.1*np.array(high-low),np.array(high)-.1*np.array(high-low),(4,3))
    sampler=MCMC(NUTS(_numpyro_model,target_accept_prob=.99),num_warmup=1000,num_samples=2000,num_chains=4,chain_method='sequential',progress_bar=False)
    sampler.run(jax.random.key(45),target,low,high,init_params=_unconstrained_starts(target,low,high,starts))
    phi=np.array(sampler.get_samples(group_by_chain=True)['theta']).reshape(-1,3)
    ci=BinnedCosmicIntegrator.from_compas_fpath(str(REPO/'tests/large_test_data/h5out_32M_reduced.h5'))
    mass,z,_=configurations()['mass10'];fast=FixedPopulationRates(ci,mass,z)
    meta=json.loads((HERE.parent/'population_stability/results/summary.json').read_text())['metadata']
    ci.compas.nSystems=512000000;ci.compas.zamsZmin=meta['zamsZmin'];ci.compas.zamsZmax=meta['zamsZmax']
    fast.p_draw=1/np.log(ci.compas.zamsZmax/ci.compas.zamsZmin)
    fast.operator=load_npz(HERE.parent/'population_stability/results/operator_512M.npz')
    beta=np.array([.1*fast(t[0],t[1],.012,t[2]).sum()/.012 for t in phi])
    count=np.load(HERE.parent/'grid_resolution/results/pilot_catalogue.npz')['counts_mass10'].sum()
    lo=gamma.cdf(.005,count+1,scale=1/beta);hi=gamma.cdf(.015,count+1,scale=1/beta)
    a=gamma.ppf(lo+np.random.default_rng(46).random(len(phi))*(hi-lo),count+1,scale=1/beta)
    draws=np.column_stack([phi[:,:2],a,phi[:,2]]);chains=draws.reshape(4,2000,4)
    rhat,bulk,tail=chain_diagnostics(chains);div=int(np.sum(np.array(sampler.get_extra_fields()['diverging'])))
    data=np.load(root/'validation.npz');ref=data['draws'];w=data['direct_weights'];mean=(w[:,None]*ref).sum(0);sd=np.sqrt((w[:,None]*(ref-mean)**2).sum(0))
    result={'target_accept_prob':.99,'rhat':rhat.tolist(),'bulk_ess':bulk.tolist(),'tail_ess':tail.tolist(),'divergences':div,
        'pass':bool(np.all(rhat<1.01) and np.all(bulk>400) and np.all(tail>400) and div==0),
        'mean_difference_from_direct_in_sd':((draws.mean(0)-mean)/sd).tolist(),
        'seconds':time.perf_counter()-start,'scope':'One seed, 512M target; exact conditional amplitude calculation remains in this validation hybrid.'}
    np.save(root/'posterior_chains.npy',chains)
    (root/'sampler_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result)

if __name__=='__main__':main()
