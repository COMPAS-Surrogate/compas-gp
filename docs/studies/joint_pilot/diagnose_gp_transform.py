"""Matched transform reference test after fixing noise; no added training points."""
import json
import pickle
import time
import logging
import numpy as np
import jax
import jax.numpy as jnp
from scipy.special import xlogy
from run_pilot import HERE,REPO,configurations
from cosmic_integration.lnl_surrogate.jax_active_learner import JaxActiveLearner,JaxGPConfig,JaxTrainingData,fit_exact_gp
from cosmic_integration.lnl_surrogate.adaptive_robust_scalar import AdaptiveRobustScaler
from cosmic_integration.lnl_surrogate.nuts_sampler import make_log_likelihood,run_nuts
from cosmic_integration.ratesSampler import BinnedCosmicIntegrator
from cosmic_integration.ratesSampler.fixed_population import FixedPopulationRates
from cosmic_integration.observation.grid_likelihood import counts_log_likelihood


def main():
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    start=time.perf_counter();out=HERE/'noise_freeze_results/upper_reference';out.mkdir(exist_ok=True)
    base=HERE/'results_seed17';old=JaxActiveLearner.load_model(base/'local_refinement/gp_model/models',0)
    x=old.data.query_points
    ci=BinnedCosmicIntegrator.from_compas_fpath(str(REPO/'tests/large_test_data/h5out_32M_reduced.h5'))
    mass,z,_=configurations()['mass10'];direct=FixedPopulationRates(ci,mass,z)
    counts=np.load(HERE.parent/'grid_resolution/results/pilot_catalogue.npz')['counts_mass10']
    raw=np.array([counts_log_likelihood(direct(*t),counts,.1) for t in x])
    scaler=AdaptiveRobustScaler.load(str(base/'gp_model'))
    original_reference=scaler.reference_value
    # The independent-bin saturated Poisson fit supplies a rigorous upper bound
    # on this specific ideal-count likelihood, with the same omitted constants.
    upper=float(np.sum(xlogy(counts,counts)-counts))
    scaler.reference_value=upper
    y=-scaler.transform(raw)[:,None]
    np.testing.assert_allclose(scaler.inverse_transform(-y[:,0]),np.maximum(raw,scaler.lower_clip_value),atol=1e-10)
    bounds=np.array([[-.5,.1,.005,4.2],[-.001,.6,.015,5.2]])
    model=fit_exact_gp(JaxTrainingData(x,y),bounds,config=JaxGPConfig(optimisation_steps=2000),seed=17)
    root=out/'gp_model';(root/'models').mkdir(parents=True,exist_ok=True)
    with (root/'models/round_0.pkl').open('wb') as f:pickle.dump(model,f)
    scaler.save(str(root))
    held=np.load(out.parent/'old_refinement_validation.npz');check=held['theta'];truth=held['direct_lnl']
    target=make_log_likelihood(model,scaler,target='mean')
    prediction=np.asarray(jax.jit(jax.vmap(target))(jnp.asarray(check)))
    err=prediction-truth;err-=err.mean()
    logging.info('Upper-reference held-out RMS %.5f',np.std(err))
    sampler=run_nuts(lnl_model_path=root/'models',outdir=out/'sampling',round_idx=0,num_warmup=1000,num_samples=2000,num_chains=4,seed=42)
    ref=np.load(HERE/'direct_reference_results/posterior_chains.npy').reshape(-1,4)
    draws=np.load(out/'sampling/posterior_samples.npy')
    q=np.quantile(draws,[.025,.16,.5,.84,.975],axis=0);rq=np.quantile(ref,[.025,.16,.5,.84,.975],axis=0)
    result={'original_reference':original_reference,'saturated_upper_bound':upper,'previous_heldout_above_reference':int(np.sum(truth>original_reference)),
        'heldout_centered_lnl_rmse':float(np.std(err)),'sampler':sampler,
        'mean_shift_in_direct_sd':((draws.mean(0)-ref.mean(0))/ref.std(0)).tolist(),
        'width68_ratio':((q[3]-q[1])/(rq[3]-rq[1])).tolist(),'width95_ratio':((q[4]-q[0])/(rq[4]-rq[0])).tolist(),
        'max_endpoint_shift_in_direct_sd':np.max(abs((q-rq)/ref.std(0)),axis=0).tolist(),'seconds':time.perf_counter()-start}
    np.savez(out/'validation.npz',theta=check,direct_lnl=truth,gp_lnl=prediction)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    logging.info('%s',result)

if __name__=='__main__':main()
