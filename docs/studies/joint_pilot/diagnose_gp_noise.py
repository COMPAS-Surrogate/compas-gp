"""Same-data ablation of a noise-freezing bug, with direct posterior checks."""
from pathlib import Path
import json
import pickle
import time
import logging
import numpy as np
import paramax
import jax
import jax.numpy as jnp
from run_pilot import HERE,REPO,configurations
from cosmic_integration.lnl_surrogate.jax_active_learner import JaxActiveLearner,JaxGPConfig,JaxTrainingData,fit_exact_gp
from cosmic_integration.lnl_surrogate.adaptive_robust_scalar import AdaptiveRobustScaler
from cosmic_integration.lnl_surrogate.nuts_sampler import make_log_likelihood,run_nuts
from cosmic_integration.ratesSampler import BinnedCosmicIntegrator
from cosmic_integration.ratesSampler.fixed_population import FixedPopulationRates
from cosmic_integration.observation.grid_likelihood import counts_log_likelihood


def main():
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    start=time.perf_counter();out=HERE/'noise_freeze_results';out.mkdir(exist_ok=True)
    base=HERE/'results_seed17';scaler=AdaptiveRobustScaler.load(str(base/'gp_model'))
    old=JaxActiveLearner.load_model(base/'local_refinement/gp_model/models',0)
    x=old.data.query_points;y=old.data.observations
    bounds=np.array([[-.5,.1,.005,4.2],[-.001,.6,.015,5.2]])
    ref=np.load(HERE/'direct_reference_results/posterior_chains.npy').reshape(-1,4)
    rng=np.random.default_rng(20260920)
    check=ref[rng.choice(len(ref),256,replace=False)]
    assert not np.any(np.all(check[:,None,:]==x[None,:,:],axis=2))
    ci=BinnedCosmicIntegrator.from_compas_fpath(str(REPO/'tests/large_test_data/h5out_32M_reduced.h5'))
    mass,z,_=configurations()['mass10'];direct=FixedPopulationRates(ci,mass,z)
    counts=np.load(HERE.parent/'grid_resolution/results/pilot_catalogue.npz')['counts_mass10']
    truth=np.array([counts_log_likelihood(direct(*t),counts,.1) for t in check])
    models={'old_refinement':old}
    for steps in [250,2000]:
        tic=time.perf_counter()
        models[f'fixed_noise_{steps}']=fit_exact_gp(JaxTrainingData(x,y),bounds,
            config=JaxGPConfig(optimisation_steps=steps),seed=17)
        logging.info('Fitted %d steps in %.1fs',steps,time.perf_counter()-tic)
    result={}
    for name,model in models.items():
        target=make_log_likelihood(model,scaler,target='mean')
        prediction=np.asarray(jax.jit(jax.vmap(target))(jnp.asarray(check)))
        residual=prediction-truth;centered=residual-residual.mean()
        trained=np.asarray(jax.jit(jax.vmap(target))(jnp.asarray(x)))
        traintruth=scaler.inverse_transform(-y[:,0])
        p=paramax.unwrap(model.posterior)
        result[name]={'noise_stddev':float(p.likelihood.obs_stddev),'kernel_variance':float(p.prior.kernel.variance),
            'lengthscales_in_prior_widths':(np.asarray(p.prior.kernel.lengthscale)/np.ptp(bounds,axis=0)).tolist(),
            'objective_start_end':model.objective_history[[0,-1]].tolist(),
            'heldout_centered_lnl_rmse':float(np.sqrt(np.mean(centered**2))),
            'heldout_max_centered_error':float(np.max(abs(centered))),
            'training_lnl_rmse':float(np.sqrt(np.mean((trained-traintruth)**2)))}
        np.savez(out/f'{name}_validation.npz',theta=check,direct_lnl=truth,gp_lnl=prediction)
        if name!='old_refinement':
            root=out/name/'gp_model';(root/'models').mkdir(parents=True,exist_ok=True)
            with (root/'models/round_0.pkl').open('wb') as f:pickle.dump(model,f)
            scaler.save(str(root))
            logging.info('Sampling %s',name)
            sample=run_nuts(lnl_model_path=root/'models',outdir=out/name/'sampling',round_idx=0,
                num_warmup=1000,num_samples=2000,num_chains=4,seed=42)
            draws=np.load(out/name/'sampling/posterior_samples.npy')
            q=np.quantile(draws,[.025,.16,.5,.84,.975],axis=0);rq=np.quantile(ref,[.025,.16,.5,.84,.975],axis=0)
            result[name].update({'sampler':sample,'mean_shift_in_direct_sd':((draws.mean(0)-ref.mean(0))/ref.std(0)).tolist(),
                'width68_ratio':((q[3]-q[1])/(rq[3]-rq[1])).tolist(),'width95_ratio':((q[4]-q[0])/(rq[4]-rq[0])).tolist()})
        logging.info('%s: %s',name,result[name])
    result['seconds']=time.perf_counter()-start
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
