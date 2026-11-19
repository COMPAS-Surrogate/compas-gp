"""Fresh location checks after the bounded noise-fix/refinement experiments."""
import json
import hashlib
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import paramax
from run_pilot import HERE,REPO,configurations
from cosmic_integration.lnl_surrogate.jax_active_learner import JaxActiveLearner
from cosmic_integration.lnl_surrogate.adaptive_robust_scalar import AdaptiveRobustScaler
from cosmic_integration.lnl_surrogate.nuts_sampler import make_log_likelihood
from cosmic_integration.ratesSampler import BinnedCosmicIntegrator
from cosmic_integration.ratesSampler.fixed_population import FixedPopulationRates
from cosmic_integration.observation.grid_likelihood import counts_log_likelihood


def main():
    out=HERE/'noise_freeze_results';root=out/'added_points/gp_model'
    model=JaxActiveLearner.load_model(root/'models',0)
    scaler=AdaptiveRobustScaler.load(str(root))
    previous=np.load(out/'old_refinement_validation.npz')['theta']
    excluded={tuple(t) for t in np.vstack([model.data.query_points,previous])}
    rng=np.random.default_rng(20260922)
    ref=np.load(HERE/'direct_reference_results/posterior_chains.npy').reshape(-1,4)
    gp=np.load(out/'added_points/sampling/posterior_samples.npy')
    def choose(pool):
        result=[]
        for t in pool[rng.permutation(len(pool))]:
            if tuple(t) not in excluded:
                result.append(t);excluded.add(tuple(t))
                if len(result)==256:break
        return np.array(result)
    test=np.vstack([choose(ref),choose(gp)])
    ci=BinnedCosmicIntegrator.from_compas_fpath(str(REPO/'tests/large_test_data/h5out_32M_reduced.h5'))
    mass,z,_=configurations()['mass10'];direct=FixedPopulationRates(ci,mass,z)
    counts=np.load(HERE.parent/'grid_resolution/results/pilot_catalogue.npz')['counts_mass10']
    truth=np.array([counts_log_likelihood(direct(*t),counts,.1) for t in test])
    target=make_log_likelihood(model,scaler,target='mean')
    pred=np.asarray(jax.jit(jax.vmap(target))(jnp.asarray(test)))
    result={}
    for name,sl in [('direct_posterior',slice(0,256)),('gp_posterior',slice(256,512))]:
        residual=pred[sl]-truth[sl];residual-=residual.mean()
        result[name]={'points':len(residual),'centered_lnl_rmse':float(np.std(residual)),
            'max_centered_error':float(np.max(abs(residual))),'absolute_centered_error_quantiles':np.quantile(abs(residual),[.5,.9,.95,.99]).tolist()}
    result['max_correlation_difference']=float(np.max(abs(np.corrcoef(ref.T)-np.corrcoef(gp.T))))
    result['actual_noise_stddev']=float(paramax.unwrap(model.posterior).likelihood.obs_stddev)
    np.savez(out/'fresh_validation.npz',theta=test,direct_lnl=truth,gp_lnl=pred)
    (out/'fresh_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    sources=sorted((REPO/'src/cosmic_integration').rglob('*.py'))
    inputs=[HERE/'results_seed17/local_refinement/gp_model/models/round_0.pkl',HERE/'results_seed17/gp_model/scaler.json']
    manifest={'source_sha256':hashlib.sha256(b''.join(p.read_bytes() for p in sources)).hexdigest(),
        'fresh_seed':20260922,'training_policy':'All 356 previous points retained; 128 new points from repaired GP posterior; no prior restriction',
        'scope':'One catalogue and training sequence. Matched ablation and fresh location checks, not multi-seed retraining or coverage validation',
        'experiment_scripts':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),HERE/'diagnose_gp_noise.py',HERE/'diagnose_gp_transform.py',HERE/'refine_fixed_noise.py']},
        'old_model_sha256':hashlib.sha256(inputs[0].read_bytes()).hexdigest()}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(result)

if __name__=='__main__':main()
