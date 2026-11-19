"""Bounded next GP hypothesis: learn the amplitude-marginalized 3D target.

Keeps every previous training location, projected onto the three shape
parameters. No prior restriction. Independent reference draws score the fit.
"""
from pathlib import Path
import json
import hashlib
import time
import numpy as np
from scipy.optimize import differential_evolution
from run_pilot import HERE, REPO, configurations
from cosmic_integration.ratesSampler import BinnedCosmicIntegrator
from cosmic_integration.ratesSampler.fixed_population import FixedPopulationRates
from cosmic_integration.observation.grid_likelihood import amplitude_marginal_log_likelihood
from cosmic_integration.lnl_surrogate.adaptive_robust_scalar import AdaptiveRobustScaler
from cosmic_integration.lnl_surrogate.jax_active_learner import JaxGPConfig, JaxTrainingData, fit_exact_gp
from cosmic_integration.lnl_surrogate.nuts_sampler import make_log_likelihood
import jax
import jax.numpy as jnp


def main():
    started=time.perf_counter()
    out=HERE/'reduced_gp_results';out.mkdir(exist_ok=True)
    counts=np.load(HERE.parent/'grid_resolution/results/pilot_catalogue.npz')['counts_mass10']
    ci=BinnedCosmicIntegrator.from_compas_fpath(str(REPO/'tests/large_test_data/h5out_32M_reduced.h5'))
    mass,z,_=configurations()['mass10']; fast=FixedPopulationRates(ci,mass,z)
    def direct(phi):
        return amplitude_marginal_log_likelihood(fast(phi[0],phi[1],.012,phi[2]),counts,.1)[0]
    x=[]
    for path in [HERE/'results_seed17/direct_evaluations.jsonl',HERE/'results_seed17/local_refinement/direct_evaluations.jsonl']:
        x.extend(np.array(json.loads(line)['theta'])[[0,1,3]] for line in path.read_text().splitlines())
    rng=np.random.default_rng(20260914)
    reference=np.load(HERE/'direct_reference_results/posterior_chains.npy').reshape(-1,4)[:,[0,1,3]]
    selection=rng.permutation(len(reference))
    x=np.unique(np.vstack([x,reference[selection[:128]]]),axis=0)
    bounds=np.array([[-.5,.1,4.2],[-.001,.6,5.2]])
    optimum=differential_evolution(lambda t:-direct(t),list(zip(*bounds)),seed=19,popsize=8,maxiter=30,polish=True)
    x=np.vstack([x,optimum.x])
    raw=np.array([direct(t) for t in x])
    scaler=AdaptiveRobustScaler(compression='sqrt',soft_clipping=False,lower_clip_value=float(raw.min()),max_scale=1e9)
    scaler.initialize_with_data(raw)
    model=fit_exact_gp(JaxTrainingData(x,-scaler.transform(raw)[:,None]),bounds,
        config=JaxGPConfig(kernel='matern52',initial_noise_stddev=np.sqrt(1e-5),optimise_noise=False,optimisation_steps=250),seed=17)
    target=make_log_likelihood(model,scaler,target='mean')
    # Exclude exact duplicates arising from repeated Metropolis states.
    test=reference[selection[128:]]
    test=np.array([t for t in test if not np.any(np.all(x==t,axis=1))])[:256]
    truth=np.array([direct(t) for t in test])
    predicted=np.asarray(jax.jit(jax.vmap(target))(jnp.asarray(test)))
    residual=predicted-truth; centered=residual-residual.mean()
    summary={'training_points':len(x),'heldout_points':len(test),'posterior_centered_lnl_rmse':float(np.sqrt(np.mean(centered**2))),
        'max_centered_error':float(np.max(abs(centered))),'seconds':time.perf_counter()-started,
        'scope':'One exploratory 3D GP fit; reference-informed training, fresh locations; not coverage validation or a sampled GP posterior',
        'hypothesis':'Analytically remove amplitude, retain all 456 old projected locations, add 128 reference draws and a direct optimum; same full prior'}
    np.savez(out/'fit_and_validation.npz',training_x=x,training_lnl=raw,test_x=test,direct_lnl=truth,gp_lnl=predicted)
    sources=sorted((REPO/'src/cosmic_integration').rglob('*.py'))+[Path(__file__)]
    provenance={'source_sha256':hashlib.sha256(b''.join(p.read_bytes() for p in sources)).hexdigest(),
        'direct_reference_manifest':json.loads((HERE/'direct_reference_results/manifest.json').read_text()),
        'seed':20260914,'gp_seed':17,'kernel':'matern52','compression':'sqrt',
        'noise_variance':1e-5,'optimisation_steps':250}
    (out/'manifest.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(summary)

if __name__=='__main__':main()
