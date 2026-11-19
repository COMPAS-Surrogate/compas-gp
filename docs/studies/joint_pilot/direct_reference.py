"""Direct reference with exact fixed-population contraction; no GP in target."""
from pathlib import Path
import json
import hashlib
import time
import logging
import numpy as np
from scipy.stats import gamma
from run_pilot import HERE, REPO, configurations
from cosmic_integration.ratesSampler import BinnedCosmicIntegrator
from cosmic_integration.ratesSampler.fixed_population import FixedPopulationRates
from cosmic_integration.ratesSampler.grid_diagnostics import system_grid_moments
from cosmic_integration.observation.grid_likelihood import counts_log_likelihood
from cosmic_integration.lnl_surrogate.nuts_sampler import chain_diagnostics


from cosmic_integration.observation.grid_likelihood import amplitude_marginal_log_likelihood as amplitude_integral


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    out = HERE/'direct_reference_results'
    out.mkdir(exist_ok=True)
    start = time.perf_counter()
    import tomllib
    cfg = tomllib.loads((HERE.parent/'grid_resolution/joint_pilot.toml').read_text())
    counts = np.load(REPO/cfg['data']['catalogue'])['counts_mass10']
    mass, z, _ = configurations()['mass10']
    ci = BinnedCosmicIntegrator.from_compas_fpath(str(REPO/cfg['data']['compas_file']))
    fast = FixedPopulationRates(ci, mass, z)
    rng = np.random.default_rng(20260912)
    bounds = np.array([cfg['prior']['low'], cfg['prior']['high']])
    probes = np.vstack([rng.uniform(*bounds, (20,4)), bounds, [-.325,.213,.012,4.253]])
    errors, lnl_errors, old_times, new_times = [], [], [], []
    for theta in probes:
        tick = time.perf_counter()
        rates, masses = ci.FindDetectionRate(**dict(zip(['p_Alpha','p_Sigma','p_SFRa','p_SFRd'],theta)))
        grid, _ = system_grid_moments(rates,masses,mass,z)
        old_times.append(time.perf_counter()-tick)
        tick = time.perf_counter()
        quick = fast(*theta)
        new_times.append(time.perf_counter()-tick)
        errors.append(np.max(np.abs(quick-grid))/grid.max())
        lnl_errors.append(abs(counts_log_likelihood(quick,counts,.1)-counts_log_likelihood(grid,counts,.1)))
    assert max(errors)<1e-10 and max(lnl_errors)<1e-8
    logging.info('Parity passed: max lnL error %.3g; direct %.4fs -> %.4fs',max(lnl_errors),np.median(old_times),np.median(new_times))
    low, high = bounds[:,[0,1,3]]
    def target(u):
        if np.any(u<=0) or np.any(u>=1): return -np.inf, np.nan
        alpha,sigma,d = low+u*(high-low)
        return amplitude_integral(fast(alpha,sigma,.012,d),counts,.1)
    warm, draws = 3000, 12000
    chains, accepts = [], []
    for chain in range(4):
        u = rng.uniform(.05,.95,3)
        val,beta = target(u)
        covariance = np.eye(3)*.03**2
        history, samples = [], []
        accepted = 0
        for step in range(warm+draws):
            proposal = rng.multivariate_normal(u,covariance)
            new, new_beta = target(proposal)
            if np.log(rng.random())<new-val:
                u,val,beta = proposal,new,new_beta
                accepted += int(step>=warm)
            if step<warm:
                history.append(u.copy())
                if step>=499 and (step+1)%100==0:
                    covariance = (2.38**2/3)*(np.cov(np.asarray(history).T)+np.eye(3)*1e-6)
            else:
                phi = low+u*(high-low)
                a = gamma.ppf(rng.uniform(gamma.cdf(.005,counts.sum()+1,scale=1/beta),gamma.cdf(.015,counts.sum()+1,scale=1/beta)),counts.sum()+1,scale=1/beta)
                samples.append([phi[0],phi[1],a,phi[2]])
            if (step+1)%3000==0: logging.info('Chain %d step %d/%d',chain+1,step+1,warm+draws)
        chains.append(samples)
        accepts.append(accepted/draws)
    # Independent full-prior integration checks modes without chain initialization.
    from scipy.stats import qmc
    from scipy.special import logsumexp
    points = qmc.Sobol(3, scramble=True, seed=20260913).random_base2(13)
    values = np.array([target(u) for u in points])
    weights = np.exp(values[:,0]-logsumexp(values[:,0]))
    importance_ess = float(1/np.sum(weights**2))
    phi = low+points*(high-low)
    qmc_mean = np.sum(weights[:,None]*phi,axis=0)
    chains=np.array(chains)
    np.save(out/'posterior_chains.npy',chains)
    rhat,bulk,tail = chain_diagnostics(chains)
    flat=chains.reshape(-1,4)
    comparisons={}
    for name,folder in [('baseline',HERE/'results_seed17'),('refinement',HERE/'results_seed17/local_refinement')]:
        gp=np.load(folder/'sampling/posterior_samples.npy')
        comparisons[name]={'mean_shift_in_direct_sd':((gp.mean(0)-flat.mean(0))/flat.std(0)).tolist(),
            'width68_ratio':(np.diff(np.quantile(gp,[.16,.84],axis=0),axis=0)[0]/np.diff(np.quantile(flat,[.16,.84],axis=0),axis=0)[0]).tolist()}
    result={'scope':'Same finite 32M population and ideal binned Poisson catalogue; no population-size or coverage validation',
        'max_relative_rate_error':max(errors),'max_lnl_error':max(lnl_errors),
        'median_original_seconds':float(np.median(old_times)),'median_contracted_seconds':float(np.median(new_times)),
        'rhat':rhat.tolist(),'bulk_ess':bulk.tolist(),'tail_ess':tail.tolist(),'acceptance':accepts,
        'converged':bool(np.all(rhat<1.01) and np.all(bulk>400) and np.all(tail>400)),
        'quantiles':np.quantile(flat,[.025,.16,.5,.84,.975],axis=0).tolist(),'gp_comparison':comparisons,
        'full_prior_sobol_ess':importance_ess, 'full_prior_sobol_mean':qmc_mean.tolist(),
        'sobol_minus_chain_mean_in_chain_sd':((qmc_mean-flat[:,[0,1,3]].mean(0))/flat[:,[0,1,3]].std(0)).tolist(),
        'seconds':time.perf_counter()-start}
    h5 = REPO/cfg['data']['compas_file']
    sources = sorted((REPO/'src/cosmic_integration').rglob('*.py'))+[Path(__file__)]
    provenance = {'config':cfg, 'seed':20260912,'warmup_per_chain':warm,'draws_per_chain':draws,
        'input_size':h5.stat().st_size,'input_mtime_ns':h5.stat().st_mtime_ns,
        'source_sha256':hashlib.sha256(b''.join(p.read_bytes() for p in sources)).hexdigest(),
        'catalogue_sha256':hashlib.sha256((REPO/cfg['data']['catalogue']).read_bytes()).hexdigest()}
    (out/'manifest.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    logging.info('Complete: %s',result)

if __name__=='__main__': main()
