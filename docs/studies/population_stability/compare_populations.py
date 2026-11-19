"""Same-catalogue finite-population comparison, with bounded-memory extraction."""
from pathlib import Path
import sys
import json
import logging
import time
import copy
import hashlib
import numpy as np
from scipy.stats import qmc, gamma
from scipy.special import logsumexp
from scipy.sparse import save_npz

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
sys.path.insert(0,str(HERE.parent/'grid_resolution'))
from probe_grids import configurations
from cosmic_integration.ratesSampler import BinnedCosmicIntegrator
from cosmic_integration.ratesSampler.population_stream import load_selected_population
from cosmic_integration.ratesSampler.fixed_population import FixedPopulationRates
from cosmic_integration.observation.grid_likelihood import amplitude_marginal_log_likelihood, counts_log_likelihood
from cosmic_integration.ratesSampler.grid_diagnostics import system_grid_moments


def weighted_quantile(x, weights, probabilities):
    order=np.argsort(x)
    return np.interp(probabilities,np.cumsum(weights[order]),x[order])


def main():
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    start=time.perf_counter();out=HERE/'results';out.mkdir(exist_ok=True)
    small_path=REPO/'tests/large_test_data/h5out_32M_reduced.h5'
    large_path=REPO/'tests/large_test_data/h5out_512M_reduced.h5'
    legacy=BinnedCosmicIntegrator.from_compas_fpath(str(small_path))
    streamed=load_selected_population(small_path,legacy.compas.massEvolvedPerBinary)
    for key in ['dcoSeeds','mass1','mass2','Zsystems','delayTime']:
        np.testing.assert_array_equal(getattr(streamed,key),getattr(legacy.compas,key))
    assert streamed.legacy_metallicity_order_matches
    mass,z,_=configurations()['mass10']
    small=FixedPopulationRates(legacy,mass,z)
    # Check that bounded operator construction still matches the original.
    parity=[]
    for theta in [[-.325,.213,.012,4.253],[-.5,.1,.005,4.2],[-.001,.6,.015,5.2]]:
        rates,masses=legacy.FindDetectionRate(**dict(zip(['p_Alpha','p_Sigma','p_SFRa','p_SFRd'],theta)))
        grid,_=system_grid_moments(rates,masses,mass,z)
        np.testing.assert_allclose(small(*theta),grid,rtol=1e-11,atol=1e-15)
        parity.append(float(np.max(abs(small(*theta)-grid))))
    large_ci=copy.copy(legacy)
    large_ci.compas=load_selected_population(large_path,legacy.compas.massEvolvedPerBinary)
    big=large_ci.compas
    order=np.argsort(big.dcoSeeds)
    positions=order[np.searchsorted(big.dcoSeeds[order],streamed.dcoSeeds)]
    for key in ['dcoSeeds','mass1','mass2','Zsystems','delayTime']:
        np.testing.assert_array_equal(getattr(big,key)[positions],getattr(streamed,key))
    # The legacy normalization uses observed metallicity extrema, which differ
    # slightly as a population grows. The operator itself excludes that factor.
    draw_range_ratio=np.log(big.zamsZmax/big.zamsZmin)/np.log(streamed.zamsZmax/streamed.zamsZmin)
    assert abs(draw_range_ratio-1)<1e-6
    logging.info('All selected 32M systems match 512M; constructing large operator')
    large=FixedPopulationRates(large_ci,mass,z)
    # Exact subtraction of the shared systems, before normalization, creates
    # the disjoint 480M remainder. No data are dropped from a GP training set.
    remainder=copy.copy(large)
    remainder.ci=copy.copy(large_ci);remainder.ci.compas=copy.copy(big)
    remainder.ci.compas.nSystems=big.nSystems-streamed.nSystems
    remainder.operator=large.operator-small.operator
    assert remainder.operator.data.min()>-1e-10
    remainder.operator.data=np.maximum(remainder.operator.data,0.)
    counts=np.load(HERE.parent/'grid_resolution/results/pilot_catalogue.npz')['counts_mass10']
    for name,model in [('32M',small),('512M',large),('480M_disjoint',remainder)]:
        save_npz(out/f'operator_{name}.npz',model.operator)
    low=np.array([-.5,.1,4.2]); high=np.array([-.001,.6,5.2])
    results={};draw_data={}
    for name,model in [('32M',small),('512M',large),('480M_disjoint',remainder)]:
        runs=[]
        for seed in [20260915,20260916]:
            u=qmc.Sobol(4,scramble=True,seed=seed).random_base2(13)
            phi=low+u[:,:3]*(high-low)
            values=np.array([amplitude_marginal_log_likelihood(model(t[0],t[1],.012,t[2]),counts,.1) for t in phi])
            weights=np.exp(values[:,0]-logsumexp(values[:,0]))
            supported=weights>1e-15
            beta=values[supported,1];shape=counts.sum()+1
            cdf_low=gamma.cdf(.005,shape,scale=1/beta);cdf_high=gamma.cdf(.015,shape,scale=1/beta)
            a=gamma.ppf(cdf_low+u[supported,3]*(cdf_high-cdf_low),shape,scale=1/beta)
            samples=np.column_stack([phi[supported,:2],a,phi[supported,2]])
            weights=weights[supported];weights/=weights.sum()
            assert np.all(np.isfinite(samples))
            mean=np.sum(samples*weights[:,None],axis=0)
            quantiles=np.array([weighted_quantile(samples[:,i],weights,[.025,.16,.5,.84,.975]) for i in range(4)]).T
            runs.append({'seed':seed,'ess':float(1/np.sum(weights**2)), 'mean':mean.tolist(),'quantiles':quantiles.tolist()})
            draw_data[f'{name}_{seed}_samples']=samples;draw_data[f'{name}_{seed}_weights']=weights
        results[name]={'runs':runs,'fiducial_rate':float(model(-.325,.213,.012,4.253).sum())}
        logging.info('%s complete: ESS %.0f / %.0f',name,runs[0]['ess'],runs[1]['ess'])
    reference=np.load(HERE.parent/'joint_pilot/direct_reference_results/posterior_chains.npy').reshape(-1,4)
    sd=reference.std(0)
    for name,result in results.items():
        runs=result['runs']
        result['between_scramble_mean_difference_in_32M_sd']=((np.array(runs[1]['mean'])-runs[0]['mean'])/sd).tolist()
        result['mean_shift_from_32M_chain_in_sd']=((np.array(runs[0]['mean'])-reference.mean(0))/sd).tolist()
        q=np.array(runs[0]['quantiles']);refq=np.quantile(reference,[.025,.16,.5,.84,.975],axis=0)
        result['width68_ratio_to_32M_chain']=((q[3]-q[1])/(refq[3]-refq[1])).tolist()
        result['width95_ratio_to_32M_chain']=((q[4]-q[0])/(refq[4]-refq[0])).tolist()
        result['endpoint_shift_in_32M_sd']=((q-refq)/sd).tolist()
    meta={'seconds':time.perf_counter()-start,'selected_32M':len(streamed.dcoSeeds),'selected_512M':len(big.dcoSeeds),
        'large_legacy_metallicity_order_matches':big.legacy_metallicity_order_matches,
        'max_absolute_rate_parity_errors':parity,'mass_evolved_per_binary':big.massEvolvedPerBinary,
        'metallicity_log_range_ratio_512M_to_32M':float(draw_range_ratio),
        'zamsZmin':big.zamsZmin,'zamsZmax':big.zamsZmax,
        'scope':'One 32M-generated ideal catalogue; paired full-prior integration; 512M contains 32M; disjoint remainder removes shared systems. Not repeated-catalogue coverage or independent-run convergence.'}
    (out/'summary.json').write_text(json.dumps({'metadata':meta,'populations':results},indent=2)+'\n')
    np.savez(out/'weighted_posteriors.npz',**draw_data)
    sources=sorted((REPO/'src/cosmic_integration').rglob('*.py'))+[Path(__file__)]
    (out/'manifest.json').write_text(json.dumps({'source_sha256':hashlib.sha256(b''.join(p.read_bytes() for p in sources)).hexdigest(),
        'catalogue_sha256':hashlib.sha256((HERE.parent/'grid_resolution/results/pilot_catalogue.npz').read_bytes()).hexdigest(),
        'file_audit':json.loads((HERE/'file_audit.json').read_text())},indent=2)+'\n')
    logging.info('Complete in %.1fs',meta['seconds'])

if __name__=='__main__':main()
