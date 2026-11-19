"""Block-bootstrap sensitivity at the existing 512M simulation budget.

Sixteen disjoint 32M blocks are an empirical resampling basis, not independent
COMPAS campaigns. Every block weight is shared across bins and parameter points.
Posterior pooling below is a sensitivity diagnostic, not Bayesian marginalization.
"""
from pathlib import Path
import sys
import copy
import json
import hashlib
import logging
import time
import numpy as np
from scipy.sparse import load_npz, save_npz, vstack, csr_matrix
from scipy.special import gammainc, gammaincc, gammaln, logsumexp
from scipy.stats import qmc, gamma

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
sys.path.insert(0,str(HERE.parent/'grid_resolution'))
from probe_grids import configurations
from cosmic_integration.ratesSampler import BinnedCosmicIntegrator
from cosmic_integration.ratesSampler.population_stream import load_selected_population
from cosmic_integration.ratesSampler.fixed_population import FixedPopulationRates
from cosmic_integration.observation.grid_likelihood import amplitude_marginal_log_likelihood
from compare_populations import weighted_quantile


def vector_marginal(rates: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rates columns are occupied bins followed by total; amplitude at .012."""
    beta=.1*rates[:,-1]/.012
    shape=float(counts.sum()+1)
    left,right=beta*.005,beta*.015
    probability=np.where(left<shape,gammainc(shape,right)-gammainc(shape,left),
                         gammaincc(shape,left)-gammaincc(shape,right))
    with np.errstate(divide='ignore',invalid='ignore'):
        value=np.sum(counts[None,:]*np.log(.1*rates[:,:-1]/.012),axis=1)+gammaln(shape)-shape*np.log(beta)+np.log(probability)-np.log(.01)
    # Preserve finite extreme tails using the tested scalar fallback. Their
    # posterior weight is normally negligible, but do not floor likelihoods.
    bad=np.flatnonzero((probability<=1e-250)&np.all(rates[:,:-1]>0,axis=1))
    for i in bad:
        unobserved=max(rates[i,-1]-rates[i,:-1].sum(),0.)
        full=np.r_[rates[i,:-1],unobserved][None,:]
        value[i],beta[i]=amplitude_marginal_log_likelihood(full,np.r_[counts,0][None,:],.1)
    return value,beta


def posterior_summary(rates: np.ndarray, counts: np.ndarray, phi: np.ndarray,
                      amplitude_u: np.ndarray) -> tuple[dict,np.ndarray,np.ndarray]:
    loglike,beta=vector_marginal(rates,counts)
    weights=np.exp(loglike-logsumexp(loglike))
    keep=weights>1e-15
    shape=counts.sum()+1
    a=np.full(len(phi),.01)
    lo=gamma.cdf(.005,shape,scale=1/beta[keep]);hi=gamma.cdf(.015,shape,scale=1/beta[keep])
    a[keep]=gamma.ppf(lo+amplitude_u[keep]*(hi-lo),shape,scale=1/beta[keep])
    draws=np.column_stack([phi[:,:2],a,phi[:,2]])
    if not np.isfinite(draws).all():raise ValueError('Conditional amplitude draw failed')
    mean=(weights[:,None]*draws).sum(0)
    sd=np.sqrt((weights[:,None]*(draws-mean)**2).sum(0))
    quantiles=np.array([weighted_quantile(draws[:,i],weights,[.025,.16,.5,.84,.975]) for i in range(4)]).T
    return {'mean':mean.tolist(),'sd':sd.tolist(),'quantiles':quantiles.tolist(),'ess':float(1/np.sum(weights**2))},draws,weights


def main():
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    start=time.perf_counter();out=HERE/'block_results';out.mkdir(exist_ok=True)
    sources=sorted((REPO/'src/cosmic_integration').rglob('*.py'))+[Path(__file__),HERE/'compare_populations.py',HERE.parent/'grid_resolution/probe_grids.py']
    h5=REPO/'tests/large_test_data/h5out_512M_reduced.h5'
    catalogue=HERE.parent/'grid_resolution/results/pilot_catalogue.npz'
    manifest={'source_sha256':hashlib.sha256(b''.join(p.read_bytes() for p in sources)).hexdigest(),
        'h5_bytes':h5.stat().st_size,'h5_mtime_ns':h5.stat().st_mtime_ns,
        'catalogue_sha256':hashlib.sha256(catalogue.read_bytes()).hexdigest(),
        'block_size':32000000,'blocks':16,'bootstrap_replicates':128,'bootstrap_seed':20260918,
        'sobol_seeds':[20260917,20260919],'sobol_power':13}
    if (out/'manifest.json').exists() and json.loads((out/'manifest.json').read_text())!=manifest:
        raise ValueError('Provenance changed: use a new block output directory')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    ci=BinnedCosmicIntegrator.from_compas_fpath(str(REPO/'tests/large_test_data/h5out_32M_reduced.h5'))
    mass,z,_=configurations()['mass10']
    blocks=[]
    if not all((out/f'block_{b:02d}.npz').exists() for b in range(16)):
        pop=load_selected_population(h5,ci.compas.massEvolvedPerBinary)
        for b in range(16):
            chosen=pop.system_rows//32000000==b
            part=copy.copy(pop)
            for attr in ['mass1','mass2','Zsystems','delayTime','dcoSeeds','system_rows']:
                setattr(part,attr,getattr(pop,attr)[chosen])
            part.nSystems=32000000
            temporary=copy.copy(ci);temporary.compas=part
            model=FixedPopulationRates(temporary,mass,z)
            save_npz(out/f'block_{b:02d}.npz',model.operator)
            logging.info('Block %d/16, selected systems %d',b+1,chosen.sum())
        (out/'population_meta.json').write_text(json.dumps({'zmin':pop.zamsZmin,'zmax':pop.zamsZmax,'mass_per_binary':pop.massEvolvedPerBinary})+'\n')
    meta=json.loads((out/'population_meta.json').read_text())
    for b in range(16):blocks.append(load_npz(out/f'block_{b:02d}.npz'))
    full=sum(blocks[1:],blocks[0].copy())
    previous=load_npz(HERE/'results/operator_512M.npz')
    error=full-previous
    relative_error=float(np.max(abs(error.data))/np.max(abs(previous.data))) if error.nnz else 0.
    assert relative_error<1e-12
    ci.compas.zamsZmin=meta['zmin'];ci.compas.zamsZmax=meta['zmax']
    ci.compas.nSystems=512000000
    ci.CalculateRedshiftRelatedParams()
    counts_full=np.load(catalogue)['counts_mass10'].ravel()
    occupied=counts_full>0;counts=counts_full[occupied]
    compact=vstack([vstack([op[occupied],csr_matrix(op.sum(axis=0))]) for op in blocks]).tocsr()
    results={};bootstrap_means=[];bootstrap_q=[];pool_draws=[];pool_weights=[]
    coefficients=np.random.default_rng(20260918).multinomial(16,np.full(16,1/16),size=128)
    np.save(out/'bootstrap_coefficients.npy',coefficients)
    for seed in [20260917,20260919]:
        u=qmc.Sobol(4,scramble=True,seed=seed).random_base2(13)
        phi=np.array([-.5,.1,4.2])+u[:,:3]*np.array([.499,.5,1.])
        cache=out/f'block_rates_{seed}.npz'
        if cache.exists(): block_rates=np.load(cache)['rates']
        else:
            block_rates=[]
            for index,t in enumerate(phi):
                density,_,pdraw=ci.FindZdistribution(ci.redshifts,np.log(meta['zmin']),np.log(meta['zmax']),p_Alpha=t[0],p_Sigma=t[1])
                formed=ci.CalculateSFR(ci.redshifts,.012,t[2])/(meta['mass_per_binary']*512000000)
                block_rates.append((compact@(density*formed[:,None]/pdraw).ravel()).reshape(16,len(counts)+1))
                if (index+1)%2048==0:logging.info('Rate grid seed %d: %d/8192',seed,index+1)
            block_rates=np.array(block_rates)
            np.savez(cache,rates=block_rates,phi=phi,u=u)
        total=block_rates.sum(axis=1)
        nominal,draws,weights=posterior_summary(total,counts,phi,u[:,3])
        results[str(seed)]={'nominal':nominal}
        if seed==20260919:continue  # Independent numerical check of full target.
        for name,coef in [('half_A',np.r_[np.full(8,2.),np.zeros(8)]),('half_B',np.r_[np.zeros(8),np.full(8,2.)])]:
            rates=np.einsum('nbk,b->nk',block_rates,coef)
            results[str(seed)][name]=posterior_summary(rates,counts,phi,u[:,3])[0]
        results[str(seed)]['blocks']=[]
        for b in range(16):
            results[str(seed)]['blocks'].append(posterior_summary(16*block_rates[:,b,:],counts,phi,u[:,3])[0])
        bootstrap_ess=[];relative_lnl=[]
        anchor=int(np.argmax(weights))
        for b,coef in enumerate(coefficients):
            rates=np.einsum('nbk,b->nk',block_rates,coef)
            summary,samples,w=posterior_summary(rates,counts,phi,u[:,3])
            bootstrap_means.append(summary['mean']);bootstrap_q.append(summary['quantiles']);bootstrap_ess.append(summary['ess'])
            pool_draws.append(samples);pool_weights.append(w/len(coefficients))
            ell,_=vector_marginal(rates,counts)
            relative_lnl.append(ell-ell[anchor])
            if (b+1)%32==0:logging.info('Bootstrap %d/128',b+1)
        relative_lnl=np.array(relative_lnl)
        supported=np.all(np.isfinite(relative_lnl),axis=0)&(weights>1e-12)
        results[str(seed)]['bootstrap']={'minimum_ess':min(bootstrap_ess),
            'mean_scatter_in_nominal_sd':(np.std(bootstrap_means,axis=0,ddof=1)/nominal['sd']).tolist(),
            'mean_shift_quantiles_in_nominal_sd':np.quantile((np.array(bootstrap_means)-nominal['mean'])/nominal['sd'],[.05,.5,.95],axis=0).tolist(),
            'posterior_weighted_rms_relative_lnl_sd':float(np.sqrt(np.sum(weights[supported]*np.var(relative_lnl[:,supported],axis=0,ddof=1))/weights[supported].sum()))}
        pooled=np.concatenate(pool_draws);pw=np.concatenate(pool_weights)
        pooled_q=np.array([weighted_quantile(pooled[:,i],pw,[.025,.16,.5,.84,.975]) for i in range(4)]).T
        q=np.array(nominal['quantiles'])
        results[str(seed)]['bootstrap']['pooled_sensitivity_width68_ratio']=((pooled_q[3]-pooled_q[1])/(q[3]-q[1])).tolist()
        np.savez(out/'bootstrap_posteriors.npz',means=bootstrap_means,quantiles=bootstrap_q,pooled_quantiles=pooled_q)
    results['metadata']={'seconds':time.perf_counter()-start,'operator_reconstruction_relative_error':relative_error,
        'scope':'Conditional block-bootstrap sensitivity, not a corrected Bayesian posterior or calibrated coverage; blocks must be exchangeable for sampling interpretation'}
    (out/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    logging.info('Finished in %.1fs',results['metadata']['seconds'])

if __name__=='__main__':main()
