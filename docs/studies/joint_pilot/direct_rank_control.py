"""One predeclared 1000-catalogue matched-population rank-calibration control.

Triggered by alpha68 deficit in a fresh 100-catalogue ensemble. No GP training,
no choice of successful seeds, no interval optimization. Equal-tailed coverage
is computed from posterior CDFs at truth on two independent quadrature grids.
"""
import json,logging,time
import numpy as np
from scipy.stats import qmc,gamma,binomtest
from scipy.special import logsumexp
from scipy.sparse import load_npz
from confirm_482 import HERE,REPO,configurations,BinnedCosmicIntegrator,FixedPopulationRates,vector_marginal


def posterior_ranks(rates,counts,phi,truth):
    loglike,beta=vector_marginal(rates,counts);w=np.exp(loglike-logsumexp(loglike))
    ranks=np.empty(4);ranks[[0,1,3]]=w@(phi<=truth[[0,1,3]])
    keep=w>1e-15;bk=beta[keep];wk=w[keep]/w[keep].sum();shape=counts.sum()+1
    low=gamma.cdf(.005,shape,scale=1/bk);sf_low=gamma.sf(.005,shape,scale=1/bk)
    denominator=np.where(low<.5,gamma.cdf(.015,shape,scale=1/bk)-low,sf_low-gamma.sf(.015,shape,scale=1/bk))
    numerator=np.where(low<.5,gamma.cdf(truth[2],shape,scale=1/bk)-low,sf_low-gamma.sf(truth[2],shape,scale=1/bk))
    if np.any(denominator<=0):raise ValueError('Conditional amplitude probability unresolved')
    ranks[2]=wk@(numerator/denominator)
    return ranks


def main():
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s');start=time.perf_counter();root=HERE/'direct_rank_control_results';root.mkdir(exist_ok=True)
    n=1000;rng=np.random.default_rng(20261025);truths=rng.uniform([-.5,.1,.005,4.2],[-.001,.6,.015,5.2],(n,4))
    ci=BinnedCosmicIntegrator.from_compas_fpath(str(REPO/'tests/large_test_data/h5out_32M_reduced.h5'))
    mass,z,_=configurations()['mass10'];fast=FixedPopulationRates(ci,mass,z)
    meta=json.loads((HERE.parent/'population_stability/block_results/population_meta.json').read_text())
    ci.compas.nSystems=256000000;ci.compas.zamsZmin=meta['zmin'];ci.compas.zamsZmax=meta['zmax'];fast.p_draw=1/np.log(meta['zmax']/meta['zmin'])
    fast.operator=sum((load_npz(HERE.parent/f'population_stability/block_results/block_{i:02d}.npz') for i in range(1,8)),load_npz(HERE.parent/'population_stability/block_results/block_00.npz'))
    counts=np.array([rng.poisson(.1*fast(*t)) for t in truths]).reshape(n,-1);occupied=np.any(counts>0,axis=0);obs=counts[:,occupied]
    np.savez(root/'catalogues.npz',truths=truths,counts=counts)
    all_ranks=[]
    for seed in [20261026,20261027]:
        phi=np.array([-.5,.1,4.2])+qmc.Sobol(3,scramble=True,seed=seed).random_base2(14)*np.array([.499,.5,1.]);cache=[]
        for t in phi:
            r=fast(t[0],t[1],.012,t[2]).ravel();cache.append(np.r_[r[occupied],r.sum()])
        cache=np.array(cache);ranks=[]
        for i,(c,truth) in enumerate(zip(obs,truths)):
            mask=c>0;select=np.r_[np.flatnonzero(mask),len(c)]
            ranks.append(posterior_ranks(cache[:,select],c[mask],phi,truth))
            if (i+1)%250==0:logging.info('Reference seed %d: %d/1000 catalogues',seed,i+1)
        all_ranks.append(ranks)
    ranks=np.array(all_ranks);np.savez(root/'ranks.npz',ranks=ranks)
    coverage={};disagreements={}
    for level in [68,90,95]:
        lo=(1-level/100)/2;hi=1-lo;flags=(ranks>=lo)&(ranks<=hi)
        coverage[str(level)]=flags[0].sum(0).tolist();disagreements[str(level)]=(flags[0]!=flags[1]).sum(0).tolist()
    summary={'catalogues':n,'seed':20261025,'validation_seeds':[20261026,20261027],'points_per_scramble':16384,'coverage':coverage,
        'coverage_exact95_intervals':{k:[list(binomtest(c,n).proportion_ci()) for c in v] for k,v in coverage.items()},
        'coverage_disagreements_between_scrambles':disagreements,'maximum_rank_difference':np.max(abs(ranks[0]-ranks[1]),axis=0).tolist(),
        'seconds':time.perf_counter()-start,'scope':'New direct-only matched-population ideal-observation control; no GP, same blocks0-7 for generation/inference. Does not independently calibrate 482-label GP or finite-population mismatch.'}
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');logging.info('Coverage: %s',coverage)

if __name__=='__main__':main()
