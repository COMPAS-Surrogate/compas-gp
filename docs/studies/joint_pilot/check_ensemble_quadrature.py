"""Independent higher-resolution direct references for the complete mock ensemble.

Integrate amplitude CDF and moments analytically rather than adding conditional
amplitude draw noise. This checks reference integration, not a refitted GP.
"""
import json,logging,sys,time
from pathlib import Path
import numpy as np
from scipy.sparse import load_npz
from scipy.stats import qmc,gamma
from scipy.optimize import brentq
from scipy.special import logsumexp
from run_pilot import HERE,REPO,configurations
from cosmic_integration.ratesSampler import BinnedCosmicIntegrator
from cosmic_integration.ratesSampler.fixed_population import FixedPopulationRates
sys.path.insert(0,str(HERE.parent/'population_stability'))
from bootstrap_blocks import vector_marginal
from compare_populations import weighted_quantile


def direct_summary(rates,counts,phi,truth,*,log_target=None):
    """Summarize shape weights and exact amplitude conditional; optionally use surrogate shape weights."""
    loglike,beta=vector_marginal(rates,counts)
    if log_target is not None:
        loglike=np.asarray(log_target,dtype=float)
        if loglike.shape!=beta.shape or not np.isfinite(loglike).all():
            raise ValueError("Invalid shape log target")
    w=np.exp(loglike-logsumexp(loglike));keep=w>1e-15
    bw=beta[keep];wk=w[keep]/w[keep].sum();shape=counts.sum()+1
    def probability(k):
        lo=gamma.cdf(.005,shape+k,scale=1/bw)
        return np.where(lo<.5,gamma.cdf(.015,shape+k,scale=1/bw)-lo,
                        gamma.sf(.005,shape+k,scale=1/bw)-gamma.sf(.015,shape+k,scale=1/bw))
    denominator=probability(0)
    if np.any(denominator<=0):raise ValueError('Unresolved conditional amplitude probability')
    mean_a=np.sum(wk*shape/bw*probability(1)/denominator)
    second_a=np.sum(wk*shape*(shape+1)/bw**2*probability(2)/denominator)
    low=gamma.cdf(.005,shape,scale=1/bw);sf_low=gamma.sf(.005,shape,scale=1/bw)
    def amplitude_cdf(x):
        mass=np.where(low<.5,gamma.cdf(x,shape,scale=1/bw)-low,
                      sf_low-gamma.sf(x,shape,scale=1/bw))
        return float(np.sum(wk*mass/denominator))
    levels=[.025,.05,.16,.5,.84,.95,.975]
    q=np.empty((len(levels),4));q[:,[0,1,3]]=np.array([weighted_quantile(phi[:,j],w,levels) for j in range(3)]).T
    q[:,2]=[brentq(lambda x:amplitude_cdf(x)-p,.005,.015,xtol=1e-12) for p in levels]
    mean=np.insert(np.sum(w[:,None]*phi,axis=0),2,mean_a)
    variance=np.insert(np.sum(w[:,None]*(phi-mean[[0,1,3]])**2,axis=0),2,second_a-mean_a**2)
    ranks=np.insert(np.sum(w[:,None]*(phi<=truth[[0,1,3]]),axis=0),2,amplitude_cdf(truth[2]))
    return {'mean':mean.tolist(),'sd':np.sqrt(variance).tolist(),'quantiles':q.tolist(),
            'ranks':ranks.tolist(),'ess':float(1/np.sum(w*w))}


def main():
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    start=time.perf_counter();root=HERE/'reduced_calibration_100_results'
    data=np.load(root/'catalogues.npz');counts=data['counts'];truths=data['truths']
    ci=BinnedCosmicIntegrator.from_compas_fpath(str(REPO/'tests/large_test_data/h5out_32M_reduced.h5'))
    mass,z,_=configurations()['mass10'];fast=FixedPopulationRates(ci,mass,z)
    meta=json.loads((HERE.parent/'population_stability/block_results/population_meta.json').read_text())
    ci.compas.nSystems=256000000;ci.compas.zamsZmin=meta['zmin'];ci.compas.zamsZmax=meta['zmax'];fast.p_draw=1/np.log(meta['zmax']/meta['zmin'])
    fast.operator=sum((load_npz(HERE.parent/f'population_stability/block_results/block_{i:02d}.npz') for i in range(9,16)),load_npz(HERE.parent/'population_stability/block_results/block_08.npz'))
    occupied=np.any(counts>0,axis=0);obs=counts[:,occupied];runs=[]
    for seed in [20261003,20261004]:
        phi=np.array([-.5,.1,4.2])+qmc.Sobol(3,scramble=True,seed=seed).random_base2(15)*np.array([.499,.5,1.])
        compact=[]
        for i,t in enumerate(phi):
            r=fast(t[0],t[1],.012,t[2]).ravel();compact.append(np.r_[r[occupied],r.sum()])
            if (i+1)%8192==0:logging.info('Reference seed %d: %d/32768 rates',seed,i+1)
        compact=np.array(compact);summaries=[]
        for i,c in enumerate(obs):
            mask=c>0;select=np.r_[np.flatnonzero(mask),len(c)]
            summaries.append(direct_summary(compact[:,select],c[mask],phi,truths[i]))
        (root/f'direct_reference_{seed}.json').write_text(json.dumps(summaries,indent=2)+'\n');runs.append(summaries)
    checks=[]
    for i,(a,b) in enumerate(zip(*runs)):
        sd=np.array(a['sd']);aq=np.array(a['quantiles']);bq=np.array(b['quantiles'])
        row={'case':i,'mean_shift_sd':float(np.max(abs(np.array(a['mean'])-b['mean'])/sd)),
             'endpoint_shift_sd':float(np.max(abs(aq-bq)/sd)),
             'width68_fraction_change':float(np.max(abs((bq[4]-bq[2])/(aq[4]-aq[2])-1)))}
        row['pass']=row['mean_shift_sd']<.05 and row['endpoint_shift_sd']<.05 and row['width68_fraction_change']<.03
        checks.append(row)
    result={'points_per_scramble':32768,'seeds':[20261003,20261004],'amplitude':'analytic conditional CDF and moments',
            'passes':sum(c['pass'] for c in checks),'catalogues':len(checks),'checks':checks,'seconds':time.perf_counter()-start}
    (root/'quadrature_refinement.json').write_text(json.dumps(result,indent=2)+'\n')
    logging.info('Direct reference refinement: %d/%d pass',result['passes'],len(checks))

if __name__=='__main__':main()
