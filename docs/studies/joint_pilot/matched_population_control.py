"""Direct-only internal-calibration control on the same fresh catalogues.

Use their generation population for inference; no GP retraining or new truths.
This distinguishes population mismatch from a coverage fluctuation shared by
both population targets. Two quadrature scrambles check numerical sensitivity.
"""
import json,logging,time
import numpy as np
from scipy.stats import qmc,binomtest
from scipy.sparse import load_npz
from confirm_482 import HERE,REPO,configurations,BinnedCosmicIntegrator,FixedPopulationRates,compare
from check_ensemble_quadrature import direct_summary


def main():
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s');start=time.perf_counter();root=HERE/'confirm_482_results'
    data=np.load(root/'catalogues.npz');counts=data['counts'];truths=data['truths']
    ci=BinnedCosmicIntegrator.from_compas_fpath(str(REPO/'tests/large_test_data/h5out_32M_reduced.h5'))
    mass,z,_=configurations()['mass10'];fast=FixedPopulationRates(ci,mass,z)
    meta=json.loads((HERE.parent/'population_stability/block_results/population_meta.json').read_text())
    ci.compas.nSystems=256000000;ci.compas.zamsZmin=meta['zmin'];ci.compas.zamsZmax=meta['zmax'];fast.p_draw=1/np.log(meta['zmax']/meta['zmin'])
    fast.operator=sum((load_npz(HERE.parent/f'population_stability/block_results/block_{i:02d}.npz') for i in range(1,8)),load_npz(HERE.parent/'population_stability/block_results/block_00.npz'))
    occupied=np.any(counts>0,axis=0);obs=counts[:,occupied];runs=[]
    for seed in [20261021,20261022]:
        phi=np.array([-.5,.1,4.2])+qmc.Sobol(3,scramble=True,seed=seed).random_base2(14)*np.array([.499,.5,1.]);r=[]
        for t in phi:
            rate=fast(t[0],t[1],.012,t[2]).ravel();r.append(np.r_[rate[occupied],rate.sum()])
        r=np.array(r);result=[]
        for c,truth in zip(obs,truths):
            mask=c>0;select=np.r_[np.flatnonzero(mask),len(c)]
            result.append(direct_summary(r[:,select],c[mask],phi,truth))
        runs.append(result);logging.info('Matched-population integration %d finished',seed)
    coverage={};disagreements={}
    for level,i,j in [(68,2,4),(90,1,5),(95,0,6)]:
        flags=[np.array([(truth>=np.array(r['quantiles'])[i])&(truth<=np.array(r['quantiles'])[j]) for truth,r in zip(truths,run)]) for run in runs]
        coverage[str(level)]=np.sum(flags[0],axis=0).tolist();disagreements[str(level)]=np.sum(flags[0]!=flags[1],axis=0).tolist()
    numerical=[compare(a,b) for a,b in zip(*runs)]
    result={'catalogues':len(counts),'population':'generation blocks0-7 used for inference; same catalogue/truth arrays',
        'coverage':coverage,'coverage_exact95_intervals':{k:[list(binomtest(n,len(counts)).proportion_ci()) for n in v] for k,v in coverage.items()},
        'coverage_disagreements_between_scrambles':disagreements,'numerical_checks':numerical,'summaries':runs,'seconds':time.perf_counter()-start}
    (root/'matched_population_control.json').write_text(json.dumps(result,indent=2)+'\n');logging.info('Matched coverage: %s',coverage)

if __name__=='__main__':main()
