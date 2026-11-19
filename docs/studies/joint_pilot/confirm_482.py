"""Fresh 100-catalogue confirmation of a frozen 482-label acquisition rule.

Same ideal-observation scope, disjoint 256M populations, fresh catalogue seed.
Two independent 16384-point integrations with analytic conditional amplitude.
No adaptive training-budget increase and no exclusion of failed cases.
"""
import json,logging,time,pickle,hashlib
import numpy as np
from scipy.stats import qmc,binomtest
from scipy.sparse import load_npz
from scipy.special import logsumexp
from bo_uncertainty_curve import (HERE,REPO,configurations,FixedPopulationRates,BinnedCosmicIntegrator,
    AdaptiveRobustScaler,JaxTrainingData,JaxGPConfig,fit_exact_gp,vector_marginal,
    sqrt_log_expected_likelihood,posterior_metrics)
from check_ensemble_quadrature import direct_summary


def compare(a,b):
    sd=np.array(a['sd']);aq=np.array(a['quantiles']);bq=np.array(b['quantiles'])
    result={'max_mean_shift_sd':float(np.max(abs(np.array(a['mean'])-b['mean'])/sd)),
        'max_endpoint_shift_sd':float(np.max(abs(aq-bq)/sd)),
        'width68_ratio':((bq[4]-bq[2])/(aq[4]-aq[2])).tolist(),
        'width95_ratio':((bq[6]-bq[0])/(aq[6]-aq[0])).tolist()}
    return result


def main():
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    start=time.perf_counter();out=HERE/'confirm_482_results';out.mkdir(exist_ok=True)
    lo=np.array([-.5,.1,4.2]);width=np.array([.499,.5,1.]);bounds=np.array([lo,lo+width])
    rng=np.random.default_rng(20261020);truths=rng.uniform([-.5,.1,.005,4.2],[-.001,.6,.015,5.2],(100,4))
    ci=BinnedCosmicIntegrator.from_compas_fpath(str(REPO/'tests/large_test_data/h5out_32M_reduced.h5'))
    mass,z,_=configurations()['mass10'];fast=FixedPopulationRates(ci,mass,z)
    meta=json.loads((HERE.parent/'population_stability/block_results/population_meta.json').read_text())
    ci.compas.nSystems=256000000;ci.compas.zamsZmin=meta['zmin'];ci.compas.zamsZmax=meta['zmax'];fast.p_draw=1/np.log(meta['zmax']/meta['zmin'])
    def operator(ids):return sum((load_npz(HERE.parent/f'population_stability/block_results/block_{i:02d}.npz') for i in ids[1:]),load_npz(HERE.parent/f'population_stability/block_results/block_{ids[0]:02d}.npz'))
    fast.operator=operator(list(range(8)));catalogues=np.array([rng.poisson(.1*fast(*t)) for t in truths]).reshape(100,-1)
    fast.operator=operator(list(range(8,16)))
    np.savez(out/'catalogues.npz',truths=truths,counts=catalogues)
    initial=lo+qmc.Sobol(3,scramble=True,seed=17).random_base2(7)*width
    unit=np.array(np.meshgrid(*([np.linspace(0,1,5)]*3),indexing='ij')).reshape(3,-1).T
    initial=np.vstack([initial,lo+unit[np.any((unit==0)|(unit==1),axis=1)]*width])
    candidates=lo+qmc.Sobol(3,scramble=True,seed=117).random_base2(12)*width
    validation=[lo+qmc.Sobol(3,scramble=True,seed=s).random_base2(14)*width for s in [20261021,20261022]]
    occupied=np.any(catalogues>0,axis=0);obs=catalogues[:,occupied]
    def rates(points):
        rows=[]
        for t in points:
            r=fast(t[0],t[1],.012,t[2]).ravel();rows.append(np.r_[r[occupied],r.sum()])
        return np.array(rows)
    ir=rates(initial);cr=rates(candidates);vr=[]
    for i,points in enumerate(validation):vr.append(rates(points));logging.info('Validation rate cache %d/2 complete',i+1)
    cache_seconds=time.perf_counter()-start;cases=[]
    for case,c in enumerate(obs):
        record={'case':case,'truth':truths[case].tolist(),'events':int(c.sum()),'training_points':482}
        tick=time.perf_counter();mask=c>0;select=np.r_[np.flatnonzero(mask),len(c)];cnt=c[mask]
        try:
            x=initial.copy();y=vector_marginal(ir[:,select],cnt)[0];cy=vector_marginal(cr[:,select],cnt)[0]
            available=np.ones(len(candidates),bool);rng_acq=np.random.default_rng(17)
            for batch in range(3):
                if not np.isfinite(y).all():raise ValueError('Nonfinite training likelihood')
                scaler=AdaptiveRobustScaler(compression='sqrt',soft_clipping=False,lower_clip_value=float(y.min()),max_scale=1e9)
                scaler.initialize_with_data(y)
                model=fit_exact_gp(JaxTrainingData(x,-scaler.transform(y)[:,None]),bounds,config=JaxGPConfig(optimisation_steps=250),seed=17)
                if batch==2:break
                mu,var=model.predict_f(candidates)
                first=sqrt_log_expected_likelihood(mu[:,0],var[:,0],scaler)
                second=sqrt_log_expected_likelihood(mu[:,0],var[:,0],scaler,power=2.)
                with np.errstate(divide='ignore'):score=second+np.log(-np.expm1(np.minimum(2*first-second,0.)))
                idx=np.flatnonzero(available);s=score[idx]
                p=.9*np.exp(s-logsumexp(s))+.1/len(idx) if np.isfinite(s).any() else np.full(len(idx),1/len(idx))
                chosen=rng_acq.choice(idx,128,replace=False,p=p);available[chosen]=False
                x=np.vstack([x,candidates[chosen]]);y=np.r_[y,cy[chosen]]
            assert len(x)==482 and len(np.unique(x,axis=0))==482
            folder=out/f'case_{case:03d}';folder.mkdir(exist_ok=True)
            with (folder/'model.pkl').open('wb') as f:pickle.dump(model,f)
            scaler.save(str(folder));np.savez(folder/'training.npz',x=x,lnl=y)
            summaries=[];metrics=[]
            for points,rates_i in zip(validation,vr):
                rr=rates_i[:,select];exact=vector_marginal(rr,cnt)[0]
                pred=scaler.inverse_transform(-model.predict_f(points)[0][:,0])
                direct=direct_summary(rr,cnt,points,truths[case]);gp=direct_summary(rr,cnt,points,truths[case],log_target=pred)
                metric={**compare(direct,gp),**{k:v for k,v in posterior_metrics(exact,pred,points).items() if k.startswith('kl_')}}
                metric['posterior_pass']=bool(metric['max_mean_shift_sd']<.2 and metric['max_endpoint_shift_sd']<.2 and max(abs(np.array(metric['width68_ratio'])-1))<.1 and max(abs(np.array(metric['width95_ratio'])-1))<.1)
                metric['kl_pass']=bool(metric['kl_direct_to_gp']<.01 and metric['kl_gp_to_direct']<.01)
                summaries.append({'direct':direct,'gp':gp});metrics.append(metric)
            checks={name:compare(summaries[0][name],summaries[1][name]) for name in ['direct','gp']}
            numerical_pass=all(m['max_mean_shift_sd']<.05 and m['max_endpoint_shift_sd']<.05 and max(abs(np.array(m['width68_ratio'])-1))<.03 for m in checks.values())
            record.update(status='evaluated',summaries=summaries,metrics=metrics,numerical_checks=checks,numerical_pass=bool(numerical_pass),posterior_pass=all(m['posterior_pass'] for m in metrics),kl_pass=all(m['kl_pass'] for m in metrics))
        except (ValueError,FloatingPointError) as exc:record.update(status='failed',reason=str(exc),posterior_pass=False,kl_pass=False,numerical_pass=False)
        record['seconds']=time.perf_counter()-tick;cases.append(record)
        (out/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
        logging.info('Case %d/100 N=%d posterior=%s KL=%s numerical=%s',case+1,record['events'],record['posterior_pass'],record['kl_pass'],record['numerical_pass'])
    valid=[c for c in cases if c['status']=='evaluated'];coverage={}
    for name in ['direct','gp']:
        for level,i,j in [(68,2,4),(90,1,5),(95,0,6)]:
            flags=[(np.array(c['truth'])>=np.array(c['summaries'][0][name]['quantiles'])[i])&(np.array(c['truth'])<=np.array(c['summaries'][0][name]['quantiles'])[j]) for c in valid]
            coverage[f'{name}_{level}']=np.sum(flags,axis=0).tolist()
    summary={'catalogues':100,'evaluated':len(valid),'failed':100-len(valid),**{k:sum(c[k] for c in cases) for k in ['posterior_pass','kl_pass','numerical_pass']},'coverage_counts_evaluated_only':coverage,
        'coverage_exact95_intervals':{k:[list(binomtest(n,len(valid)).proportion_ci()) for n in v] for k,v in coverage.items()} if valid else {},'seconds':time.perf_counter()-start,'cache_seconds':cache_seconds,
        'scope':'Fresh prior-drawn truths, disjoint halves, ideal binned Poisson data at .1yr. Fixed 482 labels, training seed17. Analytic conditional amplitude. Separate posterior, KL and numerical gates; no failed cases excluded from denominators.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'manifest.json').write_text(json.dumps({'seed':20261020,'validation_seeds':[20261021,20261022],'validation_points':16384,'training_seed':17,'training_points':482,'source_sha256':hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest(),'population_meta':meta},indent=2)+'\n')
    logging.info('Complete: %s',summary)

if __name__=='__main__':main()
