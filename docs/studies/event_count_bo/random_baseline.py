"""Independent uniform acquisition after the shared 226-point initialization.

A matched supplemental comparison; 512 IID prior points replace BO/Sobol additions.
Reuses frozen catalogues and independent numerical references, never their labels
for training. One acquisition seed; no claim of random-design seed robustness.
"""
import json,time,logging
import numpy as np
from scipy.sparse import load_npz
from run_study import (HERE,REPO,LOW,WIDTH,SHAPE,TRUTH,configurations,BinnedCosmicIntegrator,
 FixedPopulationRates,vector_marginal,direct_summary,compare,posterior_metrics,
 AdaptiveRobustScaler,JaxTrainingData,JaxGPConfig,fit_exact_gp)

def main():
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    root=HERE/'results';out=HERE/'random_results';out.mkdir(exist_ok=False);tick=time.perf_counter()
    manifest={'seed':20260910,'initialization':'Identical saved 226-point initial design','addition':'512 independent uniform prior draws; shared across catalogues','budgets':[226,354,482,610,738],'scope':'Random additions, not a wholly random initialization; one design seed'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    data=np.load(root/'catalogues.npz');occupied=np.any(data['counts']>0,axis=0)
    ci=BinnedCosmicIntegrator.from_compas_fpath(str(REPO/'tests/large_test_data/h5out_32M_reduced.h5'))
    mass,z,_=configurations()['mass10'];fast=FixedPopulationRates(ci,mass,z)
    block=HERE.parent/'population_stability/block_results';meta=json.loads((block/'population_meta.json').read_text())
    ci.compas.nSystems=256000000;ci.compas.zamsZmin=meta['zmin'];ci.compas.zamsZmax=meta['zmax'];fast.p_draw=1/np.log(meta['zmax']/meta['zmin'])
    fast.operator=sum((load_npz(block/f'block_{i:02d}.npz') for i in range(1,8)),load_npz(block/'block_00.npz'))
    xnew=np.random.default_rng(manifest['seed']).uniform(LOW[SHAPE],(LOW+WIDTH)[SHAPE],(512,3))
    rates=[]
    for p in xnew:
        r=fast(p[0],p[1],.012,p[2]).ravel();rates.append(np.r_[r[occupied],r.sum()])
    rates=np.array(rates);np.savez(out/'new_points.npz',x=xnew,rates=rates)
    old=json.loads((root/'validated_records.json').read_text());records=[]
    physics=json.loads((root/'physics.json').read_text())
    for n in [30,100,300,1000]:
        refpath=root/'refinements'/f'n{n}'
        usefine=refpath.exists() and (refpath/'reference_0.npz').exists()
        refs=[np.load(refpath/f'reference_{s}.npz' if usefine else root/f'reference_{n}_{s}.npz') for s in [0,1]]
        duration=physics[str(n)]['duration']
        for case,full in enumerate(data['counts'][data['expected_events']==n]):
            c=full[occupied];mask=c>0;cnt=c[mask];sel=np.r_[np.flatnonzero(mask),len(c)]
            init=np.load(root/f'n{n}_case{case}_bo/checkpoint_226.npz')
            x=np.vstack([init['x'],xnew]);y=np.r_[init['lnl'],vector_marginal(rates[:,sel]*duration/.1,cnt)[0]]
            validation=[]
            for ref in refs:
                p,q,r=ref['phi'],ref['logq'],ref['rates'][:,sel]*duration/.1
                exact=vector_marginal(r,cnt)[0];ds=direct_summary(r,cnt,p,TRUTH,log_target=exact-q)
                validation.append((p,q,r,exact,ds))
            dc=compare(validation[0][-1],validation[1][-1]);refpass=dc['max_mean_shift_sd']<.05 and dc['max_endpoint_shift_sd']<.05 and np.max(abs(np.array(dc['width68_ratio'])-1))<.03
            for budget in manifest['budgets']:
                if budget==226:
                    record=next(r for r in old if r['expected_events']==n and r['case']==case and r['strategy']=='bo' and r['points']==226).copy();record['strategy']='random';records.append(record);continue
                yy=y[:budget];sc=AdaptiveRobustScaler(compression='sqrt',soft_clipping=False,lower_clip_value=float(yy.min()),max_scale=1e9);sc.initialize_with_data(yy)
                model=fit_exact_gp(JaxTrainingData(x[:budget],-sc.transform(yy)[:,None]),np.array([LOW[SHAPE],(LOW+WIDTH)[SHAPE]]),config=JaxGPConfig(optimisation_steps=250),seed=17)
                metrics=[];summaries=[]
                for p,q,r,exact,ds in validation:
                    pred=sc.inverse_transform(-model.predict_f(p)[0][:,0]);gs=direct_summary(r,cnt,p,TRUTH,log_target=pred-q)
                    metrics.append({**compare(ds,gs),**{k:v for k,v in posterior_metrics(exact-q,pred-q,p).items() if k.startswith('kl_')}});summaries.append(gs)
                ck=compare(*summaries);numerical=bool(refpass and ck['max_mean_shift_sd']<.05 and ck['max_endpoint_shift_sd']<.05 and np.max(abs(np.array(ck['width68_ratio'])-1))<.03)
                accuracy=all(m['max_mean_shift_sd']<.2 and m['max_endpoint_shift_sd']<.2 and np.max(abs(np.array(m['width68_ratio'])-1))<.1 and np.max(abs(np.array(m['width95_ratio'])-1))<.1 and m['kl_direct_to_gp']<.01 and m['kl_gp_to_direct']<.01 for m in metrics)
                records.append({'expected_events':n,'case':case,'events':int(cnt.sum()),'strategy':'random','points':budget,'status':'evaluated','summaries':summaries,'metrics':metrics,'numerical_pass':numerical,'accuracy_pass':bool(accuracy),'validated_pass':bool(numerical and accuracy)})
            (out/'records.json').write_text(json.dumps(records,indent=2)+'\n');logging.info('N=%s case=%s done',n,case)
    (out/'runtime.json').write_text(json.dumps({'seconds':time.perf_counter()-tick})+'\n')
if __name__=='__main__':main()
