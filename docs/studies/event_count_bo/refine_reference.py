"""Refine unresolved numerical checks only; retain training data and all failures.

A refinement is a validation-only operation, not an acquisition retry. Existing
training sets are refit deterministically with identical settings for rescoring.
"""
import argparse,json,logging,time,pickle
import numpy as np
from scipy.sparse import load_npz
from run_study import (HERE,REPO,TRUTH,LOW,WIDTH,SHAPE,reference_design,configurations,
    BinnedCosmicIntegrator,FixedPopulationRates,vector_marginal,direct_summary,
    compare,posterior_metrics,AdaptiveRobustScaler,JaxTrainingData,JaxGPConfig,fit_exact_gp)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected',type=int,required=True)
    args=parser.parse_args();n=args.expected
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    start=time.perf_counter();root=HERE/'results'
    records=[r for r in json.loads((root/'records.json').read_text()) if r['expected_events']==n]
    refs=[r for r in json.loads((root/'references.json').read_text()) if r['expected_events']==n]
    if len(refs)!=10 or len(records)!=100:raise ValueError('Wait for the complete exposure group')
    affected=sorted({r['case'] for r in refs if not r['numerical_pass']} |
                    {r['case'] for r in records if r.get('accuracy_pass',False) and not r['numerical_pass']})
    out=root/'refinements'/f'n{n}';out.mkdir(parents=True,exist_ok=False)
    config={'expected_events':n,'affected_cases':affected,'power':16,'purpose':'Independent precision refinement; unchanged training data/config/seed',
            'source_sha256':__import__('hashlib').sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest()}
    (out/'manifest.json').write_text(json.dumps(config,indent=2)+'\n')
    if not affected:
        (out/'records.json').write_text('[]\n');(out/'references.json').write_text('[]\n');return
    data=np.load(root/'catalogues.npz');allcounts=data['counts'];occupied=np.any(allcounts>0,axis=0)
    observations=allcounts[data['expected_events']==n]
    physics=json.loads((root/'physics.json').read_text());duration=physics[str(n)]['duration']
    ci=BinnedCosmicIntegrator.from_compas_fpath(str(REPO/'tests/large_test_data/h5out_32M_reduced.h5'))
    mass,z,_=configurations()['mass10'];fast=FixedPopulationRates(ci,mass,z)
    block=HERE.parent/'population_stability/block_results';meta=json.loads((block/'population_meta.json').read_text())
    ci.compas.nSystems=256000000;ci.compas.zamsZmin=meta['zmin'];ci.compas.zamsZmax=meta['zmax'];fast.p_draw=1/np.log(meta['zmax']/meta['zmin'])
    fast.operator=sum((load_npz(block/f'block_{i:02d}.npz') for i in range(1,8)),load_npz(block/'block_00.npz'))
    validation=[]
    for scramble in [0,1]:
        seed=20261120+2*[30,100,300,1000].index(n)+scramble
        phi,logq,cov=reference_design(n,np.array(physics['fisher_unit_per_year'])*duration,seed,power=16)
        old=np.load(root/f'reference_{n}_{scramble}.npz');lookup={p.tobytes():r for p,r in zip(old['phi'],old['rates'])}
        rr=[];new=0
        for p in phi:
            if p.tobytes() in lookup:r=lookup[p.tobytes()]
            else:
                rates=fast(p[0],p[1],.012,p[2]).ravel();r=np.r_[rates[occupied],rates.sum()];new+=1
            rr.append(r)
        rr=np.array(rr);np.savez(out/f'reference_{scramble}.npz',phi=phi,logq=logq,rates=rr)
        validation.append((phi,logq,rr));logging.info('N=%d scramble%d: %d points, %d new rate evaluations',n,scramble,len(phi),new)
    new_records=[];new_refs=[]
    for case in affected:
        c=observations[case,occupied];mask=c>0;select=np.r_[np.flatnonzero(mask),len(c)];cnt=c[mask]
        evaluations=[(p,q,r[:,select]*duration/.1) for p,q,r in validation]
        direct=[]
        for p,q,r in evaluations:
            exact=vector_marginal(r,cnt)[0];direct.append((exact,direct_summary(r,cnt,p,TRUTH,log_target=exact-q)))
        check=compare(direct[0][1],direct[1][1]);refpass=bool(check['max_mean_shift_sd']<.05 and check['max_endpoint_shift_sd']<.05 and np.max(abs(np.array(check['width68_ratio'])-1))<.03)
        new_refs.append({'expected_events':n,'case':case,'events':int(cnt.sum()),'summaries':[d[1] for d in direct],
                         'numerical_check':check,'numerical_pass':refpass,'refinement_power':16})
        for strategy in ['bo','sobol']:
            for budget in [226,354,482,610,738]:
                train=np.load(root/f'n{n}_case{case}_{strategy}'/f'checkpoint_{budget}.npz')
                x,y=train['x'],train['lnl']
                scaler=AdaptiveRobustScaler(compression='sqrt',soft_clipping=False,lower_clip_value=float(y.min()),max_scale=1e9);scaler.initialize_with_data(y)
                model=fit_exact_gp(JaxTrainingData(x,-scaler.transform(y)[:,None]),np.array([LOW[SHAPE],(LOW+WIDTH)[SHAPE]]),config=JaxGPConfig(optimisation_steps=250),seed=17)
                metrics=[];summaries=[]
                for (p,q,r),(exact,ds) in zip(evaluations,direct):
                    pred=scaler.inverse_transform(-model.predict_f(p)[0][:,0]);gs=direct_summary(r,cnt,p,TRUTH,log_target=pred-q)
                    metrics.append({**compare(ds,gs),**{k:v for k,v in posterior_metrics(exact-q,pred-q,p).items() if k.startswith('kl_')}})
                    summaries.append(gs)
                gpcheck=compare(summaries[0],summaries[1]);numerical=bool(refpass and gpcheck['max_mean_shift_sd']<.05 and gpcheck['max_endpoint_shift_sd']<.05 and np.max(abs(np.array(gpcheck['width68_ratio'])-1))<.03)
                accuracy=all(m['max_mean_shift_sd']<.2 and m['max_endpoint_shift_sd']<.2 and np.max(abs(np.array(m['width68_ratio'])-1))<.1 and np.max(abs(np.array(m['width95_ratio'])-1))<.1 and m['kl_direct_to_gp']<.01 and m['kl_gp_to_direct']<.01 for m in metrics)
                original=next(r for r in records if r['case']==case and r['strategy']==strategy and r['points']==budget)
                new_records.append({**original,'summaries':summaries,'metrics':metrics,'numerical_check':gpcheck,
                    'numerical_pass':numerical,'accuracy_pass':bool(accuracy),'validated_pass':bool(accuracy and numerical),'refinement_power':16})
            logging.info('N=%d case%d %s: accuracy=%s numerical=%s',n,case,strategy,accuracy,numerical)
        (out/'records.json').write_text(json.dumps(new_records,indent=2)+'\n');(out/'references.json').write_text(json.dumps(new_refs,indent=2)+'\n')
    (out/'runtime.json').write_text(json.dumps({'seconds':time.perf_counter()-start})+'\n')

if __name__=='__main__':main()
