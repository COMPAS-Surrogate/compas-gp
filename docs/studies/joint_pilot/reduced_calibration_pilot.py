"""Noisy catalogues with disjoint 256M generation/inference populations.

An exploratory coverage pilot, not a precision calibration or real-LVK test.
"""
import argparse,json,sys,time,hashlib,logging
import numpy as np
from scipy.stats import qmc
from scipy.special import logsumexp
from scipy.sparse import load_npz
from run_pilot import HERE,REPO,configurations
from cosmic_integration.ratesSampler import BinnedCosmicIntegrator
from cosmic_integration.ratesSampler.fixed_population import FixedPopulationRates
from cosmic_integration.lnl_surrogate.jax_active_learner import JaxGPConfig,JaxTrainingData,fit_exact_gp
from cosmic_integration.lnl_surrogate.adaptive_robust_scalar import AdaptiveRobustScaler
sys.path.insert(0,str(HERE.parent/'population_stability'))
from bootstrap_blocks import vector_marginal,posterior_summary


def main():
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalogues',type=int,default=20)
    parser.add_argument('--seed',type=int,default=20260927)
    parser.add_argument('--validation-seed',type=int,default=20260928)
    parser.add_argument('--outdir',type=str,default='reduced_calibration_results')
    parser.add_argument('--check-quadrature',action='store_true')
    args=parser.parse_args()
    if args.catalogues<1:parser.error('--catalogues must be positive')
    start=time.perf_counter();out=HERE/args.outdir;out.mkdir(exist_ok=True)
    n=args.catalogues
    bounds=np.array([[-.5,.1,4.2],[-.001,.6,5.2]])
    rng=np.random.default_rng(args.seed)
    truths=rng.uniform([-.5,.1,.005,4.2],[-.001,.6,.015,5.2],(n,4))
    ci=BinnedCosmicIntegrator.from_compas_fpath(str(REPO/'tests/large_test_data/h5out_32M_reduced.h5'))
    mass,z,_=configurations()['mass10'];fast=FixedPopulationRates(ci,mass,z)
    meta=json.loads((HERE.parent/'population_stability/block_results/population_meta.json').read_text())
    ci.compas.nSystems=256000000;ci.compas.zamsZmin=meta['zmin'];ci.compas.zamsZmax=meta['zmax']
    fast.p_draw=1/np.log(meta['zmax']/meta['zmin'])
    blocks=[load_npz(HERE.parent/f'population_stability/block_results/block_{i:02d}.npz') for i in range(16)]
    fast.operator=sum(blocks[1:8],blocks[0].copy())
    catalogues=np.array([rng.poisson(.1*fast(*t)) for t in truths]).reshape(n,-1)
    fast.operator=sum(blocks[9:],blocks[8].copy())
    initial=bounds[0]+qmc.Sobol(3,scramble=True,seed=17).random_base2(9)*np.ptp(bounds,axis=0)
    candidates=bounds[0]+qmc.Sobol(3,scramble=True,seed=117).random_base2(12)*np.ptp(bounds,axis=0)
    unit=np.array(np.meshgrid(*([np.linspace(0,1,5)]*3),indexing='ij')).reshape(3,-1).T
    anchors=bounds[0]+unit[np.any((unit==0)|(unit==1),axis=1)]*np.ptp(bounds,axis=0)
    u=qmc.Sobol(4,scramble=True,seed=args.validation_seed).random_base2(13)
    phi=bounds[0]+u[:,:3]*np.ptp(bounds,axis=0)
    check_u=qmc.Sobol(4,scramble=True,seed=args.validation_seed+1).random_base2(13)
    check_phi=bounds[0]+check_u[:,:3]*np.ptp(bounds,axis=0)
    all_points=np.vstack([initial,candidates,anchors,phi]+([check_phi] if args.check_quadrature else []))
    # Retain only bins needed by at least one mock, plus the exact total rate.
    occupied=np.any(catalogues>0,axis=0)
    compact=[]
    for i,t in enumerate(all_points):
        r=fast(t[0],t[1],.012,t[2]).ravel()
        compact.append(np.r_[r[occupied],r.sum()])
        if (i+1)%4096==0:logging.info('Shared rate evaluations %d/%d',i+1,len(all_points))
    compact=np.array(compact)
    obs=catalogues[:,occupied]
    prior_rates=compact[:512];candidate_rates=compact[512:4608];anchor_rates=compact[4608:4706];test_rates=compact[4706:12898];check_rates=compact[12898:]
    assert len(test_rates)==8192
    def fit(x,y):
        scaler=AdaptiveRobustScaler(compression='sqrt',soft_clipping=False,lower_clip_value=float(np.min(y)),max_scale=1e9)
        scaler.initialize_with_data(y)
        model=fit_exact_gp(JaxTrainingData(x,-scaler.transform(y)[:,None]),bounds,config=JaxGPConfig(optimisation_steps=250),seed=17)
        return model,scaler
    cases=[]
    for case,counts in enumerate(obs):
        # vector_marginal expects all provided count columns to be occupied.
        mask=counts>0;cnt=counts[mask]
        select=np.r_[np.flatnonzero(mask),len(counts)]
        pr=prior_rates[:,select];cr=candidate_rates[:,select];ar=anchor_rates[:,select];tr=test_rates[:,select]
        record={'case':case,'truth':truths[case].tolist(),'events':int(counts.sum())}
        try:
            y,_=vector_marginal(pr,cnt)
            if not np.isfinite(y).all():raise ValueError('Nonfinite training target; do not floor unsupported events')
            model,scaler=fit(initial,y)
            pred=scaler.inverse_transform(-model.predict_f(candidates)[0][:,0])
            probabilities=.9*np.exp(pred-logsumexp(pred))+.1/len(pred)
            index=np.random.default_rng(17).choice(len(candidates),128,replace=False,p=probabilities)
            new_y,_=vector_marginal(cr[index],cnt);anchor_y,_=vector_marginal(ar,cnt)
            model,scaler=fit(np.vstack([initial,candidates[index],anchors]),np.r_[y,new_y,anchor_y])
            direct_log,beta=vector_marginal(tr,cnt)
            pred=scaler.inverse_transform(-model.predict_f(phi)[0][:,0])
            w=np.exp(direct_log-logsumexp(direct_log));gpw=np.exp(pred-logsumexp(pred))
            summary,draws,_=posterior_summary(tr,cnt,phi,u[:,3])
            # Exact conditional amplitude is evaluated where the direct posterior
            # has support. Reject GP leakage instead of fabricating tail draws.
            leakage=float(gpw[w<=1e-15].sum())
            if leakage>1e-8:raise ValueError(f'GP mass {leakage} outside numerical amplitude support')
            gm=(gpw[:,None]*draws).sum(0);dm=np.array(summary['mean']);sd=np.array(summary['sd'])
            from compare_populations import weighted_quantile
            gq=np.array([weighted_quantile(draws[:,i],gpw,[.025,.16,.5,.84,.975]) for i in range(4)]).T
            dq=np.array(summary['quantiles'])
            from scipy.stats import binomtest
            gq90=np.array([weighted_quantile(draws[:,i],gpw,[.05,.95]) for i in range(4)]).T
            dq90=np.array([weighted_quantile(draws[:,i],w,[.05,.95]) for i in range(4)]).T
            record.update({'direct_rank':np.sum(w[:,None]*(draws<=truths[case]),axis=0).tolist(),
                'gp_rank':np.sum(gpw[:,None]*(draws<=truths[case]),axis=0).tolist(),
                'direct_cover90':((truths[case]>=dq90[0])&(truths[case]<=dq90[1])).tolist(),
                'gp_cover90':((truths[case]>=gq90[0])&(truths[case]<=gq90[1])).tolist(),
                'width95_ratio':((gq[4]-gq[0])/(dq[4]-dq[0])).tolist(),
                'max_endpoint_shift_sd':float(np.max(abs(gq-dq)/sd)),
                'direct_sd':sd.tolist(),
                'gp_mass_outside_amplitude_support':leakage})
            if args.check_quadrature:
                alt,alt_draws,alt_w=posterior_summary(check_rates[:,select],cnt,check_phi,check_u[:,3])
                alt_q=np.array(alt['quantiles'])
                record['quadrature_check']={'max_mean_shift_sd':float(np.max(abs(np.array(alt['mean'])-dm)/sd)),
                    'max_endpoint_shift_sd':float(np.max(abs(alt_q-dq)/sd)),
                    'max_width68_fraction_change':float(np.max(abs((alt_q[3]-alt_q[1])/(dq[3]-dq[1])-1))),
                    'ess':alt['ess']}
                record['quadrature_pass']=bool(record['quadrature_check']['max_mean_shift_sd']<.05 and
                    record['quadrature_check']['max_endpoint_shift_sd']<.05 and
                    record['quadrature_check']['max_width68_fraction_change']<.03)
            record['full_accuracy_pass']=bool(np.max(abs(gm-dm)/sd)<.2 and
                np.max(abs(gq-dq)/sd)<.2 and
                np.max(abs((gq[3]-gq[1])/(dq[3]-dq[1])-1))<.1 and
                np.max(abs((gq[4]-gq[0])/(dq[4]-dq[0])-1))<.1)
            residual=pred-direct_log;residual-=np.sum(w*residual)
            record.update({'status':'evaluated','direct_ess':summary['ess'],'gp_ess':float(1/np.sum(gpw**2)),
                'direct_weighted_lnl_rmse':float(np.sqrt(np.sum(w*residual**2))),
                'max_mean_shift_sd':float(np.max(abs(gm-dm)/sd)),
                'width68_ratio':((gq[3]-gq[1])/(dq[3]-dq[1])).tolist(),
                'direct_cover68':((truths[case]>=dq[1])&(truths[case]<=dq[3])).tolist(),
                'gp_cover68':((truths[case]>=gq[1])&(truths[case]<=gq[3])).tolist(),
                'direct_cover95':((truths[case]>=dq[0])&(truths[case]<=dq[4])).tolist(),
                'gp_cover95':((truths[case]>=gq[0])&(truths[case]<=gq[4])).tolist(),
                'direct_quantiles':dq.tolist(),'gp_quantiles':gq.tolist(),
                'accuracy_pass':bool(np.max(abs(gm-dm)/sd)<.2 and np.max(abs((gq[3]-gq[1])/(dq[3]-dq[1])-1))<.1)})
        except (ValueError,FloatingPointError) as exc:
            record.update({'status':'failed','reason':str(exc),'accuracy_pass':False})
        cases.append(record)
        (out/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
        logging.info('Catalogue %d/%d: N=%d, %s, accuracy=%s',case+1,n,record['events'],record['status'],record['accuracy_pass'])
    valid=[r for r in cases if r['status']=='evaluated']
    summary={'catalogues':n,'evaluated':len(valid),'failed':n-len(valid),'accuracy_passes':sum(r['accuracy_pass'] for r in cases),
        'coverage_scope':'Coverage counts below are over evaluated cases only; failures are separate and cannot be omitted in any calibration claim.',
        'coverage_counts':{key:np.sum([r[key] for r in valid],axis=0).tolist() for key in ['direct_cover68','gp_cover68','direct_cover90','gp_cover90','direct_cover95','gp_cover95']},
        'seconds':time.perf_counter()-start,'scope':f'{n} prior-drawn truths, Poisson counts at 0.1 yr; generation first 256M, inference disjoint last 256M; perfect binned observations, one fixed GP training seed; not precise calibration or LVK validation.'}
    from scipy.stats import binomtest
    summary['coverage_exact95_intervals']={key:[[float(v) for v in binomtest(k,len(valid)).proportion_ci()] for k in counts] for key,counts in summary['coverage_counts'].items()} if valid else {}
    summary['full_accuracy_passes']=sum(r.get('full_accuracy_pass',False) for r in cases)
    summary['quadrature_passes']=sum(r.get('quadrature_pass',False) for r in cases) if args.check_quadrature else None
    np.savez(out/'catalogues.npz',truths=truths,counts=catalogues)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'manifest.json').write_text(json.dumps({'script_sha256':hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest(),
        'population_meta':meta,'generation_blocks':list(range(8)),'inference_blocks':list(range(8,16)),
        'seed':args.seed,'validation_seed':args.validation_seed,'catalogues':n,'check_quadrature':args.check_quadrature,'training_seed':17,'training_points':738,'noise_fixed':True},indent=2)+'\n')
    logging.info('Pilot complete: %s',summary)

if __name__=='__main__':main()
