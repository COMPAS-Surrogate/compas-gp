"""Three independent 3D training designs, without reference-informed training."""
import json
import time
import sys
import hashlib
import pickle
import logging
import numpy as np
from scipy.special import logsumexp
from scipy.stats import qmc,gamma
from run_pilot import HERE,REPO,configurations
from cosmic_integration.ratesSampler import BinnedCosmicIntegrator
from cosmic_integration.ratesSampler.fixed_population import FixedPopulationRates
from cosmic_integration.observation.grid_likelihood import amplitude_marginal_log_likelihood
from cosmic_integration.lnl_surrogate.jax_active_learner import JaxTrainingData,JaxGPConfig,fit_exact_gp
from cosmic_integration.lnl_surrogate.adaptive_robust_scalar import AdaptiveRobustScaler
sys.path.insert(0,str(HERE.parent/'population_stability'))
from compare_populations import weighted_quantile


def main():
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--population',choices=['32','512'],default='32')
    parser.add_argument('--validation-seed',type=int,default=20260924)
    parser.add_argument('--evaluate-only',action='store_true')
    parser.add_argument('--boundary-anchors',action='store_true')
    args=parser.parse_args()
    start=time.perf_counter()
    model_root=HERE/(('reduced_seed_results' if args.population=='32' else 'reduced_seed_results_512M')+('_anchored' if args.boundary_anchors else ''))
    out=model_root/(f'check_{args.validation_seed}' if args.evaluate_only else '')
    out.mkdir(parents=True,exist_ok=True)
    bounds=np.array([[-.5,.1,4.2],[-.001,.6,5.2]])
    ci=BinnedCosmicIntegrator.from_compas_fpath(str(REPO/'tests/large_test_data/h5out_32M_reduced.h5'))
    mass,z,_=configurations()['mass10'];fast=FixedPopulationRates(ci,mass,z)
    if args.population=='512':
        from scipy.sparse import load_npz
        population_meta=json.loads((HERE.parent/'population_stability/results/summary.json').read_text())['metadata']
        ci.compas.nSystems=512000000
        ci.compas.zamsZmin=population_meta['zamsZmin'];ci.compas.zamsZmax=population_meta['zamsZmax']
        fast.p_draw=1/np.log(ci.compas.zamsZmax/ci.compas.zamsZmin)
        fast.operator=load_npz(HERE.parent/'population_stability/results/operator_512M.npz')
    counts=np.load(HERE.parent/'grid_resolution/results/pilot_catalogue.npz')['counts_mass10']
    def direct(t):return amplitude_marginal_log_likelihood(fast(t[0],t[1],.012,t[2]),counts,.1)
    def fit(x,raw,seed):
        scaler=AdaptiveRobustScaler(compression='sqrt',soft_clipping=False,lower_clip_value=float(min(raw)),max_scale=1e9)
        scaler.initialize_with_data(raw)
        model=fit_exact_gp(JaxTrainingData(x,-scaler.transform(raw)[:,None]),bounds,config=JaxGPConfig(optimisation_steps=250),seed=seed)
        return model,scaler
    def predict(model,scaler,x):return scaler.inverse_transform(-model.predict_f(x)[0][:,0])
    u=qmc.Sobol(4,scramble=True,seed=args.validation_seed).random_base2(13)
    phi=bounds[0]+u[:,:3]*np.ptp(bounds,axis=0)
    exact=np.array([direct(t) for t in phi]);truth,beta=exact.T
    w=np.exp(truth-logsumexp(truth));keep=w>1e-15
    shape=counts.sum()+1;a=np.full(len(phi),.01)
    lo=gamma.cdf(.005,shape,scale=1/beta[keep]);hi=gamma.cdf(.015,shape,scale=1/beta[keep])
    a[keep]=gamma.ppf(lo+u[keep,3]*(hi-lo),shape,scale=1/beta[keep])
    draws=np.column_stack([phi[:,:2],a,phi[:,2]])
    def metrics(weights):
        mean=np.sum(draws*weights[:,None],axis=0)
        centered=draws-mean;cov=(centered*weights[:,None]).T@centered
        sd=np.sqrt(np.diag(cov));corr=cov/(sd[:,None]*sd[None,:])
        q=np.array([weighted_quantile(draws[:,i],weights,[.025,.16,.5,.84,.975]) for i in range(4)]).T
        return mean,sd,corr,q
    dm,ds,dc,dq=metrics(w)
    result={}
    for seed in [17,29,43]:
        tick=time.perf_counter()
        if args.evaluate_only:
            with (model_root/f'seed_{seed}/model.pkl').open('rb') as f:model=pickle.load(f)
            scaler=AdaptiveRobustScaler.load(str(model_root/f'seed_{seed}'))
            x=model.data.query_points;raw=scaler.inverse_transform(-model.data.observations[:,0])
        else:
            x=bounds[0]+qmc.Sobol(3,scramble=True,seed=seed).random_base2(9)*np.ptp(bounds,axis=0)
            raw=np.array([direct(t)[0] for t in x])
            model,scaler=fit(x,raw,seed)
            candidates=bounds[0]+qmc.Sobol(3,scramble=True,seed=seed+100).random_base2(12)*np.ptp(bounds,axis=0)
            prediction=predict(model,scaler,candidates)
            probability=np.exp(prediction-logsumexp(prediction))
            # Small uniform component keeps exploration when the preliminary GP is wrong.
            probability=.9*probability+.1/len(probability)
            extra=candidates[np.random.default_rng(seed).choice(len(candidates),128,replace=False,p=probability)]
            x=np.vstack([x,extra]);raw=np.r_[raw,[direct(t)[0] for t in extra]]
            if args.boundary_anchors:
                unit=np.array(np.meshgrid(*([np.linspace(0,1,5)]*3),indexing='ij')).reshape(3,-1).T
                boundary=unit[np.any((unit==0)|(unit==1),axis=1)]
                anchors=bounds[0]+boundary*np.ptp(bounds,axis=0)
                x=np.vstack([x,anchors]);raw=np.r_[raw,[direct(t)[0] for t in anchors]]
            model,scaler=fit(x,raw,seed)
        pred=predict(model,scaler,phi);gpw=np.exp(pred-logsumexp(pred))
        gm,gs,gc,gq=metrics(gpw)
        residual=pred-truth;center=residual-np.sum(w*residual)
        gpcenter=residual-np.sum(gpw*residual)
        result[str(seed)]={'gp_mass_outside_amplitude_evaluation_support':float(gpw[~keep].sum()),'direct_weighted_lnl_rmse':float(np.sqrt(np.sum(w*center**2))),
            'gp_weighted_lnl_rmse':float(np.sqrt(np.sum(gpw*gpcenter**2))),
            'gp_quadrature_ess':float(1/np.sum(gpw**2)),'mean_shift_sd':((gm-dm)/ds).tolist(),
            'width68_ratio':((gq[3]-gq[1])/(dq[3]-dq[1])).tolist(),
            'width95_ratio':((gq[4]-gq[0])/(dq[4]-dq[0])).tolist(),
            'max_endpoint_shift_sd':np.max(abs(gq-dq)/ds,axis=0).tolist(),
            'max_correlation_difference':float(np.max(abs(gc-dc))),
            'seconds':time.perf_counter()-tick}
        assert gpw[~keep].sum()<1e-8
        folder=out/f'seed_{seed}';folder.mkdir(exist_ok=True)
        with (folder/'model.pkl').open('wb') as f:pickle.dump(model,f)
        scaler.save(str(folder))
        np.savez(folder/'validation.npz',training_x=x,training_lnl=raw,phi=phi,draws=draws,direct_lnl=truth,gp_lnl=pred,direct_weights=w,gp_weights=gpw)
        logging.info('Seed %d %s',seed,result[str(seed)])
    result['metadata']={'population':args.population,'validation_seed':args.validation_seed,'evaluate_only':args.evaluate_only,'boundary_anchors':args.boundary_anchors,'training_points':len(x),'seconds':time.perf_counter()-start,'direct_quadrature_ess':float(1/np.sum(w**2)),
        'training':'512 full-prior Sobol + 128 GP-guided points with 10% uniform exploration; every training point retained; no reference-informed training',
        'assessment':'8192 common full-prior quadrature points, exact amplitude conditional. Independent numerical scramble still needed for passing configurations.',
        'source_sha256':hashlib.sha256(b''.join(p.read_bytes() for p in sorted((REPO/'src/cosmic_integration').rglob('*.py'))+[__import__('pathlib').Path(__file__)])).hexdigest()}
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
