"""Paired continuous event-uncertainty pilot, frozen design and independent checks."""
from pathlib import Path
import sys,json,time,logging,hashlib
import numpy as np
from scipy.sparse import load_npz
from scipy.special import logsumexp
from scipy.stats import qmc
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2]
sys.path.insert(0,str(HERE.parent/'event_count_bo'))
from run_study import (LOW,WIDTH,TRUTH,SHAPE,reference_design,configurations,BinnedCosmicIntegrator,
 FixedPopulationRates,AdaptiveRobustScaler,JaxTrainingData,JaxGPConfig,fit_exact_gp,compare,direct_summary,posterior_metrics,sqrt_log_expected_likelihood)
from cosmic_integration.observation.log_measurement import bin_coefficients,log_marginal_events,mock_pe_samples


def summary(rates,number,phi,logw):
    """Exact conditional amplitude with supplied shape-posterior log weights."""
    total=rates[:,-1]
    return direct_summary(np.column_stack([total,total]),np.array([number]),phi,TRUTH,log_target=logw)


def checks(a,b):
    m=compare(a,b)
    return bool(m['max_mean_shift_sd']<.05 and m['max_endpoint_shift_sd']<.05 and np.max(abs(np.array(m['width68_ratio'])-1))<.03),m


def main():
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    out=HERE/'results';out.mkdir(exist_ok=False);start=time.perf_counter()
    manifest={'expected_events':[50,500],'catalogues_per_exposure':10,'seed':20260921,'training_seed':17,
      'random_seed':20260922,'budgets':[226,354,482,610,738],'truth':TRUTH.tolist(),
      'population':'blocks0-7; same latent population for generation and inference','measurement':'log-coordinate Normal, sigma=(.03,.3)*12/rho, rho=12*U^(-1/3); conditional on detection; independent of lambda',
      'latent_distribution':'piecewise uniform in linear mass/redshift within the fixed populated bins',
      'reference_power':15,'scope':'20 paired catalogues, 80 acquisition trajectories; pilot at one truth/design seed, not SBC or LVK PE',
      'source_sha256':{str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),REPO/'src/cosmic_integration/observation/log_measurement.py']}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    ci=BinnedCosmicIntegrator.from_compas_fpath(str(REPO/'tests/large_test_data/h5out_32M_reduced.h5'))
    mass,z,_=configurations()['mass10'];fast=FixedPopulationRates(ci,mass,z)
    block=HERE.parent/'population_stability/block_results';meta=json.loads((block/'population_meta.json').read_text())
    ci.compas.nSystems=256000000;ci.compas.zamsZmin=meta['zmin'];ci.compas.zamsZmax=meta['zmax'];fast.p_draw=1/np.log(meta['zmax']/meta['zmin'])
    fast.operator=sum((load_npz(block/f'block_{i:02d}.npz') for i in range(1,8)),load_npz(block/'block_00.npz'))
    active=np.flatnonzero(np.asarray(fast.operator.getnnz(axis=1))>0);mi,zi=np.unravel_index(active,fast.shape);ze=z*.1
    lower=np.column_stack([mass[mi],ze[zi]]);upper=np.column_stack([mass[mi+1],ze[zi+1]])
    assert np.isfinite(upper).all(), 'Nonzero unbounded guard bins need a within-bin model'
    truth_rates=fast(*TRUTH).ravel()[active];r0=truth_rates.sum();rng=np.random.default_rng(manifest['seed'])
    catalogues=[]
    for n in manifest['expected_events']:
        for case in range(10):
            number=rng.poisson(n);bins=rng.choice(len(active),number,p=truth_rates/r0)
            latent=rng.uniform(lower[bins],upper[bins]);rho=12*rng.random(number)**(-1/3)
            scales=np.column_stack([.03*12/rho,.3*12/rho]);y=np.log(latent)+rng.normal(size=(number,2))*scales
            coeff=bin_coefficients(y,scales,lower,upper);offset=np.log(coeff.max(axis=1));coeff/=np.exp(offset[:,None])
            assert np.all(coeff.sum(axis=1)>0)
            catalogues.append({'n':n,'case':case,'number':number,'duration':n/r0,'bins':bins,'latent':latent,'rho':rho,'scales':scales,'y':y,'coeff':coeff})
            np.savez(out/f'catalogue_{n}_{case}.npz',bins=bins,latent=latent,rho=rho,scales=scales,y=y,coeff=coeff,log_coefficient_scale=offset)
    np.savez(out/'grid.npz',active=active,lower=lower,upper=upper,truth_rates=truth_rates,mass_edges=mass,z_edges=ze)
    def cache(points,path):
        rr=np.array([fast(p[0],p[1],.012,p[2]).ravel()[active] for p in points]);np.savez(path,phi=points,rates=rr);return rr
    lo=LOW[SHAPE];width=WIDTH[SHAPE]
    initial=lo+qmc.Sobol(3,scramble=True,seed=17).random_base2(7)*width
    unit=np.array(np.meshgrid(*([np.linspace(0,1,5)]*3),indexing='ij')).reshape(3,-1).T
    initial=np.vstack([initial,lo+unit[np.any((unit==0)|(unit==1),axis=1)]*width])
    candidate=lo+qmc.Sobol(3,scramble=True,seed=117).random_base2(12)*width
    random=np.random.default_rng(manifest['random_seed']).uniform(lo,lo+width,(512,3))
    ir=cache(initial,out/'initial.npz');cr=cache(candidate,out/'candidates.npz');rr=cache(random,out/'random.npz')
    oldphys=json.loads((HERE.parent/'event_count_bo/results/physics.json').read_text());fisher=np.array(oldphys['fisher_unit_per_year'])
    references={}
    for n in manifest['expected_events']:
        refs=[]
        for scramble in [0,1]:
            phi,logq,cov=reference_design(n,fisher*n/r0,20260930+2*manifest['expected_events'].index(n)+scramble,power=15)
            rates=cache(phi,out/f'reference_{n}_{scramble}.npz')
            np.savez(out/f'proposal_{n}_{scramble}.npz',logq=logq,cov=cov)
            refs.append((phi,logq,rates));logging.info('Reference N=%s scramble=%s rows=%s',n,scramble,len(phi))
        references[n]=refs
    # Prior-recycling identity on every event of the first small catalogue.
    cat=catalogues[0];phi,logq,ref=references[50][0];duration=cat['duration'];number=cat['number'];coef=cat['coeff']
    compact=lambda r,c:np.column_stack([r@c.T,r.sum(axis=1)])*duration/.1
    direct=log_marginal_events(ref@coef.T,ref.sum(axis=1),duration)
    direct_s=summary(compact(ref,coef),number,phi,direct-logq)
    # PE support covers all populated bins. No artificial truncation of observations.
    support_lo=lower.min(axis=0);support_hi=upper.max(axis=0);area=np.prod(upper-lower,axis=1)
    pe_records=[]
    for samples in [8192,32768]:
        for prior_id,powers in enumerate([np.array([0.,0.]),np.array([1.,2.])]):
            pe=np.zeros_like(coef);ess=[]
            for i,(y,sc) in enumerate(zip(cat['y'],cat['scales'])):
                x,prior,evidence=mock_pe_samples(y,sc,support_lo,support_hi,powers,samples,np.random.default_rng(20261010+i))
                # Exact finite support and inactive bins: outside active bins have zero model density.
                m=np.searchsorted(mass,x[:,0],side='right')-1;j=np.searchsorted(ze,x[:,1],side='right')-1
                flat=m*fast.shape[1]+j;lookup=np.searchsorted(active,flat);valid=lookup<len(active);valid[valid]&=active[lookup[valid]]==flat[valid]
                pe[i]=np.bincount(lookup[valid],weights=1/prior[valid],minlength=len(active))/samples/area
                pe[i]*=evidence/np.exp(np.log(bin_coefficients(y[None],sc[None],lower,upper).max()))
                w=1/prior;ess.append(float(w.sum()**2/(w@w)))
            target=log_marginal_events(ref@pe.T,ref.sum(axis=1),duration)
            ps=summary(compact(ref,pe),number,phi,target-logq);metric=compare(direct_s,ps)
            good=metric['max_mean_shift_sd']<.05 and metric['max_endpoint_shift_sd']<.05 and np.max(abs(np.array(metric['width68_ratio'])-1))<.03
            pe_records.append({'samples_per_event':samples,'prior_powers':powers.tolist(),'minimum_inverse_prior_ess':min(ess),'metrics':metric,'pass':bool(good)})
            np.savez(out/f'pe_coeff_{samples}_{prior_id}.npz',coeff=pe)
    (out/'prior_recycling.json').write_text(json.dumps(pe_records,indent=2)+'\n')
    if not all(r['pass'] for r in pe_records if r['samples_per_event']==32768):
        raise RuntimeError('Prior-recycling precision gate failed; inspect before GP study')
    logging.info('Prior-recycling check passed; starting paired GP experiment')
    records=[];reference_records=[]
    for cat in catalogues:
        n=cat['n'];case=cat['case'];number=cat['number'];T=cat['duration']
        for mode in ['perfect','uncertain']:
            coeff=cat['coeff'] if mode=='uncertain' else np.eye(len(active))[cat['bins']]
            def evaluate(r):return log_marginal_events(r@coeff.T,r.sum(axis=1),T)
            validation=[]
            for phi,logq,r in references[n]:
                exact=evaluate(r);compact=np.column_stack([r.sum(axis=1),r.sum(axis=1)])*T/.1
                ds=summary(compact,number,phi,exact-logq);validation.append((phi,logq,compact,exact,ds))
            refpass,refcheck=checks(validation[0][-1],validation[1][-1])
            reference_records.append({'expected_events':n,'case':case,'mode':mode,'events':number,'summaries':[v[-1] for v in validation],'numerical_pass':refpass,'numerical_check':refcheck})
            iy=evaluate(ir)
            for strategy in ['bo','random']:
                x=initial.copy();y=iy.copy();available=np.ones(len(candidate),bool);acq_rng=np.random.default_rng(17)
                folder=out/f'n{n}_case{case}_{mode}_{strategy}';folder.mkdir();tick=time.perf_counter()
                for step,budget in enumerate(manifest['budgets']):
                    sc=AdaptiveRobustScaler(compression='sqrt',soft_clipping=False,lower_clip_value=float(y.min()),max_scale=1e9);sc.initialize_with_data(y)
                    model=fit_exact_gp(JaxTrainingData(x,-sc.transform(y)[:,None]),np.array([lo,lo+width]),config=JaxGPConfig(optimisation_steps=250),seed=17)
                    metrics=[];summaries=[]
                    for phi,logq,compact,exact,ds in validation:
                        pred=sc.inverse_transform(-model.predict_f(phi)[0][:,0]);gs=summary(compact,number,phi,pred-logq)
                        metrics.append({**compare(ds,gs),**{k:v for k,v in posterior_metrics(exact-logq,pred-logq,phi).items() if k.startswith('kl_')}});summaries.append(gs)
                    precision,ck=checks(*summaries);numerical=bool(refpass and precision)
                    accuracy=all(m['max_mean_shift_sd']<.2 and m['max_endpoint_shift_sd']<.2 and np.max(abs(np.array(m['width68_ratio'])-1))<.1 and np.max(abs(np.array(m['width95_ratio'])-1))<.1 and m['kl_direct_to_gp']<.01 and m['kl_gp_to_direct']<.01 for m in metrics)
                    records.append({'expected_events':n,'case':case,'events':number,'mode':mode,'strategy':strategy,'points':budget,'metrics':metrics,'summaries':summaries,'accuracy_pass':bool(accuracy),'numerical_pass':numerical,'validated_pass':bool(accuracy and numerical),'numerical_check':ck,'elapsed_seconds':time.perf_counter()-tick})
                    np.savez(folder/f'checkpoint_{budget}.npz',x=x,lnl=y)
                    if step==4:break
                    if strategy=='random':chosen=np.arange(step*128,(step+1)*128);newx=random[chosen];newr=rr[chosen]
                    else:
                        mu,var=model.predict_f(candidate);one=sqrt_log_expected_likelihood(mu[:,0],var[:,0],sc);two=sqrt_log_expected_likelihood(mu[:,0],var[:,0],sc,power=2.)
                        with np.errstate(divide='ignore'):score=two+np.log(-np.expm1(np.minimum(2*one-two,0.)))
                        ix=np.flatnonzero(available);p=.9*np.exp(score[ix]-logsumexp(score[ix]))+.1/len(ix)
                        chosen=acq_rng.choice(ix,128,replace=False,p=p);available[chosen]=False;newx=candidate[chosen];newr=cr[chosen]
                    x=np.vstack([x,newx]);y=np.r_[y,evaluate(newr)]
                logging.info('N=%s case=%s %s %s accuracy=%s numerical=%s',n,case,mode,strategy,accuracy,numerical)
                (out/'records.json').write_text(json.dumps(records,indent=2)+'\n');(out/'references.json').write_text(json.dumps(reference_records,indent=2)+'\n')
    (out/'runtime.json').write_text(json.dumps({'seconds':time.perf_counter()-start})+'\n')
if __name__=='__main__':main()
