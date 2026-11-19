"""Four-input paper figures from frozen scores and joint GP importance weights.

Production GPs are frozen; illustrative 25/50-point fits use saved initial labels.
Reference amplitude proposals for predictive maps are importance-corrected to the
joint GP target; amplitude is never replaced by its analytic conditional.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.special import logsumexp
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve
from matplotlib.patches import Patch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import PolyCollection,LineCollection
from matplotlib.colors import LogNorm,Normalize

REPO=Path(__file__).resolve().parents[3]
BASE=REPO/'docs/studies/amplitude_gp'
ROOT=BASE/'manuscript_stage2_20260917'
INPUTS=BASE/'paper_inputs_20260915'
OUT=ROOT/'paper_figures'
FIG=REPO/'overleaf/figures/bo_study'
LOW=np.array([-.5,.1,.005,4.2]);WIDTH=np.array([.499,.5,.01,1.])
LABELS=[r'$\alpha$',r'$\sigma$',r'$a$',r'$d$']
BLUE='#226A91';ORANGE='#D17A22';COLORS=[BLUE,ORANGE,'#677E45','#A4517B']
SOURCES={}


def read(path):
    SOURCES[str(path.relative_to(REPO))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())


def save(fig,name):
    for ax in fig.axes:
        ax.tick_params(which='both',direction='in',top=True,right=True)
    for ext in ('pdf','png'):fig.savefig(FIG/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def histories():
    result={}
    for phase in ('manuscript_stage1_20260917','manuscript_stage2_20260917'):
        for t in read(BASE/phase/'results_snapshot.json')['tasks']:result[t['job']['name']]=t
    for path in sorted((ROOT/'earlier_histories').glob('*/history.json')):
        data=read(path)
        result[path.parent.name]=dict(job={'name':path.parent.name},rows=data['rows'])
    return result


def summaries(history):
    with (ROOT/'combined_trajectory_summary.csv').open() as f:rows=list(csv.DictReader(f))
    rows=[r for r in rows if r['group']=='count_history' and r['seed']=='17']
    assert len(rows)==140
    counts=[10,50,100,200,500,700,1000]
    fig,axes=plt.subplots(1,2,figsize=(9,3.5),sharey=True,layout='constrained')
    for ax,view in zip(axes,('perfect','uncertain')):
        for method,color,offset,marker in [('bo',BLUE,-.10,'o'),('random',ORANGE,.10,'s')]:
            for i,n in enumerate(counts):
                group=sorted([r for r in rows if r['view']==view and r['strategy']==method and int(r['expected_events'])==n],key=lambda r:r['name'])
                assert len(group)==5
                for k,r in enumerate(group):
                    confirmed=bool(r['confirmation_cost']);y=int(r['confirmation_cost'] or r['maximum'])
                    ax.scatter(i+offset+(k-2)*.023,y,marker=marker if confirmed else '^',s=25,
                               facecolors=color if confirmed else 'none',edgecolors=color,lw=.8)
        ax.set(xticks=range(7),xticklabels=counts,xlabel=r'$\mathbb{E}[N_{\rm obs}]$',ylim=(440,2000))
        ax.legend(handles=[Line2D([],[],marker='o',color=BLUE,ls='',label='BO'),Line2D([],[],marker='s',color=ORANGE,ls='',label='Random'),Line2D([],[],marker='^',color='.3',mfc='none',ls='',label='Unconfirmed at ceiling')],frameon=False,fontsize=8)
    axes[0].set_ylabel('Training evaluations including confirmation')
    save(fig,'stopping_budget')
    fig,axes=plt.subplots(2,2,figsize=(9,5.5),height_ratios=[2,1],sharex='col',sharey='row',layout='constrained')
    records=[]
    for col,view in enumerate(('perfect','uncertain')):
        axes[0,col].set_title('Perfect measurements' if col==0 else 'Uncertain measurements',fontsize=10)
        axes[0,col].axhline(1,color='.45',ls=':',lw=1,zorder=0)
        for k in range(4):
            medians=[];bounds=[]
            for n in counts:
                vals=[]
                for case in range(200,205):
                    name=f'count_n{n}_{view}_c{case}_s17_bo'
                    scores=history[name]['rows'][-1]['attempts'][-1]['scores']
                    gp=np.mean([np.asarray(s['gp']['quantiles'])[3,k]-np.asarray(s['gp']['quantiles'])[1,k] for s in scores])/(.68*WIDTH[k])
                    vals.append(gp);records.append(dict(name=name,parameter=k,gp_width_prior=gp))
                medians.append(np.median(vals));bounds.append(np.quantile(vals,[.1,.9]))
            medians=np.array(medians);bounds=np.array(bounds)
            axes[0,col].errorbar(np.arange(7)+(k-1.5)*.09,medians,yerr=[medians-bounds[:,0],bounds[:,1]-medians],color=COLORS[k],marker='o',ms=3.5,lw=1.2,capsize=2,label=LABELS[k])
        for method,color,offset,marker in [('bo',BLUE,-.09,'o'),('random',ORANGE,.09,'s')]:
            budgets=[]
            for n in counts:
                group=[r for r in rows if r['view']==view and int(r['expected_events'])==n and r['strategy']==method]
                assert len(group)==5
                assert len({r['maximum'] for r in group})==1
                budgets.append(int(group[0]['maximum']))
            axes[1,col].plot(np.arange(7)+offset,budgets,marker=marker,mfc='white',color=color,lw=1.2,label=method.upper() if method=='bo' else 'Random')
        axes[0,col].legend(frameon=False,ncol=4,fontsize=9,loc='lower left')
        axes[1,col].set(xticks=range(7),xticklabels=counts,xlabel=r'$\mathbb{E}[N_{\rm obs}]$',ylim=(0,2100),yticks=[0,1000,2000])
        axes[1,col].text(.04,.16,'Full budgets; stopping counts unavailable',transform=axes[1,col].transAxes,fontsize=8,color='.35')
        axes[1,col].legend(frameon=False,fontsize=8,loc='upper left',ncol=2)
    axes[0,0].set_ylabel(r'$W_{68}/W_{68}^{\rm prior}$');axes[1,0].set_ylabel(r'LnL evaluations $N_{\rm train}$')
    save(fig,'acquisition_summary');(OUT/'widths.json').write_text(json.dumps(records,indent=2)+'\n')


def smooth_display_density(histogram, covariance):
    """Reflect a Gaussian display kernel: 0.2 posterior SD plus 0.45 bin.

    Histogram axes and covariance use the same order. This never enters scores.
    """
    kernel_cov=.2**2*np.asarray(covariance)+np.eye(2)*.45**2
    radius=int(np.ceil(4*np.sqrt(np.max(np.diag(kernel_cov)))))
    x,y=np.mgrid[-radius:radius+1,-radius:radius+1]
    pos=np.stack([x,y],axis=-1)
    kernel=np.exp(-.5*np.einsum('...i,ij,...j->...',pos,np.linalg.inv(kernel_cov),pos))
    kernel/=kernel.sum()
    result=np.maximum(0,fftconvolve(np.pad(histogram,radius,mode='symmetric'),kernel,mode='valid'))
    return result*histogram.sum()/result.sum()


def levels(hist):
    ordered=np.sort(hist.ravel())[::-1];cdf=np.cumsum(ordered)/ordered.sum()
    return sorted([ordered[min(np.searchsorted(cdf,p),len(ordered)-1)] for p in (.5,.9)])


def corner_arrays(name, n, view, history):
    """Evaluate frozen fits on larger joint designs for smooth display only."""
    sys.path.insert(0,str(BASE));import paper_campaign as campaign
    hashes=read(INPUTS/'input_hashes.json')
    def source_npz(path):
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest==hashes[str(path.relative_to(INPUTS))]
        SOURCES[str(path.relative_to(REPO))]=digest
        return dict(np.load(path))
    final=history[name]['rows'][-1]
    checkpoint=OUT/'corner_checkpoints'/name/f"checkpoint_{final['labels']}.npz"
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==final['checkpoint_sha256']
    SOURCES[str(checkpoint.relative_to(REPO))]=final['checkpoint_sha256']
    state,_,_,_=campaign.restore(checkpoint,final['labels'])
    cat=source_npz(INPUTS/'catalogues'/f'count_n{n}_{view}_c200.npz')
    arrays=[]
    for scramble in (0,1):
        cache=OUT/f'corner_{n}_{view}_{scramble}.npz'
        if not cache.exists():
            ref=source_npz(INPUTS/'references'/f't0_n{n}_{scramble}.npz')
            ids=np.random.default_rng(202609180+scramble).choice(len(ref['phi']),32768,replace=False)
            phi,rates,logq=ref['phi'][ids],ref['rates'][ids],ref['logq'][ids]
            duration=float(cat['duration']);events=len(cat['coeff'])
            a,aq=campaign.study.amplitude_design(duration*rates.sum(axis=1)/.012,events,16,202609190+scramble)
            repeated=np.repeat(phi,16,axis=0)
            points=np.column_stack([repeated[:,:2],a.ravel(),repeated[:,2]])
            predicted=campaign.study.pilot.base.predict(state,(points-LOW)/WIDTH)
            eventlog=np.concatenate([np.log(chunk@cat['coeff'].T).sum(axis=1) for chunk in np.array_split(rates,32)])
            exact=(eventlog[:,None]+events*np.log(duration*a/.012)-duration*rates.sum(axis=1)[:,None]*a/.012).ravel()
            np.savez_compressed(cache,points=points,predicted=predicted,exact=exact,logq=np.repeat(logq,16)+aq.ravel())
            print('Corner reconstructed',name,scramble,flush=True)
        SOURCES[str(cache.relative_to(REPO))]=hashlib.sha256(cache.read_bytes()).hexdigest()
        arrays.append(dict(np.load(cache)))
    return arrays


def symmetric_gp_kl(logp, logq):
    """Symmetric KL on a shared uniform integration design, in nats."""
    p=np.asarray(logp)-logsumexp(logp)
    q=np.asarray(logq)-logsumexp(logq)
    return float(.5*(np.exp(p)@(p-q)+np.exp(q)@(q-p)))


def gp_only_evolution(name, history):
    """Compare GP posteriors using two uniform Sobol designs, no exact target."""
    from scipy.stats import qmc
    from cosmic_integration.lnl_surrogate.stopping import posterior_summary
    sys.path.insert(0,str(BASE));import paper_campaign as campaign
    folder=OUT/'gp_only_evolution'/name;folder.mkdir(parents=True,exist_ok=True)
    final=history[name]['rows'][-1]
    finalpath=OUT/'corner_checkpoints'/name/f"checkpoint_{final['labels']}.npz"
    assert hashlib.sha256(finalpath.read_bytes()).hexdigest()==final['checkpoint_sha256']
    checkpoints={int(p.stem.split('_')[-1]):p for p in (OUT/'early_checkpoints'/name).glob('checkpoint_*.npz')}
    checkpoints[final['labels']]=finalpath
    with np.load(finalpath) as f:final_unit=f['unit'];final_lnl=f['lnl']
    initial_unit,initial_lnl=final_unit[:50],final_lnl[:50]
    for n,path in checkpoints.items():
        with np.load(path) as f:
            np.testing.assert_array_equal(f['unit'],final_unit[:n])
            np.testing.assert_array_equal(f['lnl'],final_lnl[:n])
    for n in (25,50):
        path=folder/f'checkpoint_{n}.npz'
        if not path.exists():
            state=campaign.fit(initial_unit[:n],initial_lnl[:n],n)
            campaign.save(path,state,initial_unit[:n],initial_lnl[:n],np.random.default_rng(17))
        checkpoints[n]=path
    units=[qmc.Sobol(4,scramble=True,seed=202609231+s).random_base2(18) for s in (0,1)]
    predictions={};summaries={}
    for n,path in sorted(checkpoints.items()):
        sha=hashlib.sha256(path.read_bytes()).hexdigest();SOURCES[str(path.relative_to(REPO))]=sha
        cache=folder/f'predictions_{n}.npz'
        if cache.exists():
            with np.load(cache) as f:
                assert str(f['checkpoint_sha256'])==sha
                values=f['predictions']
        else:
            state,_,_,_=campaign.restore(path,n)
            values=np.array([campaign.study.pilot.base.predict(state,u) for u in units])
            np.savez_compressed(cache,predictions=values,checkpoint_sha256=sha)
            print(name,n,'GP-only design evaluated',flush=True)
        predictions[n]=values
        summaries[n]=[posterior_summary(LOW+WIDTH*u,v) for u,v in zip(units,values)]
    final_values=predictions[final['labels']]
    curve=[]
    for n,values in sorted(predictions.items()):
        if n<50 or n==final['labels']:continue
        kls=[]
        for v,last in zip(values,final_values):
            kls.append(symmetric_gp_kl(v,last))
        if abs(kls[0]-kls[1])>max(.001,.1*max(kls)) or min(s['ess'] for s in summaries[n])<100:
            raise ValueError(f'GP KL integration unresolved: {name} {n} {kls}')
        curve.append(dict(labels=n,kl_designs=kls,ess=[s['ess'] for s in summaries[n]]))
    checks={}
    for n in (25,final['labels']):
        change=campaign.study.pilot.compare(*summaries[n])
        checks[n]=dict(change=change,ess=[x['ess'] for x in summaries[n]])
        if max(change['mean_sd'],change['endpoint_sd'])>.05 or change['width_fraction']>.03 or min(checks[n]['ess'])<100:
            raise ValueError(f'GP-only plotting integration unresolved: {name} {n} {checks[n]}')
    report=dict(initial=25,illustrative=[25,50],production_initial=144,final=final['labels'],curve=curve,checks=checks,
                integration='Two scrambled uniform Sobol designs, 2**18 unit-prior points each; no direct likelihood evaluations')
    (folder/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    points=LOW+WIDTH*np.concatenate(units)
    weights=[np.concatenate([np.exp(v-logsumexp(v))/2 for v in predictions[n]]) for n in (25,final['labels'])]
    return points,weights,report


def corners(history):
    checks=[]
    for view in ('perfect','uncertain'):
        fig=plt.figure(figsize=(12,8.6));gs=fig.add_gridspec(6,9,height_ratios=[1,1,1,1,.36,.9],width_ratios=[1,1,1,1,.45,1,1,1,1],hspace=.08,wspace=.10)
        for panel,n in enumerate((100,1000)):
            name=f'count_n{n}_{view}_c200_s17_bo'
            points,targetweights,report=gp_only_evolution(name,history);checks.append(report)
            fig.text(.30 if panel==0 else .73,.94,rf'$\mathbb{{E}}[N_{{\rm obs}}]={n}$ — {view} measurements',ha='center',fontsize=11)
            unit=(points-LOW)/WIDTH
            covariances=[np.cov(unit.T,aweights=w,ddof=0) for w in targetweights]
            truth=(np.array([-.3,.35,.01,4.7])-LOW)/WIDTH
            for row in range(4):
                for col in range(row+1):
                    ax=fig.add_subplot(gs[row,panel*5+col])
                    if row==col:
                        for idx,(w,color,style) in enumerate(zip(targetweights,('.55',BLUE),('--','-'))):
                            h,e=np.histogram(unit[:,col],bins=80,range=(0,1),weights=w,density=True)
                            sigma=np.sqrt(.2**2*covariances[idx][col,col]*80**2+.45**2)
                            h=gaussian_filter(h,sigma,mode='reflect')
                            if idx==1:ax.fill_between((e[:-1]+e[1:])/2,0,h,color=BLUE,alpha=.18)
                            ax.plot((e[:-1]+e[1:])/2,h,color=color,ls=style,lw=1.5 if idx==1 else 1.)
                        ax.set_yticks([])
                    else:
                        for idx,(w,color,style) in enumerate(zip(targetweights,('.55',BLUE),('--','-'))):
                            h,ex,ey=np.histogram2d(unit[:,col],unit[:,row],bins=64,range=[[0,1],[0,1]],weights=w)
                            h=smooth_display_density(h,covariances[idx][np.ix_([col,row],[col,row])]*64**2);lev=levels(h)
                            x,y=(ex[:-1]+ex[1:])/2,(ey[:-1]+ey[1:])/2
                            if idx==1:ax.contourf(x,y,h.T,levels=[*lev,h.max()*1.001],colors=['#d1e2ed','#82b1cc'],zorder=1)
                            ax.contour(x,y,h.T,levels=lev,colors=color,linestyles=style,linewidths=1.,zorder=3 if idx==0 else 2)
                        ax.set_ylim(0,1)
                    ax.axvline(truth[col],color='.15',lw=.8,zorder=4)
                    if row!=col:ax.axhline(truth[row],color='.15',lw=.8,zorder=4)
                    else:ax.set_ylim(bottom=0)
                    ax.set_xlim(0,1);ticks=[.2,.6];ax.set_xticks(ticks)
                    if row==3:
                        ax.set_xticklabels([f'{(LOW[col]+WIDTH[col]*t)*(100 if col==2 else 1):.2g}' for t in ticks],fontsize=7);ax.set_xlabel(r'$10^2a$' if col==2 else LABELS[col])
                    else:ax.set_xticklabels([])
                    if col==0 and row>0:
                        ax.set_yticks(ticks);ax.set_yticklabels([f'{(LOW[row]+WIDTH[row]*t)*(100 if row==2 else 1):.2g}' for t in ticks],fontsize=7);ax.set_ylabel(r'$10^2a$' if row==2 else LABELS[row])
                    elif row!=col:ax.set_yticks(ticks);ax.set_yticklabels([])
            ax=fig.add_subplot(gs[5,panel*5:panel*5+4])
            curve=report['curve'];x=[r['labels'] for r in curve];values=np.array([r['kl_designs'] for r in curve])
            ax.plot(x,values.mean(axis=1),'-o',color=BLUE,lw=1.4,ms=3)
            ax.fill_between(x,values.min(axis=1),values.max(axis=1),color=BLUE,alpha=.15,lw=0)
            ax.set(yscale='log',xlim=(50,report['final']),xlabel=r'$N_{\rm train}$',ylabel=r'$D_{\rm KL}^{\rm sym}$ [nat]')
            ax.set_xticks([50,300,600,900,1200] if n==100 else [50,500,1000,1500,1884])
            ax.tick_params(labelsize=8)
            ax.text(.98,.94,'Compared with final BO GP',ha='right',va='top',transform=ax.transAxes,fontsize=8)
        fig.legend(handles=[Line2D([],[],color='.55',ls='--',label='Initial GP: 25 points'),Patch(facecolor='#82b1cc',label='Final BO GP'),Line2D([],[],color='.15',label='Injected value')],loc='upper center',bbox_to_anchor=(.52,.045),ncol=3,frameon=False)
        save(fig,f'posterior_stop_{view}')
    (OUT/'corner_validation.json').write_text(json.dumps(checks,indent=2)+'\n')


def sample_joint_intensities(rates,amplitude,logweights,duration,draws,seed):
    """Draw rate intensities from an already evaluated joint GP, retaining A."""
    rng=np.random.default_rng(seed);w=np.exp(logweights-logsumexp(logweights))
    indices=rng.choice(len(amplitude),draws,p=w)
    replicas=len(amplitude)//len(rates)
    return duration*rates[indices//replicas]*(amplitude[indices,None]/.012)


def replicate_intensities(intensities,lower,upper,mode,seed):
    rng=np.random.default_rng(seed);numbers=rng.poisson(intensities.sum(axis=1));replicas=[]
    for rate,n in zip(intensities,numbers):
        bins=rng.choice(len(lower),n,p=rate/rate.sum());latent=rng.uniform(lower[bins],upper[bins])
        rho=12*rng.random(n)**(-1/3)
        noise=np.array([.03,.3])[None,:]*(12/rho[:,None])*rng.normal(size=(n,2))
        replicas.append(latent if mode=='perfect' else np.exp(np.log(latent)+noise))
    return replicas,numbers


def predictive():
    sys.path.insert(0,str(BASE));import paper_campaign as campaign
    from cosmic_integration.lnl_surrogate.stopping import posterior_summary
    gridpath=REPO/'docs/studies/event_uncertainty/paired_20260912/pilot/grid.npz'
    SOURCES[str(gridpath.relative_to(REPO))]=hashlib.sha256(gridpath.read_bytes()).hexdigest()
    with np.load(gridpath) as g:lower,upper,truth=g['lower'],g['upper'],g['truth_rates']
    input_hashes=read(INPUTS/'input_hashes.json')
    np.testing.assert_allclose(campaign.RateEngine(INPUTS)([-.3,.35,.01,4.7]),truth,rtol=1e-10,atol=1e-15)
    def verified_npz(path):
        h=hashlib.sha256(path.read_bytes()).hexdigest();assert h==input_hashes[str(path.relative_to(INPUTS))]
        SOURCES[str(path.relative_to(REPO))]=h
        return dict(np.load(path))
    for n in (30,1200):
        for mode in ('perfect','uncertain'):
            name=f'predictive_n{n}_{mode}_s17_bo';folder=ROOT/'production/runs'/name
            resultpath=OUT/f'predictive_{n}_{mode}.json'
            if resultpath.exists():print('Reuse',name,flush=True);continue
            complete=read(folder/'complete.json');checkpoint=folder/f"checkpoint_{complete['labels']}.npz"
            assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==complete['checkpoint_sha256']
            SOURCES[str(checkpoint.relative_to(REPO))]=complete['checkpoint_sha256']
            state,_,_,_=campaign.restore(checkpoint,complete['labels'])
            cat=verified_npz(INPUTS/'catalogues'/f'predictive_n{n}_{mode}.npz');duration=float(cat['duration']);events=len(cat['coeff'])
            latest=read(folder/f"score_{complete['labels']}.json")
            draws=[];checks=[]
            for scramble in (0,1):
                cache=OUT/f'joint_intensities_{n}_{mode}_{scramble}.npz';metacache=OUT/f'joint_intensities_{n}_{mode}_{scramble}.json'
                if cache.exists() and metacache.exists():
                    draws.append(np.load(cache)['intensities']);checks.append(read(metacache));continue
                ref=verified_npz(INPUTS/'references'/f't0_n{100 if n==30 else 1000}_{scramble}.npz')
                rng=np.random.default_rng(202609180+scramble)
                indices=rng.choice(len(ref['phi']),min(32768,len(ref['phi'])),replace=False)
                phi,rates,logq=ref['phi'][indices],ref['rates'][indices],ref['logq'][indices]
                a,aq=campaign.study.amplitude_design(duration*rates.sum(axis=1)/.012,events,16,202609190+scramble)
                repeated=np.repeat(phi,16,axis=0);points=np.column_stack([repeated[:,:2],a.ravel(),repeated[:,2]])
                predicted=campaign.study.pilot.base.predict(state,(points-LOW)/WIDTH)
                q=np.repeat(logq,16)+aq.ravel();gp=posterior_summary(points,predicted-q)
                expected=latest['attempts'][-1]['scores'][scramble]['gp']
                change=campaign.study.pilot.compare(expected,gp)
                if change['mean_sd']>.05 or change['endpoint_sd']>.05 or change['width_fraction']>.03:raise ValueError(f'Unresolved predictive reconstruction: {name} {change}')
                sampled=sample_joint_intensities(rates,a.ravel(),predicted-q,duration,50000,202609200+scramble)
                metadata=dict(scramble=scramble,checkpoint_sha256=complete['checkpoint_sha256'],gp_summary=gp,change=change,draws=len(sampled),shape_points=len(indices))
                np.savez_compressed(cache,intensities=sampled);metacache.write_text(json.dumps(metadata,indent=2)+'\n')
                draws.append(sampled);checks.append(metadata)
                print(name,scramble,'reconstructed',change,flush=True)
            intensity=np.concatenate(draws);quantiles=np.quantile(intensity,[.05,.5,.95],axis=0)
            visible=quantiles[1]>=.001
            split=np.max(abs(np.quantile(draws[0],[.05,.5,.95],axis=0)[:,visible]-np.quantile(draws[1],[.05,.5,.95],axis=0)[:,visible])/quantiles[1,visible])
            if split>.1:raise ValueError(f'Rate-map precision unresolved: {name} {split}')
            np.savez_compressed(OUT/f'rate_posterior_{n}_{mode}.npz',quantiles=quantiles)
            reps=[];numbers=[]
            for scramble,d in enumerate(draws):
                rr,nn=replicate_intensities(d[:1000],lower,upper,mode,202609220+scramble)
                reps+=rr;numbers.extend(nn.tolist())
            observed=cat['latent'] if mode=='perfect' else np.exp(cat['y']);columns=[]
            nonempty=[r for r in reps if len(r)]
            for k in (0,1):
                endpoint=max(observed[:,k].max(),max(r[:,k].max() for r in nonempty));axis=np.linspace(0,endpoint*1.01,350)
                cdfs=np.array([np.searchsorted(np.sort(r[:,k]),axis,side='right')/len(r) for r in nonempty])
                splitdiff=np.max(abs(np.quantile(cdfs[:1000],[.05,.5,.95],axis=0)-np.quantile(cdfs[1000:],[.05,.5,.95],axis=0)))
                columns.append(dict(x=axis.tolist(),observed=(np.searchsorted(np.sort(observed[:,k]),axis,side='right')/events).tolist(),quantiles=np.quantile(cdfs,[.05,.5,.95],axis=0).tolist(),split_cdf_max_difference=float(splitdiff)))
            report=dict(expected=n,mode=mode,events=events,labels=complete['labels'],posterior_outcome=latest['outcome'],posterior_kl=latest['maximum_directed_kl'],checks=checks,max_rate_quantile_split=float(split),columns=columns,replicate_counts=numbers,draws=len(numbers),empty_replicates=sum(x==0 for x in numbers))
            resultpath.write_text(json.dumps(report,indent=2)+'\n')
    plot_predictive(lower,upper,truth)


def plot_predictive(lower,upper,truth):
    sys.path.insert(0,str(REPO/'docs/studies/paper_completion'));import weight_rate_figures as old
    for n in (30,1200):
        fig,axes=plt.subplots(2,3,figsize=(9,5),layout='constrained')
        for row,mode in enumerate(('perfect','uncertain')):
            data=read(OUT/f'predictive_{n}_{mode}.json')
            for k in (0,1):
                c=data['columns'][k];q=np.asarray(c['quantiles']);ax=axes[row,k]
                ax.fill_between(c['x'],q[0],q[2],color=BLUE,alpha=.17)
                ax.plot(c['x'],q[1],color=BLUE,lw=1.1,label='GP predictive median')
                ax.step(c['x'],c['observed'],where='post',color='.1',lw=1,label='Observed')
                ax.set_ylim(0,1)
                if k==0:ax.set_ylabel('Cumulative fraction');ax.legend(frameon=False,fontsize=7)
            axes[row,2].hist(data['replicate_counts'],bins=24,density=True,color=BLUE,alpha=.55)
            axes[row,2].axvline(data['events'],color='.1');axes[row,2].set_ylabel('Predictive density')
        for ax,label in zip(axes[-1],[r'Chirp mass $\mathcal{M}_c$ [$M_\odot$]','Redshift $z$',r'Detected count $N_{\rm obs}$']):ax.set_xlabel(label)
        save(fig,f'posterior_predictive_{n}')
    qs=[np.load(OUT/f'rate_posterior_30_{m}.npz')['quantiles'] for m in ('perfect','uncertain')]
    areas=np.prod(upper-lower,axis=1);boundary=old.boundary_segments(lower,upper,old.highest_density_bins(truth,areas))
    widths=[np.where(q[1]>=.001,50*(q[2]-q[0])/q[1],np.nan) for q in qs]
    vmax=10**np.ceil(np.log10(max(q[1].max() for q in qs)));umax=10*np.ceil(max(np.nanmax(w) for w in widths)/10)
    fig,axes=plt.subplots(2,2,figsize=(8,5.5),sharex=True,sharey=True,layout='constrained')
    for row,(q,w) in enumerate(zip(qs,widths)):
        im=old.draw_map(axes[row,0],lower,upper,np.where(q[1]>=.001,q[1],np.nan),LogNorm(.001,vmax))
        ui=old.draw_map(axes[row,1],lower,upper,w,Normalize(0,umax),'cividis')
        for ax in axes[row]:ax.add_collection(LineCollection(boundary,colors='black',linewidths=1))
        axes[row,0].set_ylabel('Redshift $z$')
    for ax in axes[-1]:ax.set_xlabel(r'Chirp mass $\mathcal{M}_c$ [$M_\odot$]')
    fig.colorbar(im,ax=list(axes[:,0]),fraction=.04,pad=.02,label='Median expected count per bin')
    fig.colorbar(ui,ax=list(axes[:,1]),fraction=.04,pad=.02,label='Relative 90% interval half-width [%]')
    save(fig,'paired_rate_posterior')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('part',choices=['summaries','corners','predictive','all']);args=p.parse_args()
    OUT.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'sans-serif','font.size':9,'axes.labelsize':10,'xtick.direction':'in','ytick.direction':'in','xtick.top':True,'ytick.right':True,'axes.spines.top':True,'axes.spines.right':True})
    h=histories()
    if args.part in ('summaries','all'):summaries(h)
    if args.part in ('corners','all'):corners(h)
    if args.part in ('predictive','all'):predictive()
    SOURCES[str(Path(__file__).relative_to(REPO))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT/f'{args.part}_sources.json').write_text(json.dumps(SOURCES,indent=2)+'\n')


if __name__=='__main__':main()
