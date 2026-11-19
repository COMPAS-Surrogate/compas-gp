"""Reconstruct posterior evolution from saved BO checkpoints; no new training labels.

Uses both existing importance-reference scrambles for the first N=300 catalogue.
Full parameter draws combine GP shape weights with the exact amplitude conditional.
"""
import json
import sys
import numpy as np
from scipy.special import logsumexp
from scipy.stats import gamma
from scipy.ndimage import gaussian_filter
from matplotlib.lines import Line2D
from run_study import (HERE,LOW,WIDTH,SHAPE,vector_marginal,AdaptiveRobustScaler,
                       JaxTrainingData,JaxGPConfig,fit_exact_gp)
from publish_figures import plt,save,LABELS,BLUE
sys.path.insert(0, str(HERE.parent/'event_uncertainty'))
from posterior_convergence import reference_means, event_target

BUDGETS=[32,128,482]
COLORS=['#3569a8','#bb7434','#9563b5']
TRACE_BUDGETS=[32,128,226,354,482,610,738]


def largest_kl(a, b, lengths):
    """Maximum directed KL across the two independent importance references."""
    values=[]
    for x,y in zip(np.split(a, np.cumsum(lengths)[:-1]),
                   np.split(b, np.cumsum(lengths)[:-1])):
        x=x-logsumexp(x);y=y-logsumexp(y)
        values += [float(np.exp(x)@(x-y)), float(np.exp(y)@(y-x))]
    return max(values)

def density2d(samples,i,j,bins=45):
    h,xe,ye=np.histogram2d(samples[:,i],samples[:,j],bins=bins,
        range=[(LOW[i],LOW[i]+WIDTH[i]),(LOW[j],LOW[j]+WIDTH[j])],density=True)
    h=gaussian_filter(h,1,mode='reflect')
    return (xe[:-1]+xe[1:])/2,(ye[:-1]+ye[1:])/2,h.T

def levels(h):
    ordered=np.sort(h.ravel())[::-1];c=np.cumsum(ordered)/ordered.sum()
    return sorted([ordered[min(np.searchsorted(c,p),len(c)-1)] for p in [.9,.5]])

def main():
    global BUDGETS, TRACE_BUDGETS
    root=HERE/'results'; n=300;case=0
    data=np.load(root/'catalogues.npz');occupied=np.any(data['counts']>0,axis=0)
    counts=data['counts'][data['expected_events']==n][case,occupied];mask=counts>0
    select=np.r_[np.flatnonzero(mask),len(counts)];cnt=counts[mask]
    duration=json.loads((root/'physics.json').read_text())[str(n)]['duration']
    refs=[np.load(root/f'reference_{n}_{s}.npz') for s in [0,1]]
    phi=np.concatenate([r['phi'] for r in refs]);logq=np.concatenate([r['logq'] for r in refs])
    rates=np.concatenate([r['rates'][:,select] for r in refs])*duration/.1
    exact,beta=vector_marginal(rates,cnt)
    def draws(loglike):
        w=np.exp(loglike-logq-logsumexp(loglike-logq))
        rng=np.random.default_rng(20260909)
        ix=np.searchsorted(np.cumsum(w),(np.arange(65536)+rng.random())/65536)
        b=beta[ix];k=cnt.sum()+1;u=rng.random(len(ix))
        lo=gamma.cdf(.005,k,scale=1/b);hi=gamma.cdf(.015,k,scale=1/b)
        sl=gamma.sf(.005,k,scale=1/b);sh=gamma.sf(.015,k,scale=1/b)
        a=np.where(lo<.5,gamma.ppf(lo+u*(hi-lo),k,scale=1/b),gamma.isf(sl-u*(sl-sh),k,scale=1/b))
        result=np.insert(phi[ix],2,a,axis=1)
        assert np.isfinite(result).all() and np.all(result>=LOW) and np.all(result<=LOW+WIDTH)
        return result,float(1/np.sum(w*w))
    samples={};ess={};samples['reference'],ess['reference']=draws(exact)
    extension=HERE.parent/'event_uncertainty/corner_extension_20260910'
    outcome=None
    if (extension/'outcomes.json').exists():
        outcome=next((r for r in json.loads((extension/'outcomes.json').read_text())
                      if r['strategy']=='bo' and r['state']=='reference_verified_convergence'),None)
    trace_phi,trace_logq,trace_exact=phi,logq,exact
    lengths=[len(r['phi']) for r in refs]
    if outcome is not None:
        BUDGETS=[32,128,outcome['points']]
        TRACE_BUDGETS=[32,128]+list(range(226,outcome['points']+1,128))
        power=next(r['power'] for r in json.loads((extension/'references.json').read_text()) if r['numerical_pass'])
        er=[np.load(extension/f'reference_n300_p{power}_{s}.npz') for s in [0,1]]
        eq=[np.load(extension/f'proposal_n300_p{power}_{s}.npz') for s in [0,1]]
        cat=np.load(extension/'catalogue_300_perfect_0.npz')
        assert len(cat['coeff'])==int(cnt.sum())
        trace_phi=np.concatenate([r['phi'] for r in er])
        trace_logq=np.concatenate([q['logq'] for q in eq])
        trace_exact=np.concatenate([event_target(r['rates'],cat['coeff'],float(cat['duration'])) for r in er])
        lengths=[len(r['phi']) for r in er]
        old=np.load(root/'n300_case0_bo/checkpoint_738.npz')
        new=np.load(extension/'n300_perfect_case0_bo/checkpoint_738.npz')
        np.testing.assert_allclose(old['x'],new['x'],rtol=0,atol=1e-14)
        assert np.ptp(old['lnl']-new['lnl'])<1e-8, 'Extension changed the likelihood beyond its data-only offset'
    trace=[]; previous=None
    for budget in TRACE_BUDGETS:
        folder=root/f'n{n}_case{case}_bo' if outcome is None else extension/'n300_perfect_case0_bo'
        train=np.load(folder/f'checkpoint_{max(budget,226)}.npz');x,y=train['x'][:budget],train['lnl'][:budget]
        scaler=AdaptiveRobustScaler(compression='sqrt',soft_clipping=False,lower_clip_value=float(y.min()),max_scale=1e9);scaler.initialize_with_data(y)
        model=fit_exact_gp(JaxTrainingData(x,-scaler.transform(y)[:,None]),np.array([LOW[SHAPE],(LOW+WIDTH)[SHAPE]]),config=JaxGPConfig(optimisation_steps=250),seed=17)
        pred=scaler.inverse_transform(-reference_means(model, trace_phi))
        trace.append({'points':budget,'reference_kl':largest_kl(trace_exact-trace_logq,pred-trace_logq,lengths),
                      'previous_kl':None if previous is None else largest_kl(previous-trace_logq,pred-trace_logq,lengths)})
        previous=pred
        if budget in BUDGETS:
            display_pred=scaler.inverse_transform(-reference_means(model,phi))
            samples[str(budget)],ess[str(budget)]=draws(display_pred)
    records=json.loads((root/'validated_records.json').read_text())
    if outcome is not None:
        extension_records=json.loads((extension/'records.json').read_text())
        for row in trace:
            if row['points']>=226:
                saved=next(r for r in extension_records if r['strategy']=='bo' and r['points']==row['points'])
                target=max(m[k] for m in saved['metrics'] for k in ['kl_direct_to_gp','kl_gp_to_direct'])
                np.testing.assert_allclose(row['reference_kl'],target,rtol=1e-6,atol=1e-9)
    for budget in [b for b in BUDGETS if b>=226]:
        if outcome is None:
            target=next(r for r in records if r['expected_events']==n and r['case']==case and r['strategy']=='bo' and r['points']==budget)['summaries'][0]
        else:
            target=next(r for r in extension_records if r['strategy']=='bo' and r['points']==budget)['summaries'][0]
        assert np.max(abs(samples[str(budget)].mean(0)-target['mean'])/target['sd'])<.05, 'Posterior reconstruction disagrees with validated means'
    np.savez(root/'posterior_evolution_samples.npz',**samples)
    (root/'posterior_evolution_convergence.json').write_text(json.dumps({
        'expected_events':n,'observed_events':int(cnt.sum()),'case':case,
        'strategy':'bo','trace':trace,'benchmark_outcome':outcome,
        'scope':'Unsmoothed importance weights; maximum of both directed KLs and both reference scrambles. Initialization prefixes are included. KL alone is not the full stopping rule.'},indent=2)+'\n')
    (root/'posterior_evolution.json').write_text(json.dumps({'expected_events':n,'observed_events':int(cnt.sum()),'case':case,'budgets':BUDGETS,'ess':ess,'draws':65536,'seed':20260909,'smoothing':'45 bins per prior dimension, Gaussian sigma=1 bin, reflecting boundaries','scope':'32 and 128 are initialization-prefix diagnostics without boundary anchors or BO acquisitions; the final panel is a saved BO stage. Not part of the matched-budget comparison.'},indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(7.1,2.6),sharex=True,sharey=True,layout='constrained')
    grid=[density2d(samples[str(b)],0,1) for b in BUDGETS];vmax=max(h.max() for x,y,h in grid)
    xr,yr,hr=density2d(samples['reference'],0,1)
    for ax,(x,y,h) in zip(axes,grid):
        mesh=ax.pcolormesh(x,y,h,cmap='Blues',shading='nearest',vmin=0,vmax=vmax,rasterized=True)
        ax.contour(xr,yr,hr,levels=levels(hr),colors='.25',linewidths=.9,linestyles=['--','-'])
        ax.set(xlabel=LABELS[0],xlim=(LOW[0],LOW[0]+WIDTH[0]),ylim=(LOW[1],LOW[1]+WIDTH[1]))
    axes[0].set_ylabel(LABELS[1]);fig.colorbar(mesh,ax=axes,label='Posterior density',shrink=.85)
    save(fig,'posterior_surfaces')
    fig,axes=plt.subplots(4,4,figsize=(7.1,7.1))
    fig.subplots_adjust(left=.10,bottom=.09,right=.98,top=.98,hspace=.08,wspace=.16)
    for i in range(4):
        for j in range(4):
            ax=axes[i,j]
            if j>i:ax.set_visible(False);continue
            if i==j:
                ax.hist(samples['reference'][:,i],bins=50,range=(LOW[i],LOW[i]+WIDTH[i]),density=True,histtype='stepfilled',color='.85')
                for b,c in zip(BUDGETS,COLORS):ax.hist(samples[str(b)][:,i],bins=50,range=(LOW[i],LOW[i]+WIDTH[i]),density=True,histtype='step',color=c,lw=1.2)
                ax.set_yticks([])
            else:
                x,y,h=density2d(samples['reference'],j,i);ax.contourf(x,y,h,levels=[levels(h)[0],h.max()*1.001],colors=['.9'])
                for b,c in zip(BUDGETS,COLORS):
                    x,y,h=density2d(samples[str(b)],j,i);ax.contour(x,y,h,levels=levels(h),colors=[c],linewidths=1,linestyles=['--','-'])
                ax.set_ylim(LOW[i],LOW[i]+WIDTH[i])
            ax.set_xlim(LOW[j],LOW[j]+WIDTH[j]);ax.locator_params(axis='x',nbins=3)
            if i==3:ax.set_xlabel(LABELS[j]);ax.tick_params(axis='x',labelrotation=45,labelsize=7,pad=3)
            else:ax.tick_params(labelbottom=False)
            if j==0 and i>0:ax.set_ylabel(LABELS[i])
            elif j>0:ax.tick_params(labelleft=False)
    handles=[Line2D([],[],color=c,label=f'{b} training points') for b,c in zip(BUDGETS,COLORS)]+[Line2D([],[],color='.7',lw=5,label='Reference')]
    fig.legend(handles=handles,loc='upper right',bbox_to_anchor=(.95,.985),fontsize=9)
    inset=fig.add_axes([.64,.615,.32,.225])
    inset.plot(TRACE_BUDGETS,[r['reference_kl'] for r in trace],color='.2',marker='o',ms=3,label='To reference')
    inset.plot(TRACE_BUDGETS[1:],[r['previous_kl'] for r in trace[1:]],color='#398653',ls='--',marker='s',ms=3,label='To previous fit')
    for threshold in [.01,.001]:
        inset.axhline(threshold,color='.5',ls=':',lw=.8)
    inset.set(yscale='log',xlabel='Training evaluations',ylabel=r'Max. directed $D_{\rm KL}$')
    inset.set_xticks([32,354,738,TRACE_BUDGETS[-1]] if outcome else [32,226,482,738]);inset.tick_params(labelsize=7)
    inset.xaxis.label.set_size(8);inset.yaxis.label.set_size(8)
    inset.legend(fontsize=7,loc='upper right')
    save(fig,'posterior_corner')

if __name__=='__main__':main()
