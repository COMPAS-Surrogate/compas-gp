"""Posterior-first figures from frozen decisions; no new acquisition or tuning."""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from scipy.stats import gamma
from scipy.ndimage import gaussian_filter, gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = HERE/'presentation_20260911'
FIG = REPO/'overleaf/figures/bo_study'
ROOT = HERE.parent/'operational_stopping/gpry_compas'
FRESH = ROOT/'fresh_confirmation_20260911'
sys.path.insert(0, str(ROOT))
import score_scheduled_audit as validation
import plot_paper
BASE = validation.base
LOW, WIDTH = BASE.core.LOW, BASE.core.WIDTH
LABELS = [r'$\alpha$', r'$\sigma$', r'$10^2 a$', r'$d$']
SCALE = np.array([1, 1, 100, 1])
plt.rcParams.update({'font.size':10, 'pdf.fonttype':42, 'axes.spines.top':True,
    'axes.spines.right':True, 'xtick.top':True, 'ytick.right':True,
    'xtick.direction':'in', 'ytick.direction':'in'})


def draw_posterior(phi, beta, logw, events, seed=202609301):
    """Systematic shape resampling and exact bounded-Gamma amplitude draws."""
    rng = np.random.default_rng(seed)
    w = np.exp(logw-logsumexp(logw))
    cdf = np.cumsum(w); cdf[-1] = 1
    ix = np.searchsorted(cdf, (np.arange(131072)+rng.random())/131072)
    b = beta[ix]; shape = events+1
    u = rng.random(len(ix))
    lo, hi = gamma.cdf(.005,shape,scale=1/b), gamma.cdf(.015,shape,scale=1/b)
    sl, sh = gamma.sf(.005,shape,scale=1/b), gamma.sf(.015,shape,scale=1/b)
    a = np.where(lo < .5, gamma.ppf(lo+u*(hi-lo),shape,scale=1/b),
                 gamma.isf(sl-u*(sl-sh),shape,scale=1/b))
    samples = np.insert(phi[ix],2,a,axis=1)
    assert np.isfinite(samples).all() and np.all(samples>=LOW) and np.all(samples<=LOW+WIDTH)
    return samples


def prepare(mode):
    name = f't0_n1000_{mode}_c100_bo'
    decisions = json.loads((FRESH/'decisions.json').read_text())
    decision = decisions[name]; selected = decision['rules']['fixed96']
    assert selected is not None and not decision['reference_accessed']
    stop = selected['training']
    score = json.loads((FRESH/'offline'/f'{name}_{stop}.json').read_text())
    assert score['accuracy_pass']
    hashes = {}
    refs = validation.references(ROOT/'extension_v2',0,1000,hashes)
    catpath = FRESH/'training'/name/'catalogue.npz'
    cat = dict(np.load(catpath)); truth = cat['truth']
    checked = validation.score.direct_checks(refs,cat,truth)
    assert checked[-1]['passed']
    phi = np.concatenate([p for p,_,_ in refs])
    beta = np.concatenate([r.sum(1)*float(cat['duration'])/.012 for _,r,_ in refs])
    precision_checks = {}
    weights = {'direct':np.concatenate([w-logsumexp(w) for w in checked[0]])}
    for count in [118,stop]:
        p = FRESH/'training'/name/f'checkpoint_{count}.npz'
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        training_checkpoints = json.loads((FRESH/'training'/name/'checkpoints.json').read_text())
        if count == stop:
            assert sha == next(h['checkpoint_sha256'] for h in decision['history'] if h['training']==count)
        else:
            assert sha == next(h['sha256'] for h in training_checkpoints if h['points']==count)
        hashes[str(p.relative_to(REPO))] = sha
        state,_,_ = BASE.load_checkpoint(p)
        ws = [BASE.predict(state,(p-BASE.LOW)/BASE.WIDTH)-q for p,_,q in refs]
        summaries = [validation.score.four_summary(p,r,cat,truth,w) for (p,r,_),w in zip(refs,ws)]
        passed,diagnostics = BASE.data_source.precision(*summaries)
        assert passed, ('Unresolved plotting posterior', count, diagnostics)
        precision_checks[str(count)] = diagnostics
        weights[str(count)] = np.concatenate([w-logsumexp(w) for w in ws])
    samples = {k:draw_posterior(phi,beta,w,len(cat['coeff'])) for k,w in weights.items()}
    checks = {}
    for key,source in [('direct','direct'),(str(stop),'gp')]:
        target = score['four_summaries'][source][0]
        error = np.max(abs(samples[key].mean(0)-target['mean'])/target['sd'])
        q = np.quantile(samples[key],[.025,.16,.84,.975],axis=0)
        endpoints = np.asarray(target['quantiles'])[[0,2,4,6]]
        qerror = np.max(abs(q-endpoints)/target['sd'])
        assert error < .05 and qerror < .07, (key,error,qerror)
        checks[key] = dict(max_mean_shift_sd=float(error),max_endpoint_shift_sd=float(qerror))
    hashes[str(catpath.relative_to(REPO))] = hashlib.sha256(catpath.read_bytes()).hexdigest()
    hashes[str((FRESH/'decisions.json').relative_to(REPO))] = hashlib.sha256((FRESH/'decisions.json').read_bytes()).hexdigest()
    meta = dict(name=name,selection='First fresh catalogue at fiducial injection and 1000 expected events, same latent events for both sensors',
        events=len(cat['coeff']),expected=1000,mode=mode,selected=selected,history=decision['history'],
        independent_kl=score['maximum_directed_kl'],accuracy_pass=score['accuracy_pass'],truth=truth.tolist(),
        sampling_checks=checks,posterior_precision_checks=precision_checks,source_sha256=hashes,draws=131072,
        smoothing='64 equal bins per prior dimension; Gaussian sigma 0.8 bins; reflecting boundaries; 50/90 percent contours')
    np.savez_compressed(OUT/f'{mode}_samples.npz',**samples)
    (OUT/f'{mode}.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(mode,selected,checks,flush=True)
    return samples,meta


def contour_levels(h):
    ordered = np.sort(h.ravel())[::-1]
    mass = np.cumsum(ordered)/ordered.sum()
    return np.sort([ordered[min(np.searchsorted(mass,p),len(ordered)-1)] for p in [.5,.9]])


def corner(samples, meta):
    fig,axes = plt.subplots(4,4,figsize=(7.1,6.7))
    fig.subplots_adjust(left=.10,bottom=.09,right=.985,top=.97,wspace=.11,hspace=.10)
    stop = meta['selected']['training']
    styles = [('direct','.35','-',1.4),('118','#D55E00','--',1.),(str(stop),'#0072B2','-',1.3)]
    # Both measurement views use identical full-prior axes and density smoothing.
    for i in range(4):
        for j in range(4):
            ax = axes[i,j]
            if j>i: ax.set_visible(False); continue
            for key,color,ls,lw in styles:
                s = samples[key]*SCALE
                ranges = [(LOW[k]*SCALE[k],(LOW[k]+WIDTH[k])*SCALE[k]) for k in range(4)]
                if i==j:
                    h,edges = np.histogram(s[:,i],64,range=ranges[i],density=True)
                    x = (edges[:-1]+edges[1:])/2; h=gaussian_filter1d(h,.8,mode='reflect')
                    if key=='direct':ax.fill_between(x,0,h,color='.85')
                    ax.plot(x,h,color=color,ls=ls,lw=lw)
                    ax.set_yticks([])
                else:
                    h,xe,ye = np.histogram2d(s[:,j],s[:,i],64,range=[ranges[j],ranges[i]],density=True)
                    h = gaussian_filter(h.T,.8,mode='reflect')
                    x,y = (xe[:-1]+xe[1:])/2,(ye[:-1]+ye[1:])/2
                    ax.contour(x,y,h,levels=contour_levels(h),colors=color,linestyles=ls,linewidths=lw)
                    ax.set_ylim(ranges[i])
                ax.set_xlim(ranges[j])
            if i==j:
                ax.relim();ax.autoscale_view(scalex=False,scaley=True);ax.set_ylim(bottom=0)
            ax.axvline(meta['truth'][j]*SCALE[j],color='.55',ls=':',lw=.8)
            if i>j:ax.axhline(meta['truth'][i]*SCALE[i],color='.55',ls=':',lw=.8)
            ax.xaxis.set_major_locator(MaxNLocator(3)); ax.yaxis.set_major_locator(MaxNLocator(3))
            if i==j:ax.set_yticks([])
            ax.minorticks_on();ax.tick_params(which='both',direction='in',top=True,right=True,labelsize=8)
            if i==3:ax.set_xlabel(LABELS[j])
            else:ax.tick_params(labelbottom=False)
            if j==0 and i>0:ax.set_ylabel(LABELS[i])
            elif j>0:ax.tick_params(labelleft=False)
    handles=[Line2D([],[],color=c,ls=ls,lw=lw,label=label) for (_,c,ls,lw),label in zip(styles,
        ['Direct posterior','BO: 118 training',f'BO stop: {stop} training'])]
    handles.append(Line2D([],[],color='.55',ls=':',label='Injection'))
    fig.legend(handles=handles,loc='upper right',bbox_to_anchor=(.98,.995),frameon=False,fontsize=9)
    audit=fig.add_axes([.64,.74,.32,.12]); stability=fig.add_axes([.64,.57,.32,.115])
    history=meta['history']; x=[h['training'] for h in history]
    results=[h['stages'][0]['result'] for h in history]
    audit.plot(x,[r['maximum_directed_kl']/.003 for r in results],'-o',ms=3,color='#009E73',label='KL / 0.003')
    audit.plot(x,[r['bootstrap_95_kl']/.005 for r in results],'--s',ms=3,color='#CC79A7',label='95% / 0.005')
    audit.plot(x,[32/min(r['gp_ess'],r['direct_ess']) for r in results],':',color='.4',label='32 / ESS')
    audit.axhline(1,color='k',lw=.8);audit.set(yscale='log',ylabel='Audit / limit')
    audit.legend(fontsize=6.7,frameon=False,ncol=1,loc='upper right')
    streak=[h['diagnostics']['streaks']['joint'] for h in history]
    stability.plot(x,streak,'o-',color='#0072B2',ms=3)
    stability.axhline(2,color='k',lw=.8);stability.set(ylabel='Stable checks',xlabel='Training evaluations',ylim=(-.3,max(3,max(streak)+.4)))
    stability.yaxis.set_major_locator(MaxNLocator(integer=True,nbins=3))
    for ax in [audit,stability]:
        ax.axvline(stop,color='.25',ls='--',lw=.9)
        ax.set_xticks([238,358,stop]);ax.tick_params(which='both',direction='in',top=True,right=True,labelsize=7)
        ax.xaxis.label.set_size(8);ax.yaxis.label.set_size(8)
    audit.tick_params(labelbottom=False)
    fig.savefig(FIG/f'posterior_stop_{meta["mode"]}.pdf',bbox_inches='tight')
    fig.savefig(FIG/f'posterior_stop_{meta["mode"]}.png',dpi=180,bbox_inches='tight');plt.close(fig)


def completion_quantiles(rows: list[dict]) -> np.ndarray:
    """Empirical quantiles retaining right-censored cases in the denominator."""
    if len(rows) != 100 or len({r['case'] for r in rows}) != 100:
        raise ValueError('Expected 100 distinct catalogues')
    if any(r['state'] not in ['reference_verified_convergence', 'resource_limit'] for r in rows):
        raise ValueError('Unresolved references cannot define completion quantiles')
    values = [r['points'] if r['state']=='reference_verified_convergence' else np.inf for r in rows]
    q = np.quantile(values,[.1,.5,.9],method='inverted_cdf')
    if not np.isfinite(q).all():
        raise ValueError('Reported quantiles are not resolved before censoring')
    return q


def efficiency():
    rows=plot_paper.ensemble_rows(); fig,axes=plt.subplots(1,2,figsize=(7.1,3.1),sharey=True,layout='constrained')
    summaries=[]
    for ax,mode,counts in zip(axes,['perfect','uncertain'],[[30,100,300,1000],[50,500]]):
        for strategy,color,ls in [('bo','#0072B2','-'),('random','#D55E00','--')]:
            qs=[]
            for n in counts:
                rr=[r for r in rows if (r['mode'],r['exposure'],r['strategy'])==(mode,n,strategy)]
                assert len(rr)==100 and len({r['case'] for r in rr})==100
                assert all(r['state'] in ['reference_verified_convergence','resource_limit'] for r in rr)
                values=[r['points'] if r['state']=='reference_verified_convergence' else np.inf for r in rr]
                q=completion_quantiles(rr)
                qs.append(q);summaries.append(dict(mode=mode,expected=n,strategy=strategy,quantiles=q.tolist(),censored=sum(not np.isfinite(v) for v in values),denominator=100))
            q=np.asarray(qs)
            ax.plot(counts,q[:,1],ls,marker='o',ms=4,color=color,label='BO' if strategy=='bo' else 'Random')
            ax.fill_between(counts,q[:,0],q[:,2],color=color,alpha=.12)
        ax.set_xscale('log');ax.set_yscale('log');ax.set_xticks(counts,counts);ax.set_xlabel('Expected observed events');ax.minorticks_on()
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter());ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[0].set_ylabel('Training evaluations');axes[0].legend(frameon=False)
    for ax in axes:ax.set_yticks([226,500,1000,2000,4000],[226,500,1000,2000,4000]);ax.set_ylim(190,4500)
    fig.savefig(FIG/'acquisition_summary.pdf',bbox_inches='tight');fig.savefig(FIG/'acquisition_summary.png',dpi=180,bbox_inches='tight');plt.close(fig)
    (OUT/'efficiency.json').write_text(json.dumps(summaries,indent=2)+'\n')


def main():
    OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
    for mode in ['perfect','uncertain']:
        if '--plots-only' in sys.argv:
            samples=dict(np.load(OUT/f'{mode}_samples.npz'));meta=json.loads((OUT/f'{mode}.json').read_text())
        else:
            samples,meta=prepare(mode)
        corner(samples,meta)
    efficiency()
    source_hashes = dict(plot_paper.SOURCES)
    for mode in ['perfect','uncertain']:
        source_hashes.update(json.loads((OUT/f'{mode}.json').read_text())['source_sha256'])
    (OUT/'sources.json').write_text(json.dumps(source_hashes,indent=2)+'\n')
    (OUT/'source_sha256.txt').write_text(hashlib.sha256(Path(__file__).read_bytes()).hexdigest()+'\n')

if __name__=='__main__':main()
