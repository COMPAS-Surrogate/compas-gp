"""Posterior-first figures with checked dense-GP overlays and frozen BO stops."""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from scipy.stats import gamma
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.signal import fftconvolve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
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


def prepare(mode, expected=1000):
    name = f't0_n{expected}_{mode}_c100_bo'
    decisions = json.loads((FRESH/'decisions.json').read_text())
    decision = decisions[name]; selected = decision['rules']['fixed96']
    assert selected is not None and not decision['reference_accessed']
    stop = selected['training']
    score = json.loads((FRESH/'offline'/f'{name}_{stop}.json').read_text())
    assert score['accuracy_pass']
    hashes = {}
    refs = validation.references(ROOT/'extension_v2',0,expected,hashes)
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
    meta = dict(name=name,selection='First fresh catalogue at fiducial injection for each exposure; same latent events for both sensors',
        events=len(cat['coeff']),expected=expected,mode=mode,selected=selected,history=decision['history'],
        independent_kl=score['maximum_directed_kl'],accuracy_pass=score['accuracy_pass'],truth=truth.tolist(),
        sampling_checks=checks,posterior_precision_checks=precision_checks,source_sha256=hashes,draws=131072,
        smoothing='64 equal bins per prior dimension; Gaussian sigma 0.8 bins; reflecting boundaries; 50/90 percent contours')
    np.savez_compressed(OUT/f'n{expected}_{mode}_samples.npz',**samples)
    (OUT/f'n{expected}_{mode}.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(mode,selected,checks,flush=True)
    return samples,meta


def contour_levels(h):
    ordered = np.sort(h.ravel())[::-1]
    mass = np.cumsum(ordered)/ordered.sum()
    return np.sort([ordered[min(np.searchsorted(mass,p),len(ordered)-1)] for p in [.5,.9]])


def audit_ratio(result: dict) -> float:
    """Largest normalized audit requirement; stability remains a separate gate."""
    return max(result['maximum_directed_kl']/.003,
               result['bootstrap_95_kl']/.005,
               32/min(result['gp_ess'],result['direct_ess']))


def smooth_density(histogram: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Covariance-oriented Gaussian smoothing of a binned display density.

    xy is expressed in histogram-bin coordinates. A 0.2-posterior-SD kernel
    smooths Monte Carlo texture without imposing the same physical bandwidth
    on broad and narrow marginals. Reflection retains the finite prior support.
    Quantitative validation always uses the original unsmoothed weights.
    """
    covariance=.2**2*np.cov(xy.T)+np.eye(2)*.45**2
    radius=int(np.ceil(4*np.sqrt(np.max(np.diag(covariance)))))
    y,x=np.mgrid[-radius:radius+1,-radius:radius+1]
    pos=np.stack([x,y],axis=-1)
    kernel=np.exp(-.5*np.einsum('...i,ij,...j->...',pos,np.linalg.inv(covariance),pos))
    kernel/=kernel.sum()
    padded=np.pad(histogram,radius,mode='symmetric')
    result=np.maximum(0,fftconvolve(padded,kernel,mode='valid'))
    return result*histogram.sum()/result.sum()


def corner_axes(fig, rectangle, samples, meta, include_reference=True):
    """Compare neutral BO curves with an actual 2000-label reference GP."""
    gs=fig.add_gridspec(4,4,left=rectangle[0],right=rectangle[1],
        bottom=rectangle[2],top=rectangle[3],wspace=.10,hspace=.10)
    stop=meta['selected']['training']
    styles=[('reference','#226A91','-',1.4),('118','.60','--',.8),(str(stop),'.12','--',1.05)]
    if not include_reference:
        styles=[(str(meta.get('early_points',118)),'.60','--',.9),(str(stop),'#226A91','-',1.4)]
    axes=[]
    for i in range(4):
        row=[]
        for j in range(4):
            if j>i:row.append(None);continue
            ax=fig.add_subplot(gs[i,j]);row.append(ax)
            ranges=[(LOW[k]*SCALE[k],(LOW[k]+WIDTH[k])*SCALE[k]) for k in range(4)]
            for key,color,ls,lw in (styles if include_reference else styles[::-1]):
                s=samples[key]*SCALE
                if i==j:
                    h,edges=np.histogram(s[:,i],64,range=ranges[i],density=True)
                    x=(edges[:-1]+edges[1:])/2
                    sigma=np.sqrt((.2*np.std(s[:,i])/(edges[1]-edges[0]))**2+.45**2)
                    h=gaussian_filter1d(h,sigma,mode='reflect')
                    if key=='reference' or (not include_reference and key==str(stop)):
                        ax.fill_between(x,0,h,color=color,alpha=.18)
                    ax.plot(x,h,color=color,ls=ls,lw=lw)
                else:
                    h,xe,ye=np.histogram2d(s[:,j],s[:,i],64,range=[ranges[j],ranges[i]],density=True)
                    xy=np.column_stack([(s[:,j]-ranges[j][0])/(ranges[j][1]-ranges[j][0])*64,(s[:,i]-ranges[i][0])/(ranges[i][1]-ranges[i][0])*64])
                    h=smooth_density(h.T,xy);x,y=(xe[:-1]+xe[1:])/2,(ye[:-1]+ye[1:])/2
                    levels=contour_levels(h)
                    if key=='reference':ax.contourf(x,y,h,levels=[levels[0],h.max()*1.001],colors=[color],alpha=.08)
                    if not include_reference and key==str(stop):
                        ax.contourf(x,y,h,levels=[*levels,h.max()*1.001],
                                    colors=['#d1e2ed','#82b1cc'])
                    else:
                        ax.contour(x,y,h,levels=levels,colors=color,linestyles=ls,linewidths=lw)
                    ax.set_ylim(ranges[i])
            ax.set_xlim(ranges[j])
            if i==j:ax.relim();ax.autoscale_view(scalex=False,scaley=True);ax.set_ylim(bottom=0)
            ax.axvline(meta['truth'][j]*SCALE[j],color='black',ls='-',lw=.9,zorder=6)
            if i>j:ax.axhline(meta['truth'][i]*SCALE[i],color='black',ls='-',lw=.9,zorder=6)
            ax.xaxis.set_major_locator(MaxNLocator(3));ax.yaxis.set_major_locator(MaxNLocator(3))
            if i==j:ax.set_yticks([])
            ax.minorticks_on();ax.tick_params(which='both',direction='in',top=True,right=True,labelsize=7,pad=2)
            if i==3:ax.set_xlabel(LABELS[j],fontsize=9,labelpad=3)
            else:ax.tick_params(labelbottom=False)
            if j==0 and i>0:ax.set_ylabel(LABELS[i],fontsize=9,labelpad=3)
            elif j>0:ax.tick_params(labelleft=False)
        axes.append(row)
    handles=[Line2D([],[],color=c,ls=ls,lw=lw,label=label) for (_,c,ls,lw),label in zip(styles,
        [meta.get('reference_label', 'Reference GP: 2000'), 'Early BO: 118',
         meta.get('selected_label', f'BO stop: {stop}')])]
    if not include_reference:
        handles=[Line2D([],[],color=styles[0][1],ls='--',lw=.9,label=f'{styles[0][0]}-point GP'),
                 Patch(facecolor='#82b1cc',edgecolor='none',label=f'{stop}-point GP')]
    handles.append(Line2D([],[],color='black',ls='-',lw=.9,label='Injected value'))
    fig.legend(handles=handles,loc='upper right',bbox_to_anchor=(rectangle[1],rectangle[3]-.005),
               frameon=False,fontsize=8,handlelength=2.3,labelspacing=.25,borderpad=0,
               title=rf'$\mathbb{{E}}[N_{{\rm obs}}]={meta["expected"]}$, {meta['mode']}',
               title_fontsize=8)
    return axes


def paired_corners(bundles,mode):
    fig=plt.figure(figsize=(7.2,5.0))
    rectangles=[(.055,.475,.315,.97),(.575,.995,.315,.97)]
    corner_panels=[corner_axes(fig,rect,samples,meta) for rect,(samples,meta) in zip(rectangles,bundles)]
    # One audit scale and shared time limits across all four cases: no clipped excursions.
    ratios=[]
    for sensor in ['perfect','uncertain']:
        for expected in [100,1000]:
            ds=json.loads((FRESH/'decisions.json').read_text())[f't0_n{expected}_{sensor}_c100_bo']
            ratios += [audit_ratio(h['stages'][0]['result']) for h in ds['history']]
    for rect,(samples,meta) in zip(rectangles,bundles):
        ax=fig.add_axes([rect[0],.09,rect[1]-rect[0],.145])
        history=meta['history'];x=np.array([h['training'] for h in history]);results=[h['stages'][0]['result'] for h in history]
        y=np.array([audit_ratio(r) for r in results]);stable=np.array([h['diagnostics']['streaks']['joint']>=2 for h in history])
        ax.axhspan(.25,1,color='.4',alpha=.06)
        ax.axhline(1,color='.35',ls='--',lw=.9)
        ax.plot(x,y,color='.25',lw=1.2)
        ax.scatter(x[~stable],y[~stable],s=26,facecolors='white',edgecolors='.25',zorder=3)
        ax.scatter(x[stable],y[stable],s=26,facecolors='.25',edgecolors='.25',zorder=3)
        stop=meta['selected']['training'];j=int(np.flatnonzero(x==stop)[0])
        assert results[j]['passed'] and stable[j]
        assert not any(r['passed'] and s for r,s in zip(results[:j],stable[:j]))
        ax.axvline(stop,color='.55',ls=':',lw=.8)
        ax.scatter([stop],[y[j]],s=115,marker='*',facecolors='.05',edgecolors='white',linewidths=.5,zorder=5)
        ax.set(yscale='log',xlim=(220,620),ylim=(.25,max(ratios)*1.45),xlabel='Training evaluations',ylabel='Audit score')
        ax.set_xticks([238,358,478,598]);ax.minorticks_on();ax.tick_params(which='both',direction='in',top=True,right=True,labelsize=8)
        ax.xaxis.label.set_size(9);ax.yaxis.label.set_size(9)
    # Stable meanings across panels; caption supplies all thresholds and panel identities.
    handles=[Line2D([],[],ls='',marker='o',markerfacecolor='white',markeredgecolor='.25',label='Stability not yet satisfied'),
        Line2D([],[],ls='',marker='o',color='.25',label='Stability satisfied'),
        Line2D([],[],ls='',marker='*',color='.05',markersize=9,label='BO stop')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.53,-.065),ncol=3,frameon=False,fontsize=8)
    fig.savefig(FIG/f'posterior_stop_{mode}.pdf',bbox_inches='tight')
    fig.savefig(FIG/f'posterior_stop_{mode}.png',dpi=180,bbox_inches='tight');plt.close(fig)


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


def width_rows(outcomes: list[dict]) -> list[dict]:
    """Retrieve each BO posterior at its accepted benchmark checkpoint."""
    records=plot_paper.read(plot_paper.ROOT/'convergence_20260910/ensemble/records.json')
    updates=plot_paper.read(HERE/'refined_case49_20260911/records.json')
    key=lambda r:(r['mode'],r['exposure'],r['case'],r['strategy'],r['points'])
    lookup={key(r):r for r in records}
    lookup.update({key(r):r for r in updates})
    result=[]
    for outcome in outcomes:
        if outcome['strategy']!='bo':continue
        assert outcome['state']=='reference_verified_convergence'
        r=lookup[key(outcome)]
        assert r['accuracy_pass'] and r['numerical_pass'] and r['streak']>=2
        q=np.asarray(r['summaries'][0]['quantiles'])
        result.append(dict(mode=r['mode'],expected=r['exposure'],case=r['case'],points=r['points'],
            width68=(q[4]-q[2]).tolist(),normalized_width68=((q[4]-q[2])/(.68*WIDTH)).tolist()))
    assert len(result)==600
    return result


def efficiency():
    rows=plot_paper.ensemble_rows()
    physical=width_rows(rows)
    fig,panels=plt.subplots(2,2,figsize=(7.1,5.2),sharex='col',sharey='row',layout='constrained')
    axes=panels[1]
    width_quantiles=[]
    for ax,mode,counts in zip(panels[0],['perfect','uncertain'],[[30,100,300,1000],[50,500]]):
        for j,(color,ls) in enumerate(zip(['#0072B2','#D55E00','#009E73','#9467B0'],['-','--','-.',':'])):
            values=[[r['normalized_width68'][j] for r in physical if (r['mode'],r['expected'])==(mode,n)] for n in counts]
            assert all(len(v)==100 for v in values)
            lo,med,hi=np.quantile(values,[.1,.5,.9],axis=1)
            ax.plot(counts,med,color=color,ls=ls,marker='o',ms=3,label=[r'$\alpha$',r'$\sigma$',r'$a$',r'$d$'][j])
            ax.fill_between(counts,lo,hi,color=color,alpha=.09,lw=0)
            width_quantiles.append(dict(mode=mode,parameter=['alpha','sigma','a','d'][j],expected=counts,quantiles=np.array([lo,med,hi]).T.tolist()))
        ax.axhline(1,color='.5',ls=':',lw=.75)
        ax.set_ylim(0,1.08);ax.set_xscale('log')
        ax.minorticks_on();ax.tick_params(which='both',direction='in',top=True,right=True)
    panels[0,0].set_ylabel('68% width / prior 68% width')
    panels[0,0].legend(frameon=False,ncol=2,loc='lower left',fontsize=9)
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
    (OUT/'physics_widths.json').write_text(json.dumps(dict(rows=physical,quantiles=width_quantiles,source='Accepted, internally validated BO posteriors; all 600 catalogues'),indent=2)+'\n')


def main():
    OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
    bundles = {}
    for mode in ['perfect','uncertain']:
        bundles[mode] = []
        for expected in [100,1000]:
            path = OUT/f'n{expected}_{mode}_samples.npz'
            meta_path = OUT/f'n{expected}_{mode}.json'
            if path.exists():
                samples=dict(np.load(path));meta=json.loads(meta_path.read_text())
            elif expected==1000 and (OUT/f'{mode}_samples.npz').exists():
                samples=dict(np.load(OUT/f'{mode}_samples.npz'));meta=json.loads((OUT/f'{mode}.json').read_text())
                np.savez_compressed(path,**samples);meta_path.write_text(json.dumps(meta,indent=2)+'\n')
            else:
                if '--plots-only' in sys.argv:raise FileNotFoundError(path)
                samples,meta=prepare(mode,expected)
            reference=HERE/'dense_reference_20260912'/f'n{expected}_{mode}'
            checked=json.loads((reference/'validation.json').read_text())
            assert checked['reference_accepted'] and checked['points']==2000
            samples['reference']=np.load(reference/'samples.npz')['reference']
            meta['dense_reference']=json.loads((reference/'training.json').read_text())
            for p in [reference/'checkpoint_2000.npz',reference/'validation.json',reference/'samples.npz']:
                meta['source_sha256'][str(p.relative_to(REPO))]=hashlib.sha256(p.read_bytes()).hexdigest()
            bundles[mode].append((samples,meta))
        paired_corners(bundles[mode],mode)
    efficiency()
    source_hashes = dict(plot_paper.SOURCES)
    for bundle in bundles.values():
        for samples,meta in bundle:source_hashes.update(meta['source_sha256'])
    (OUT/'render_manifest.json').write_text(json.dumps(dict(corners=4,displayed_posteriors=['118 training evaluations','recorded BO stop','2000-label reference GP'],reference_overlay='Actual GP, internally checked; no numerical-target contour',bo_color='monochrome',bivariate_smoothing='Covariance-oriented Gaussian, bandwidth 0.2 posterior SD plus 0.45-bin isotropic grid regularization, reflected prior boundaries; unsmoothed validation',audit_score='max(KL/0.003, bootstrap95/0.005, 32/minimum_ESS)',stability_gate='at least two consecutive resolved checks'),indent=2)+'\n')
    (OUT/'sources.json').write_text(json.dumps(source_hashes,indent=2)+'\n')
    (OUT/'source_sha256.txt').write_text(hashlib.sha256(Path(__file__).read_bytes()).hexdigest()+'\n')

if __name__=='__main__':main()
