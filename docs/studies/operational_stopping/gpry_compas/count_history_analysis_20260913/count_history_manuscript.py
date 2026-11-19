"""Publish saved paired campaign posteriors and portable analysis tables."""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import numpy as np
from scipy.special import logsumexp
import presentation_figures as old
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = old.REPO
ROOT = old.ROOT/'count_history_20260912'
OUT = old.ROOT/'count_history_analysis_20260913'
FIG = old.FIG


def table(path, rows):
    with path.open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def prepare(n, mode, early_points=25):
    name = f't0_n{n}_{mode}_c200_bo'
    folder = ROOT/'offline'/name
    history = json.loads((folder/'history.json').read_text())
    outcome = json.loads((folder/'outcome.json').read_text())
    stop = outcome['stop']['selected']
    catpath = ROOT/'training'/name/'catalogue.npz'
    with np.load(catpath) as d:
        cat = dict(d)
    reference = json.loads((folder/'reference.json').read_text())
    hashes = {}
    refs = old.validation.score.cached_references(0, reference['proposal_expected_events'], hashes)
    assert all(reference['source_sha256'][k] == v for k,v in hashes.items())
    phi = np.concatenate([p for p,_,_ in refs])
    beta = np.concatenate([r.sum(1)*float(cat['duration'])/.012 for _,r,_ in refs])
    arrays = {}
    with np.load(folder/'reference_weights.npz') as d:
        arrays['reference'] = [d[f'direct_{s}'] for s in (0,1)]
    early_folder = folder if early_points == 58 else OUT/f'early{early_points}'/name
    early_row = history[0] if early_points == 58 else json.loads((early_folder/'score.json').read_text())
    for count in (early_points, stop['training']):
        with np.load((early_folder if count == early_points else folder)/f'posterior_{count}.npz') as d:
            arrays[str(count)] = [d[f'logweights_{s}'] for s in (0,1)]
    samples, checks = {}, {}
    for key, weights in arrays.items():
        samples[key] = old.draw_posterior(phi, beta, np.concatenate([w-logsumexp(w) for w in weights]), len(cat['coeff']))
        row = history[0] if key == 'reference' else early_row if key == str(early_points) else next(h for h in history if h['points'] == int(key))
        assert all(row[k] for k in ('reference_numerical','gp_numerical','four_numerical'))
        target = row['four_summaries']['direct' if key == 'reference' else 'gp'][0]
        mean = float(np.max(abs(samples[key].mean(0)-target['mean'])/target['sd']))
        q = np.quantile(samples[key], [.025,.16,.84,.975], axis=0)
        endpoint = float(np.max(abs(q-np.asarray(target['quantiles'])[[0,2,4,6]])/target['sd']))
        assert mean < .05 and endpoint < .07, (name,key,mean,endpoint)
        checks[key] = dict(mean_error_sd=mean, endpoint_error_sd=endpoint)
    for path in [catpath, folder/'history.json', folder/'outcome.json', folder/'reference_weights.npz',
                 early_folder/f'posterior_{early_points}.npz', (ROOT/'training'/name if early_points == 58 else early_folder)/f'checkpoint_{early_points}.npz', folder/f"posterior_{stop['training']}.npz"]:
        hashes[str(path.relative_to(REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
    np.savez_compressed(OUT/f'{name}_draws.npz', **samples)
    meta = dict(name=name, early_points=early_points, expected=n, mode=mode, events=len(cat['coeff']), selected=stop,
                truth=cat['truth'].tolist(), reference_label='Direct reference',
                reference_overlay_displayed=False, curve_reference='Validated 1258-point GP; self excluded',
                selected_label=f"Audit stop: {stop['training']}", draw_checks=checks,
                source_sha256=hashes, selection='Catalogue 200; frozen fixed96 audit stop')
    (OUT/f'{name}_figure.json').write_text(json.dumps(meta,indent=2)+'\n')
    return samples, meta, history


def corners(mode, bundles):
    fig = plt.figure(figsize=(7.2,5.7))
    for rect, (samples, meta, rows) in zip([(.065,.475,.385,.985),(.585,.995,.385,.985)],bundles):
        corner = old.corner_axes(fig, rect, samples, meta, include_reference=False)
        ticks = [(-.4, -.2), (.2, .4), (.8, 1.2), (4.4, 4.8)]
        for i, row in enumerate(corner):
            for j, axis in enumerate(row):
                if axis is None:
                    continue
                axis.set_xticks(ticks[j])
                if i != j:
                    axis.set_yticks(ticks[i])
                axis.minorticks_off()
                axis.tick_params(which='major', length=2.5)
        # Panel identity belongs in the caption, following manuscript convention.
        fig.legends[-1].set_title(None)
        ax = fig.add_axes([rect[0],.10,rect[1]-rect[0],.195])
        assert all(r['late_reference_accepted'] for r in rows)
        rows = [r for r in rows if not r['late_reference_is_self']]
        x = [r['points'] for r in rows]; y = [r['symmetric_kl_to_late_gp'] for r in rows]
        ax.plot(x,y,color='#226A91',lw=1.2)
        n = meta['selected']['training']
        ax.plot(n,next(r['symmetric_kl_to_late_gp'] for r in rows if r['points']==n),'o',color='.15',ms=4)
        ax.axhline(.01,color='.4',ls='--',lw=.8)
        ax.set(xlim=(50,1270),ylim=(1e-8,10),yscale='log',xticks=[100,400,700,1000,1258],
               xlabel=r'$N_{\rm train}$',ylabel=r'$D_{\mathrm{KL}}^{\mathrm{sym}}$')
        ax.tick_params(which='both',direction='in',top=True,right=True,labelsize=7)
        ax.xaxis.label.set_size(8); ax.yaxis.label.set_size(10)
    fig.legend(handles=[Line2D([],[],ls='--',color='.4',label='0.01 nats (validation guide)'),
                        Line2D([],[],ls='',marker='o',color='.15',label='Recorded audit stop')],
               loc='lower center',bbox_to_anchor=(.54,-.015),ncol=2,frameon=False,fontsize=8)
    for ext in ('pdf','png'):
        fig.savefig(FIG/f'posterior_stop_{mode}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def summary():
    """Replace the old ensemble layout using only paired campaign outcomes."""
    counts = [10,50,100,200,500,700,1000]
    records=[]
    for path in sorted((ROOT/'offline').glob('*/outcome.json')):
        r=json.loads(path.read_text())
        h=json.loads((path.parent/'history.json').read_text())
        stop=r['stop']['selected']
        row=next(x for x in h if x['points']==stop['training'])
        assert row['accuracy_pass']
        q=np.mean([np.asarray(x['quantiles']) for x in row['four_summaries']['gp']],axis=0)
        records.append(dict(name=r['name'],expected=r['expected_events'],mode=r['mode'],
                            width68=((q[4]-q[2])/(.68*old.WIDTH)).tolist(),**stop))
    fig,axes=plt.subplots(2,2,figsize=(7.1,5.2),sharex='col',sharey='row',layout='constrained')
    quantiles=[]
    for col,mode in enumerate(('perfect','uncertain')):
        top,bottom=axes[:,col]
        top.set_title('Perfect measurements' if mode=='perfect' else 'Uncertain measurements',fontsize=11,pad=9)
        for j,(color,ls) in enumerate(zip(['#0072B2','#D55E00','#009E73','#9467B0'],['-','--','-.',':'])):
            vals=[[r['width68'][j] for r in records if (r['mode'],r['expected'])==(mode,n)] for n in counts]
            assert all(len(v)==5 for v in vals)
            lo,med,hi=np.quantile(vals,[.1,.5,.9],axis=1)
            top.plot(counts,med,color=color,ls=ls,marker='o',ms=3,label=[r'$\alpha$',r'$\sigma$',r'$a$',r'$d$'][j])
            top.fill_between(counts,lo,hi,color=color,alpha=.09,lw=0)
            quantiles.append(dict(mode=mode,metric=['alpha','sigma','a','d'][j],quantiles=np.array([lo,med,hi]).T.tolist()))
        top.axhline(1,color='.5',ls=':',lw=.75);top.set_ylim(0,1.08)
        for key,color,ls,label in [('training','#0072B2','-',r'$N_{\rm train}$'),('total','#D55E00','--',r'$N_{\rm train}+N_{\rm audit}$')]:
            vals=[[r[key] for r in records if (r['mode'],r['expected'])==(mode,n)] for n in counts]
            lo,med,hi=np.quantile(vals,[.1,.5,.9],axis=1)
            bottom.plot(counts,med,color=color,ls=ls,marker='o',ms=3,label=label)
            bottom.fill_between(counts,lo,hi,color=color,alpha=.1,lw=0)
            quantiles.append(dict(mode=mode,metric=key,quantiles=np.array([lo,med,hi]).T.tolist()))
        bottom.set(ylim=(180,1050),yticks=[200,400,600,800,1000],xlabel=r'$\mathbb{E}[N_{\rm obs}]$')
        for ax in (top,bottom):
            ax.set_xscale('log');ax.set_xticks(counts,[10,50,100,200,500,'',1000])
            ax.xaxis.set_minor_formatter(old.matplotlib.ticker.NullFormatter())
            ax.tick_params(which='both',direction='in',top=True,right=True)
    axes[0,0].set_ylabel('68% width / prior 68% width');axes[0,0].legend(frameon=False,ncol=2,loc='lower left',fontsize=9)
    axes[1,0].set_ylabel('Likelihood evaluations at stop');axes[1,0].legend(frameon=False,fontsize=9)
    for ext in ('pdf','png'):fig.savefig(FIG/f'acquisition_summary.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    (OUT/'paired_acquisition_summary.json').write_text(json.dumps(dict(expected_counts=counts,rows=records,quantiles=quantiles),indent=2)+'\n')


def main():
    histories, marginals = [], []
    for path in sorted((ROOT/'offline').glob('*/history.json')):
        for r in json.loads(path.read_text()):
            keys = ('name','expected_events','mode','case','points','posterior_comparisons',
                    'elapsed_seconds','reference_numerical','gp_numerical','four_numerical',
                    'accuracy_pass','symmetric_kl','maximum_directed_kl',
                    'symmetric_kl_to_late_gp','late_reference_accepted','late_reference_is_self')
            histories.append({k:r[k] for k in keys})
            for target, summaries in r['four_summaries'].items():
                for scramble,s in enumerate(summaries):
                    for i,parameter in enumerate(('alpha','sigma','a','d')):
                        q=np.asarray(s['quantiles'])[:,i]
                        marginals.append(dict(name=r['name'],points=r['points'],target=target,
                            scramble=scramble,parameter=parameter,mean=s['mean'][i],sd=s['sd'][i],
                            q025=q[0],q16=q[2],q50=q[3],q84=q[4],q975=q[6]))
    assert len(histories)==2870 and len(marginals)==45920
    table(OUT/'histories.csv',histories); table(OUT/'marginals.csv',marginals)
    bundles = {m:[prepare(n,m) for n in (100,1000)] for m in ('perfect','uncertain')}
    for n in (100,1000):
        with np.load(ROOT/'training'/f't0_n{n}_perfect_c200_bo/catalogue.npz') as p, np.load(ROOT/'training'/f't0_n{n}_uncertain_c200_bo/catalogue.npz') as u:
            for key in ('latent','scales','y','rho','bins','duration','truth'):
                np.testing.assert_array_equal(p[key],u[key])
    for mode,b in bundles.items():
        corners(mode,b)
        for _,m,_ in b: print(m['name'],m['events'],m['selected'],m['draw_checks'],flush=True)
    for src,dest in [('ensemble_convergence_candidate','count_history_convergence'),('stopping_summary','count_history_stopping')]:
        for ext in ('pdf','png'): shutil.copy2(OUT/f'{src}.{ext}',FIG/f'{dest}.{ext}')
    summary()
    shutil.copy2(Path(__file__),OUT/Path(__file__).name)


if __name__=='__main__': main()
