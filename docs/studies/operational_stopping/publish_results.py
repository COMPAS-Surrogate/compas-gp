"""Report hidden-reference checks without excluding resource or numerical issues."""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm

HERE=Path(__file__).resolve().parent
ROOT=HERE/'confirmation_v1'
OUT=HERE/'report_v1'
FAMILIES=['correlated','curved','two_modes','narrow','hidden_mode']
NAMES={'correlated':'Correlated Gaussian','curved':'Curved posterior','two_modes':'Separated modes',
       'narrow':'Narrow Gaussian','hidden_mode':'Narrow secondary mode'}
BLUE='#3569a8';PINK='#a43d63';GRAY='.5'
plt.rcParams.update({'font.family':'serif','mathtext.fontset':'stix','font.size':9,
                     'xtick.direction':'in','ytick.direction':'in','xtick.top':True,
                     'ytick.right':True,'pdf.fonttype':42})


def category(row):
    if not row['reference_numerical_pass']:
        return 'reference_unresolved'
    if row['state']!='operational_stop':
        return 'no_stop'
    return 'premature' if row['premature_stop'] else 'validated_stop'


def load_case(folder):
    decision=(folder/'decision.json').read_bytes()
    score=json.loads((folder/'reference_score.json').read_text())
    assert hashlib.sha256(decision).hexdigest()==score['decision_sha256']
    return json.loads(decision),json.loads((folder/'records.json').read_text()),score['scores']


def save(fig,name):
    for ax in fig.axes:
        if not ax.get_visible():continue
        assert not ax.get_title()
        ax.tick_params(which='both',direction='in',top=True,right=True)
        for spine in ax.spines.values():spine.set_visible(True)
    for ext in ['pdf','png']:fig.savefig(OUT/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def failure_density(row):
    """Display the first premature case by protocol family/seed order."""
    spec=importlib.util.spec_from_file_location('report_fit_engine',HERE/'run_study.py')
    engine=importlib.util.module_from_spec(spec);sys.modules[spec.name]=engine;spec.loader.exec_module(engine)
    from toys import make_target
    folder=ROOT/f"{row['family']}_{row['seed']}"
    decision=json.loads((folder/'decision.json').read_text())
    data=np.load(folder/f"checkpoint_{decision['checkpoint']:03d}.npz")
    model,scaler=engine.fit(data['x'],data['lnl'])
    coordinate=(np.arange(512)+.5)/512
    xx,yy=np.meshgrid(coordinate,coordinate,indexing='xy')
    query=np.c_[xx.ravel(),yy.ravel()]
    target=make_target(row['family'],row['seed'])
    logvalues=[target(query),engine.predict(model,scaler,query)]
    densities=[np.exp(v-engine.logsumexp(v))*len(v) for v in logvalues]
    vmax=max(v.max() for v in densities)
    fig,axes=plt.subplots(1,2,figsize=(7,3),sharex=True,sharey=True,layout='constrained')
    for ax,density,label in zip(axes,densities,['Hidden reference','Stopped GP']):
        mesh=ax.pcolormesh(coordinate,coordinate,density.reshape(512,512),
                           norm=LogNorm(vmin=vmax/1e5,vmax=vmax),cmap='Blues',rasterized=True)
        ax.set(xlabel=r'$x_1$',ylabel=r'$x_2$',xlim=(0,1),ylim=(0,1),aspect='equal')
        ax.text(.96,.96,label,transform=ax.transAxes,ha='right',va='top',fontsize=9,
                bbox={'facecolor':'white','edgecolor':'none','alpha':.85})
    fig.colorbar(mesh,ax=axes,label='Posterior density',shrink=.85)
    save(fig,'premature_stop_density')
    selected={'family':row['family'],'seed':row['seed'],'calls':row['calls'],
              'selection':'first numerically resolved premature stop in protocol family/seed order',
              'region':'x1 > 0.55',
              'reference_mass_in_region':float(densities[0][query[:,0]>.55].sum()/len(query)),
              'gp_mass_in_region':float(densities[1][query[:,0]>.55].sum()/len(query))}
    (OUT/'density_example.json').write_text(json.dumps(selected,indent=2)+'\n')


def prediction_error_location(folder, records, scores):
    """Locate pre-refit errors after the GP already met the hidden target."""
    by_checkpoint={s['checkpoint']:s for s in scores}
    counts=Counter()
    for record in records:
        index=record['checkpoint']
        acquisition=folder/f'acquisition_{index:03d}.npz'
        score=by_checkpoint[index]
        if not acquisition.exists() or not score['accuracy_pass'] or not score['numerical_pass']:
            continue
        data=np.load(acquisition);actual,pred=data['actual'],data['predicted']
        # Use the exact frozen calculation and its fitted constant offset.
        from cosmic_integration.lnl_surrogate.stopping import prediction_check,StoppingTolerances
        check=prediction_check(pred,actual,StoppingTolerances())
        if check['passed']:continue
        previous=np.load(folder/f'checkpoint_{index:03d}.npz')['lnl']
        aligned=pred+check['offset']
        scaled=abs(actual-aligned)/(.1+.01*(actual.max()-actual))
        exceeds=scaled>1
        relevant=np.maximum(actual,aligned)>previous.max()-12
        counts['failing_batches_after_reference_accuracy']+=1
        if not np.any(exceeds&relevant):counts['exceedances_only_deep_tail']+=1
    return dict(counts)


def main():
    status=json.loads((ROOT/'status.json').read_text())
    if status['state']!='complete' or status['completed']!=25:
        raise RuntimeError('Wait for all 25 frozen confirmation cases')
    rows=json.loads((ROOT/'outcomes.json').read_text())
    assert len(rows)==25
    OUT.mkdir(exist_ok=True)
    summary={}
    fig,axes=plt.subplots(2,3,figsize=(10,5.8),sharex=True,sharey=True,layout='constrained')
    for ax,family in zip(axes.flat,FAMILIES):
        group=[r for r in rows if r['family']==family]
        assert len(group)==5
        counts=Counter(category(r) for r in group)
        stopped=[r['calls'] for r in group if r['state']=='operational_stop']
        delayed=[];unnecessary=[]
        for r in group:
            decision,records,scores=load_case(ROOT/f"{family}_{r['seed']}")
            x=[s['calls'] for s in scores];y=[max(s['reference_kl']) for s in scores]
            # Successful audits consume calls while leaving the accepted GP fixed.
            if decision['calls']>x[-1]:x.append(decision['calls']);y.append(y[-1])
            kind=category(r);color=PINK if kind=='premature' else BLUE if kind=='validated_stop' else GRAY
            ax.plot(x,y,color=color,alpha=.65,lw=1)
            ax.plot(x[-1],y[-1],marker='o' if r['state']=='operational_stop' else 'x',
                    color=color,ms=5,mfc='white' if kind=='reference_unresolved' else color)
            first=next((s['calls'] for s in scores if s['numerical_pass'] and s['accuracy_pass']),None)
            if kind=='validated_stop' and first is not None:delayed.append(decision['calls']-first)
            if kind=='no_stop' and scores[-1]['accuracy_pass']:unnecessary.append(r['seed'])
        summary[family]={'cases':5,'outcomes':dict(counts),'median_stop_calls':float(np.median(stopped)) if stopped else None,
                         'median_extra_calls_after_first_accurate_checkpoint':float(np.median(delayed)) if delayed else None,
                         'accurate_but_no_stop_seeds':unnecessary}
        ax.axhline(.01,color='.25',ls=':',lw=1)
        ax.set(yscale='log',xlabel='Exact evaluations, including audits',ylabel='Max. directed reference KL [nats]')
        ax.tick_params(labelbottom=True)
        ax.text(.96,.10 if family=='hidden_mode' else .96,NAMES[family],transform=ax.transAxes,ha='right',va='bottom' if family=='hidden_mode' else 'top',fontsize=9,
                bbox={'facecolor':'white','edgecolor':'none','alpha':.85})
    axes.flat[-1].axis('off')
    handles=[Line2D([],[],color=BLUE,marker='o',label='Stopped; reference target met'),
             Line2D([],[],color=PINK,marker='o',label='Stopped prematurely'),
             Line2D([],[],color=GRAY,marker='x',label='Resource guard; no stop'),
             Line2D([],[],color='.25',ls=':',label='KL target: 0.01 nats')]
    axes.flat[-1].legend(handles=handles,loc='center',frameon=False,fontsize=9)
    save(fig,'hidden_reference_kl')
    # Predeclared first seed in each family, not a best-looking realization.
    fig,axes=plt.subplots(2,3,figsize=(10,5.8),sharex=True,sharey=True,layout='constrained')
    for ax,family in zip(axes.flat,FAMILIES):
        decision,records,scores=load_case(ROOT/f'{family}_2000')
        x=[r['calls'] for r in records]
        pred=[max(r['prediction'].get('q90_scaled_error',np.nan),r['prediction'].get('max_scaled_error',np.nan)/3) for r in records]
        stability=[r['stability_kl']/.001 if r['stability_kl'] is not None else np.nan for r in records]
        reference=[max(s['reference_kl'])/.01 for s in scores]
        ax.plot(x,pred,color='#bb7434',ls='--',label='New-label prediction gate')
        ax.plot(x,stability,color='#398653',ls='-.',label='Successive-posterior KL gate')
        ax.plot(x,reference,color='.2',label='Hidden-reference KL / target')
        ax.axhline(1,color='.5',ls=':',lw=.8)
        ax.axvline(decision['calls'],color='.5',lw=.8,ls='--')
        ax.set(yscale='log',xlabel='Exact evaluations',ylabel='Diagnostic / threshold')
        ax.tick_params(labelbottom=True)
        ax.text(.96,.10 if family=='hidden_mode' else .96,NAMES[family],transform=ax.transAxes,ha='right',va='bottom' if family=='hidden_mode' else 'top',fontsize=9,
                bbox={'facecolor':'white','edgecolor':'none','alpha':.85})
    axes.flat[-1].axis('off')
    handles,labels=axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles,labels,loc='center',frameon=False,fontsize=9)
    save(fig,'operational_vs_reference')
    counts=Counter(category(r) for r in rows)
    accurate_no_stop=sum(len(g['accurate_but_no_stop_seeds']) for g in summary.values())
    result={'confirmation_cases':len(rows),'outcomes':dict(counts),'groups':summary}
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    text=['# Operational stopping: hidden-reference validation (v1)',
          '\nThe rule was frozen before 25 confirmation runs. The exact posterior reference was accessed only after each stopping decision was saved; all 25 decision hashes were checked. Development seed1000 is excluded.',
          f"\nAcross 25 toy runs: **{counts['validated_stop']} validated stops, {counts['premature']} premature stops, {counts['no_stop']} resource-limited runs without a stop, and {counts['reference_unresolved']} numerically unresolved reference assessments.**",
          '\n| Toy family | Validated stops | Premature stops | No stop | Reference unresolved | Median calls among stops |',
          '|---|---:|---:|---:|---:|---:|']
    if counts['premature']:
        text.insert(1,f"\n**This candidate rule is not ready for general use.** It accepted incomplete posteriors in the narrow-secondary-mode stress test. It was also unnecessarily conservative on other targets; {accurate_no_stop} of {counts['no_stop']} resource-limited toy runs had accurate final posteriors.")
    for family,g in summary.items():
        c=g['outcomes'];median=g['median_stop_calls']
        text.append(f"| {NAMES[family]} | {c.get('validated_stop',0)}/5 | {c.get('premature',0)}/5 | {c.get('no_stop',0)}/5 | {c.get('reference_unresolved',0)}/5 | {median:g} |" if median is not None else
                    f"| {NAMES[family]} | {c.get('validated_stop',0)}/5 | {c.get('premature',0)}/5 | {c.get('no_stop',0)}/5 | {c.get('reference_unresolved',0)}/5 | No stops |")
    text += ['\n![Hidden-reference accuracy](hidden_reference_kl.png)',
        '\nEach line is one frozen confirmation trajectory. Reference KL was computed afterwards on two independent quadratures; the larger directed value is shown. Dots are operational stops, crosses are resource guards. A final horizontal segment accounts for audit calls that did not refit the GP. The full accuracy target also includes marginal and numerical checks.',
        '\n![Operational diagnostics compared with reference](operational_vs_reference.png)',
        '\nThese panels show predeclared seed2000 in each family. The reference curve was hidden from stopping. Values below one pass the plotted component, but the full rule additionally requires marginal stability, numerical precision, consecutive checks and a fresh audit. The vertical line is the final decision or resource guard.',
        '\n## Interpretation',
        '\nPassing a finite GP-weighted audit is not a proof of global mode discovery. Resource-limited runs are retained even when their final posterior is already accurate. The evaluation counts include all audit calls; an accepted audit does not improve the fitted GP itself.',
        '\nFive runs per family give a coarse failure-mode screen, not a precise rare-error rate or a universal guarantee. These are deterministic 2D densities, not noisy-catalogue coverage tests. No confirmation thresholds were retuned.',
        '\nSee [frozen protocol](../PROTOCOL.md), [source manifest](../confirmation_v1/manifest.json), and [all outcomes](../confirmation_v1/outcomes.json). Source snapshots identify the exact code loaded for this campaign; later dimensional generalization preserves its 2D defaults.']
    first_bad=next((r for r in rows if category(r)=='premature'),None)
    if first_bad is not None:
        failure_density(first_bad)
        density=json.loads((OUT/'density_example.json').read_text())
        text += ['\n## A premature stop', '\n![Premature-stop posterior](premature_stop_density.png)',
                 f"\nThis is the first premature stop in protocol family/seed order: {NAMES[first_bad['family']]}, seed{first_bad['seed']}. Both panels use the same prior and colour normalization. In the region x1 > 0.55, the reference places {100*density['reference_mass_in_region']:.2f}% of its probability, compared with {100*density['gp_mass_in_region']:.3f}% for the stopped GP. This illustrates an observed failure; it is not an omitted or retrained case."]
    physical=HERE/'compas_v1/outcomes.json'
    if physical.exists():
        cp=json.loads(physical.read_text());text+=['\n## Separate COMPAS pilots','\nThese two preselected original catalogues are implementation pilots, not a new calibration ensemble. References were loaded only after each operational decision. The GP uses three shape parameters; external validation also recovers the conditional SFR amplitude.']
        fig,axes=plt.subplots(1,2,figsize=(7.1,3.2),sharey=True,layout='constrained')
        locations={}
        for r in cp:
            text.append(f"\n- {r['family']}: {r['state']}, {r['calls']} calls including audits; maximum directed reference KL {r['final_reference_kl']:.4g}; numerical check {r['reference_numerical_pass']}.")
            folder=HERE/'compas_v1'/r['family']
            decision,records,scores=load_case(folder)
            locations[r['family']]=prediction_error_location(folder,records,scores)
            ax=axes[len(locations)-1]
            x=[s['calls'] for s in scores];y=[max(s['reference_kl']) for s in scores]
            if decision['calls']>x[-1]:x.append(decision['calls']);y.append(y[-1])
            ax.plot(x,y,color=BLUE,lw=1)
            ax.plot(x[-1],y[-1],marker='o' if r['state']=='operational_stop' else 'x',color=BLUE,ms=6)
            ax.axhline(.01,color='.3',ls=':',lw=1)
            ax.set(yscale='log',xlabel='Exact evaluations, including audits',ylabel='Max. directed reference KL [nats]')
            ax.text(.96,.96,'100 expected, perfect' if 'perfect' in r['family'] else '500 expected, uncertain',transform=ax.transAxes,ha='right',va='top',fontsize=8)
        save(fig,'compas_reference_kl')
        (OUT/'prediction_error_locations.json').write_text(json.dumps(locations,indent=2)+'\n')
        text+=['\n![COMPAS hidden-reference checks](compas_reference_kl.png)',
               '\nDots indicate an operational stop and crosses a resource guard. The final horizontal segment counts the accepted audit. Reference curves were hidden from stopping. The detailed pre-refit error locations are saved in `prediction_error_locations.json`; the 12-lnL diagnostic cutoff matches the broad audit relevance threshold and is not a general posterior-mass bound.']
    (OUT/'RESULTS.md').write_text('\n'.join(text)+'\n')


if __name__=='__main__':main()
