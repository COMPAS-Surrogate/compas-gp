"""Publish title-free scientific figures. Descriptions live in TeX captions.

Axes and compact legends identify quantities. Descriptions remain in captions. Old diagnostic figures are not copied into the paper.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
OUT=REPO/'overleaf/figures/bo_study'
BLUE='#3569a8';ORANGE='#bb7434';GRAY='#404040'
LABELS=[r'$\alpha$',r'$\sigma$',r'$a_{\rm SFR}$',r'$d_{\rm SFR}$']
plt.rcParams.update({'font.family':'serif','mathtext.fontset':'stix','font.size':9,
    'axes.labelsize':10,'xtick.labelsize':8,'ytick.labelsize':8,'pdf.fonttype':42,
    'ps.fonttype':42,'axes.spines.top':True,'axes.spines.right':True,
    'xtick.direction':'in','ytick.direction':'in','xtick.top':True,'ytick.right':True,
    'axes.linewidth':.8,'legend.frameon':False,
    'lines.linewidth':1.2,'savefig.bbox':'tight'})


def save(fig,name):
    for ax in fig.axes:
        assert not ax.get_title(), 'Figure titles prohibited'
        ax.minorticks_on()
        ax.tick_params(which='both',direction='in',top=True,right=True)
        for spine in ax.spines.values(): spine.set_visible(True)
    assert fig._suptitle is None, 'Figure titles prohibited'
    OUT.mkdir(parents=True,exist_ok=True)
    for ext in ['pdf','png']:fig.savefig(OUT/f'{name}.{ext}',dpi=200)
    plt.close(fig)


def existing_results():
    root=HERE.parent/'joint_pilot/uncertain_100_results'
    records=json.loads((root/'cases.json').read_text())
    assert len(records)==100 and all(r['status']=='evaluated' for r in records)
    fig,ax=plt.subplots(figsize=(3.5,3.2),layout='constrained')
    u=np.linspace(0,1,501);eps=np.sqrt(np.log(40)/200)
    ax.fill_between(u,np.maximum(0,u-eps),np.minimum(1,u+eps),color='.9',label='95% envelope')
    ax.plot(u,u,color='.5',lw=.8,ls=':')
    for j,(color,style) in enumerate(zip([BLUE,ORANGE,'#398653','#9563b5'],['-','--','-.',':'])):
        ranks=np.array([r['summaries'][0]['gp']['ranks'][j] for r in records])
        # A truth is inside a central equal-tailed interval of content c iff |2u-1| <= c.
        content=np.sort(abs(2*ranks-1))
        ax.step(np.r_[0,content,1],np.r_[0,np.arange(1,101)/100,1],where='post',color=color,ls=style,label=LABELS[j])
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='Credible interval probability',ylabel='Fraction of injections in interval')
    handles,labels=ax.get_legend_handles_labels()
    ax.legend(handles[1:]+handles[:1],labels[1:]+labels[:1],loc='upper left',fontsize=8)
    save(fig,'uncertain_pp')
    fig,axes=plt.subplots(3,2,figsize=(7.1,5.5),sharex='col',layout='constrained')
    for i,color in enumerate([BLUE,ORANGE,'#9563b5']):
        data=json.loads((root/f'uncertain_{i:03d}/predictive.json').read_text())['gp']
        events=int(sum(data['observed_mass_counts']))
        for ax,coord in zip(axes[i],['mass','z']):
            q=np.array(data[f'{coord}_count_quantiles']);x=np.arange(q.shape[1])
            ax.fill_between(x,q[0],q[2],color=color,alpha=.16,step='mid',lw=0)
            ax.step(x,q[1],where='mid',color=color,label=f'{events} events')
            ax.plot(x,data[f'observed_{coord}_counts'],'o',ls='none',color=color,ms=2.7)
            ax.set(ylabel='Event count');ax.set_ylim(bottom=0)
        axes[i,0].legend(loc='upper left',fontsize=8)
    axes[-1,0].set_xlabel('Measured mass-bin index')
    axes[-1,1].set_xlabel('Measured redshift-bin index')
    save(fig,'predictive_counts')


def event_results():
    root=HERE/'results'
    if not (root/'runtime.json').exists():return
    records=json.loads((root/'validated_records.json').read_text())
    random_root=HERE/'random_results'
    has_random=(random_root/'runtime.json').exists()
    if has_random:records+=json.loads((random_root/'records.json').read_text())
    physics=json.loads((root/'physics.json').read_text())
    expected=[30,100,300,1000];budgets=[226,354,482,610,738]
    truth=np.array([-.3,.35,.01,4.7])
    fig,axes=plt.subplots(2,2,figsize=(7.1,4.8),sharex=True,sharey=True,layout='constrained')
    for ax,n in zip(axes.flat,expected):
        assert has_random, 'Complete random comparison required'
        strategies=[('bo',BLUE,'-','o'),('random',ORANGE,'--','s')]
        for strategy,color,style,marker in strategies:
            values=[]
            for budget in budgets:
                rows=[r for r in records if r['expected_events']==n and r['strategy']==strategy and r['points']==budget]
                assert len(rows)==10 and all(r['status']=='evaluated' for r in rows)
                values.append([max(m['kl_direct_to_gp'] for m in r['metrics']) for r in rows])
            lo,med,hi=np.quantile(values,[.1,.5,.9],axis=1)
            ax.fill_between(budgets,lo,hi,color=color,alpha=.16,lw=0)
            ax.plot(budgets,med,color=color,ls=style,marker=marker,ms=3,label={'bo':'BO','sobol':'Space-filling','random':'Random'}[strategy])
        ax.axhline(.01,color='.4',ls=':',lw=1)
        ax.set(yscale='log',xlim=(215,749))
        ax.set_xticks(budgets)
        ax.text(.96,.94,f'{n} expected detections',transform=ax.transAxes,
                ha='right',va='top',fontsize=9,
                bbox={'facecolor':'white','edgecolor':'none','alpha':.85,'pad':2})
    axes[0,0].legend(loc='upper left')
    for ax in axes[-1]:ax.set_xlabel('Number of training points')
    for ax in axes[:,0]:ax.set_ylabel(r'$D_{\mathrm{KL}}(p_{\mathrm{ref}}\Vert p_{\mathrm{GP}})$')
    save(fig,'acquisition_efficiency')
    # Normalize widths by prior central 68% widths so four parameters share a scale.
    prior68=.68*np.array([.499,.5,.01,1.])
    fig,ax=plt.subplots(figsize=(3.5,3.2),layout='constrained')
    for j,(color,style) in enumerate(zip([BLUE,ORANGE,'#398653','#9563b5'],['-','--','-.',':'])):
        widths=[]
        for n in expected:
            rows=[r for r in records if r['expected_events']==n and r['strategy']=='bo' and r['points']==738]
            assert len(rows)==10
            widths.append([(np.array(r['summaries'][0]['quantiles'])[4,j]-np.array(r['summaries'][0]['quantiles'])[2,j])/prior68[j] for r in rows])
        lo,med,hi=np.quantile(widths,[.1,.5,.9],axis=1)
        ax.fill_between(expected,lo,hi,color=color,alpha=.10,lw=0)
        ax.plot(expected,med,color=color,ls=style,marker='o',ms=3,label=LABELS[j])
    ax.axhline(1,color='.5',lw=.8,ls=':')
    ax.set(xscale='log',xlabel='Expected number of detections',ylabel='68% interval width / prior width',ylim=(0,1.08))
    ax.set_xticks(expected,labels=[str(n) for n in expected]);ax.legend(loc='lower left',ncol=2,fontsize=8)
    save(fig,'event_widths')
    training=np.load(root/'n100_case0_bo/checkpoint_482.npz')['x']
    fig,axes=plt.subplots(1,3,figsize=(7.1,2.3),layout='constrained')
    shapetruth=truth[[0,1,3]];shape_labels=[LABELS[0],LABELS[1],LABELS[3]]
    for ax,(i,j) in zip(axes,[(0,1),(0,2),(1,2)]):
        ax.scatter(training[:226,i],training[:226,j],s=8,facecolors='none',edgecolors='.65',lw=.5)
        ax.scatter(training[226:,i],training[226:,j],s=5,color=BLUE,alpha=.7)
        ax.plot(shapetruth[i],shapetruth[j],'x',color=ORANGE,ms=6,mew=1.2)
        ax.set(xlabel=shape_labels[i],ylabel=shape_labels[j])
    save(fig,'exploration_locations')


if __name__=='__main__':
    existing_results()
    event_results()
