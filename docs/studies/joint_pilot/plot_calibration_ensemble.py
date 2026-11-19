"""Static paper figure: posterior ranks and interval coverage for every case.

Chart contract: compare GP/direct calibration on 100 mock catalogues; ECDFs
and coverage dot/interval panels, one panel per parameter. Neutral direct solid
line/open circles, blue GP dashed line/squares; nominal reference in grey.
Pointwise bands are binomial, not simultaneous confidence bands. Source is the
complete cases.json and summary.json; fail rather than silently exclude cases.
Export PNG/PDF at 14x7 inches and visually inspect the PNG.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

HERE=Path(__file__).resolve().parent

def main():
    root=HERE/'reduced_calibration_100_results'
    cases=json.loads((root/'cases.json').read_text())
    summary=json.loads((root/'summary.json').read_text())
    assert len(cases)==summary['catalogues'] and summary['failed']==0
    suffix=''
    if (root/'reference_assessment.json').exists():
        assessment=json.loads((root/'reference_assessment.json').read_text())
        reference=json.loads((root/'direct_reference_20261003.json').read_text())
        for case,ref in zip(cases,reference):case['direct_rank']=ref['ranks']
        summary['coverage_counts'].update(assessment['direct_coverage_counts'])
        summary['coverage_exact95_intervals'].update(assessment['direct_coverage_exact95_intervals'])
        suffix='_refined'
    n=len(cases);x=np.linspace(0,1,201)
    fig,axes=plt.subplots(2,4,figsize=(14,7),constrained_layout=True)
    labels=[r'$\alpha$',r'$\sigma$',r'$a_{\rm SFR}$',r'$d_{\rm SFR}$']
    for i,label in enumerate(labels):
        ax=axes[0,i]
        ax.fill_between(x,binom.ppf(.025,n,x)/n,binom.ppf(.975,n,x)/n,color='.9',label='95% pointwise band')
        ax.plot(x,x,color='.5',lw=1)
        for kind,color,style in [('direct','#222222','-'),('gp','#326bb0','--')]:
            ranks=np.sort([c[kind+'_rank'][i] for c in cases])
            ax.step(np.r_[0,ranks,1],np.r_[0,np.arange(1,n+1)/n,1],where='post',color=color,ls=style,lw=1.8,label=kind.upper() if kind=='gp' else 'Direct')
        ax.set(xlim=(0,1),ylim=(0,1),title=label,xlabel='Posterior CDF at injected truth')
        ax=axes[1,i]
        ax.plot([.6,1],[.6,1],color='.5',lw=1)
        for kind,color,marker,offset in [('direct','#222222','o',-.003),('gp','#326bb0','s',.003)]:
            levels=np.array([.68,.90,.95]);counts=np.array([summary['coverage_counts'][f'{kind}_cover{int(v*100)}'][i] for v in levels]);y=counts/n
            ci=np.array([summary['coverage_exact95_intervals'][f'{kind}_cover{int(v*100)}'][i] for v in levels])
            ax.errorbar(levels+offset,y,yerr=np.array([y-ci[:,0],ci[:,1]-y]),color=color,marker=marker,mfc='white' if kind=='direct' else color,ls='none',capsize=3,label=kind)
        ax.set(xlim=(.63,.99),ylim=(.5,1.01),xlabel='Nominal interval probability',xticks=[.68,.90,.95])
    axes[0,0].set_ylabel('Fraction of catalogues')
    axes[1,0].set_ylabel('Coverage with exact 95% intervals')
    axes[0,0].legend(frameon=False,fontsize=8,loc='upper left')
    fig.suptitle(f'Posterior calibration on {n} simulated catalogues\nDisjoint 256M generation / inference halves • ideal binned events • fixed GP procedure',fontsize=14)
    for ext in ['png','pdf']:fig.savefig(root/f'calibration{suffix}.{ext}',dpi=180)
    plt.close(fig)

if __name__=='__main__':main()
