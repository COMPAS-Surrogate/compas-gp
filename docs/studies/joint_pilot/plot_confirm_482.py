"""Fresh-catalogue calibration figure; retain every evaluated case.

Scientific chart contract: four parameter facets with posterior-rank ECDFs and
coverage dot/interval panels. Direct black solid/open circles versus GP blue
dashed/squares. Pointwise uniform-reference bands and exact binomial intervals
are not simultaneous guarantees. Export PNG/PDF and inspect the PNG.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

HERE=Path(__file__).resolve().parent


def main():
    root=HERE/'confirm_482_results';cases=json.loads((root/'cases.json').read_text());summary=json.loads((root/'summary.json').read_text())
    assert len(cases)==summary['catalogues'] and summary['failed']==0,'Do not silently omit failed cases'
    n=len(cases);x=np.linspace(0,1,201);fig,axes=plt.subplots(2,4,figsize=(14,7),constrained_layout=True)
    for i,label in enumerate([r'$\alpha$',r'$\sigma$',r'$a_{\rm SFR}$',r'$d_{\rm SFR}$']):
        ax=axes[0,i];ax.fill_between(x,binom.ppf(.025,n,x)/n,binom.ppf(.975,n,x)/n,color='.9',label='95% pointwise band');ax.plot(x,x,color='.5',lw=1)
        for name,color,style in [('direct','#222222','-'),('gp','#326bb0','--')]:
            ranks=np.sort([c['summaries'][0][name]['ranks'][i] for c in cases]);ax.step(np.r_[0,ranks,1],np.r_[0,np.arange(1,n+1)/n,1],where='post',color=color,ls=style,lw=1.8,label='Direct' if name=='direct' else 'GP')
        ax.set(xlim=(0,1),ylim=(0,1),title=label,xlabel='Posterior CDF at injected truth')
        ax=axes[1,i];ax.plot([.5,1],[.5,1],color='.5',lw=1)
        for name,color,marker,offset in [('direct','#222222','o',-.003),('gp','#326bb0','s',.003)]:
            levels=np.array([68,90,95]);y=np.array([summary['coverage_counts_evaluated_only'][f'{name}_{level}'][i] for level in levels])/n
            ci=np.array([summary['coverage_exact95_intervals'][f'{name}_{level}'][i] for level in levels]);ax.errorbar(levels/100+offset,y,yerr=np.array([y-ci[:,0],ci[:,1]-y]),color=color,marker=marker,mfc='white' if name=='direct' else color,ls='none',capsize=3)
        ax.set(xlim=(.63,.99),ylim=(.45,1.01),xlabel='Nominal interval probability',xticks=[.68,.90,.95])
    axes[0,0].set_ylabel('Fraction of catalogues');axes[1,0].set_ylabel('Coverage with exact 95% intervals');axes[0,0].legend(frameon=False,fontsize=8)
    fig.suptitle('Fresh-catalogue validation of the 482-label GP\n100 simulated catalogues • disjoint 256M halves • ideal binned observations',fontsize=14)
    for ext in ['png','pdf']:fig.savefig(root/f'calibration.{ext}',dpi=180)
    plt.close(fig)

if __name__=='__main__':main()
