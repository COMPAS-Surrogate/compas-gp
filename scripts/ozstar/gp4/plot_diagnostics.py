"""Render completed four-input diagnostics without refitting or selecting checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();root=args.root;out=args.output;out.mkdir(parents=True,exist_ok=True)
    hashes=json.loads((root/'diagnostic_hashes.json').read_text())
    for name,expected in hashes.items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
    plt.rcParams.update({'font.family':'serif','font.size':10,'axes.labelsize':11,
        'xtick.direction':'in','ytick.direction':'in','xtick.top':True,'ytick.right':True,
        'axes.spines.top':True,'axes.spines.right':True,'legend.fontsize':9,'savefig.bbox':'tight'})
    colors={100:'#277DA1',1000:'#D17A22'}
    fig,axes=plt.subplots(1,2,figsize=(9.0,3.3),layout='constrained')
    summary=[]
    for n in (100,1000):
        for mode,style in [('perfect','-'),('uncertain','--')]:
            folder=root/'diagnostics'/f'count_n{n}_{mode}_c200_s17_bo'
            data=json.loads((folder/'holdout.json').read_text());rows=data['rows'];summary+=rows
            for scramble in (0,1):
                with np.load(folder/f'holdout_{scramble}.npz') as d:
                    error=np.abs(d['predicted']-d['exact']);weights=d['weights']
                    np.testing.assert_allclose(weights.sum(),1,atol=1e-10)
                    order=np.argsort(error);cdf=np.cumsum(weights[order]);assert abs(cdf[-1]-1)<1e-9
                    keep=np.unique(np.r_[0,np.linspace(0,len(error)-1,1500,dtype=int),len(error)-1])
                    axes[0].plot(error[order][keep],cdf[keep],style,color=colors[n],lw=1.1,alpha=.8)
            levels=np.asarray(rows[0]['levels']);coverage=np.array([r['coverage'] for r in rows])
            axes[1].plot(levels,coverage.mean(axis=0),style,color=colors[n],lw=1.6,
                         marker='o' if mode=='perfect' else 's',ms=3,label=f'{n}, {mode}')
            axes[1].fill_between(levels,coverage.min(axis=0),coverage.max(axis=0),color=colors[n],alpha=.12)
    axes[0].set(xlabel=r'$|\widehat\ell-\ell|$ [nat]',ylabel='Cumulative posterior weight',xlim=(0,.5),ylim=(0,1.01))
    axes[1].plot([0,1],[0,1],':',color='.3',lw=1)
    axes[1].set(xlabel='Nominal interval probability',ylabel='Empirical posterior-weighted coverage',xlim=(0,1),ylim=(0,1))
    axes[1].legend(loc='upper left',frameon=False)
    for ax in axes:ax.minorticks_on();ax.tick_params(which='both',top=True,right=True)
    for suffix in ('pdf','png'):fig.savefig(out/f'gp_holdout.{suffix}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(10.0,3.1),layout='constrained')
    viewcolors={'perfect':'#277DA1','uncertain':'#D17A22'}
    all_slices=[]
    for i,n in enumerate((100,1000)):
        for mode in ('perfect','uncertain'):
            folder=root/'diagnostics'/f'count_n{n}_{mode}_c200_s17_bo'
            with np.load(folder/'amplitude_injection.npz') as d:
                if mode=='perfect':axes[i].plot(d['amplitude'],d['direct_density'],color='.15',ls='--',lw=1.3,label='Exact Gamma')
                axes[i].plot(d['amplitude'],d['gp_density'],color=viewcolors[mode],lw=1.2,label=f'GP, {mode}')
        axes[i].set(xlabel=r'$a$',ylabel=r'$p(a\mid\phi_{\rm inj},\mathcal{D})$',xlim=(.005,.015))
        axes[i].ticklabel_format(axis='x',style='sci',scilimits=(-2,-2),useMathText=True)
    axes[0].legend(frameon=False,loc='upper left',fontsize=8)
    for mode in ('perfect','uncertain'):
        observed=[];gp=[];exact=[]
        for n in (10,50,100,200,500,1000):
            d=json.loads((root/'diagnostics'/f'count_n{n}_{mode}_c200_s17_bo'/'amplitude.json').read_text())
            all_slices+=d['rows'];r=d['rows'][0]
            observed.append(r['observed']);gp.append(r['gp']['fractional_sd']);exact.append(r['direct']['fractional_sd'])
        axes[2].plot(observed,gp,color=viewcolors[mode],marker='o' if mode=='perfect' else 's',ms=3,lw=1,label=f'GP, {mode}')
        if mode=='perfect':axes[2].plot(observed,exact,color='.15',ls='--',lw=1.2,label='Bounded Gamma')
    x=np.geomspace(5,1300,200);axes[2].plot(x,1/np.sqrt(x+1),':',color='.4',lw=1.2,label=r'$(N_{\rm obs}+1)^{-1/2}$')
    axes[2].set(xscale='log',yscale='log',xlabel=r'$N_{\rm obs}$',ylabel=r'$\mathrm{sd}(a)/\mathbb{E}[a]$',xlim=(5,1300))
    axes[2].legend(frameon=False,fontsize=8)
    for ax in axes:ax.minorticks_on();ax.tick_params(which='both',top=True,right=True)
    for suffix in ('pdf','png'):fig.savefig(out/f'amplitude_conditional_gp4.{suffix}',dpi=180)
    plt.close(fig)
    metrics=dict(holdout=summary,amplitude=all_slices,
                 source_sha256=hashes,plotter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'gp4_diagnostic_figure_sources.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print('Raw RMSE',min(r['raw_rmse'] for r in summary),max(r['raw_rmse'] for r in summary))
    print('Centered RMSE',min(r['centered_rmse'] for r in summary),max(r['centered_rmse'] for r in summary))
    print('CDF max error',max(r['analytic_cdf_max_error'] for r in all_slices))
    print('Slice mean error SD',max(r['mean_error_sd'] for r in all_slices))
    print('Slice width error',max(r['width68_fraction_error'] for r in all_slices))


if __name__=='__main__':main()
