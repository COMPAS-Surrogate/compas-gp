"""Compare marginal constraints for the same mock catalogue and finite target."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent

def main():
    series=[('Direct likelihood', np.load(HERE/'direct_reference_results/posterior_chains.npy').reshape(-1,4),'#262626','-'),
            ('GP: initial',np.load(HERE/'results_seed17/sampling/posterior_samples.npy'),'#3569B0','--'),
            ('GP: refined',np.load(HERE/'results_seed17/local_refinement/sampling/posterior_samples.npy'),'#B65D24',':')]
    fig,axes=plt.subplots(1,4,figsize=(12,3.6))
    for i,(ax,label,bounds) in enumerate(zip(axes,[r'$\alpha$',r'$\sigma$',r'$a_{\rm SFR}$',r'$d_{\rm SFR}$'],[(-.5,-.001),(.1,.6),(.005,.015),(4.2,5.2)])):
        for name,draws,color,style in series:
            ax.hist(draws[:,i],bins=np.linspace(*bounds,45),density=True,histtype='step',color=color,linestyle=style,lw=1.7,label=name)
        ax.set(xlabel=label,xlim=bounds)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Posterior density')
    axes[2].ticklabel_format(axis='x',style='sci',scilimits=(0,0))
    fig.suptitle('The GP compresses the alpha and sigma constraints',x=.07,ha='left',fontsize=14)
    fig.text(.07,.88,'81 ideal binned events • fixed 32M population • identical coarse grid and priors',fontsize=10)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False)
    fig.tight_layout(rect=[0,.09,1,.85])
    fig.savefig(HERE/'direct_reference_results/posterior_comparison.png',dpi=180)
    fig.savefig(HERE/'direct_reference_results/posterior_comparison.pdf')

if __name__=='__main__': main()
