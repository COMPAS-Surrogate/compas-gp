"""Matched posterior comparison isolating the repaired noise freeze."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent

def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--refined",action="store_true")
    refined=parser.parse_args().refined
    out=HERE/'noise_freeze_results'
    series=[('Direct likelihood',np.load(HERE/'direct_reference_results/posterior_chains.npy').reshape(-1,4),'#262626','-'),
        ('Original refined GP',np.load(HERE/'results_seed17/local_refinement/sampling/posterior_samples.npy'),'#B65D24',':'),
        ('Fixed noise + 128 points' if refined else 'Fixed noise, longer fit',np.load(out/('added_points' if refined else 'fixed_noise_2000')/'sampling/posterior_samples.npy'),'#3569B0','--')]
    fig,axes=plt.subplots(1,4,figsize=(12,3.6))
    for i,(ax,label,bounds) in enumerate(zip(axes,[r'$\alpha$',r'$\sigma$',r'$a_{\rm SFR}$',r'$d_{\rm SFR}$'],[(-.5,-.001),(.1,.6),(.005,.015),(4.2,5.2)])):
        for name,draws,color,style in series:
            ax.hist(draws[:,i],bins=np.linspace(*bounds,40),density=True,histtype='step',color=color,ls=style,lw=1.6,label=name)
        ax.set(xlabel=label,xlim=bounds)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Posterior density')
    axes[2].ticklabel_format(axis='x',style='sci',scilimits=(0,0))
    fig.suptitle('Fixing the noise freeze removes most of the artificial narrowing',x=.07,ha='left',fontsize=13)
    fig.text(.07,.88,'Same grid and 81-event mock • all 356 old points retained, 128 added • validation remains incomplete' if refined else 'Same 356 training points, transform, grid and 81-event mock • remaining joint-likelihood error is unresolved',fontsize=9)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False)
    fig.tight_layout(rect=[0,.09,1,.85])
    fig.savefig(out/('posterior_noise_fix_refined.png' if refined else 'posterior_noise_fix.png'),dpi=180)
    fig.savefig(out/('posterior_noise_fix_refined.pdf' if refined else 'posterior_noise_fix.pdf'))

if __name__=='__main__':main()
