"""Display conditional finite-population sensitivity, not confidence coverage."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent

def main():
    out=HERE/'block_results'
    data=json.loads((out/'summary.json').read_text())['20260917']
    center=np.array(data['nominal']['mean']);sd=np.array(data['nominal']['sd'])
    blocks=(np.array([b['mean'] for b in data['blocks']])-center)/sd
    bootstrap=(np.load(out/'bootstrap_posteriors.npz')['means']-center)/sd
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),sharey=True)
    rng=np.random.default_rng(0)
    for ax,values,title,color in zip(axes,[blocks,bootstrap],['16 disjoint 32M populations','128 bootstrap populations, each 512M'],['#555555','#3569B0']):
        ax.axhline(0,color='#777777',lw=1,ls='--')
        for i in range(4):
            ax.scatter(i+rng.uniform(-.12,.12,len(values)),values[:,i],s=16,alpha=.55,color=color,edgecolors='none')
            ax.plot([i-.2,i+.2],[np.median(values[:,i])]*2,color='#222222',lw=1.5)
        ax.set_xticks(range(4),[r'$\alpha$',r'$\sigma$',r'$a_{\rm SFR}$',r'$d_{\rm SFR}$'])
        ax.set_title(title,fontsize=11,loc='left')
        ax.set_ylim(-.85,.85)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Posterior mean shift / full-512M posterior SD')
    fig.suptitle('Finite-population sensitivity is smaller at the full 512M size',x=.08,ha='left',fontsize=13)
    fig.text(.08,.9,'Same 81-event mock • direct likelihood • each point is a population reanalysis',fontsize=10)
    fig.text(.08,.025,'Conditional on the existing simulation campaign; this does not establish repeated-catalogue coverage.',fontsize=9)
    fig.tight_layout(rect=[0,.07,1,.86])
    fig.savefig(out/'bootstrap_sensitivity.png',dpi=180)
    fig.savefig(out/'bootstrap_sensitivity.pdf')

if __name__=='__main__':main()
