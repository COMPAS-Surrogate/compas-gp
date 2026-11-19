"""Direct-likelihood calibration control: exact binomial interval comparison.

One panel per nominal coverage, four parameters per panel. Black markers/error
bars and grey nominal reference; identical percentage scale. Explicitly direct
only, with no GP coverage claim. Save PNG/PDF and visually inspect PNG.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent


def main():
    root=HERE/'direct_rank_control_results';s=json.loads((root/'summary.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(11,4),sharey=True,constrained_layout=True)
    x=np.arange(4)
    for ax,level in zip(axes,[68,90,95]):
        y=np.array(s['coverage'][str(level)])/s['catalogues']*100;ci=np.array(s['coverage_exact95_intervals'][str(level)])*100
        ax.errorbar(x,y,yerr=np.array([y-ci[:,0],ci[:,1]-y]),fmt='o',color='#222222',capsize=4)
        ax.axhline(level,color='.5',ls='--',lw=1)
        ax.set_xticks(x,[r'$\alpha$',r'$\sigma$',r'$a_{\rm SFR}$',r'$d_{\rm SFR}$']);ax.set_title(f'{level}% credible interval')
        ax.set_ylim(55,100);ax.grid(axis='y',alpha=.15)
    axes[0].set_ylabel('Coverage (%) with exact 95% intervals')
    fig.suptitle('Direct-likelihood internal calibration\n1,000 fresh catalogues • same 256M population for generation and inference • no GP',fontsize=13)
    for ext in ['png','pdf']:fig.savefig(root/f'coverage.{ext}',dpi=180)
    plt.close(fig)

if __name__=='__main__':main()
