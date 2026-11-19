"""Scientific KL curves: three catalogue facets, two KL directions.

Solid blue mean target versus dashed orange expected-likelihood target. Thin
lines show training seeds (averaged over the two validation scrambles); thick
lines show seed medians. The comparison is paired at identical training points.
Save PNG/PDF and inspect the PNG; no uncertainty-band/coverage claim.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent

def main():
    root=HERE/'bo_uncertainty_results';rows=json.loads((root/'curves.json').read_text())
    fig,axes=plt.subplots(2,3,figsize=(12,7),sharex=True,constrained_layout=True)
    points=sorted({r['points'] for r in rows})
    for case in range(3):
        events=next(r['events'] for r in rows if r['case']==case)
        for j,key in enumerate(['kl_direct_to_gp','kl_gp_to_direct']):
            ax=axes[j,case]
            for target,color,style,label in [('mean','#326bb0','-','GP mean'),('expected_likelihood','#bd641e','--','Expected likelihood')]:
                series=[]
                for seed in [17,29,43]:
                    y=[np.mean([r[key] for r in rows if r['case']==case and r['seed']==seed and r['points']==n and r['target']==target]) for n in points]
                    series.append(y);ax.plot(points,y,color=color,ls=style,lw=1,alpha=.3)
                ax.plot(points,np.median(series,axis=0),color=color,ls=style,lw=2.2,marker='o',ms=4,label=label)
            ax.set_yscale('log');ax.grid(axis='y',alpha=.15)
            ax.set_title(f'Catalogue {case+1}: {events} events' if j==0 else '')
            if j==1:ax.set_xlabel('Total training points (including 98 anchors)')
            if case==0:ax.set_ylabel('KL(direct ‖ GP), nats' if j==0 else 'KL(GP ‖ direct), nats')
    axes[0,0].legend(frameon=False,fontsize=9)
    fig.suptitle('Posterior error during batched acquisition\nThree training seeds per catalogue • thick lines: medians • fixed coarse grid',fontsize=14)
    for ext in ['png','pdf']:fig.savefig(root/f'kl_curves.{ext}',dpi=180)
    plt.close(fig)

if __name__=='__main__':main()
