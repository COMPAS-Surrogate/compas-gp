"""Render the measured finite-population likelihood-loss comparison."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]


def main() -> None:
    source=HERE/'likelihood_loss_20260912/results.json'
    data=json.loads(source.read_text())
    assert data['state']=='complete' and all(r['numerically_resolved'] for r in data['curves'])
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.spines.top':True,'axes.spines.right':True,
        'xtick.top':True,'ytick.right':True,'xtick.direction':'in','ytick.direction':'in'})
    fig,ax=plt.subplots(figsize=(4.6,3.15),layout='constrained')
    for row,color,ls in zip(data['curves'],['#0072B2','#D55E00','#009E73'],['-','--','-.']):
        x=np.asarray(row['expected']);y=np.asarray(row['loss']);err=np.asarray(row['numerical_se'])
        ax.plot(x,y,color=color,ls=ls,lw=1.3,marker='x',ms=4,mew=1,label=f"{row['size']}M")
        ax.fill_between(x,y-2*err,y+2*err,color=color,alpha=.12,lw=0)
    ax.set(xscale='log',yscale='log',xlabel='Expected observed detections',ylabel='Expected log-likelihood loss [nats]')
    ax.set_xticks([10,30,100,300,1000,3000],[10,30,100,300,1000,3000])
    ax.set_yticks([.01,.1,1,10],['0.01','0.1','1','10'])
    ax.minorticks_on();ax.tick_params(which='both',top=True,right=True,direction='in')
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.legend(frameon=False,title='COMPAS systems',fontsize=9,title_fontsize=9,loc='upper left')
    dest=REPO/'overleaf/figures/bo_study'
    fig.savefig(dest/'population_likelihood_loss.pdf',bbox_inches='tight')
    fig.savefig(dest/'population_likelihood_loss.png',dpi=180,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    main()
