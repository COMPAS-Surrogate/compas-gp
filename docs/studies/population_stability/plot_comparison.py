"""Same-catalogue marginal intervals for nested and disjoint populations."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent

def main():
    data=json.loads((HERE/'results/summary.json').read_text())['populations']
    fig,axes=plt.subplots(1,4,figsize=(12,3.6))
    names=['32M','512M','480M_disjoint']
    colors=['#262626','#3569B0','#B65D24']
    for i,(ax,label) in enumerate(zip(axes,[r'$\alpha$',r'$\sigma$',r'$a_{\rm SFR}$',r'$d_{\rm SFR}$'])):
        for row,(name,color) in enumerate(zip(names,colors)):
            q=np.array(data[name]['runs'][0]['quantiles'])[:,i]
            ax.plot(q[[0,4]],[row,row],color=color,lw=1)
            ax.plot(q[[1,3]],[row,row],color=color,lw=4,solid_capstyle='butt')
            ax.plot(q[2],row,'o',color=color,ms=5)
        ax.set(xlabel=label,yticks=[0,1,2],ylim=(2.6,-.6))
        ax.set_yticklabels(['32M','512M','480M disjoint'] if i==0 else [])
        ax.spines[['top','right','left']].set_visible(False)
        ax.tick_params(axis='y',length=0)
    axes[2].ticklabel_format(axis='x',style='sci',scilimits=(0,0))
    fig.suptitle('Population-size sensitivity of the direct posterior',x=.12,ha='left',fontsize=14)
    fig.text(.12,.88,'Same 81-event mock and coarse grid • median, 68% (thick) and 95% (thin) intervals',fontsize=10)
    fig.text(.12,.025,'512M includes 32M; 480M excludes those shared systems. One catalogue; calibration remains untested.',fontsize=9)
    fig.tight_layout(rect=[0,.08,1,.85])
    fig.savefig(HERE/'results/posterior_comparison.png',dpi=180)
    fig.savefig(HERE/'results/posterior_comparison.pdf')

if __name__=='__main__':main()
