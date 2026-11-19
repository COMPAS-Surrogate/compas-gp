"""Audit matched acquisition designs and plot paired KL comparisons.

Scientific figure contract: forward KL against total labels, faceted by catalogue
and posterior target. Blue solid BO versus orange dashed Sobol; show every seed
thinly and the seed median boldly. Average numerical scrambles within each seed;
never count them as extra independent experiments. Export PNG/PDF and inspect.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc

HERE=Path(__file__).resolve().parent


def main():
    out=HERE/'acquisition_comparison_results';out.mkdir(exist_ok=True)
    roots={'bo':HERE/'bo_uncertainty_results','sobol':HERE/'sobol_uncertainty_results'}
    rows={k:json.loads((p/'curves.json').read_text()) for k,p in roots.items()}
    points=[226,354,482,610,738];seeds=[17,29,43]
    for case in range(3):
        for seed in seeds:
            initial=[np.load(root/f'case_{case}_seed_{seed}/checkpoint_226.npz') for root in roots.values()]
            np.testing.assert_array_equal(initial[0]['training_x'],initial[1]['training_x'])
            np.testing.assert_allclose(initial[0]['training_lnl'],initial[1]['training_lnl'],rtol=0,atol=1e-12)
            candidate=np.array([-.5,.1,4.2])+qmc.Sobol(3,scramble=True,seed=seed+100).random_base2(12)*np.array([.499,.5,1.])
            for strategy,root in roots.items():
                previous=None
                for n in points:
                    a=np.load(root/f'case_{case}_seed_{seed}/checkpoint_{n}.npz');x=a['training_x']
                    assert len(x)==n and len(np.unique(x,axis=0))==n
                    if previous is not None:np.testing.assert_array_equal(x[:len(previous)],previous)
                    if strategy=='sobol':np.testing.assert_array_equal(x[226:],candidate[:n-226])
                    previous=x
    def paired_values(strategy,target,n,key,case=None):
        return np.array([np.mean([r[key] for r in rows[strategy] if r['case']==c and r['seed']==seed and r['points']==n and r['target']==target]) for c in (range(3) if case is None else [case]) for seed in seeds])
    comparison=[]
    for target in ['mean','expected_likelihood']:
        for n in points:
            for key in ['kl_direct_to_gp','kl_gp_to_direct']:
                b=paired_values('bo',target,n,key);s=paired_values('sobol',target,n,key)
                if n==226:np.testing.assert_allclose(b,s,rtol=1e-10,atol=1e-12)
                comparison.append({'target':target,'points':n,'metric':key,'bo_median':float(np.median(b)),
                    'sobol_median':float(np.median(s)),'median_paired_bo_over_sobol':float(np.median(b/s)),
                    'bo_lower_trajectories':int(np.sum(b<s)),'sobol_lower_trajectories':int(np.sum(s<b)),
                    'bo_values':b.tolist(),'sobol_values':s.tolist()})
    summary={'paired_trajectories':9,'numerical_scrambles_per_trajectory':2,'audits_pass':True,'comparisons':comparison,
             'scope':'Three catalogues, three seeds. Fixed Sobol prefixes of the identical candidate pool, matching initial points and budgets. No universal BO optimality or wall-time speedup claim.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,axes=plt.subplots(2,3,figsize=(12,7),sharex=True,constrained_layout=True)
    for case in range(3):
        events=next(r['events'] for r in rows['bo'] if r['case']==case)
        for j,target in enumerate(['mean','expected_likelihood']):
            ax=axes[j,case]
            for strategy,color,style,label in [('bo','#326bb0','-','BO: EI + variance'),('sobol','#bd641e','--','Space-filling Sobol')]:
                values=np.array([paired_values(strategy,target,n,'kl_direct_to_gp',case) for n in points]).T
                for y in values:ax.plot(points,y,color=color,ls=style,lw=1,alpha=.3)
                ax.plot(points,np.median(values,axis=0),color=color,ls=style,lw=2.2,marker='o',ms=4,label=label)
            ax.set_yscale('log');ax.grid(axis='y',alpha=.15)
            ax.set_title(f'{events} events — '+('mean target' if j==0 else 'expected-likelihood target'))
            if j==1:ax.set_xlabel('Total training points (including anchors)')
            if case==0:ax.set_ylabel('KL(direct ‖ GP), nats')
    axes[0,0].legend(frameon=False,fontsize=8)
    fig.suptitle('Matched acquisition comparison\nThree seeds per catalogue • thin: individual seeds • thick: medians',fontsize=14)
    for ext in ['png','pdf']:fig.savefig(out/f'acquisition_kl.{ext}',dpi=180)
    plt.close(fig)
    print(json.dumps([r for r in comparison if r['points']==738],indent=2))

if __name__=='__main__':main()
