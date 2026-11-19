"""Paired label-budget audit with a retrospective, explicitly exploratory gate."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent


def main():
    strategies=['bo','sobol','posterior_variance'];points=[226,354,482,610,738]
    rows={s:json.loads((HERE/f'{s}_uncertainty_results/curves.json').read_text()) for s in strategies}
    out=HERE/'label_efficiency_results';out.mkdir(exist_ok=True)
    result=[]
    for strategy in strategies:
        for c in range(3):
            for seed in [17,29,43]:
                old=None
                for n in points:
                    d=np.load(HERE/f'{strategy}_uncertainty_results/case_{c}_seed_{seed}/checkpoint_{n}.npz')
                    x=d['training_x'];assert len(x)==n and len(np.unique(x,axis=0))==n
                    if old is not None:np.testing.assert_array_equal(x[:len(old)],old)
                    if n==226:
                        ref=np.load(HERE/f'bo_uncertainty_results/case_{c}_seed_{seed}/checkpoint_226.npz')
                        np.testing.assert_array_equal(x,ref['training_x'])
                        np.testing.assert_allclose(d['training_lnl'],ref['training_lnl'],atol=1e-12,rtol=0)
                    old=x
        for target in ['mean','expected_likelihood']:
            first={}
            for n in points:
                vals=[];passed=[]
                for c in range(3):
                    for seed in [17,29,43]:
                        a=[r for r in rows[strategy] if r['case']==c and r['seed']==seed and r['points']==n and r['target']==target]
                        assert len(a)==2
                        vals.append(np.mean([r['kl_direct_to_gp'] for r in a]))
                        ok=all(r['kl_direct_to_gp']<.01 and r['kl_gp_to_direct']<.01 and r['max_shape_mean_shift_sd']<.2 and np.max(abs(np.array(r['shape_width68_ratio'])-1))<.1 for r in a)
                        passed.append(ok)
                        if ok:first.setdefault(f'{c}_{seed}',n)
                result.append({'strategy':strategy,'target':target,'points':n,'passes':sum(passed),'median_forward_kl':float(np.median(vals)),'max_forward_kl':float(np.max(vals))})
            (out/f'{strategy}_{target}_first_pass.json').write_text(json.dumps(first,indent=2)+'\n')
    (out/'summary.json').write_text(json.dumps({'audits_pass':True,'records':result,'gate':'Both KL directions <.01 nats, shape means <.2 direct SD, shape68 widths within10%, both numerical scrambles. Retrospective only, not online stopping or full posterior calibration.'},indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for s,color,style,label in [('bo','#326bb0','-','EI + variance'),('sobol','#777777',':','Sobol'),('posterior_variance','#bd641e','--','Likelihood variance')]:
        a=[r for r in result if r['strategy']==s and r['target']=='mean']
        axes[0].plot(points,[r['median_forward_kl'] for r in a],color=color,ls=style,marker='o',label=label)
        axes[1].plot(points,[r['passes'] for r in a],color=color,ls=style,marker='o')
    axes[0].set(yscale='log',ylabel='Median KL(direct ‖ GP), nats');axes[0].legend(frameon=False,fontsize=9)
    axes[1].set(ylabel='Trajectories meeting exploratory gate',ylim=(-.3,9.3),yticks=[0,3,6,9])
    for ax in axes:ax.set_xlabel('Total training labels (including anchors)');ax.grid(axis='y',alpha=.15)
    fig.suptitle('Training-label efficiency on three simulated catalogues\nMean target • three seeds each • two numerical checks per trajectory',fontsize=13)
    for ext in ['png','pdf']:fig.savefig(out/f'label_efficiency.{ext}',dpi=180)
    plt.close(fig)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
