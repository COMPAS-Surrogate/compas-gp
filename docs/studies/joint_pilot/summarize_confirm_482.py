"""Assemble an evidence summary without excluding failed accuracy checks."""
import hashlib,json,importlib.metadata
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent


def main():
    root=HERE/'confirm_482_results';cases=json.loads((root/'cases.json').read_text());summary=json.loads((root/'summary.json').read_text())
    evaluated=[c for c in cases if c['status']=='evaluated'];metrics=[m for c in evaluated for m in c['metrics']]
    aggregate={'evaluated':len(evaluated),'event_range':[min(c['events'] for c in cases),max(c['events'] for c in cases)],'empty_catalogues':sum(c['events']==0 for c in cases),
        'max_mean_shift_sd':max(m['max_mean_shift_sd'] for m in metrics),'max_endpoint_shift_sd':max(m['max_endpoint_shift_sd'] for m in metrics),
        'width68_ratio_range':[float(np.min([m['width68_ratio'] for m in metrics])),float(np.max([m['width68_ratio'] for m in metrics]))],
        'width95_ratio_range':[float(np.min([m['width95_ratio'] for m in metrics])),float(np.max([m['width95_ratio'] for m in metrics]))],
        'forward_kl_median':float(np.median([np.mean([m['kl_direct_to_gp'] for m in c['metrics']]) for c in evaluated])),
        'forward_kl_max':max(m['kl_direct_to_gp'] for m in metrics),'reverse_kl_max':max(m['kl_gp_to_direct'] for m in metrics),
        'flagged_cases':[{'case':c['case'],'events':c['events'],'status':c['status'],'posterior_pass':c['posterior_pass'],'kl_pass':c['kl_pass'],'numerical_pass':c['numerical_pass']} for c in cases if not(c['posterior_pass'] and c['kl_pass'] and c['numerical_pass'])]}
    aggregate['coverage_disagreements_between_scrambles']={}
    for name in ['direct','gp']:
        for level,i,j in [(68,2,4),(90,1,5),(95,0,6)]:
            flags=[]
            for scramble in [0,1]:flags.append(np.array([(np.array(c['truth'])>=np.array(c['summaries'][scramble][name]['quantiles'])[i])&(np.array(c['truth'])<=np.array(c['summaries'][scramble][name]['quantiles'])[j]) for c in evaluated]))
            aggregate['coverage_disagreements_between_scrambles'][f'{name}_{level}']=np.sum(flags[0]!=flags[1],axis=0).tolist()
    (root/'aggregate.json').write_text(json.dumps(aggregate,indent=2)+'\n')
    sources=[HERE/'confirm_482.py',HERE/'bo_uncertainty_curve.py',HERE/'check_ensemble_quadrature.py',HERE/'plot_confirm_482.py',Path(__file__)]
    sources+=list((HERE.parents[2]/'src/cosmic_integration').rglob('*.py'))
    provenance={'source_sha256':{str(p.relative_to(HERE.parents[2])):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'versions':{p:importlib.metadata.version(p) for p in ['numpy','scipy','jax','gpjax','numpyro']}}
    (root/'evidence_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(aggregate,indent=2))

if __name__=='__main__':main()
