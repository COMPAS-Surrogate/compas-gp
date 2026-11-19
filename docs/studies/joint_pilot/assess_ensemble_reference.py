"""Compare stored GP intervals with a numerically refined direct reference."""
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest

HERE=Path(__file__).resolve().parent

def main():
    root=HERE/'reduced_calibration_100_results'
    cases=json.loads((root/'cases.json').read_text())
    references=[json.loads((root/f'direct_reference_{seed}.json').read_text()) for seed in [20261003,20261004]]
    checks=[];coverage={};changed={}
    for level,lo,hi in [(68,2,4),(90,1,5),(95,0,6)]:
        masks=[]
        for ref in references:
            masks.append(np.array([(np.array(c['truth'])>=np.array(r['quantiles'])[lo])&(np.array(c['truth'])<=np.array(r['quantiles'])[hi]) for c,r in zip(cases,ref)]))
        coverage[f'direct_cover{level}']=masks[0].sum(0).tolist()
        changed[str(level)]=np.sum(masks[0]!=masks[1],axis=0).tolist()
    for c,r in zip(cases,references[0]):
        dq=np.array(r['quantiles'])[[0,2,3,4,6]];gq=np.array(c['gp_quantiles']);sd=np.array(r['sd'])
        row={'case':c['case'],'max_endpoint_shift_sd':float(np.max(abs(gq-dq)/sd)),
            'width68_ratio':((gq[3]-gq[1])/(dq[3]-dq[1])).tolist(),
            'width95_ratio':((gq[4]-gq[0])/(dq[4]-dq[0])).tolist()}
        row['interval_pass']=bool(row['max_endpoint_shift_sd']<.2 and np.max(abs(np.array(row['width68_ratio'])-1))<.1 and np.max(abs(np.array(row['width95_ratio'])-1))<.1)
        checks.append(row)
    original=json.loads((root/'summary.json').read_text())
    result={'primary_reference_seed':20261003,'checks':checks,'interval_passes':sum(c['interval_pass'] for c in checks),
        'direct_coverage_counts':coverage,'reference_coverage_disagreements':changed,
        'direct_coverage_exact95_intervals':{key:[list(binomtest(k,len(cases)).proportion_ci()) for k in counts] for key,counts in coverage.items()},
        'scope':'Stored 8192-point GP intervals compared against 32768-point direct quadrature with analytic amplitude CDF. GP not retrained or re-integrated.'}
    (root/'reference_assessment.json').write_text(json.dumps(result,indent=2)+'\n')
    sources=[Path(__file__),HERE/'reduced_calibration_pilot.py',HERE/'check_ensemble_quadrature.py',HERE/'plot_calibration_ensemble.py']
    sources+=list((HERE.parents[2]/'src/cosmic_integration/lnl_surrogate').glob('*.py'))
    (root/'evidence_source_hashes.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))

if __name__=='__main__':main()
