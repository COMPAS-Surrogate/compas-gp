"""Paired independent-scramble check for selected bootstrap replicates."""
import json
import numpy as np
from bootstrap_blocks import HERE, posterior_summary


def main():
    out=HERE/'block_results'
    counts=np.load(HERE.parent/'grid_resolution/results/pilot_catalogue.npz')['counts_mass10'].ravel()
    counts=counts[counts>0]
    coefficients=np.load(out/'bootstrap_coefficients.npy')[:8]
    summaries=[]
    for seed in [20260917,20260919]:
        with np.load(out/f'block_rates_{seed}.npz') as f:
            rates,phi,u=f['rates'],f['phi'],f['u']
        summaries.append([posterior_summary(np.einsum('nbk,b->nk',rates,c),counts,phi,u[:,3])[0] for c in coefficients])
    nominal=json.loads((out/'summary.json').read_text())['20260917']['nominal']
    sd=np.array(nominal['sd'])
    mean_error=np.array([np.array(a['mean'])-b['mean'] for a,b in zip(*summaries)])/sd
    endpoint_error=np.array([np.array(a['quantiles'])-b['quantiles'] for a,b in zip(*summaries)])/sd
    result={'replicates_checked':len(coefficients),'max_mean_difference_in_nominal_sd':np.max(abs(mean_error),axis=0).tolist(),
        'max_quantile_difference_in_nominal_sd':np.max(abs(endpoint_error),axis=(0,1)).tolist()}
    (out/'numerical_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result)

if __name__=='__main__':main()
