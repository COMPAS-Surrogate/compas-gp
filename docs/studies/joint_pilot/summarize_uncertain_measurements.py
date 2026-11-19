"""Aggregate retained uncertainty validation and predictive diagnostics."""
import argparse
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', default='uncertain_100_results')
    args = parser.parse_args()
    root = HERE/args.outdir
    records = json.loads((root/'cases.json').read_text())
    output = {}
    for mode in sorted({r['mode'] for r in records}):
        rows = [r for r in records if r['mode'] == mode]
        valid = [r for r in rows if r['status'] == 'evaluated']
        metrics = [m for r in valid for m in r['metrics']]
        if not metrics:
            output[mode] = {'evaluated': 0, 'attempted': len(rows)}
            continue
        entry = {'attempted': len(rows), 'evaluated': len(valid),
                 'event_range': [min(r['events'] for r in rows), max(r['events'] for r in rows)],
                 'empty_catalogues': sum(r['events'] == 0 for r in rows),
                 'failed_cases': [r['case'] for r in rows if r['status'] != 'evaluated'],
                 'posterior_gate_failures': [r['case'] for r in rows if not r['posterior_pass']],
                 'kl_gate_failures': [r['case'] for r in rows if not r['kl_pass']],
                 'numerical_gate_failures': [r['case'] for r in rows if not r['numerical_pass']],
                 'max_mean_shift_sd': max(m['max_mean_shift_sd'] for m in metrics),
                 'max_endpoint_shift_sd': max(m['max_endpoint_shift_sd'] for m in metrics),
                 'median_case_max_forward_kl': float(np.median([max(m['kl_direct_to_gp'] for m in r['metrics']) for r in valid])),
                 'max_forward_kl': max(m['kl_direct_to_gp'] for m in metrics),
                 'max_reverse_kl': max(m['kl_gp_to_direct'] for m in metrics)}
        for level in [68, 95]:
            widths = np.array([m[f'width{level}_ratio'] for m in metrics])
            entry[f'width{level}_ratio_range'] = [float(widths.min()), float(widths.max())]
        for name in ['direct', 'gp']:
            ranks = np.array([r['summaries'][0][name]['ranks'] for r in valid])
            other = np.array([r['summaries'][1][name]['ranks'] for r in valid])
            entry[f'{name}_max_rank_difference_between_scrambles'] = np.max(abs(ranks-other), axis=0).tolist()
            ordered = np.sort(ranks, axis=0)
            n = len(valid)
            dkw = np.maximum(np.max(np.arange(1, n+1)[:, None]/n-ordered, axis=0),
                             np.max(ordered-np.arange(n)[:, None]/n, axis=0))
            entry[f'{name}_max_cdf_deviation'] = dkw.tolist()
        entry['dkw_per_panel_95_threshold'] = float(np.sqrt(np.log(40)/(2*len(valid))))
        output[mode] = entry
    if 'perfect' in output:
        paired = []
        for case in sorted({r['case'] for r in records}):
            pair = {r['mode']: r for r in records if r['case'] == case and r['status'] == 'evaluated'}
            if len(pair) == 2:
                q = {mode: np.array(row['summaries'][0]['direct']['quantiles']) for mode, row in pair.items()}
                paired.append({'case': case, 'events': pair['perfect']['events'],
                    'uncertain_over_perfect_width68': ((q['uncertain'][4]-q['uncertain'][2])/(q['perfect'][4]-q['perfect'][2])).tolist()})
        output['paired_direct_widths'] = paired
    ppc = {}
    for case in range(3):
        path = root/f'uncertain_{case:03d}'/'predictive.json'
        if path.exists():
            data = json.loads(path.read_text())
            ppc[case] = {name: {key: value for key, value in result.items()
                               if key in ['statistic_names', 'observed_statistics', 'tail_probability_ge_observed']}
                         for name, result in data.items()}
    output['posterior_predictive'] = ppc
    (root/'aggregate.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({k: v for k, v in output.items() if k != 'posterior_predictive'}, indent=2))


if __name__ == '__main__':
    main()
