"""Rebuild rate maps and predictive checks at the shared reference-KL target."""
from __future__ import annotations

import json
import shutil

import weight_rate_figures as rates
from reference_selection import select_checkpoint


def main() -> None:
    folder = rates.CAMPAIGN / 'pilot'
    output = rates.HERE / 'reference_kl_20260912/predictive'
    output.mkdir(parents=True, exist_ok=True)
    rows = json.loads((folder/'records.json').read_text())
    selections = {}
    for expected in [30, 1200]:
        for mode in rates.MODES:
            subset = [r for r in rows if (r['exposure'], r['mode'], r['case'], r['strategy'])
                      == (expected, mode, 0, 'bo')]
            chosen = select_checkpoint(subset)
            if chosen is None:
                raise ValueError('No comparison checkpoint for predictive example')
            selections[expected, mode] = chosen['points']
    for mode in rates.MODES:
        print(rates.reconstruct(folder, mode, output,
                                checkpoint_points=selections[30, mode]), flush=True)
    rates.plot_figures(folder, output)
    rates.diagnostics.plot_predictive(folder, output, checkpoint_points=selections)
    (output/'selection.json').write_text(json.dumps(
        {f'n{n}_{m}':p for (n,m),p in selections.items()}, indent=2)+'\n')
    destination = rates.REPO / 'overleaf/figures/bo_study'
    for name in ['paired_rate_posterior', 'posterior_predictive_30', 'posterior_predictive_1200']:
        for extension in ['pdf', 'png']:
            shutil.copy2(output/f'{name}.{extension}', destination/f'{name}.{extension}')


if __name__ == '__main__':
    main()
