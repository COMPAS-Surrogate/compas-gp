"""Replot saved validation results; no likelihood evaluations or GP refitting.

Run from any directory with the project Python environment. All numerical inputs
are retained separately; the manifest records hashes and successful-run counts.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np

REPO = Path(__file__).resolve().parents[3]
FIG = REPO / 'overleaf/figures/bo_study'
OUT = Path(__file__).parent / 'scope_cleanup_20260916'
COLORS = ['#0072B2', '#D55E00', '#009E73', '#9467B0']


def save(fig: plt.Figure, name: str) -> None:
    for ax in fig.axes:
        ax.minorticks_on()
        ax.tick_params(which='both', direction='in', top=True, right=True)
    for ext in ('pdf', 'png'):
        fig.savefig(FIG / f'{name}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)


def acquisition_summary(records: list[dict], counts: list[int]) -> list[dict]:
    """Summarize all five catalogues per view/count at their frozen audit stops."""
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.1), sharex='col',
        sharey='row', gridspec_kw={'height_ratios': [2.6, 1]}, layout='constrained')
    quantiles = []
    for col, mode in enumerate(('perfect', 'uncertain')):
        top, bottom = axes[:, col]
        for j, (color, ls) in enumerate(zip(COLORS, ['-', '--', '-.', ':'])):
            vals = [[r['width68'][j] for r in records
                     if (r['mode'], r['expected']) == (mode, n)] for n in counts]
            assert all(len(v) == 5 for v in vals)
            lo, med, hi = np.quantile(vals, [.1, .5, .9], axis=1)
            top.plot(counts, med, color=color, ls=ls, marker='o', ms=3,
                     label=[r'$\alpha$', r'$\sigma$', r'$a$', r'$d$'][j])
            top.fill_between(counts, lo, hi, color=color, alpha=.10, lw=0)
            quantiles.append(dict(mode=mode, metric=['alpha', 'sigma', 'a', 'd'][j],
                                  quantiles=np.array([lo, med, hi]).T.tolist()))
        top.axhline(1, color='.5', ls=':', lw=.75)
        top.set_ylim(0, 1.08)
        vals = [[r['training'] for r in records
                 if (r['mode'], r['expected']) == (mode, n)] for n in counts]
        assert all(len(v) == 5 for v in vals)
        lo, med, hi = np.quantile(vals, [.1, .5, .9], axis=1)
        bottom.plot(counts, med, color=COLORS[0], marker='o', ms=3)
        bottom.fill_between(counts, lo, hi, color=COLORS[0], alpha=.12, lw=0)
        quantiles.append(dict(mode=mode, metric='training',
                              quantiles=np.array([lo, med, hi]).T.tolist()))
        bottom.set(ylim=(200, 500), yticks=[200, 300, 400, 500],
                   xlabel=r'$\mathbb{E}[N_{\rm obs}]$')
        for ax in (top, bottom):
            ax.set_xscale('log')
            ax.set_xticks(counts, [10, 50, 100, 200, 500, '', 1000])
            ax.xaxis.set_minor_formatter(NullFormatter())
    axes[0, 0].set_ylabel(r'$W_{68}/W_{68}^{\rm prior}$')
    axes[0, 0].legend(frameon=False, ncol=2, loc='lower left', fontsize=9)
    axes[1, 0].set_ylabel(r'$N_{\rm train}^{\rm sel}$')
    save(fig, 'acquisition_summary')
    return quantiles


def stopping_cost(rows: list[dict]) -> list[dict]:
    """Plot successful runs only; record omitted counts instead of censoring them."""
    fig, axis = plt.subplots(figsize=(4.6, 3.0), layout='constrained')
    axes = [axis]
    summary = []
    for ax, mode in zip(axes, ('perfect',)):
        counts = sorted({r['exposure'] for r in rows if r['mode'] == mode})
        for strategy, color, style, label in zip(
                ('bo', 'random'), COLORS[:2], ('-', '--'), ('BO', 'Random')):
            q = []
            for n in counts:
                group = [r for r in rows if (r['mode'], r['exposure'], r['strategy'])
                         == (mode, n, strategy)]
                good = [r['points'] for r in group if r['state'] == 'selected']
                assert len(group) == 100 and good
                values = np.quantile(good, [.1, .5, .9])
                q.append(values)
                summary.append(dict(mode=mode, expected=n, strategy=strategy,
                    selected=len(good), omitted=len(group)-len(good), quantiles=values.tolist()))
            lo, med, hi = np.array(q).T
            ax.plot(counts, med, color=color, ls=style, marker='o', ms=4, label=label)
            ax.fill_between(counts, lo, hi, color=color, alpha=.12, lw=0)
        ax.set(xscale='log', xlim=(25, 1200), ylim=(300, 2700),
               xlabel=r'$\mathbb{E}[N_{\rm obs}]$')
        ax.set_xticks([30, 100, 300, 1000], [30, 100, 300, 1000])
        ax.xaxis.set_minor_formatter(NullFormatter())
    axes[0].set_ylabel(r'$N_{\rm train}^{\rm acc}$')
    axes[0].legend(frameon=False, fontsize=9)
    save(fig, 'stopping_budget')
    return summary


def main() -> None:
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42,
                         'axes.spines.top': True, 'axes.spines.right': True})
    OUT.mkdir(exist_ok=True)
    paths = [REPO/'docs/studies/paper_completion/reference_kl_20260912/ensemble_outcomes.json',
             REPO/'docs/studies/operational_stopping/gpry_compas/count_history_analysis_20260913/paired_acquisition_summary.json']
    stops = stopping_cost(json.loads(paths[0].read_text()))
    paired = json.loads(paths[1].read_text())
    widths = acquisition_summary(paired['rows'], paired['expected_counts'])
    assert sum(r['omitted'] for r in stops) == 3
    manifest = dict(stopping_budget=stops, paired_summary=widths,
        source_sha256={str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in paths + [Path(__file__)]},
        scope='Replot saved results. Accuracy-budget plot shows perfect measurements only; '
              'percentiles omit unsuccessful runs, counted explicitly above. Paired width figure '
              'retains both measurement models and the original selected checkpoints.')
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
