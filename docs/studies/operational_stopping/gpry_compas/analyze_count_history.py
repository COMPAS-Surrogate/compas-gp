"""Validate and summarize the completed paired campaign without changing inputs."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import plot_count_history as plot


def paired_differences(results: list[dict]) -> list[int]:
    """Return uncertain minus perfect training costs, matched by count and case."""
    pairs = {}
    for r in results:
        key = (r['expected_events'], r['case'])
        pair = pairs.setdefault(key, {})
        if r['mode'] in pair:
            raise ValueError('Duplicate measurement view')
        pair[r['mode']] = r['stop']['selected']['training']
    if any(set(p) != set(plot.MODES) for p in pairs.values()):
        raise ValueError('Incomplete measurement pair')
    return [p['uncertain'] - p['perfect'] for _, p in sorted(pairs.items())]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    snapshot = plot.collect_snapshot(root)
    results = [c['result'] for c in snapshot['cases']]
    expected = {j['name'] for j in snapshot['manifest']['jobs']}
    assert snapshot['status']['state'] == 'complete'
    assert {r['name'] for r in results} == expected and len(results) == len(expected)
    exceptions, rows, stops, final_kls = [], [], [], []
    for c in snapshot['cases']:
        r = c['result']
        history = json.loads((root/'offline'/r['name']/'history.json').read_text())
        frozen = json.loads((root/'stops'/f"{r['name']}.json").read_text())
        assert frozen == r['stop'] and not frozen['reference_accessed']
        s = next(h for h in history if h['points'] == frozen['selected']['training'])
        assert s['accuracy_pass'] and all(m['passed'] for m in s['four_metrics'])
        assert all(s[k] for k in ('reference_numerical', 'gp_numerical', 'four_numerical'))
        assert s['maximum_directed_kl'] < .01
        late = json.loads((root/'offline'/r['name']/'late_reference.json').read_text())
        assert late['accepted'] == r['late_reference_accepted']
        assert late['accepted'] == bool(history[-1]['accuracy_pass'] and history[-1]['maximum_directed_kl'] < .001)
        final_kls.append(history[-1]['maximum_directed_kl'])
        stops.append(s)
        if not late['accepted']:
            exceptions.append(f"Late reference rejected: {r['name']}; endpoint maximum directed KL = {final_kls[-1]:.8g} nats (required <0.001).")
        for h in history:
            if not all(h[k] for k in ('reference_numerical', 'gp_numerical', 'four_numerical')):
                exceptions.append(f"Numerically unresolved checkpoint: {r['name']}, {h['points']} training labels.")
        rows.append(dict(name=r['name'], expected_events=r['expected_events'], mode=r['mode'],
            case=r['case'], **frozen['selected'], comparisons=frozen['posterior_comparisons'],
            maximum_directed_kl=s['maximum_directed_kl'], symmetric_kl=s['symmetric_kl'],
            late_reference_accepted=late['accepted']))
    differences = np.asarray(paired_differences(results))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'snapshot.json').write_text(json.dumps(snapshot, indent=2)+'\n')
    with (args.output/'stops.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'stix', 'font.size': 11,
                         'axes.linewidth': .7, 'pdf.fonttype': 42})
    # Every trajectory is scored against a direct target, including the case
    # whose continued GP is not an accepted reference. No self-KL is plotted.
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.1), sharex=True, sharey=True)
    for ax, n in zip(axes.flat, plot.COUNTS):
        for mode in plot.MODES:
            group = plot.subset(snapshot, n, mode)
            curves = []
            for c in group:
                x, y, resolved = plot.curve_data(c['history'], 'symmetric_kl')
                curves.append(y)
                ax.plot(x, y, color=plot.COLORS[mode], alpha=.28, lw=.8,
                        ls='-' if mode == 'perfect' else '--')
                ax.plot(x[~resolved], y[~resolved], 'x', color=plot.COLORS[mode], ms=5, zorder=6)
                stop = c['result']['stop']['posterior_comparisons']
                ax.plot(stop, y[np.flatnonzero(x == stop)[0]],
                        'o' if mode == 'perfect' else 's', color=plot.COLORS[mode], mfc='white', ms=4)
            ax.plot(x, np.median(curves, axis=0), color=plot.COLORS[mode],
                    lw=1.8, ls='-' if mode == 'perfect' else '--')
        ax.set(yscale='log', xlim=(-.5, 40), ylim=(1e-5, 10), xticks=[0, 10, 20, 30, 40])
        ax.tick_params(which='both', direction='in', top=True, right=True)
        ax.grid(axis='y', alpha=.15)
    axes.flat[-1].axis('off')
    handles = [Line2D([], [], color=plot.COLORS[m], lw=1.8,
                      ls='-' if m == 'perfect' else '--', marker='o' if m == 'perfect' else 's',
                      mfc='white', label=plot.LABELS[m]) for m in plot.MODES]
    handles += [Line2D([], [], color='.4', marker='o', mfc='white', ls='', label='Recorded audit stop'),
                Line2D([], [], color='.4', marker='x', ls='', label='Numerically unresolved')]
    axes.flat[-1].legend(handles=handles, loc='center', frameon=False, fontsize=10)
    fig.supxlabel('Posterior comparisons', y=.02)
    fig.supylabel(r'Symmetric KL to direct posterior [nats]', x=.01)
    fig.subplots_adjust(left=.075, bottom=.12, top=.97, right=.99, hspace=.18, wspace=.13)
    plot.save(fig, args.output, 'ensemble_convergence_candidate')
    plot.stopping(snapshot, args.output)

    table = ['| Expected events | Perfect training / total | Uncertain training / total |',
             '|---:|---:|---:|']
    for n in plot.COUNTS:
        medians = []
        for mode in plot.MODES:
            group = [r for r in rows if r['expected_events'] == n and r['mode'] == mode]
            medians.append(' / '.join(str(int(np.median([r[k] for r in group]))) for k in ('training', 'total')))
        table.append(f'| {n} | {medians[0]} | {medians[1]} |')
    report = f'''# Completed paired count-history analysis

All {len(results)} jobs completed; saved outcomes: {dict(Counter(r['outcome'] for r in results))}.
There are 35 independent latent catalogues, each viewed with perfect and uncertain
measurements, at one fixed fiducial injection (five pairs per expected count).
All 2870 checkpoints are present. All 70 frozen shadow stops pass the direct
numerical, shape and four-parameter accuracy checks. The largest directed KL
at a stop is {max(r['maximum_directed_kl'] for r in rows):.8g} nats (limit 0.01).
The four-parameter checks require mean/endpoint shifts below 0.2 reference SD
and relative changes in both 68% and 95% interval widths below 10%.

## Evaluation costs

Medians across five catalogues per cell; total includes fresh fixed96 audits.

{chr(10).join(table)}

Training costs range from {min(r['training'] for r in rows)} to {max(r['training'] for r in rows)};
total policy costs range from {min(r['total'] for r in rows)} to {max(r['total'] for r in rows)}.
The stopping comparisons range from {min(r['comparisons'] for r in rows)} to {max(r['comparisons'] for r in rows)}
and correspond to 58 + 30 times the comparison count training labels.
Uncertain measurements stop earlier in {sum(differences < 0)} paired catalogues,
at the same checkpoint in {sum(differences == 0)}, and later in {sum(differences > 0)}.
Thus these runs do not support a general claim that measurement uncertainty
reduces acquisition cost. Costs rise modestly with count, with considerable
catalogue scatter and no clean monotonic trend at the larger counts.
Audits contribute {min(r['audit']/r['total'] for r in rows):.1%}–{max(r['audit']/r['total'] for r in rows):.1%}
of policy likelihood calls. Continuing every trajectory to 1258 training labels
is diagnostic work, not the policy stopping cost. Offline scoring, legacy
monitoring audits and reference construction are excluded from policy totals.
These label counts do not measure end-to-end speedup.

## Exceptions and evidence limits

{chr(10).join('- '+e for e in exceptions)}

69/70 continued GPs qualify as late references. The rejected endpoint still
passes the ordinary accuracy checks but misses the stricter late-reference KL
threshold. It is retained in both figures using the direct posterior target.
The two unresolved checkpoints are initialization states; none is an accepted
stop. Final self-KL is never evidence of accuracy. The direct-reference final
maximum directed KL ranges from {min(final_kls):.4g} to {max(final_kls):.4g} nats.
The stops were replayed and frozen before reference scoring; this is evidence
for this policy in a fixed-injection experiment, not prospective online timing,
prior-wide calibration, or a population-mismatch robustness test. Five pairs
per count cannot establish a small population-wide failure probability.
There is no random-acquisition comparator in this new campaign.

## Figure recommendation and captions

Current manuscript Figure 3 is `posterior_stop_uncertain.pdf`, paired with
Figure 2's perfect-measurement examples. Do not replace only that figure with
different catalogues while retaining its claim of matched latent events.
The new ensemble figure is a candidate addition or coordinated redesign of
the convergence presentation. Manuscript source and figures are unchanged.

**ensemble_convergence_candidate.pdf:** Symmetric KL to the direct numerical
posterior through all 40 posterior comparisons. Panels run left to right,
top to bottom through expected counts 10, 50, 100, 200, 500, 700 and 1000.
Each panel contains five paired catalogues: blue solid for perfect measurements
and orange dashed for uncertain measurements. Thin curves show every catalogue;
thick curves show medians. Open circles/squares mark the recorded shadow audit
stops; crosses mark unresolved numerical checks. Panel limits are shared.
The reference is the same direct target throughout each trajectory, including
the endpoint; comparison count excludes the 58-label initialization, and audit
costs are reported separately. Thin curves are catalogue variation, not
confidence intervals. No plotted reference value was used to select a stop.

**stopping_summary.pdf:** Each symbol is one catalogue's recorded stop, with
short horizontal strokes showing medians. Panels show comparison count,
training plus fresh audit labels, and maximum directed KL over both directions
and independent integration designs. This maximum differs from the symmetric
KL in the convergence figure. Expected counts are equally spaced categories.

Reproduce with `.venv/bin/python docs/studies/operational_stopping/gpry_compas/analyze_count_history.py
--root {args.root} --output NEW_OUTPUT_DIRECTORY`.
`snapshot.json` records input membership and outcome/history hashes;
`stops.csv` contains every plotted stop. Validation reads saved metrics and
frozen decisions; it does not rerun the exact likelihood or numerical quadrature.
'''
    (args.output/'ANALYSIS.md').write_text(report)
    print(report)


if __name__ == '__main__':
    main()
