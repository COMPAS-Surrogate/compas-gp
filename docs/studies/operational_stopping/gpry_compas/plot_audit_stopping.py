"""Plot measured audit statistics beside reference KL, without rerunning BO.

The recorded stops belong to the frozen combined rule. This renderer does not
apply or validate the proposed periodic, adaptive-size audit controller.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from cosmic_integration.lnl_surrogate.posterior_audit import importance_audit_check

BLUE, ORANGE = '#207eae', '#b56718'
GROUPS = [(100, 'perfect'), (100, 'uncertain'), (1000, 'perfect'), (1000, 'uncertain')]
KL_LIMIT, BOOTSTRAP_LIMIT = .003, .005


def reference_curve(rows: list[dict]) -> dict:
    """Retain both KL definitions and unresolved checkpoints explicitly."""
    ordered = sorted(rows, key=lambda r: r['points'])
    points = [r['points'] for r in ordered]
    if not points or len(points) != len(set(points)):
        raise ValueError('A trajectory needs unique scored checkpoints')
    return dict(points=points,
                maximum_directed_kl=[r['maximum_directed_kl'] for r in ordered],
                symmetric_kl=[r['symmetric_kl'] for r in ordered],
                unresolved=[not (r['reference_numerical'] and r['gp_numerical'])
                            for r in ordered])


def plotted_audits(record: dict) -> list[dict]:
    """Use recorded acceptance, never infer a stop from the KL checks alone."""
    selected = record['selected']
    answer = []
    for i, audit in enumerate(record['audits']):
        metric = audit['result']
        answer.append(dict(training=audit['training'], total=audit['training']+96*(i+1),
                           **metric, observed_stop=bool(selected and
                           audit['training'] == selected['training'])))
    if selected and not any(a['observed_stop'] and a['passed'] for a in answer):
        raise ValueError('The recorded stop must have an accepted audit')
    return answer


def choose_example(cases: dict) -> str:
    """Choose a confirmation case with a rejection and a stop near the median.

    This illustrative selection does not inspect the reference KL values.
    All cases, including non-stops, are shown in the companion figure.
    """
    stops = [c['selected']['training'] for c in cases.values() if c['selected']]
    middle = float(np.median(stops))
    eligible = [name for name, c in cases.items() if c['cohort'] == 'confirmation'
                and c['selected'] and any(not a['passed'] for a in c['audits'])]
    if not eligible:
        raise ValueError('No confirmation example contains both rejection and acceptance')
    return min(eligible, key=lambda n: (abs(cases[n]['selected']['training']-middle), n))


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(path.with_suffix('.png'), dpi=190, bbox_inches='tight')
    plt.close(fig)


def draw_case(ax: plt.Axes, case: dict, ylim: tuple[float, float]) -> None:
    curve = case['reference']
    x, y = np.asarray(curve['points']), np.asarray(curve['maximum_directed_kl'])
    ax.plot(x, y, color=BLUE, lw=1.6, zorder=2)
    unresolved = np.asarray(curve['unresolved'])
    ax.plot(x[unresolved], y[unresolved], 'o', ms=4, mfc='white', mec=BLUE, zorder=3)
    for a in case['audits']:
        n, estimate, upper = a['training'], a['maximum_directed_kl'], a['bootstrap_95_kl']
        ax.plot([n, n], [estimate, upper], color=ORANGE, lw=1, zorder=4)
        ax.plot(n, estimate, 'o', ms=5, color=ORANGE, zorder=5)
        ax.plot(n, upper, '^', ms=6, mfc='white', mec=ORANGE, zorder=5)
    ax.axhline(KL_LIMIT, color=ORANGE, lw=.9, alpha=.65)
    ax.axhline(BOOTSTRAP_LIMIT, color=ORANGE, ls='--', lw=.9, alpha=.65)
    if case['selected']:
        ax.axvline(case['selected']['training'], color='.25', ls=':', lw=1.4)
    ax.set(yscale='log', ylim=ylim, xlim=(50, 675))
    ax.set_xticks([100, 200, 300, 400, 500, 600])
    ax.grid(alpha=.16)
    ax.tick_params(which='both', direction='in', top=True, right=True)


def audit_handles() -> list[Line2D]:
    return [Line2D([], [], color=BLUE, label='Reference KL (validation only)'),
            Line2D([], [], color=ORANGE, marker='o', ls='none', label='Fresh-audit KL estimate'),
            Line2D([], [], color=ORANGE, marker='^', mfc='white', ls='none',
                   label='Audit 95th bootstrap percentile'),
            Line2D([], [], color='.25', ls=':', label='Observed stop (tested rule)')]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.input.resolve(), args.outdir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    hashes = {}

    def read(relative: str) -> object:
        raw = (root/relative).read_bytes()
        hashes[relative] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    decisions = read('decisions.json')
    mass = read('mass_audit/decisions.json')
    verification = read('verification.json')
    assert verification['state'] == 'passed'
    assert verification['decision_sha256'] == hashes['decisions.json']
    assert verification['mass_decision_sha256'] == hashes['mass_audit/decisions.json']
    rows = read('offline/records.json')
    outcomes = read('offline/outcomes.json')
    cases, recomputed = {}, 0
    for name, decision in decisions.items():
        if decision['strategy'] != 'bo':
            continue
        audit_record = read(f'mass_audit/{name}.json')
        assert audit_record['selected'] == mass[name]['selected']
        for audit in audit_record['audits']:
            check = importance_audit_check(audit['predicted'], audit['actual'],
                audit['log_proposal'], 202609116+audit['training'])
            assert check['passed'] == audit['result']['passed']
            for key, value in check.items():
                if key != 'passed':
                    np.testing.assert_allclose(value, audit['result'][key], rtol=1e-10, atol=1e-12)
            recomputed += 1
        outcome = next(o for o in outcomes if o['name'] == name and o['rule'] == 'mass_guarded')
        cases[name] = dict(truth_index=decision['truth_index'], expected_events=decision['expected_events'],
            mode=decision['mode'], case=decision['case'], cohort=mass[name]['cohort'],
            selected=mass[name]['selected'], outcome=outcome['outcome'],
            reference=reference_curve([r for r in rows if r['name'] == name]),
            audits=plotted_audits(audit_record))
    assert len(cases) == 24
    example = choose_example(cases)
    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'stix', 'font.size': 11,
        'axes.labelsize': 12, 'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True, 'axes.spines.top': True,
        'axes.spines.right': True, 'savefig.facecolor': 'white'})

    case = cases[example]
    maximum = max(case['reference']['maximum_directed_kl'])
    ylim = (1e-4, max(.1, 10**np.ceil(np.log10(maximum))))
    fig, ax = plt.subplots(figsize=(7.5, 4.9))
    draw_case(ax, case, ylim)
    ax.set_xlabel('Training likelihood evaluations')
    ax.set_ylabel(r'$D_{\mathrm{KL}}^{\max}$')
    ax.text(665, KL_LIMIT, r'$0.003$', fontsize=9, color=ORANGE, va='top', ha='right')
    ax.text(665, BOOTSTRAP_LIMIT, r'$0.005$', fontsize=9, color=ORANGE, va='bottom', ha='right')
    stop = case['selected']
    ax.text(stop['training']+10, .87, f"Stop at {stop['training']} training\n+ {stop['audit']} audit evaluations",
            transform=ax.get_xaxis_transform(), fontsize=10, va='top')
    fig.legend(handles=audit_handles(), loc='upper center', ncol=2, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .89))
    save(fig, out/'audit_kl_example')

    # Show every case so the illustrative choice cannot hide unsuccessful stops.
    upper = max(v for c in cases.values() for v in c['reference']['maximum_directed_kl'])
    lower = min(v for c in cases.values() for v in c['reference']['maximum_directed_kl'] if v > 0)
    all_ylim = (10**np.floor(np.log10(lower)), 10**np.ceil(np.log10(upper)))
    fig, axes = plt.subplots(6, 4, figsize=(13, 13), sharex=True, sharey=True)
    for name, c in cases.items():
        row = 2*c['truth_index']+c['case']
        col = GROUPS.index((c['expected_events'], c['mode']))
        ax = axes[row, col]
        draw_case(ax, c, all_ylim)
        label = f"Stop: {c['selected']['training']}" if c['selected'] else 'No accepted stop'
        ax.text(.96, .93, label, transform=ax.transAxes, ha='right', va='top', fontsize=9)
        ax.set_xticks([100, 300, 500])
    for i, ax in enumerate(axes[:, 0]):
        ax.set_ylabel(f'Injection {i//2+1}, catalogue {i%2+1}\n'+r'$D_{\mathrm{KL}}^{\max}$', fontsize=10)
    for ax, (n, mode) in zip(axes[-1], GROUPS):
        ax.set_xlabel(f'{n} expected events, {mode}\nTraining evaluations', fontsize=10)
    fig.legend(handles=audit_handles(), loc='upper center', ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .955))
    save(fig, out/'audit_kl_all_cases')

    # The GPry-style ensemble plot keeps the original symmetric KL definition.
    fig, axes = plt.subplots(2, 2, figsize=(8, 5.9), sharex=True, sharey=True)
    ensemble = []
    for ax, (n, mode) in zip(axes.flat, GROUPS):
        members = {name: c for name, c in cases.items() if (c['expected_events'], c['mode']) == (n, mode)}
        curves = [c['reference'] for c in members.values()]
        x = np.asarray(curves[0]['points'])
        assert all(c['points'] == x.tolist() for c in curves)
        lo, median, hi = np.quantile([c['symmetric_kl'] for c in curves], [.25, .5, .75], axis=0)
        stops = [c['selected']['training'] for c in members.values() if c['selected']]
        quantiles = np.quantile(stops, [.25, .5, .75])
        ax.axvspan(quantiles[0], quantiles[2], color='.88', zorder=0)
        ax.axvline(quantiles[1], color='.3', ls=':', lw=1.4)
        ax.plot(x, median, color=BLUE, lw=1.6)
        ax.fill_between(x, lo, hi, color=BLUE, alpha=.17, linewidth=0)
        unresolved = np.any([c['unresolved'] for c in curves], axis=0)
        ax.plot(x[unresolved], median[unresolved], 'o', ms=4, mfc='white', mec=BLUE)
        ax.text(.96, .94, f'{len(stops)}/{len(members)} accepted', transform=ax.transAxes,
                ha='right', va='top', fontsize=10)
        ax.set(yscale='log', ylim=all_ylim, xlim=(50, 675))
        ax.grid(alpha=.18)
        ensemble.append(dict(expected_events=n, mode=mode, names=list(members), points=x.tolist(),
            median=median.tolist(), q25=lo.tolist(), q75=hi.tolist(),
            stop_quantiles=quantiles.tolist(), accepted=len(stops), total=len(members)))
    for ax in axes[:, 0]:
        ax.set_ylabel(r'$D_{\mathrm{KL}}^{\mathrm{sym}}$')
    for ax, (_, mode) in zip(axes[-1], GROUPS):
        ax.set_xlabel(f'{mode.capitalize()} measurements\nTraining likelihood evaluations')
    for ax, n in zip(axes[:, 0], [100, 1000]):
        ax.text(.04, .94, f'{n} expected events', transform=ax.transAxes, va='top', fontsize=10)
    fig.legend(handles=[Line2D([], [], color=BLUE, label='BO median and interquartile range'),
        Line2D([], [], color='.3', ls=':', label='Accepted stop: median and interquartile range')],
        loc='upper center', ncol=1, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .90))
    save(fig, out/'ensemble_stops')

    data = dict(example=example, cases=cases, ensemble=ensemble,
                kl_limit=KL_LIMIT, bootstrap_limit=BOOTSTRAP_LIMIT)
    (out/'plot_data.json').write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    manifest = dict(source_root=str(root), source_sha256=hashes, audit_records_recomputed=recomputed,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        selection='Confirmation case with rejection and acceptance nearest median accepted training stop; lexical tie-break.',
        no_new_likelihood_evaluations=True, rule='Frozen mass_guarded combined rule; no revised controller applied')
    (out/'provenance.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (out/'CAPTIONS.md').write_text(f'''# Stopping diagnostics for the COMPAS surrogate

The proposed explanation is: **Stop when fresh likelihood checks show that
correcting the surrogate would barely change the posterior.** The figures below
show measurements and stopping decisions from the existing, more restrictive
combined rule. The periodic/adaptive-size audit revision has not been run.

## Audit overlaid on KL

![Audit and reference KL](audit_kl_example.png)

Illustrative confirmation case `{example}`: injection {case['truth_index']+1},
{case['expected_events']} expected events, {case['mode']} measurements, catalogue
{case['case']+1}. Blue is the reference discrepancy, computed only for validation.
Orange circles are estimates from fresh 96-point likelihood audits; open triangles
are the 95th percentiles of 256 stratified bootstrap resamples. Vertical orange
segments link the two statistics, not confidence intervals. The horizontal solid
and dashed orange lines are their respective acceptance thresholds, 0.003 and
0.005. The black dotted line is the recorded stop at {stop['training']} training
plus {stop['audit']} audit evaluations ({stop['total']} total).

Both reference and audit use the larger of the two directed KL divergences,
`D_max = max(KL(p || q), KL(q || p))`, rather than symmetric KL, to match the audit
criterion. The reference curve conservatively takes the largest value across
the two independent numerical integrations as well. The audit approximates the
discrepancy by importance weighting fresh exact likelihood evaluations against
the GP: `w_i proportional to exp(lnL_exact_i - lnL_GP_i)`, with the actual mixture
proposal accounted for. An additive constant cancels on normalization.

The tested rule also required the prediction counter, posterior stability,
pointwise residual and effective-sample-size checks. Crossing the orange lines
alone is therefore **not** a reconstruction of its stopping decision. Bootstrap
percentiles describe finite-audit variability; they are not guaranteed bounds on
the full posterior discrepancy or sequential confidence guarantees. The 4D
physical posterior uses a 3-input GP with an exact shared amplitude conditional.
The example is selected using audit history and stop count, not reference KL.

## All cases

![All 24 cases](audit_kl_all_cases.png)

Rows: injections 1–3, two catalogues per injection. Columns: 100/perfect,
100/uncertain, 1000/perfect, 1000/uncertain. Identical encodings, scales and
thresholds are used throughout. Three cases have no accepted stop by 658 training
evaluations. Missing orange markers mean no audit was performed, not a passed or
failed audit. Hollow blue circles retain numerically unresolved reference/GP
integrations. All runs continued to 658 training points for validation, even after
their recorded stop; audit points were held out of training in this experiment.

## Ensemble stopping locations

![Ensemble stopping locations](ensemble_stops.png)

Blue is median symmetric KL, half the sum of directed KLs, with its interquartile
range across all six cases per panel. Grey bands and dotted lines show the
interquartile range and median of **accepted** training stopping locations.
Non-stopping cases remain in the blue curves and are excluded only from those
conditional stop summaries; accepted counts are shown. Grey shading is variation
in stopping location, not an error threshold. Hollow blue circles mark a
checkpoint with at least one unresolved numerical integration. The horizontal
axis excludes audit cost; the single-case caption reports total cost explicitly.
Top/bottom rows use 100/1000 expected events; left/right use perfect/uncertain
measurements. The first injection was used for rule development.

## Reproduce

From the repository root:

```sh
MPLCONFIGDIR=/tmp/compas-mpl .venv/bin/python docs/studies/operational_stopping/gpry_compas/plot_audit_stopping.py --input docs/studies/operational_stopping/gpry_compas/extension_v2 --outdir docs/studies/operational_stopping/gpry_compas/audit_plot_draft
```

Only saved rates/likelihood results and audit records are used. No new likelihood
calls, GP training, reference integration or changes to frozen decisions occur.
`plot_data.json` contains every plotted value; `provenance.json` identifies source
hashes and records the numerical reconstruction of all {recomputed} audit results.
''')
    print(f'Wrote three figure pairs and captions to {out}; example={example}; audits checked={recomputed}')


if __name__ == '__main__':
    main()
