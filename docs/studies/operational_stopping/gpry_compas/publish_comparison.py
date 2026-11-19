"""Render the frozen BO/random comparison and manuscript-ready numerical text."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

METHODS = ['bo', 'random']
RULES = ['original', 'precision_refined', 'guarded', 'mass_guarded']
NAMES = {'original': 'Counter + Gaussian KL', 'precision_refined': 'Precision-refined counter + Gaussian KL',
         'guarded': 'Pointwise audit', 'mass_guarded': 'Posterior-mass audit'}
COLORS = {'bo': '#207eae', 'random': '#cf8a27'}
STYLES = {'bo': '-', 'random': '--'}
GROUPS = [(100, 'perfect'), (100, 'uncertain'), (1000, 'perfect'), (1000, 'uncertain')]


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def finish(fig, path: Path) -> None:
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(path.with_suffix('.png'), dpi=190, bbox_inches='tight')
    plt.close(fig)


def stopping_stats(records: list[dict]) -> dict:
    accepted = [r['selected'] for r in records if r['selected']]
    return dict(outcomes=dict(Counter(r['outcome'] for r in records)),
        stopped=len(accepted), total=len(records),
        median_training=float(np.median([r['training'] for r in accepted])) if accepted else None,
        median_total=float(np.median([r['total'] for r in accepted])) if accepted else None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    args = parser.parse_args(); root, out = args.input, args.outdir
    out.mkdir(parents=True, exist_ok=True)
    assert json.loads((root/'status.json').read_text())['state'] == 'complete'
    assert json.loads((root/'offline/status.json').read_text())['state'] == 'complete'
    verification = json.loads((root/'verification.json').read_text())
    assert verification['state'] == 'passed'
    decision_sha = hashlib.sha256((root/'decisions.json').read_bytes()).hexdigest()
    assert verification['decision_sha256'] == decision_sha
    assert json.loads((root/'offline/manifest.json').read_text())['decision_sha256'] == decision_sha
    decisions = json.loads((root/'decisions.json').read_text())
    mass_decisions = json.loads((root/'mass_audit/decisions.json').read_text())
    mass_sha = hashlib.sha256((root/'mass_audit/decisions.json').read_bytes()).hexdigest()
    assert verification['mass_decision_sha256'] == mass_sha
    assert json.loads((root/'offline/manifest.json').read_text())['mass_decision_sha256'] == mass_sha
    rows = json.loads((root/'offline/records.json').read_text())
    outcomes = json.loads((root/'offline/outcomes.json').read_text())
    by_name = {name: sorted([r for r in rows if r['name'] == name], key=lambda r: r['points']) for name in decisions}
    budget = max(r['points'] for r in rows)
    summary = {}
    for strategy in METHODS:
        summary[strategy] = {}
        for rule in RULES:
            subset = [r for r in outcomes if r['strategy'] == strategy and r['rule'] == rule]
            summary[strategy][rule] = stopping_stats(subset)
    confirmation = {s: {rule: stopping_stats([o for o in outcomes if o['strategy'] == s
        and o['rule'] == rule and o['truth_index'] != 0]) for rule in RULES} for s in METHODS}
    mass_confirmation = {s: confirmation[s]['mass_guarded'] for s in METHODS}
    mass_attempted = {s: sum(mass_decisions[name]['audit_labels'] > 0 for name, d in decisions.items()
                             if d['strategy'] == s) for s in METHODS}
    pointwise_only = [name for name, d in decisions.items() if d['truth_index'] != 0 and d['strategy'] == 'bo'
                      and d['rules']['guarded'] and not mass_decisions[name]['selected']]
    rescues = [o for o in outcomes if o['rule'] == 'mass_guarded' and o['outcome'] == 'accurate'
               and o['truth_index'] != 0 and o['strategy'] == 'bo' and decisions[o['name']]['rules']['original'] is None]
    rescue_text = ''
    if rescues:
        example = rescues[0]; stop = example['selected']
        rescue_text = (f"At injection {example['truth_index']+1}, {example['expected_events']} expected events "
            f"and {example['mode']} measurements, one confirmation case never satisfies the original rule. "
            f"Precision refinement plus the mass audit accepts after {stop['training']} training and "
            f"{stop['audit']} audit evaluations ({stop['total']} total). Independent validation passes "
            f"all four marginal checks, with maximum directed KL {example['maximum_directed_kl']:.5f}.")
    final = {s: [r for r in rows if r['strategy'] == s and r['points'] == budget] for s in METHODS}
    final_stats = {s: dict(accurate=sum(r['accuracy_pass'] for r in final[s]), total=len(final[s]),
        median_kl=float(np.median([r['symmetric_kl'] for r in final[s]])),
        unresolved=sum(not(r['reference_numerical'] and r['gp_numerical'] and r['four_numerical']) for r in final[s]))
        for s in METHODS}
    audit_diagnostics = []
    for name, d in decisions.items():
        audits = json.loads((root/name/'audits.json').read_text())
        for a in audits:
            result = a['result']
            core_pass = (result.get('rms', float('inf')) <= .07
                         and result.get('q95', float('inf')) <= .15
                         and result.get('weight_ess_fraction', 0) >= .99)
            audit_diagnostics.append(dict(result, name=name, strategy=d['strategy'], training=a['training'],
                posterior_checks_pass=core_pass, broad_only_rejection=bool(core_pass and not result['passed'])))
    pairs = []
    for name, d in decisions.items():
        if d['strategy'] != 'bo': continue
        random_name = name[:-2]+'random'
        b, r = by_name[name][-1], by_name[random_name][-1]
        pairs.append(dict(bo=name, random=random_name, bo_kl=b['symmetric_kl'], random_kl=r['symmetric_kl'],
                          ratio=r['symmetric_kl']/b['symmetric_kl'],
                          resolved=bool(b['reference_numerical'] and b['gp_numerical']
                                        and r['reference_numerical'] and r['gp_numerical'])))
    resolved_pairs = [p for p in pairs if p['resolved']]
    paired_stats = dict(bo_lower_kl=sum(p['bo_kl'] < p['random_kl'] for p in resolved_pairs),
                       pairs=len(resolved_pairs), total_pairs=len(pairs),
                       median_random_to_bo_kl=float(np.median([p['ratio'] for p in resolved_pairs])))
    positive_kl = [r['symmetric_kl'] for r in rows if r['symmetric_kl'] > 0]
    ylim = (10**np.floor(np.log10(min(positive_kl))), 10**np.ceil(np.log10(max(positive_kl))))
    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'stix', 'font.size': 10,
                         'axes.labelsize': 11, 'xtick.direction': 'in', 'ytick.direction': 'in',
                         'xtick.top': True, 'ytick.right': True, 'axes.spines.top': True,
                         'axes.spines.right': True, 'savefig.facecolor': 'white'})
    handles = [Line2D([], [], color=COLORS[s], ls=STYLES[s], label='BO' if s == 'bo' else 'IID random') for s in METHODS]
    fig, axes = plt.subplots(2, 2, figsize=(8, 5.6), sharex=True, sharey=True)
    plot_data = []
    for ax, (n, mode) in zip(axes.flat, GROUPS):
        for s in METHODS:
            names = [name for name, d in decisions.items() if d['expected_events'] == n and d['mode'] == mode and d['strategy'] == s]
            if not names: continue
            x = [r['points'] for r in by_name[names[0]]]
            curves = np.array([[r['symmetric_kl'] for r in by_name[name]] for name in names])
            low, median, high = np.quantile(curves, [.25, .5, .75], axis=0)
            ax.plot(x, median, color=COLORS[s], ls=STYLES[s], lw=1.6)
            ax.fill_between(x, low, high, color=COLORS[s], alpha=.16, linewidth=0)
            unresolved = np.any([[not(r['reference_numerical'] and r['gp_numerical'])
                                  for r in by_name[name]] for name in names], axis=0)
            if np.any(unresolved):
                ax.plot(np.asarray(x)[unresolved], median[unresolved], ls='none', marker='o',
                        ms=4, markerfacecolor='white', markeredgecolor=COLORS[s])
            plot_data.append(dict(expected_events=n, mode=mode, strategy=s, names=names, points=x,
                                  median=median.tolist(), q25=low.tolist(), q75=high.tolist(),
                                  unresolved=unresolved.tolist()))
        ax.set_yscale('log'); ax.set_ylim(*ylim); ax.set_xlim(58, budget)
        ax.grid(alpha=.22); ax.tick_params(which='both', top=True, right=True, direction='in')
        ax.minorticks_on()
    for ax in axes[:, 0]: ax.set_ylabel(r'$D_{\mathrm{KL}}^{\mathrm{sym}}$')
    for ax in axes[1]: ax.set_xlabel('Training likelihood evaluations')
    fig.legend(handles=handles, loc='upper center', ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .93)); finish(fig, out/'bo_random_convergence')

    # Match the reference illustration, with an explicitly conditional stopping band.
    fig, axes = plt.subplots(2, 2, figsize=(8, 5.6), sharex=True, sharey=True)
    stop_plot_data = []
    for ax, (n, mode) in zip(axes.flat, GROUPS):
        curve = next(p for p in plot_data if (p['expected_events'], p['mode'], p['strategy']) == (n, mode, 'bo'))
        stops = [mass_decisions[name]['selected']['training'] for name in curve['names']
                 if mass_decisions[name]['selected']]
        quantiles = np.quantile(stops, [.25, .5, .75]) if stops else None
        if stops:
            ax.axvspan(quantiles[0], quantiles[2], color='.88', zorder=0)
            ax.axvline(quantiles[1], color='.45', ls=':', lw=1.3)
        ax.plot(curve['points'], curve['median'], color=COLORS['bo'], lw=1.6)
        ax.fill_between(curve['points'], curve['q25'], curve['q75'], color=COLORS['bo'], alpha=.16, linewidth=0)
        unresolved = np.asarray(curve['unresolved'])
        ax.plot(np.asarray(curve['points'])[unresolved], np.asarray(curve['median'])[unresolved],
                ls='none', marker='o', ms=4, markerfacecolor='white', markeredgecolor=COLORS['bo'])
        ax.set_yscale('log'); ax.set_ylim(*ylim); ax.set_xlim(58, budget)
        ax.grid(alpha=.22); ax.tick_params(which='both', top=True, right=True, direction='in'); ax.minorticks_on()
        stop_plot_data.append(dict(expected_events=n, mode=mode, names=curve['names'], accepted=len(stops),
            total=len(curve['names']), training_stops=stops,
            training_quantiles=quantiles.tolist() if quantiles is not None else None))
    for ax in axes[:, 0]: ax.set_ylabel(r'$D_{\mathrm{KL}}^{\mathrm{sym}}$')
    for ax in axes[1]: ax.set_xlabel('Training likelihood evaluations')
    fig.legend(handles=[handles[0], Line2D([], [], color='.45', ls=':', label='Median accepted stop')],
               loc='upper center', ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .93)); finish(fig, out/'mass_stopping_convergence')

    # Every underlying trajectory remains visible, separated by injection.
    fig, axes = plt.subplots(3, 4, figsize=(11, 6), sharex=True, sharey=True)
    for ti, row_axes in enumerate(axes):
        for ax, (n, mode) in zip(row_axes, GROUPS):
            for name, d in decisions.items():
                if (d['truth_index'], d['expected_events'], d['mode']) != (ti, n, mode): continue
                rr = by_name[name]; s = d['strategy']
                ax.plot([r['points'] for r in rr], [r['symmetric_kl'] for r in rr],
                        color=COLORS[s], ls=STYLES[s], alpha=.75, lw=1.)
            ax.set_yscale('log'); ax.set_ylim(*ylim); ax.set_xlim(58, budget)
            ax.grid(alpha=.2); ax.tick_params(which='both', top=True, right=True); ax.minorticks_on()
    for ax in axes[:, 0]: ax.set_ylabel(r'$D_{\mathrm{KL}}^{\mathrm{sym}}$')
    for ax in axes[-1]: ax.set_xlabel('Training evaluations')
    fig.legend(handles=handles, loc='upper center', ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .94)); finish(fig, out/'all_trajectories')

    fig, ax = plt.subplots(figsize=(7.7, 8))
    ticklabels = []
    for i, pair in enumerate(pairs):
        d = decisions[pair['bo']]
        ticklabels.append(f"{d['truth_index']+1}, {d['expected_events']}, {'P' if d['mode']=='perfect' else 'U'}, {d['case']+1}")
        for delta, s in [(-.15, 'bo'), (.15, 'random')]:
            name = pair[s]; dec = decisions[name]
            result = next(o for o in outcomes if o['name'] == name and o['rule'] == 'mass_guarded')
            value = result['selected']['total'] if result['selected'] else dec['training_labels']+mass_decisions[name]['audit_labels']
            marker = {'accurate': 'o', 'premature': 'x', 'unresolved': 's', 'no_stop': '>'}[result['outcome']]
            ax.plot(value, i+delta, marker=marker, ms=5, color=COLORS[s], linestyle='none',
                    markerfacecolor=COLORS[s] if s == 'bo' and result['outcome'] != 'unresolved' else 'none')
    ax.set_yticks(range(len(pairs)), ticklabels, fontsize=8)
    ax.invert_yaxis(); ax.set_xlabel('Likelihood evaluations including stopping audits')
    ax.set_ylabel('Injection, expected count, measurement, realization')
    ax.grid(axis='x', alpha=.22); ax.tick_params(which='both', top=True, right=True)
    cost_handles = [Line2D([], [], color=COLORS[s], marker='o', ls='none',
                          markerfacecolor=COLORS[s] if s == 'bo' else 'none',
                          label='BO' if s == 'bo' else 'IID random') for s in METHODS]
    legend = cost_handles+[Line2D([], [], color='.35', marker=m, ls='none', label=l) for m, l in
                      [('o', 'Accurate stop'), ('x', 'Premature stop'),
                       ('s', 'Unresolved validation'), ('>', 'No stop by budget')]]
    fig.legend(handles=legend, loc='upper center', ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, .92)); finish(fig, out/'guarded_stopping_cost')

    lines = ['# COMPAS stopping extension: BO versus IID random', '',
        'Three injections, two expected counts, and two independent latent catalogues per combination give '
        '12 fresh latent catalogues. Perfect/uncertain views yield 24 likelihood cases and 48 trajectories. '
        'This is a held-out extension of the previous stopping study, not evidence for every COMPAS case.', '',
        '![Matched training budgets](bo_random_convergence.png)', '',
        'Top row: 100 expected events; bottom: 1000. Left: perfect measurements; right: uncertain. '
        'Blue solid is BO; orange dashed is IID uniform random continuation after identical initialization. '
        'Curves show median symmetric KL (half the sum of directed KLs), with descriptive interquartile '
        'bands across six cases per panel. All runs continue to 658 training labels. No curves are smoothed. '
        'Hollow markers indicate that at least one contributing case has unresolved numerical integration; '
        'those early estimates are retained rather than silently excluding the case. '
        'Audits are separate exact evaluations and are excluded from this training-budget axis.', '',
        '| Method | Rule | Accurate | Premature | Unresolved | No stop | Median training | Median total |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    latex_rows = []
    for s in METHODS:
        for rule in RULES:
            r = summary[s][rule]; c = r['outcomes']
            values = [c.get(k, 0) for k in ['accurate', 'premature', 'unresolved', 'no_stop']]
            train = '--' if r['median_training'] is None else f"{r['median_training']:g}"
            total = '--' if r['median_total'] is None else f"{r['median_total']:g}"
            lines.append(f"| {s.upper()} | {NAMES[rule]} | {' | '.join(map(str, values))} | {train} | {total} |")
            label = {'original': 'Counter + Gaussian', 'precision_refined': 'Refined precision',
                     'guarded': 'Pointwise audit', 'mass_guarded': 'Mass-weighted audit'}[rule]
            confirmed = confirmation[s][rule]
            confirmed_counts = [confirmed['outcomes'].get(k, 0) for k in ['accurate', 'premature', 'unresolved', 'no_stop']]
            confirmed_total = '--' if confirmed['median_total'] is None else f"{confirmed['median_total']:g}"
            latex_rows.append(f"{'BO' if s == 'bo' else 'Random'} & {label} & "
                              + ' & '.join(map(str, confirmed_counts)) + f" & {confirmed_total} " + chr(92)*2)
    lines += ['', 'The posterior-mass variant was defined after inspecting pointwise tail-audit failures '
        'at the first injection, before revealing reference scores. Injection 1 is development evidence '
        'for this variant. Its independent confirmation uses injections 2/3 (16 likelihood cases per method).', '',
        *[f"- {s.upper()} mass-audit confirmation: "
          + ', '.join(f"{mass_confirmation[s]['outcomes'].get(k, 0)} {k.replace('_', ' ')}" for k in
                      ['accurate', 'premature', 'unresolved', 'no_stop'])
          + (f"; median total among stops {mass_confirmation[s]['median_total']:g}."
             if mass_confirmation[s]['median_total'] is not None else '; no accepted stops.') for s in METHODS], '',
        'Stopping medians include accepted runs only, including any premature/unresolved acceptances. '
        'They exclude censored runs and are not paired speedup estimates. All four marginal intervals are '
        'checked at every claimed stop and final fit. Accuracy also requires each directed KL <0.01 and '
        'independent numerical resolution. The plotted symmetric KL being <0.01 is insufficient by itself.', '',
        'The counter assesses the points selected by each acquisition method. Random sampling can '
        'therefore fail its counter through tail prediction errors even when its posterior is adequate. '
        'Its non-stops characterize this shared operational rule; the matched-budget accuracy curves '
        'are the primary acquisition comparison.', '',
        'Cases reaching a posterior-mass audit: '
        + ', '.join(f"{s.upper()} {mass_attempted[s]}/24" for s in METHODS)+'. '
        'A case that never reaches audit eligibility does not test the audit acceptance thresholds.', '',
        rescue_text, '',
        f"The mass-audit rule does not dominate the pointwise audit case by case: {len(pointwise_only)} "
        + ('BO confirmation case accepted by the pointwise rule remains non-stopping with the ' if len(pointwise_only) == 1 else
           'BO confirmation cases accepted by the pointwise rule remain non-stopping with the ')
        + 'independent mass-audit draws. The aggregate comparison retains this tradeoff.', '',
        f"At {budget} training labels, full accuracy passes: BO {final_stats['bo']['accurate']}/{len(final['bo'])}; "
        f"random {final_stats['random']['accurate']}/{len(final['random'])}. "
        f"BO has lower final symmetric KL in {paired_stats['bo_lower_kl']}/{paired_stats['pairs']} numerically resolved pairs "
        f"(out of {paired_stats['total_pairs']} pairs); "
        f"the median paired random/BO KL ratio is {paired_stats['median_random_to_bo_kl']:.3g}.", '',
        '![Operational costs](guarded_stopping_cost.png)', '',
        'Each row is a matched likelihood case under the posterior-mass stopping variant. P/U denote perfect/uncertain measurements; '
        'injections 1/2/3 follow the frozen protocol order. Filled/open markers distinguish BO/random. '
        'Circles indicate accurate acceptance, crosses '
        'premature acceptance, right-pointing triangles no stop by the actual budget spent. '
        'Each audit costs 96 exact calls; failed audits are included. Training continues independently '
        'of held-out audits, preserving the pure BO/random allocation comparison.', '',
        '## Grid-free use and scope', '',
        'Training, acquisition, GP-only stability and stopping audits require no parameter-space rate grid. '
        'They request exact rates only at selected coordinates. Cached exact rates and direct posterior '
        'integrations are used only after stopping decisions are frozen to score this tractable benchmark. '
        'The fixed mass/redshift output bins are a different grid and remain part of the likelihood.', '',
        'The physical model has four parameters, but the GP has three inputs because amplitude is integrated '
        'analytically. With the same exact amplitude conditional, full-joint KL equals shape KL. Four marginal '
        'means and intervals are checked explicitly. Grid-free amplitude reconstruction still requires total '
        'rates at posterior points, which are additional to training/audit counts. No explicit four-input GP, '
        'new COMPAS evolutionary model, universal convergence guarantee, continuous-sensor calibration, '
        'or end-to-end runtime improvement is established.', '',
        'The development case `t0_n1000_perfect_c0_bo` illustrates why a pointwise tail veto can be '
        'too conservative: it never accepted by the resource limit, whereas the posterior-mass rule '
        'accepted at 328 training plus 96 audit calls. Independent scoring gives maximum directed '
        'KL 0.00240 and passes all four marginal checks. This example motivated interpretation of '
        'the audit behavior; it is not counted in the mass-audit confirmation set.', '',
        '## BO convergence with stopping markers', '',
        '![BO stopping convergence](mass_stopping_convergence.png)', '',
        'Panel order and blue median/interquartile KL curves match the acquisition comparison. '
        'The dotted vertical line and grey band show the median and interquartile range of '
        'accepted posterior-mass stopping checkpoints, conditional on stopping. They count training '
        'labels only; audit costs appear in the table and cost plot. Non-stopping runs remain in '
        'the blue curves and are excluded only from the conditional stopping quantiles. '
        'This plot includes the development injection. Accepted/total cases in panel order: '
        + ', '.join(f"{p['accepted']}/{p['total']}" for p in stop_plot_data)+'.', '',
        '## Reproduction', '', 'Run from the repository root with the existing virtual environment:', '', '```sh']
    prefix = 'docs/studies/operational_stopping/gpry_compas'
    for script, flags in [('compare_random.py', f'--outdir {prefix}/extension_v2'),
                          ('run_mass_audit.py', f'--input {prefix}/extension_v2'),
                          ('score_comparison.py', f'--input {prefix}/extension_v2'),
                          ('verify_comparison.py', f'--input {prefix}/extension_v2'),
                          ('publish_comparison.py', f'--input {prefix}/extension_v2 --outdir {prefix}/report_v2')]:
        lines.append(f'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/compas-mpl .venv/bin/python {prefix}/{script} {flags}')
    lines += ['```', '', '[Frozen protocol](../EXTENSION_PROTOCOL.md); [all trajectories](all_trajectories.pdf).', '',
              '[Mass-audit protocol](../MASS_AUDIT_PROTOCOL.md). Use a fresh output directory for a rerun.', '',
              '## Exceptional outcomes', '']
    for o in outcomes:
        if o['outcome'] != 'accurate': lines.append(f"- {o['name']}, {NAMES[o['rule']]}: {o['outcome']}; {o['selected']}.")
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    tex = r'''\subsection{Operational stopping and a matched random baseline}
\label{sec:stopping_results}
The stopping extension uses three injections,
$(-0.3,0.35,0.01,4.7)$, $(-0.46,0.15,0.006,4.35)$ and
$(-0.12,0.52,0.013,5.05)$ in $(\alpha,\sigma,a,d)$ order.
At each injection, exposures give 100 or 1000 expected events, with two fresh
Poisson catalogues per exposure. Paired perfect and uncertain measurements of
these 12 latent catalogues yield 24 likelihood cases and 48 BO/random
trajectories. Each pair has identical initialization and GP settings. These
data are separate from the twelve-run development experiment used to choose
the initial extension's tolerances. All stopping decisions precede direct-reference
scoring; trajectories continue to 658 training evaluations for validation.

\begin{figure*}\centering
\includegraphics[width=.92\textwidth]{figures/bo_study/bo_random_stopping_convergence.pdf}
\caption{Matched training-budget comparison of the GPry-inspired BO acquisition
and IID uniform random continuation. Top/bottom rows have 100/1000 expected
events; left/right columns use perfect/uncertain measurements. Solid blue and
dashed orange curves show median symmetric KL for BO and random sampling,
respectively, with interquartile bands over six cases per panel. These are
descriptive spreads across injections and catalogue realizations, not
confidence intervals. Hollow markers indicate that one or more contributing
cases have unresolved numerical integration; these estimates remain visible.
Symmetric KL is half the sum of the two directions;
accuracy requires each direction to be below 0.01. The full four-dimensional
joint KL equals this shape-posterior KL under the shared exact amplitude
conditional. The axis counts training evaluations, including initialization;
held-out stopping audits and independent validation costs are separate.}
\label{fig:bo_random_stopping}\end{figure*}

\begin{table*}\centering
\caption{Operational stopping outcomes on the same 16 confirmation cases per
method, from the second and third injections. The first injection was used to
develop the mass-weighted audit and is excluded from this table.
All four marginal posterior checks and both directed KL tests must pass for
an accurate stop. Unresolved denotes an accepted stop whose numerical
validation fails; no stop denotes a censored run. Median total evaluations
include all accepted stops and their audits, exclude censored runs, and are
not a paired speedup estimate. All cases, including development cases, remain
in the acquisition comparison in Figure~\ref{fig:bo_random_stopping}.}
\label{tab:operational_stopping}
\begin{tabular}{llrrrrr}\toprule
Method & Rule & Accurate & Premature & Unresolved & No stop & Median total\\
\midrule
'''+ '\n'.join(latex_rows)+r'''
\bottomrule\end{tabular}\end{table*}
'''
    tex += (f"\nAt 658 training evaluations, the full accuracy criterion passes for "
            f"{final_stats['bo']['accurate']}/24 BO and {final_stats['random']['accurate']}/24 random fits. "
            f"BO has lower symmetric KL in {paired_stats['bo_lower_kl']}/{paired_stats['pairs']} numerically resolved matched pairs "
            f"(out of {paired_stats['total_pairs']} pairs), "
            f"with a median paired random-to-BO KL ratio of {paired_stats['median_random_to_bo_kl']:.0f}.\n")
    tex += r'''
The pointwise tail veto motivated a mass-weighted audit variant after inspection
of the first injection's audit residuals, before any direct-reference scores
were revealed. The first injection is therefore development evidence for this
variant; the remaining two injections provide 16 confirmation cases per method.
'''
    for s in METHODS:
        c = mass_confirmation[s]['outcomes']
        tex += (f"{'BO' if s == 'bo' else 'Random sampling'} gives "
                f"{c.get('accurate', 0)} accurate, {c.get('premature', 0)} premature, "
                f"{c.get('unresolved', 0)} numerically unresolved acceptances and "
                f"{c.get('no_stop', 0)} non-stopping cases in that confirmation set.\n")
    tex += (f"Across all 24 cases per method, {mass_attempted['bo']} BO and "
            f"{mass_attempted['random']} random trajectories reach the mass-audit eligibility conditions. "
            "Runs that never reach eligibility do not test the audit acceptance thresholds.\n")
    tex += r'''
In one development example with 1000 expected events and perfect measurements,
the pointwise veto never accepted by the resource limit. The mass-weighted rule
accepted the 328-point GP after 96 audit evaluations; its maximum directed
reference KL is 0.00240 and all four marginal checks pass. This illustrates the
cost of demanding small absolute errors throughout low-probability tails,
without treating this development example as independent confirmation.
'''
    tex += '\n'+rescue_text+'\n'
    tex += (f"The improvement is not uniform across cases: {len(pointwise_only)} BO confirmation "
            + ("case accepted by the pointwise audit remains" if len(pointwise_only) == 1 else
               "cases accepted by the pointwise audit remain")
            + " non-stopping under the independent mass-audit draws.\n")
    tex += r'''
The finite stopping experiment tests the specified prior, injections, population
and measurement model. Neither a passing counter nor stable successive GP
posteriors establishes accuracy for every COMPAS case. The guarded audit tests
fresh likelihood predictions, and its extra calls are included in
Table~\ref{tab:operational_stopping}. Cached direct references remain an offline
benchmark tool rather than an input required by the stopping method.
The counter checks each method's selected locations, so inaccurate predictions
at random tail points can delay a stop even for an adequate posterior.
Random non-stops therefore reflect both acquisition and the shared counter;
the matched-budget accuracy comparison is the primary acquisition result.
'''
    (out/'stopping_results.tex').write_text(tex)
    write(out/'summary.json', dict(stopping=summary, final=final_stats, paired=paired_stats,
                                  verification=verification, confirmation=confirmation,
                                  mass_confirmation=mass_confirmation,
                                  mass_audit_attempted_cases=mass_attempted,
                                  audit_attempts=len(audit_diagnostics),
                                  broad_only_rejections=sum(a['broad_only_rejection'] for a in audit_diagnostics)))
    write(out/'audit_diagnostics.json', audit_diagnostics)
    write(out/'plot_data.json', plot_data)
    write(out/'stop_plot_data.json', stop_plot_data)
    write(out/'provenance.json', {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [root/'manifest.json', root/'decisions.json', root/'mass_audit/decisions.json', root/'offline/records.json',
         root/'offline/outcomes.json', root/'verification.json', Path(__file__)]})


if __name__ == '__main__':
    main()
