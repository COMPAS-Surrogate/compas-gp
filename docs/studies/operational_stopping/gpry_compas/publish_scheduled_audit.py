"""Render the measured periodic-audit development comparison and its limits."""
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

import plot_audit_stopping as plotting

GROUPS = [(100, 'perfect'), (100, 'uncertain'), (1000, 'perfect'), (1000, 'uncertain')]
FORMER_FAILURES = ['t2_n100_uncertain_c0_bo', 't2_n1000_perfect_c0_bo', 't2_n1000_uncertain_c0_bo']


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    args = parser.parse_args()
    folder = args.input.resolve()
    manifest = json.loads((folder/'manifest.json').read_text())
    stable_required = manifest.get('require_joint_stability', False)
    root = Path(manifest['training_root'])
    assert json.loads((folder/'offline/status.json').read_text())['state'] == 'complete'
    decisions = json.loads((folder/'decisions.json').read_text())
    outcomes = json.loads((folder/'offline/outcomes.json').read_text())
    summary = json.loads((folder/'offline/summary.json').read_text())
    rows = json.loads((root/'offline/records.json').read_text())
    old_summary = json.loads((root.parent/'report_v2/summary.json').read_text())
    out = folder/'report'; out.mkdir(exist_ok=True)
    curves = {name: plotting.reference_curve([r for r in rows if r['name']==name]) for name in decisions}
    data = dict(curves=curves, outcomes=outcomes, ensemble=[], former_failures=[])
    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'stix', 'font.size': 10,
        'axes.labelsize': 11, 'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True, 'axes.spines.top': True, 'axes.spines.right': True})
    for rule in ['fixed96', 'adaptive384']:
        fig, axes = plt.subplots(2, 2, figsize=(8, 5.8), sharex=True, sharey=True)
        for ax, (n, mode) in zip(axes.flat, GROUPS):
            names = [name for name, d in decisions.items() if d['strategy']=='bo'
                     and (d['expected_events'], d['mode'])==(n, mode)]
            x = np.asarray(curves[names[0]]['points'])
            lo, med, hi = np.quantile([curves[name]['symmetric_kl'] for name in names], [.25, .5, .75], axis=0)
            selected = [decisions[name]['rules'][rule]['training'] for name in names if decisions[name]['rules'][rule]]
            accepted = [o for o in outcomes if o['name'] in names and o['rule']==rule and o['selected']]
            accurate = sum(o['outcome']=='accurate' for o in accepted)
            quantiles = np.quantile(selected, [.25, .5, .75]) if selected else None
            if selected:
                ax.axvspan(quantiles[0], quantiles[2], color='.88', zorder=0)
                ax.axvline(quantiles[1], color='.3', ls=':', lw=1.4)
            ax.plot(x, med, color=plotting.BLUE, lw=1.6)
            ax.fill_between(x, lo, hi, color=plotting.BLUE, alpha=.17, linewidth=0)
            unresolved = np.any([curves[name]['unresolved'] for name in names], axis=0)
            ax.plot(x[unresolved], med[unresolved], 'o', ms=4, mfc='white', mec=plotting.BLUE)
            ax.text(.96, .94, f'{len(selected)}/6 stopped; {accurate} accurate', transform=ax.transAxes,
                    ha='right', va='top', fontsize=9)
            ax.set(yscale='log', ylim=(1e-5, 1e2), xlim=(50, 675))
            ax.grid(alpha=.18)
            data['ensemble'].append(dict(rule=rule, expected_events=n, mode=mode, names=names,
                points=x.tolist(), median=med.tolist(), q25=lo.tolist(), q75=hi.tolist(),
                stop_quantiles=quantiles.tolist() if quantiles is not None else None,
                stopped=len(selected), accurate=accurate))
        for ax in axes[:, 0]:
            ax.set_ylabel(r'$D_{\mathrm{KL}}^{\mathrm{sym}}$')
        for ax, mode in zip(axes[-1], ['Perfect', 'Uncertain']):
            ax.set_xlabel(f'{mode} measurements\nTraining likelihood evaluations')
        for ax, n in zip(axes[:, 0], [100, 1000]):
            ax.text(.04, .83, f'{n} expected events', transform=ax.transAxes, va='top', fontsize=9)
        fig.legend(handles=[Line2D([], [], color=plotting.BLUE, label='BO median and interquartile range'),
            Line2D([], [], color='.3', ls=':', label='Scheduled audit stop: median and interquartile range')],
            loc='upper center', frameon=False)
        fig.tight_layout(rect=(0, 0, 1, .9))
        fig.savefig(out/f'{rule}_stopping_kl.pdf', bbox_inches='tight')
        fig.savefig(out/f'{rule}_stopping_kl.png', dpi=190, bbox_inches='tight')
        if rule == 'fixed96':
            # Manuscript panel definitions and outcome counts belong in captions.
            for ax in axes.flat:
                for text in list(ax.texts):
                    text.remove()
            for ax in axes[-1]:
                ax.set_xlabel('Training likelihood evaluations')
            fig.tight_layout(rect=(0, 0, 1, .9))
            plotting.save(fig, out/'fixed96_stopping_kl_paper')
        else:
            plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    for ax, name in zip(axes, FORMER_FAILURES):
        decision = decisions[name]
        stop = decision['rules']['adaptive384']
        audits = []
        for h in decision['history']:
            if stop and h['training'] > stop['training']:
                continue
            if h['stages']:
                stage = h['stages'][-1]
                audits.append(dict(training=h['training'], **stage['result']))
        case = dict(reference=curves[name], audits=audits, selected=stop)
        plotting.draw_case(ax, case, (1e-4, 1e2))
        ax.set_xticks([100, 300, 500])
        ax.set_xlabel(f"{decision['expected_events']} expected events, {decision['mode']}\nTraining evaluations")
        outcome = next(o for o in outcomes if o['name']==name and o['rule']=='adaptive384')
        label = f"Stop {stop['training']}; {outcome['outcome']}" if stop else 'No accepted stop'
        ax.text(.96, .94, label, transform=ax.transAxes, ha='right', va='top', fontsize=9)
        data['former_failures'].append(dict(name=name, **case, outcome=outcome))
    axes[0].set_ylabel(r'$D_{\mathrm{KL}}^{\max}$')
    fig.legend(handles=plotting.audit_handles(), loc='upper center', ncol=2, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .84))
    plotting.save(fig, out/'former_nonstops')

    # Count all audit evaluations, including rejected audits, in the cost figure.
    fig, ax = plt.subplots(figsize=(7.5, 6))
    names = sorted(name for name, d in decisions.items() if d['strategy']=='bo')
    for i, name in enumerate(names):
        for offset, rule, colour in [(-.13, 'fixed96', plotting.BLUE), (.13, 'adaptive384', plotting.ORANGE)]:
            o = next(o for o in outcomes if o['name']==name and o['rule']==rule)
            marker = {'accurate':'o', 'premature':'x', 'unresolved':'s', 'no_stop':'>'}[o['outcome']]
            ax.plot(o['total_spent'], i+offset, marker=marker, color=colour, ms=4, ls='none')
    labels = [f"{decisions[n]['truth_index']+1}, {decisions[n]['expected_events']}, "
              f"{'P' if decisions[n]['mode']=='perfect' else 'U'}, {decisions[n]['case']+1}" for n in names]
    ax.set_yticks(range(len(names)), labels, fontsize=8)
    ax.invert_yaxis(); ax.grid(axis='x', alpha=.2)
    ax.set_xlabel('Training + audit likelihood evaluations')
    ax.set_ylabel('Injection, expected count, measurement, catalogue')
    fig.legend(handles=[Line2D([], [], color=c, marker='o', ls='none', label=l)
                        for c,l in [(plotting.BLUE, 'Fixed 96'), (plotting.ORANGE, 'Adaptive to 384')]],
               loc='upper center', ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .95)); plotting.save(fig, out/'bo_total_cost')

    stability_description = ('Acceptance additionally requires two consecutive numerically resolved '
        'joint-posterior stability checks. These do not block scheduled audits. '
        'This second candidate preserves the audit-only failure results in scheduled_v1.'
        if stable_required else 'Posterior stability is diagnostic only in this first candidate.')
    lines = ['# Periodic-audit development results', '', stability_description, '',
        'Same 24 COMPAS likelihood cases and 48 frozen BO/random training trajectories as extension_v2. '
        'All are development evidence: the revision was motivated by already-seen failures. '
        'New audit coordinates and labels were generated without reading reference scores; '
        'all stopping decisions were frozen before this validation.', '',
        '| Sampling | Audit | Accurate | Premature | Unresolved | No stop | Median training at stop | Median total at stop |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for strategy in ['bo', 'random']:
        for rule in ['fixed96', 'adaptive384']:
            r = summary[strategy][rule]; c = r['outcomes']
            count = [c.get(k, 0) for k in ['accurate', 'premature', 'unresolved', 'no_stop']]
            med = ['--' if r[k] is None else f'{r[k]:g}' for k in ['median_training', 'median_total_accepted']]
            lines.append(f"| {strategy.upper()} | {rule} | "+' | '.join(map(str, count+med))+' |')
    lines += ['', 'Accepted-stop medians exclude non-stops and include any premature stops. '
        'They are not end-to-end speedups. Full four-parameter accuracy is required: both directed '
        'KLs <0.01; all marginal means/endpoints <0.2 reference SD; 68/95% interval widths within '
        '10%; and numerical precision checks. The earlier combined mass rule accepted 21/24 BO '
        'cases accurately and left three non-stops, with median 388 training and 484 total among '
        'accepted cases.', '', '## Previously non-stopping cases', '',
        '| Case | Fixed 96 training + audit | Adaptive training + audit | Adaptive result | Reference max KL |',
        '|---|---|---|---|---:|']
    for name in FORMER_FAILURES:
        d = decisions[name]
        counts = ['--' if not d['rules'][k] else f"{d['rules'][k]['training']} + {d['rules'][k]['audit']}"
                  for k in ['fixed96', 'adaptive384']]
        o = next(o for o in outcomes if o['name']==name and o['rule']=='adaptive384')
        kl = '--' if not o['selected'] else f"{o['maximum_directed_kl']:.5g}"
        lines.append(f"| {name} | {' | '.join(counts)} | {o['outcome']} | {kl} |")
    lines += ['', '![New audit diagnostics](former_nonstops.png)', '',
        'Blue: maximum directed reference KL, used only for validation. Orange circles: fresh '
        'audit estimates; triangles: 95th bootstrap percentiles. Solid/dashed orange horizontal '
        'lines are 0.003/0.005. At each checkpoint the final adaptive audit stage is shown; '
        'intermediate stages are retained in JSON. Dotted lines mark the actual adaptive stop. '
        'The x axis counts training labels; the table includes audit cost. These are the third '
        'injection and first catalogue. Bootstrap quantiles are finite-sample diagnostics, not '
        'confidence guarantees for the full posterior or repeated testing.', '',
        '![Adaptive stopping bands](adaptive384_stopping_kl.png)', '',
        'Blue median/IQR curves include all six BO cases in each panel; grey bands and dotted '
        'lines are conditional IQR/median accepted stopping locations. Counts explicitly distinguish '
        'stops from independently accurate stops. The y axis is symmetric KL (half the sum of both '
        'directions), so the blue curve alone is not the accuracy decision. Hollow blue circles '
        'retain unresolved numerical checkpoints. Top/bottom: 100/1000 expected events; '
        'left/right: perfect/uncertain measurements. Training continued after recorded stops '
        'for validation. Audit evaluations are excluded from this x axis.', '',
        '![Fixed-size stopping bands](fixed96_stopping_kl.png)', '',
        'Identical panel definitions for the fixed-96 comparator.', '',
        '![Full BO evaluation costs](bo_total_cost.png)', '',
        'Fixed/adaptive audit costs include every rejected audit and only the labels spent before '
        'each policy accepted. Circles are accurate, crosses premature, squares unresolved, '
        'right triangles non-stops. Both policies share initial 96-point audit draws. '
        'Adaptive extra draws are held out of fixed-policy decisions.', '',
        '## Scope and implementation', '',
        'The schedule starts at 238 labels and repeats every 60, always including 658. The prediction '
        'counter and pointwise residuals are diagnostics, not vetoes. Numerical '
        'integration and effective-sample-size checks remain necessary. Adaptive audits pool '
        '96, 192 and 384 fresh samples at one frozen GP using the actual posterior/tempered/prior '
        'mixture density. See ../manifest.json and ../../SCHEDULED_AUDIT_PROTOCOL.md. '+stability_description, '',
        'Audits remain held out of the saved GP training paths. Recycling failed audit labels '
        'into training and independently generated new-catalogue confirmation are not tested. '
        'The model has four physical parameters, with amplitude marginalized analytically and '
        'three GP inputs. Rate queries use the same fixed-population evaluator; cached rate '
        'references are used only for scoring. No new parameter-space reference grid or GPry '
        'installation was needed. All original extension_v2 decisions are retained.', '',
        'The random trajectories all receive fresh audits in this replay. Their lack of acceptance '
        'does not mean every final random posterior is inaccurate: 5/24 satisfy the looser '
        'independent accuracy target at 658 training points. Their maximum directed reference KLs '
        'are about 0.0062–0.0095, and their last fresh audits fail the tighter internal stopping '
        'thresholds. This conservatism remains part of the procedure; random runs would need '
        'more training to test eventual operational convergence.', '',
        'Reproduce from the repository root with the existing virtual environment:', '', '```sh',
        '.venv/bin/python docs/studies/operational_stopping/gpry_compas/'+
        ('run_stable_scheduled_audit.py' if stable_required else 'run_scheduled_audit.py')+
        ' --input docs/studies/operational_stopping/gpry_compas/extension_v2 --outdir FRESH_OUTPUT_DIRECTORY',
        '.venv/bin/python docs/studies/operational_stopping/gpry_compas/score_scheduled_audit.py --input FRESH_OUTPUT_DIRECTORY',
        '.venv/bin/python docs/studies/operational_stopping/gpry_compas/publish_scheduled_audit.py --input FRESH_OUTPUT_DIRECTORY',
        '```', '']
    (out/'RESULTS.md').write_text('\n'.join(lines))
    (out/'plot_data.json').write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    paths = [folder/'decisions.json', folder/'offline/outcomes.json', folder/'offline/summary.json',
             root/'offline/records.json', Path(__file__)]
    (out/'provenance.json').write_text(json.dumps(dict(source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths}, old_rule_summary=old_summary), indent=2)+'\n')
    print(out/'RESULTS.md')


if __name__ == '__main__':
    main()
