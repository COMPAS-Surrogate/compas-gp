"""Build a report and scientific figures from completed four-input follow-ups."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np


def maximum_kl(record: dict) -> float:
    return max(max(s['directed_kl']) for s in record['scores'])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    args = parser.parse_args()
    root = args.input.resolve()
    rows = json.loads((root/'results.json').read_text())
    old_root = root.parent/'large_batch_rescored_20260914'
    old = {r['name']: r for r in json.loads((old_root/'results.json').read_text())}
    slices = json.loads((root/'conditional_slices.json').read_text())
    assert len(rows) == len(old) == 24 and len(slices) == 48
    assert json.loads((root/'training_status.json').read_text())['phase'] == 'complete'
    assert json.loads((root/'scoring_status.json').read_text())['phase'] == 'complete'
    assert json.loads((old_root/'status.json').read_text())['complete']
    optimizer = {n: json.loads((root.parent/f'optimizer_20260914_n{n}'/'results.json').read_text())
                 for n in (100, 1000)}
    offset_root = root.parent/'reference_offset_20260914'
    assert json.loads((offset_root/'manifest.json').read_text())['complete']
    offsets = json.loads((offset_root/'results.json').read_text())
    assert len(offsets) == 24 and all('result' in r for r in offsets)
    checkpoints = [{**r, **cp} for r in rows for cp in r['checkpoints']]
    # Replicates share a shape coordinate. Dividing the raw importance ESS by
    # replicas is a conservative lower bound on the ESS after pooling by shape.
    ess_bound = min(s[target]['ess']/c['replicas'] for c in checkpoints
                    for s in c['scores'] for target in ('direct', 'gp'))
    assert ess_bound >= 100, 'Replicated-shape ESS gate needs further investigation'
    plt.rcParams.update({'font.size': 10.5, 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.facecolor': 'white', 'font.family': 'DejaVu Sans'})
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True, sharey=True, layout='constrained')
    for ax, (n, mode) in zip(axes.flat, [(100, 'perfect'), (100, 'uncertain'), (1000, 'perfect'), (1000, 'uncertain')]):
        ax.axhline(.01, color='#333333', lw=1, ls=':', label='Accuracy threshold')
        for r in [r for r in rows if r['expected'] == n and r['mode'] == mode]:
            color = '#2672a3' if r['strategy'] == 'bo' else '#bc8526'
            data = r['checkpoints']
            ax.plot([c['labels'] for c in data], [maximum_kl(c) for c in data], color=color,
                    ls='-' if r['strategy'] == 'bo' else '--', lw=1.3, alpha=.8,
                    label=r['strategy'].upper() if r['seed'] == 17 else None)
            for c in data:
                ax.scatter(c['labels'], maximum_kl(c), color=color, s=25,
                           marker='x' if c['outcome'] == 'unresolved' else 'o')
        ax.set(title=f'{n} expected detections · {mode}', yscale='log', xticks=[414, 744])
        ax.yaxis.set_minor_locator(NullLocator())
    axes[0, 0].legend(frameon=False, fontsize=9, loc='lower left')
    for ax in axes[:, 0]:
        ax.set_ylabel('Maximum directed KL [nats]')
    for ax in axes[1]:
        ax.set_xlabel('Training labels')
    fig.supxlabel('Each line is one training seed; crosses indicate unresolved numerical integration', fontsize=9)
    for ext in ('png', 'pdf'):
        fig.savefig(root/f'joint_learning.{ext}', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True, layout='constrained')
    for column, n in enumerate([100, 1000]):
        for r in [r for r in rows if r['expected'] == n and r['strategy'] == 'bo']:
            variants = sorted([v for v in offsets if v['name'] == r['name']], key=lambda v: v['offset'])
            original_slice = next(v for v in slices if v['name'] == r['name'] and v['labels'] == 744)
            color = '#2672a3' if r['mode'] == 'perfect' else '#bc8526'
            style = '-' if r['mode'] == 'perfect' else '--'
            joint = [r['checkpoints'][-1], *[v['result'] for v in variants]]
            conditional = [original_slice, *[v['conditional'] for v in variants]]
            axes[0, column].plot([0, 5, 20], [maximum_kl(v) for v in joint], color=color, ls=style,
                marker='o', ms=3, lw=1.2, label=r['mode'] if r['seed'] == 17 else None)
            axes[1, column].plot([0, 5, 20], [max(v['directed_kl']) for v in conditional],
                color=color, ls=style, marker='o', ms=3, lw=1.2)
        axes[0, column].set_title(f'{n} expected detections')
        for ax in axes[:, column]:
            ax.axhline(.01, color='#333333', ls=':', lw=1)
            ax.set(yscale='log', xticks=[0, 5, 20])
            ax.yaxis.set_minor_locator(NullLocator())
        axes[1, column].set_xlabel('Reference offset above best training lnL')
    axes[0, 0].set_ylabel('Joint posterior KL [nats]')
    axes[1, 0].set_ylabel('Conditional amplitude KL [nats]')
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.supxlabel('Fixed 744 labels; each line is a training seed; dotted line marks KL = 0.01', fontsize=9)
    for ext in ('png', 'pdf'):
        fig.savefig(root/f'reference_offset.{ext}', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), layout='constrained')
    for ax, n in zip(axes, [100, 1000]):
        job = f'n{n}_perfect_s17_bo'
        for labels, color, style in [(414, '#bc8526', '--'), (744, '#2672a3', '-')]:
            data = np.load(root/job/f'conditional_slice_{labels}.npz')
            x = data['amplitude']*1000
            logdensity = data['gp']
            density = np.exp(logdensity-logdensity.max())
            density /= np.trapezoid(density, x)
            ax.plot(x, density, color=color, ls=style, lw=1.5, label=f'{labels} labels')
        density = np.exp(data['exact']-data['exact'].max())
        density /= np.trapezoid(density, x)
        ax.plot(x, density, color='#333333', ls='-.', lw=1.5, label='Exact conditional')
        observed = next(r['observed'] for r in slices if r['name'] == job)
        ax.set(title=f'{n} expected / {observed} observed', xlabel=r'SFR amplitude $A$ ($10^{-3}$)',
               xlim=(5, 15), xticks=[5, 10, 15], ylim=(0, None))
        ax.minorticks_off()
    axes[0].set_ylabel('Conditional posterior density\nper plotted amplitude unit')
    axes[0].legend(frameon=False, fontsize=9)
    fig.supxlabel('Four-input GP slices at the injection shape; fixed seed 17, perfect measurements', fontsize=9)
    for ext in ('png', 'pdf'):
        fig.savefig(root/f'amplitude_slices.{ext}', dpi=180)
    plt.close(fig)

    lines = ['# Four-input GP: small-batch follow-up', '',
        'The matched follow-up is complete: 24 trajectories, two scored budgets each. '
        'This tests the same two latent catalogues as the preceding pilot; it is not fresh-catalogue confirmation.', '',
        '## Joint posterior accuracy', '',
        'All training uses batches of three, immediate conditioning, hyperparameter refits every 30 labels, '
        'and the full prior with every training point retained. BO uses the existing optimized GPry-form '
        'acquisition generalized to four inputs; random uses IID prior draws. Initial 144-label designs are paired.', '',
        '| Expected | View | Method | 414: accurate / inaccurate / unresolved | 744: accurate / inaccurate / unresolved | 744 KL range [nats] |',
        '|---|---|---|---|---|---:|']
    for n in [100, 1000]:
        for mode in ['perfect', 'uncertain']:
            for method in ['bo', 'random']:
                group = [r for r in checkpoints if (r['expected'], r['mode'], r['strategy']) == (n, mode, method)]
                counts = []
                for labels in [414, 744]:
                    count = Counter(r['outcome'] for r in group if r['labels'] == labels)
                    counts.append(' / '.join(str(count[k]) for k in ['accurate', 'inaccurate', 'unresolved']))
                kl = [maximum_kl(r) for r in group if r['labels'] == 744]
                lines.append(f'| {n} | {mode} | {method} | {counts[0]} | {counts[1]} | {min(kl):.4g}–{max(kl):.4g} |')
    improvements = sum(maximum_kl(r['checkpoints'][1]) < maximum_kl(r['checkpoints'][0]) for r in rows)
    wins = 0
    ratios = []
    for r in [r for r in rows if r['strategy'] == 'bo']:
        other = next(s for s in rows if s['name'] == r['name'].replace('_bo', '_random'))
        b, a = r['checkpoints'][-1], other['checkpoints'][-1]
        if b['outcome'] != 'unresolved' and a['outcome'] != 'unresolved':
            ratios.append(maximum_kl(a)/maximum_kl(b))
            wins += maximum_kl(b) < maximum_kl(a)
    lines += ['', f'Increasing 414 → 744 labels lowers the KL estimate in {improvements}/24 trajectories. '
        f'At 744 labels, BO has lower KL in {wins}/{len(ratios)} numerically resolved matched pairs. '
        + (f'The median random/BO KL ratio is {np.median(ratios):.3g}. ' if ratios else '')
        + 'These are error comparisons at fixed budgets, not speedup factors or validated stopping results.', '',
        '![Joint learning curves](joint_learning.png)', '', '## Resolving the previous numerical ambiguity', '']
    count = Counter(r['outcome'] for r in old.values())
    lines += [f'The unchanged original 400-label checkpoints now have {count["accurate"]} accurate, '
        f'{count["inaccurate"]} inaccurate and {count["unresolved"]} unresolved outcomes under refined scoring. '
        'No original label or checkpoint was changed. Initial one-amplitude-draw results remain archived.', '',
        'Scoring now balances uniform and conditional-Gamma amplitude proposals, stratified within each component. '
        'Each shape point receives 16 amplitude draws, with a bounded refinement to 64 for unresolved cases. '
        'The two shape designs remain independent full-support numerical integrations; exact rates are reused only offline. '
        'Direct references must agree with the analytic marginalized-plus-conditional reconstruction. '
        'GP moments, intervals and both KL directions must also agree across designs.', '',
        f'The conservative importance-ESS lower bound after accounting for repeated shape coordinates is {ess_bound:.1f} '
        '(required ≥100). Replicated amplitude points are not counted as independent catalogues or independent shape draws.', '',
        '414-label results are a near-budget comparison with the old 400-label fits, not an exactly matched test of batch size. '
        'The acquisition rule and refit schedule both changed; the old/new contrast does not isolate only batch size.', '',
        '## Known-answer amplitude test within the four-input GP', '',
        'Fix the three shape parameters at their injection values and evaluate the trained four-input GP over amplitude. '
        'Compare with the exact truncated Gamma using the same prior. These are conditional slices of a joint GP, '
        'not separately trained one-dimensional models and not marginalized amplitude posteriors.', '',
        '| Expected | Method | Accurate slices at 414 | Accurate slices at 744 |', '|---|---|---:|---:|']
    for n in [100, 1000]:
        for method in ['bo', 'random']:
            part = [r for r in slices if r['expected'] == n and r['strategy'] == method]
            counts = [sum(r['accuracy_pass'] for r in part if r['labels'] == budget) for budget in [414, 744]]
            lines.append(f'| {n} | {method} | {counts[0]}/6 | {counts[1]}/6 |')
    lines += ['', '![Amplitude conditional slices](amplitude_slices.png)', '',
        '## Optimizer-iteration check', '',
        'Hold the 744 selected labels and target transform fixed, using perfect-measurement seed 17 at each count. '
        'Refit from identical initial conditions with 250, 1500 and 3000 Adam steps. Fit all variants before reference scoring.', '',
        '| Expected | Steps | Final negative GP marginal log likelihood | Posterior KL [nats] | Outcome |',
        '|---|---:|---:|---:|---|']
    for n, variants in optimizer.items():
        for r in variants:
            lines.append(f'| {n} | {r["steps"]} | {r["final_objective"]:.6g} | {maximum_kl(r["result"]):.5g} | {r["result"]["outcome"]} |')
    lines += ['', 'Improving the GP fitting objective is not itself evidence of posterior accuracy; judge the direct-reference column. '
        'This checks two frozen datasets, not all optimizer choices or kernels.', '',
        '## Smoothing the square-root reference', '',
        'An exploratory fixed-label ablation places the square-root transform reference 5 or 20 lnL above the '
        'best training label. This avoids the transform cusp at that label while retaining its original scale, '
        'floor, all 744 labels, fixed noise and 250 optimizer steps. All twelve BO fits are tested at both offsets. '
        'The numerical reference maximum and injection are not used to choose the offset. No production setting changed.', '',
        '| Reference offset | Joint accurate / inaccurate / unresolved | Accurate conditional amplitude slices |',
        '|---|---|---:|']
    for offset in [5., 20.]:
        part = [r for r in offsets if r['offset'] == offset]
        counts = Counter(r['result']['outcome'] for r in part)
        desc = ' / '.join(str(counts[k]) for k in ['accurate', 'inaccurate', 'unresolved'])
        lines.append(f'| +{offset:g} | {desc} | {sum(r["conditional"]["accuracy_pass"] for r in part)}/12 |')
    lines += ['', 'This is a fit-only investigation on already examined catalogues. It does not validate acquisition '
        'with a changed transform from the start. A successful conditional amplitude check is necessary evidence, '
        'but cannot substitute for the joint posterior comparison.', '',
        '![Reference-offset ablation](reference_offset.png)', '',
        '## Decision and limits', '',
        '- Keep the validated amplitude-marginalized workflow as the reference. This experiment does not authorize replacing it.',
        '- The explicit amplitude remains a useful known-answer diagnostic. A joint GP must recover both this conditional law and the full posterior.',
        '- More labels help in the measured cases, but extrapolating a required budget or certifying a universally adequate four-input GP is unsupported.',
        '- These results isolate approximation to a fixed finite-population likelihood. They do not test independent COMPAS realizations, finer grids, prior-wide calibration, GP variance calibration, missed modes or NUTS correctness.', '',
        '## Reproducibility and checks', '',
        'The new dimension-independent acquisition matches the historical three-input function in a seeded parity test. '
        'Four-input coordinates are checked for novelty, bounds and reproducibility; stratified importance weights are tested '
        'against uniform-prior integrals and truncated-Gamma moments, including tail/underflow cases. '
        'The reference-offset round trip, unchanged scale/floor and smooth peak derivative are also tested. '
        'The full fast suite passed 182 tests (three heavy/network tests deselected; one existing Matplotlib warning).', '',
        'The final saved-run audit checks all 24 trajectories, 48 checkpoint hashes and 4800 pre-training batches. '
        'All source snapshots and paired initial designs match. An independent check of 192 labels through '
        'physical rate evaluations agrees within 1.46e-11 lnL; the catalogue exposure/count scaling used '
        'in the Gamma-slice test also passes. See `verification.json`. All three rendered figures were visually inspected.', '',
        'Sources and protocol were frozen before training in `manifest.json` and `source_snapshot/`. '
        'Per-job `complete.json`, `history.json`, `batches.json`, `scores.json`, and independent-reference metadata retain '
        'costs, predictions made before adding labels, checkpoint hashes, initial/refined numerical attempts and outcomes. '
        'Training times exclude shared initial-data construction and offline references; do not infer end-to-end speedup.', '',
        'Run `small_batch_followup.py --output NEW_DIRECTORY --stage train`, followed by the same command with '
        '`--stage score`. `conditional_slices.py`, `optimizer_check.py` and `rescore_large_batch.py` provide the separate controls. '
        'Run this summarizer after all controls finish. Production defaults, prior, grid and manuscript remain unchanged. '
        'No commits or remote writes.', '']
    (root/'RESULTS.md').write_text('\n'.join(lines))
    sources = [root/'results.json', root/'conditional_slices.json', old_root/'results.json', offset_root/'results.json', Path(__file__)]
    (root/'figure_sources.json').write_text(json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                                     for p in sources}, indent=2)+'\n')


if __name__ == '__main__':
    main()
