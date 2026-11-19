"""Render inspectable pilot diagnostics from frozen results, without refitting."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, NullLocator
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    args = parser.parse_args()
    root = args.input.resolve()
    conditional = json.loads((root/'conditional.json').read_text())
    rows = json.loads((root/'results.json').read_text())
    status = json.loads((root/'status.json').read_text())
    assert status['phase'] == 'complete' and len(rows) == 24 and len(conditional) == 9
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.facecolor': 'white', 'font.family': 'DejaVu Sans'})
    colors = {17: '#2672a3', 29: '#c18a28', 43: '#6a7841'}
    styles = {17: '-', 29: '--', 43: ':'}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), layout='constrained')
    for ax, expected in zip(axes, [10, 100, 1000]):
        selected = [r for r in conditional if r['expected'] == expected]
        for r in selected:
            data = np.load(root/f'conditional_n{expected}_s{r["seed"]}.npz')
            x = data['amplitude']*1000
            density = np.exp(data['gp']-data['gp'].max())
            density /= np.trapezoid(density, x)
            ax.plot(x, density, color=colors[r['seed']], ls=styles[r['seed']], lw=1.5,
                    label=f'GP seed {r["seed"]}')
        exact = np.exp(data['exact']-data['exact'].max())
        exact /= np.trapezoid(exact, x)
        ax.plot(x, exact, color='#262626', lw=1.5, ls='-.', label='Exact')
        ax.set(title=f'{expected} expected / {selected[0]["observed"]} observed',
               xlabel=r'SFR amplitude $A$ ($10^{-3}$)', xlim=(5, 15), ylim=(0, None))
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.minorticks_off()
    axes[0].set_ylabel('Posterior density per plotted unit')
    axes[0].legend(frameon=False, fontsize=9)
    fig.savefig(root/'conditional_posteriors.png', dpi=180)
    fig.savefig(root/'conditional_posteriors.pdf')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True, layout='constrained')
    for ax, expected in zip(axes, [100, 1000]):
        ax.axhline(.01, ls='--', color='#444444', lw=1, label='Accuracy threshold')
        for group, mode in enumerate(['perfect', 'uncertain']):
            for j, seed in enumerate([17, 29, 43]):
                pair = [next(r for r in rows if r['expected'] == expected and r['mode'] == mode
                             and r['seed'] == seed and r['strategy'] == strategy)
                        for strategy in ['bo', 'random']]
                x = group*3 + np.array([0, 1]) + (j-1)*.13
                y = [max(max(s['directed_kl']) for s in r['scores']) for r in pair]
                ax.plot(x, y, color='#bbbbbb', lw=.7, zorder=1)
                for xx, yy, r in zip(x, y, pair):
                    ax.scatter(xx, yy, color='#2672a3' if r['strategy'] == 'bo' else '#c18a28',
                               marker='x' if r['outcome'] == 'unresolved' else 'o', s=38, zorder=2)
        ax.set(yscale='log', title=f'{expected} expected detections',
               xticks=[0, 1, 3, 4], xticklabels=['BO\nperfect', 'Random\nperfect', 'BO\nuncertain', 'Random\nuncertain'])
        ax.yaxis.set_minor_locator(NullLocator())
    axes[0].set_ylabel('Maximum directed KL estimate [nats]')
    axes[0].legend(frameon=False, fontsize=9, loc='lower right')
    fig.supxlabel('400 labels each; links pair training seeds; crosses have unresolved GP integration', fontsize=10)
    fig.savefig(root/'joint_errors.png', dpi=180)
    fig.savefig(root/'joint_errors.pdf')
    plt.close(fig)

    lines = ['# Explicit SFR amplitude pilot — results', '',
        'The bounded pilot is complete. The four-input GP is not validated at the tested 400-label budget.', '',
        '## Conditional amplitude', '',
        'Fixed shape, 33 training labels, three scrambled Sobol designs per catalogue size. '
        'The count draws are 8, 91 and 943. Eight of nine GPs pass the accuracy gates; '
        'the 943-event seed-43 fit fails. These are three count realizations, not nine independent catalogues.', '',
        '| Expected / observed | Maximum KL across seeds [nats] | Accurate seeds | Exact bounded fractional SD | Unbounded Gamma fractional SD |',
        '|---|---:|---:|---:|---:|']
    for n in [10, 100, 1000]:
        part = [r for r in conditional if r['expected'] == n]
        lines.append(f'| {n} / {part[0]["observed"]} | {max(max(r["directed_kl"]) for r in part):.6g} | '
            f'{sum(r["accuracy_pass"] for r in part)}/3 | {part[0]["bounded_fractional_sd"]:.5f} | {part[0]["unbounded_fractional_sd"]:.5f} |')
    lines += ['', 'The bounded prior is Uniform(0.005, 0.015). The unbounded benchmark is '
        '1/sqrt(Nobs+1), not exactly 1/sqrt(Nobs). Direct trapezoidal CDFs agree with '
        f'the analytic truncated Gamma to {max(r["exact_cdf_error"] for r in conditional):.3g} maximum absolute error.', '',
        '![Conditional amplitude posteriors](conditional_posteriors.png)', '',
        '## Joint four-input GP', '',
        'Two saved latent catalogues (100 and 1000 expected detections), each with perfect/uncertain '
        'views, three training seeds and matched BO/random continuations give 24 fits. '
        'All 24 completed at 400 labels. None passes the full posterior gate: '
        f'{Counter(r["outcome"] for r in rows)["inaccurate"]} are numerically resolved and inaccurate; '
        f'{Counter(r["outcome"] for r in rows)["unresolved"]} have unresolved GP integration. '
        'All four direct reference checks and analytic-amplitude parity checks pass.', '',
        '| Expected | View | Strategy | Accurate / inaccurate / unresolved | KL estimate range [nats] | Direct-weighted centered lnL RMSE range |',
        '|---|---|---|---|---:|---:|']
    for n in [100, 1000]:
        for mode in ['perfect', 'uncertain']:
            for strategy in ['bo', 'random']:
                part = [r for r in rows if (r['expected'], r['mode'], r['strategy']) == (n, mode, strategy)]
                count = Counter(r['outcome'] for r in part)
                kl = [max(max(s['directed_kl']) for s in r['scores']) for r in part]
                rmse = [s['centered_lnl_rmse'] for r in part for s in r['scores']]
                lines.append(f'| {n} | {mode} | {strategy} | {count["accurate"]} / {count["inaccurate"]} / {count["unresolved"]} | '
                             f'{min(kl):.3g}–{max(kl):.3g} | {min(rmse):.3g}–{max(rmse):.3g} |')
    wins = 0
    for r in [r for r in rows if r['strategy'] == 'bo']:
        other = next(s for s in rows if s['strategy'] == 'random' and
                     (s['expected'], s['mode'], s['seed']) == (r['expected'], r['mode'], r['seed']))
        wins += max(max(s['directed_kl']) for s in r['scores']) < max(max(s['directed_kl']) for s in other['scores'])
    lines += ['', f'BO has the lower KL estimate in {wins}/12 pairs. Most GP integrations are unresolved, '
        'so this is directional diagnostic evidence, not a validated efficiency ratio. '
        'The large-batch 25/75 sampling recipe is a new pilot policy; it is not the historical '
        'three-member optimized acquisition. There was no online stopping test.', '',
        '![Joint GP error estimates](joint_errors.png)', '',
        '## Interpretation and next decision', '',
        '- The conditional Gamma calculation supplies a useful known-answer check, and already exposes a high-count GP miss.',
        '- Do not replace the validated marginalized workflow or rerun the large campaign with this four-input configuration.',
        '- The next useful experiment is a bounded comparison using the existing small-batch acquisition generalized to four inputs, with the failed pilot preserved. This separates fourth-coordinate difficulty from the changed acquisition/batch schedule.',
        '- Reference integration must also cover inaccurate, broader GP posteriors before precise joint-KL claims. Keep current unresolved cases explicit; do not interpret them as posterior passes.',
        '- These tests do not validate finite-population sufficiency, grid information loss, GP variance calibration, or NUTS sampling. Adaptive pixels remain deferred.', '',
        '## Reproduction and verification', '',
        'Run from the repository root, using a new output directory:', '', '```sh',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/compas-amplitude-mpl .venv/bin/python docs/studies/amplitude_gp/run_pilot.py --output NEW_DIRECTORY',
        '.venv/bin/python docs/studies/amplitude_gp/summarize.py NEW_DIRECTORY', '```', '',
        'A separate saved-run audit verifies all 24 checkpoint/source hashes and paired initial designs. '
        'It checks 192 labels against physical rate evaluations at the actual amplitude, agreeing '
        'within 7.3e-12 in log likelihood; perfect/uncertain catalogue pairing also passes. '
        'See `verification.json` and `verify_run.py`. Both rendered diagnostic figures were visually inspected.', '',
        'The numerical likelihood-integration, proposal-density, support and invalid-input tests pass. '
        'The fast suite passed 177 tests, with three heavy/network tests deselected and one existing Matplotlib warning. '
        'Production defaults and the manuscript are unchanged. No commits or remote writes.', '',
        'The training timer excludes the shared initial-label evaluation, loader setup, and offline reference work. '
        'It must not be quoted as an end-to-end benchmark. Source hashes, catalogue hashes, saved labels, '
        'checkpoints, per-scramble scores and progress records are retained in this directory.', '']
    (root/'RESULTS.md').write_text('\n'.join(lines))
    sources = [root/'conditional.json', root/'results.json', root/'status.json', Path(__file__)]
    (root/'figure_sources.json').write_text(json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                                     for p in sources}, indent=2)+'\n')


if __name__ == '__main__':
    main()
