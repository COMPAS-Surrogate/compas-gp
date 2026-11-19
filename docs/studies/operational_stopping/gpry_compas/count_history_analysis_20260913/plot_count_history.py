"""Render an immutable progress snapshot from the running paired BO campaign."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np

COUNTS = (10, 50, 100, 200, 500, 700, 1000)
MODES = ('perfect', 'uncertain')
COLORS = {'perfect': '#24749b', 'uncertain': '#bc6d25'}
LABELS = {'perfect': 'Perfect measurements', 'uncertain': 'Uncertain measurements'}
STATE_COLORS = {'accurate': '#d9e9f1', 'running': '#f5d694', 'pending': '#f2f3f4',
                'premature': '#d66b78', 'unresolved': '#c3afd2', 'no_stop': '#b8bdc5',
                'failed': '#d66b78'}


def curve_data(history: list[dict], metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exclude late-GP self-KL; leave unresolved values flagged, not discarded."""
    rows = sorted(history, key=lambda r: r['posterior_comparisons'])
    indices = [r['posterior_comparisons'] for r in rows]
    if len(set(indices)) != len(indices):
        raise ValueError('Duplicate posterior comparison indices')
    if metric == 'symmetric_kl_to_late_gp':
        rows = [r for r in rows if not r['late_reference_is_self']]
    x = np.array([r['posterior_comparisons'] for r in rows])
    y = np.array([r[metric] for r in rows])
    resolved = np.array([bool(r['reference_numerical'] and r['gp_numerical'] and r['four_numerical'])
                         for r in rows])
    if not np.isfinite(y).all() or np.any(y < 0):
        raise ValueError('KL values must be finite and nonnegative')
    return x, y, resolved


def collect_snapshot(root: Path) -> dict:
    """Freeze completed-case membership once; only read campaign files."""
    sources = {}
    def read(path: Path):
        raw = path.read_bytes()
        sources[str(path.relative_to(root))] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    manifest = read(root/'manifest.json')
    status = read(root/'campaign_status.json')
    cases = []
    for path in sorted(root.glob('offline/*/outcome.json')):
        result = read(path)
        history = read(path.parent/'history.json')
        if len(history) != result['checkpoints'] or len(history) != (manifest['maximum']-58)//30+1:
            raise ValueError(f'Incomplete history for {result["name"]}')
        if [r['points'] for r in history] != list(range(58, manifest['maximum']+1, 30)):
            raise ValueError('Unexpected checkpoint schedule')
        selected = result['stop']['selected']
        if selected:
            row = next(r for r in history if r['points'] == selected['training'])
            if row['posterior_comparisons'] != result['stop']['posterior_comparisons']:
                raise ValueError('Stopping and plotting axes disagree')
            if selected['total'] != selected['training']+selected['audit']:
                raise ValueError('Stopping costs do not sum')
            np.testing.assert_allclose(row['maximum_directed_kl'], result['maximum_directed_kl'])
        for row in history:
            kls = [list(s.values()) for s in row['kl_to_late_gp']]
            np.testing.assert_allclose(row['symmetric_kl_to_late_gp'], np.mean(kls))
        keep = ('points', 'posterior_comparisons', 'reference_numerical', 'gp_numerical',
                'four_numerical', 'symmetric_kl', 'maximum_directed_kl',
                'symmetric_kl_to_late_gp', 'late_reference_is_self', 'checkpoint_sha256')
        cases.append(dict(result=result, history=[{k: r[k] for k in keep} for r in history]))
    return dict(captured_at=datetime.now(ZoneInfo('Pacific/Auckland')).isoformat(),
                campaign=str(root), manifest=manifest, status=status, cases=cases,
                sources_sha256=sources)


def subset(snapshot: dict, n: int, mode: str) -> list[dict]:
    return [c for c in snapshot['cases'] if (c['result']['expected_events'], c['result']['mode']) == (n, mode)]


def save(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.savefig(output/f'{stem}.png', dpi=180, bbox_inches='tight', facecolor='white')
    fig.savefig(output/f'{stem}.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)


def progress(snapshot: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), sharey=True)
    jobs = snapshot['manifest']['jobs']
    cases = {c['result']['name']: c['result'] for c in snapshot['cases']}
    current = snapshot['status'].get('current_job')
    active = current['name'] if current else None
    iterations = snapshot['manifest']['cases_per_count']
    used = set()
    for ax, mode in zip(axes, MODES):
        for job in jobs:
            if job['mode'] != mode:
                continue
            x, y = job['iteration']-1, COUNTS.index(job['expected_events'])
            result = cases.get(job['name'])
            state = (result['outcome'] if result else
                     'running' if job['name'] == active and snapshot['status']['state'] == 'running' else
                     'failed' if job['name'] == active and snapshot['status']['state'] == 'failed' else 'pending')
            used.add(state)
            ax.add_patch(Rectangle((x-.47, y-.44), .94, .88, fc=STATE_COLORS[state], ec='white'))
            number = result['stop']['posterior_comparisons'] if result else None
            label = str(number) if number is not None else ('RUN' if state == 'running' else
                    '—' if state == 'pending' else state.replace('_', ' '))
            ax.text(x, y, label, ha='center', va='center', fontsize=11 if number is not None else 8,
                    color='#1e3c50' if state == 'accurate' else '#555555')
        ax.set(xlim=(-.55, iterations-.45), ylim=(6.6, -.8), xticks=range(iterations),
               xticklabels=range(1, iterations+1), yticks=range(7), yticklabels=COUNTS,
               xlabel='Catalogue iteration')
        ax.text(.5, 1.02, LABELS[mode], transform=ax.transAxes, ha='center', fontsize=12)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_ylabel(r'Expected observations, $\mathbb{E}[N_{\rm obs}]$')
    names = {'accurate': 'Completed: accuracy passed', 'running': 'Running', 'pending': 'Pending',
             'premature': 'Premature stop', 'unresolved': 'Unresolved', 'no_stop': 'No stop', 'failed': 'Execution failed'}
    fig.legend(handles=[Patch(fc=STATE_COLORS[s], label=names[s]) for s in names if s in used],
               loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(.5, .01), fontsize=9)
    fig.subplots_adjust(bottom=.23, top=.88, wspace=.16)
    save(fig, output, 'progress')


def kl_evolution(snapshot: dict, output: Path, metric: str, stem: str) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5), sharex=True, sharey=True)
    values = [curve_data(c['history'], metric)[1] for c in snapshot['cases']]
    positive = np.concatenate(values) if values else np.array([1e-5, 1.])
    positive = positive[positive > 0]
    limits = (10**np.floor(np.log10(positive.min())), 10**np.ceil(np.log10(positive.max())))
    for ax, n in zip(axes.flat, COUNTS):
        counts = []
        for mode in MODES:
            group = subset(snapshot, n, mode); counts.append(len(group))
            stack = []
            for c in group:
                x, y, resolved = curve_data(c['history'], metric)
                stack.append(y)
                # Zero KL cannot appear on a log axis; never replace it with a
                # fabricated positive floor. Final self-comparisons are excluded.
                shown = np.where(y > 0, y, np.nan)
                color = COLORS[mode]
                ax.plot(x, shown, color=color, lw=.75, alpha=.3)
                ax.plot(x[~resolved], shown[~resolved], 'x', color=color, ms=4)
                stop = c['result']['stop']['posterior_comparisons']
                if stop is not None and stop in x:
                    k = np.flatnonzero(x == stop)[0]
                    ax.plot(stop, shown[k], 'o' if mode == 'perfect' else 's', ms=4,
                            mec=color, mfc='white', mew=1., zorder=5)
            if stack:
                median = np.median(stack, axis=0)
                ax.plot(x, np.where(median > 0, median, np.nan), color=COLORS[mode],
                        lw=1.8, ls='-' if mode == 'perfect' else '--')
        ax.text(.96, .94, rf'$\mathbb{{E}}[N_{{\rm obs}}]={n}$', transform=ax.transAxes,
                ha='right', va='top', fontsize=11)
        ax.text(.96, .82, f'{counts[0]} perfect / {counts[1]} uncertain', transform=ax.transAxes,
                ha='right', va='top', fontsize=8.5, color='.4')
        ax.set(yscale='log', ylim=limits, xlim=(0, 40), xticks=[0, 10, 20, 30, 40])
        ax.grid(axis='y', alpha=.18, lw=.65)
        ax.tick_params(which='both', direction='in', top=True, right=True)
    axes.flat[-1].axis('off')
    handles = [Line2D([], [], color=COLORS[m], lw=1.8, ls='-' if m == 'perfect' else '--',
                      marker='o' if m == 'perfect' else 's', mfc='white', label=LABELS[m]) for m in MODES]
    axes.flat[-1].legend(handles=handles, loc='upper left', frameon=False, fontsize=10)
    axes.flat[-1].text(.04, .55, 'Thin lines: individual catalogues\nThick lines: median\nOpen markers: recorded audit stop\n\nCompleted jobs only; partial ensemble.',
                      transform=axes.flat[-1].transAxes, fontsize=10, va='top', linespacing=1.6)
    fig.supxlabel('Number of posterior comparisons', y=.02, fontsize=12)
    ref = 'late GP' if metric == 'symmetric_kl_to_late_gp' else 'direct posterior'
    fig.supylabel(r'$D_{\rm KL}^{\rm sym}$ to '+ref+' [nats]', x=.01, fontsize=12)
    fig.subplots_adjust(left=.075, bottom=.12, top=.96, right=.99, hspace=.19, wspace=.12)
    save(fig, output, stem)


def stopping(snapshot: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for mode in MODES:
        offset = -.15 if mode == 'perfect' else .15
        for j, n in enumerate(COUNTS):
            group = subset(snapshot, n, mode)
            stopped = [c['result'] for c in group if c['result']['stop']['selected']]
            if not stopped:
                continue
            values = [[r['stop']['posterior_comparisons'] for r in stopped],
                      [r['stop']['selected']['total'] for r in stopped],
                      [r['maximum_directed_kl'] for r in stopped]]
            dx = np.linspace(-.04, .04, len(stopped)) if len(stopped) > 1 else [0]
            for ax, y in zip(axes, values):
                ax.plot(j+offset+np.asarray(dx), y, 'o' if mode == 'perfect' else 's',
                        color=COLORS[mode], mfc='white', ms=5, mew=1.2)
                ax.plot([j+offset-.08, j+offset+.08], [np.median(y)]*2, color=COLORS[mode], lw=2)
    labels = ['Posterior comparisons at stop', 'Training + audit labels at stop',
              'Maximum directed KL to direct\nposterior at stop [nats]']
    for ax, label in zip(axes, labels):
        ax.set(xticks=range(7), xticklabels=COUNTS, xlim=(-.6, 6.6), ylabel=label,
               xlabel=r'Expected observations, $\mathbb{E}[N_{\rm obs}]$')
        ax.grid(axis='y', alpha=.2)
        ax.tick_params(direction='in', top=True, right=True)
    axes[0].set_ylim(bottom=0); axes[1].set_ylim(bottom=0); axes[2].set_yscale('log')
    fig.legend(handles=[Line2D([], [], marker='o' if m == 'perfect' else 's', color=COLORS[m],
                               ls='', mfc='white', label=LABELS[m]) for m in MODES],
               loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(.5, 1.02))
    fig.subplots_adjust(left=.06, right=.99, bottom=.18, top=.87, wspace=.38)
    save(fig, output, 'stopping_summary')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, help='Re-render an existing immutable snapshot')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = json.loads(args.snapshot.read_text()) if args.snapshot else collect_snapshot(args.root.resolve())
    (args.output/'snapshot.json').write_text(json.dumps(snapshot, indent=2, allow_nan=False)+'\n')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.labelsize': 10,
                         'axes.edgecolor': '.3', 'axes.linewidth': .7, 'pdf.fonttype': 42})
    progress(snapshot, args.output)
    kl_evolution(snapshot, args.output, 'symmetric_kl_to_late_gp', 'kl_late_reference')
    kl_evolution(snapshot, args.output, 'symmetric_kl', 'kl_direct_reference')
    stopping(snapshot, args.output)
    completed = len(snapshot['cases'])
    outcomes = dict(Counter(c['result']['outcome'] for c in snapshot['cases']))
    caption = f'''# Campaign snapshot

Captured {snapshot['captured_at']}. {completed}/70 completed jobs; outcomes: {outcomes}.
The plots use the exact immutable membership and source hashes in `snapshot.json`.

- `progress`: every scheduled job, separated by measurement model. Numbers in
  completed cells are the recorded fixed96 audit stopping comparison counts.
  Grey cells are pending, not failed. Each iteration adds one paired catalogue
  at each expected count. This snapshot is not a live dashboard.
- `kl_late_reference`: symmetric KL (mean of the two directions, averaged over
  two independent numerical designs) between each GP and its continued
  1258-label GP. The final self-comparison is omitted. Each panel includes every
  completed case in its group; thin lines show cases and thick lines show their
  median. Open circles/squares mark recorded perfect/uncertain audit stops.
  These are shadow operational decisions, frozen before reference scoring.
- `kl_direct_reference`: the same trajectories scored against the independent
  direct numerical target, retaining the final checkpoint. Symmetric KL has the
  same definition. Crosses, if present, mark numerically unresolved checkpoints.
- `stopping_summary`: individual completed-case stops (open symbols) and medians
  (short horizontal strokes). Total labels include training and fixed96 audit
  calls up to acceptance, excluding diagnostic continuation, legacy monitor
  audits and offline reference costs. The right panel uses the maximum of both
  directed KLs over both designs, which differs from the symmetric trajectory
  metric. Expected-count groups are equally spaced categories in this figure.

Only completed cases are included in curves and stopping summaries. With just
two or three catalogues in most groups, the scatter is preliminary and does not
establish calibrated coverage or a final ensemble result. Catalogue pairing is
preserved; each expected-count setting has its own independent catalogues.
Late-reference acceptance flags, all outcomes, and full source hashes remain in
the snapshot. A late GP's agreement with itself is never treated as evidence.
The scientific prior, acquisition, stopping policy and manuscript are unchanged.
'''
    (args.output/'CAPTIONS.md').write_text(caption)
    print(json.dumps(dict(output=str(args.output.resolve()), completed=completed, outcomes=outcomes)))


if __name__ == '__main__':
    main()
