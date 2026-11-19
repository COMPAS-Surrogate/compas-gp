"""Reference-KL learning curves and consistently selected comparison posteriors.

Uses saved training states and exact reference evaluations. No acquisition,
training label, original stopping decision or scientific prior is changed.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import presentation_figures as old
from reference_selection import KL_LIMIT, select_checkpoint, symmetric_reference_kl

HERE, REPO, BASE = old.HERE, old.REPO, old.BASE
OUT = HERE / 'reference_kl_20260912'
FIG = old.FIG


def write(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def prepare(mode: str, expected: int) -> tuple[dict, dict]:
    """Score every saved checkpoint, then draw the selected four-D posterior."""
    name = f't0_n{expected}_{mode}_c100_bo'
    folder = old.FRESH / 'training' / name
    catpath = folder / 'catalogue.npz'
    cat = dict(np.load(catpath))
    hashes = {}
    refs = old.validation.references(old.ROOT / 'extension_v2', 0, expected, hashes)
    checked = old.validation.score.direct_checks(refs, cat, cat['truth'])
    assert checked[-1]['passed'], 'Numerical reference must be resolved'
    reference_weights = checked[0]
    checkpoints = json.loads((folder / 'checkpoints.json').read_text())
    records = []
    for checkpoint in checkpoints:
        count = checkpoint['points']
        path = folder / f'checkpoint_{count}.npz'
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        assert sha == checkpoint['sha256']
        hashes[str(path.relative_to(REPO))] = sha
        dest = OUT / name / f'kl_{count}.json'
        if dest.exists() and json.loads(dest.read_text()).get('definition') == 'half_sum_directed':
            row = json.loads(dest.read_text())
            assert row['checkpoint_sha256'] == sha
        else:
            state, _, _ = BASE.load_checkpoint(path)
            weights = [BASE.predict(state, (p-BASE.LOW)/BASE.WIDTH)-q for p, _, q in refs]
            estimates = [float(np.mean(old.validation.score.directed_kls(a, b)))
                         for a, b in zip(reference_weights, weights)]
            summaries = [BASE.posterior_summary((p-BASE.LOW)/BASE.WIDTH, w)
                         for (p, _, _), w in zip(refs, weights)]
            numerical, detail = BASE.summaries_agree(*summaries, BASE.TOL)
            row = dict(points=count, symmetric_kl=float(np.mean(estimates)), definition='half_sum_directed',
                       independent_estimates=estimates, numerical_pass=bool(numerical),
                       numerical_check=detail, checkpoint_sha256=sha)
            write(dest, row)
        records.append(row)
        print(name, count, f'KL={row["symmetric_kl"]:.6g}', flush=True)
    selected = select_checkpoint(records)
    if selected is None:
        raise ValueError(f'{name}: no selected checkpoint within saved budget')
    count = selected['points']
    phi = np.concatenate([p for p, _, _ in refs])
    beta = np.concatenate([r.sum(1)*float(cat['duration'])/.012 for _, r, _ in refs])
    weights = {'reference': reference_weights}
    targets = {'reference': checked[2][0]}
    checks = {}
    for n in [118, count]:
        state, _, _ = BASE.load_checkpoint(folder / f'checkpoint_{n}.npz')
        ws = [BASE.predict(state, (p-BASE.LOW)/BASE.WIDTH)-q for p, _, q in refs]
        summaries = [old.validation.score.four_summary(p, r, cat, cat['truth'], w)
                     for (p, r, _), w in zip(refs, ws)]
        passed, detail = BASE.data_source.precision(*summaries)
        assert passed, (name, n, 'Plot integration unresolved', detail)
        checks[str(n)] = detail
        weights[str(n)] = ws
        targets[str(n)] = summaries[0]
    samples = {key: old.draw_posterior(phi, beta,
               np.concatenate([w-logsumexp(w) for w in ws]), len(cat['coeff']))
               for key, ws in weights.items()}
    draw_checks = {}
    for key, draw in samples.items():
        target = targets[key]
        mean_error = float(np.max(abs(draw.mean(0)-target['mean'])/target['sd']))
        quantiles = np.quantile(draw, [.025,.16,.84,.975], axis=0)
        endpoint_error = float(np.max(abs(quantiles-np.asarray(target['quantiles'])[[0,2,4,6]])/target['sd']))
        assert mean_error < .05 and endpoint_error < .07, (name, key, mean_error, endpoint_error)
        draw_checks[key] = dict(mean_error_sd=mean_error, endpoint_error_sd=endpoint_error)
    hashes[str(catpath.relative_to(REPO))] = hashlib.sha256(catpath.read_bytes()).hexdigest()
    meta = dict(name=name, mode=mode, expected=expected, events=len(cat['coeff']),
                truth=cat['truth'].tolist(), selected={'training': count},
                selection='Earliest checkpoint completing two consecutive symmetric reference KL values below 0.01',
                reference='Independent exact-likelihood numerical posterior',
                reference_label='Reference', selected_label=f'Selected BO: {count}',
                records=records, source_sha256=hashes, sampling_checks=checks, draw_checks=draw_checks)
    np.savez_compressed(OUT / f'n{expected}_{mode}_samples.npz', **samples)
    write(OUT / f'n{expected}_{mode}.json', meta)
    return samples, meta


def paired_corners(bundles: list, mode: str, limits: tuple[float, float]) -> None:
    """Give each corner a full learning curve and one selection marker."""
    fig = plt.figure(figsize=(7.2, 5.7))
    rectangles = [(.065, .475, .385, .985), (.585, .995, .385, .985)]
    for rect, (samples, meta) in zip(rectangles, bundles):
        old.corner_axes(fig, rect, samples, meta)
        ax = fig.add_axes([rect[0], .10, rect[1]-rect[0], .195])
        rows = meta['records']
        x = np.array([r['points'] for r in rows])
        y = np.array([r['symmetric_kl'] for r in rows])
        ax.plot(x, y, color='#226A91', lw=1.25)
        # These are individual trajectories: no invented ensemble/uncertainty band.
        bad = np.array([not r['numerical_pass'] for r in rows])
        ax.scatter(x[bad], y[bad], s=13, facecolors='white', edgecolors='#226A91', zorder=4)
        ax.axhline(KL_LIMIT, color='.4', ls='--', lw=.8)
        n = meta['selected']['training']
        yy = next(r['symmetric_kl'] for r in rows if r['points'] == n)
        ax.scatter([n], [yy], color='.15', s=18, zorder=5)
        ax.set(xlim=(50, 670), ylim=limits, yscale='log',
               xlabel='Likelihood evaluations', ylabel=r'$D_{\mathrm{KL}}^{\mathrm{sym}}$')
        ax.set_xticks([100, 250, 400, 550, 650])
        ax.grid(which='major', color='.85', lw=.55)
        ax.minorticks_on()
        ax.tick_params(which='both', direction='in', top=True, right=True, labelsize=8)
        ax.xaxis.label.set_size(9)
        ax.yaxis.label.set_size(10)
    handles = [Line2D([], [], color='.4', ls='--', lw=.8, label=r'Comparison target: $10^{-2}$'),
               Line2D([], [], color='.15', ls='', marker='o', ms=4, label='Selected checkpoint')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.54, -.015),
               ncol=2, frameon=False, fontsize=8)
    for suffix in ['pdf', 'png']:
        fig.savefig(FIG / f'posterior_stop_{mode}.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def replay_records(records: list[dict], outcomes: list[dict]) -> tuple[list, list]:
    """Apply one rule to all strategies, keeping every unselected trajectory."""
    groups = defaultdict(list)
    key = lambda r: (r['mode'], r['exposure'], r['case'], r['strategy'])
    for r in records:
        groups[key(r)].append(r)
    selected_outcomes, selected_records = [], []
    for original in outcomes:
        rows = groups[key(original)]
        chosen = select_checkpoint(rows)
        result = {k: original[k] for k in ['mode', 'exposure', 'case', 'strategy']}
        result.update(points=chosen['points'] if chosen else max(r['points'] for r in rows),
                      state='selected' if chosen else 'target_not_reached',
                      selection='symmetric_reference_kl_below_0.01_twice')
        selected_outcomes.append(result)
        if chosen:
            selected_records.append(chosen)
    return selected_outcomes, selected_records


def ensemble() -> None:
    """Recompute both evaluation counts and physical widths at the new selection."""
    p = old.plot_paper
    records = p.read(p.ROOT / 'convergence_20260910/ensemble/records.json')
    updates = p.read(HERE / 'refined_case49_20260911/records.json')
    key = lambda r: (r['mode'], r['exposure'], r['case'], r['strategy'], r['points'])
    lookup = {key(r): r for r in records}
    lookup.update({key(r): r for r in updates})
    outcomes, chosen = replay_records(list(lookup.values()), p.ensemble_rows())
    write(OUT / 'ensemble_outcomes.json', outcomes)
    chosen_lookup = {key(r): r for r in chosen}
    physical = []
    for o in outcomes:
        if o['strategy'] != 'bo':
            continue
        if o['state'] != 'selected':
            raise ValueError('Missing BO width; do not silently drop a catalogue')
        r = chosen_lookup[key(o)]
        q = np.asarray(r['summaries'][0]['quantiles'])
        physical.append(dict(mode=r['mode'], expected=r['exposure'], case=r['case'], points=r['points'],
                             width68=(q[4]-q[2]).tolist(),
                             normalized_width68=((q[4]-q[2])/(.68*old.WIDTH)).tolist()))
    # Reuse the established panel layout, with all data supplied from this replay.
    original_rows, original_widths, original_quantiles, original_out = p.ensemble_rows, old.width_rows, old.completion_quantiles, old.OUT
    p.ensemble_rows = lambda: [{**r, 'state': 'reference_verified_convergence' if r['state']=='selected' else 'resource_limit'} for r in outcomes]
    old.width_rows = lambda _: physical
    old.OUT = OUT
    try:
        old.efficiency()
    finally:
        p.ensemble_rows, old.width_rows, old.completion_quantiles, old.OUT = original_rows, original_widths, original_quantiles, original_out
    physics_path = OUT / 'physics_widths.json'
    physics = json.loads(physics_path.read_text())
    physics['source'] = 'BO posteriors selected by symmetric reference KL < 0.01 twice; all 600 catalogues'
    write(physics_path, physics)
    write(OUT / 'ensemble_sources.json', p.SOURCES)
    counts = {s: sum(r['state']=='selected' for r in outcomes if r['strategy']==s) for s in ['bo', 'random']}
    write(OUT / 'ensemble_summary.json', dict(selected=counts, total_per_strategy=600))
    lines = [r'\begin{table*}\centering',
             r'\caption{Evaluations at the second consecutive checkpoint with symmetric reference KL below 0.01. Counts give selected / target not reached within the saved budget. Values are medians [10th--90th percentiles], retaining unselected runs above the observed budgets. Each group contains 100 catalogues.}',
             r'\begin{tabular}{rlrrrr}\toprule Expected detections & Measurements & BO counts & BO evaluations & Random counts & Random evaluations\\\midrule']
    for n, m in [(30,'perfect'),(100,'perfect'),(300,'perfect'),(1000,'perfect'),(50,'uncertain'),(500,'uncertain')]:
        cells=[]
        for s in ['bo','random']:
            rr=[r for r in outcomes if (r['exposure'],r['mode'],r['strategy'])==(n,m,s)]
            nselected=sum(r['state']=='selected' for r in rr)
            vals=sorted(r['points'] if r['state']=='selected' else np.inf for r in rr)
            lo,med,hi=np.quantile(vals,[.1,.5,.9],method='inverted_cdf')
            assert np.isfinite([lo,med,hi]).all()
            assert all(r['points'] >= hi for r in rr if r['state'] != 'selected'), 'Reported quantile intersects a censored budget'
            cells += [f'{nselected} / {100-nselected}',f'{med:g} [{lo:g}--{hi:g}]']
        lines.append(f'{n} & {m.capitalize()} & '+ ' & '.join(cells)+r'\\')
    lines += [r'\bottomrule\end{tabular}\label{tab:bo_budgets}\end{table*}']
    (REPO/'overleaf/output/bo_study/convergence_table.tex').write_text('\n'.join(lines)+'\n')


def main() -> None:
    OUT.mkdir(exist_ok=True)
    bundles = {mode: [prepare(mode, n) for n in [100, 1000]] for mode in ['perfect', 'uncertain']}
    values = [r['symmetric_kl'] for bs in bundles.values() for _, m in bs for r in m['records']]
    limits = (10**np.floor(np.log10(min(values))), 10**np.ceil(np.log10(max(values))))
    for mode, bs in bundles.items():
        paired_corners(bs, mode, limits)
    ensemble()
    write(OUT/'manifest.json', dict(rule='symmetric reference KL < 0.01 at two consecutive saved checkpoints',
          reference='Independent exact-likelihood numerical posterior; same target for curves and corners',
          curve='Individual trajectory through all saved checkpoints; no ensemble band',
          selected={m['name']:m['selected']['training'] for bs in bundles.values() for _,m in bs},
          sources={k:v for bs in bundles.values() for _,m in bs for k,v in m['source_sha256'].items()},
          script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))


if __name__ == '__main__':
    main()
