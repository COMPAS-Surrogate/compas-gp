"""Paired event information and inferred detected intensity on the saved grid.

Read-only post-processing of the frozen campaign. Neither the likelihood nor
acquisition is changed. Display weights normalize the bin-averaged likelihood
coefficients, not posterior membership probabilities under an inferred population.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import logsumexp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter, ScalarFormatter

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CAMPAIGN = HERE.parent / 'event_uncertainty/paired_20260912'
sys.path.insert(0, str(HERE.parent / 'event_uncertainty'))
import paired_diagnostics as diagnostics

MODES = ('perfect', 'uncertain')
DISPLAY_MIN = .001


def event_weights(coeff: np.ndarray) -> np.ndarray:
    """Normalize each event over active bins; invariant to event-only factors."""
    coeff = np.asarray(coeff, dtype=float)
    if (coeff.ndim != 2 or not np.isfinite(coeff).all() or np.any(coeff < 0)
            or np.any(coeff.sum(axis=1) <= 0)):
        raise ValueError('Each event requires finite nonnegative, nonzero weights')
    return coeff / coeff.sum(axis=1, keepdims=True)


def intensity_draws(rates: np.ndarray, logweights: np.ndarray, duration: float,
                    events: int, draws: int = 20000, seed: int = 202609126) -> np.ndarray:
    """Expected counts T*r per bin with posterior shape AND amplitude uncertainty.

Input rates are at reference amplitude .012. No Poisson or measurement noise
is added: those belong to the separate replicated-catalogue predictive checks.
"""
    rates, logweights = np.asarray(rates), np.asarray(logweights)
    if (rates.ndim != 2 or logweights.shape != (len(rates),) or duration <= 0
            or not np.isfinite(rates).all() or np.any(rates < 0)
            or np.any(rates.sum(axis=1) <= 0) or not np.isfinite(logweights).any()
            or np.isnan(logweights).any() or np.isposinf(logweights).any()):
        raise ValueError('Finite rates and resolved posterior weights required')
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(rates), draws, p=np.exp(logweights-logsumexp(logweights)))
    amplitudes = diagnostics.draw_amplitudes(
        duration*rates[indices].sum(axis=1)/.012, events, rng)
    return duration*rates[indices]*amplitudes[:, None]/.012


def highest_density_bins(rates: np.ndarray, areas: np.ndarray,
                         probability: float = .9) -> np.ndarray:
    """Smallest greedy bin union with requested mass, ordered by linear density."""
    rates, areas = np.asarray(rates), np.asarray(areas)
    if (rates.shape != areas.shape or np.any(rates < 0) or np.any(areas <= 0)
            or rates.sum() <= 0 or not 0 < probability <= 1):
        raise ValueError('Positive areas, total rate and valid probability required')
    order = np.argsort(-rates/areas, kind='stable')
    end = min(np.searchsorted(np.cumsum(rates[order]), probability*rates.sum())+1, len(rates))
    mask = np.zeros(len(rates), dtype=bool)
    mask[order[:end]] = True
    return mask


def boundary_segments(lower: np.ndarray, upper: np.ndarray, selected: np.ndarray) -> list:
    """Boundary of a selected bin union, cancelling shared edges on the native grid."""
    xs = np.unique(np.r_[lower[:, 0], upper[:, 0]])
    zs = np.unique(np.r_[lower[:, 1], upper[:, 1]])
    inside = np.zeros((len(xs)-1, len(zs)-1), dtype=bool)
    for lo, hi in zip(lower[selected], upper[selected]):
        i, k = np.searchsorted(xs, [lo[0], hi[0]])
        j, l = np.searchsorted(zs, [lo[1], hi[1]])
        inside[i:k, j:l] = True
    edges = []
    for i, j in np.argwhere(inside):
        if i == 0 or not inside[i-1, j]:
            edges.append([(xs[i], zs[j]), (xs[i], zs[j+1])])
        if i == len(xs)-2 or not inside[i+1, j]:
            edges.append([(xs[i+1], zs[j]), (xs[i+1], zs[j+1])])
        if j == 0 or not inside[i, j-1]:
            edges.append([(xs[i], zs[j]), (xs[i+1], zs[j])])
        if j == len(zs)-2 or not inside[i, j+1]:
            edges.append([(xs[i], zs[j+1]), (xs[i+1], zs[j+1])])
    return edges


def draw_map(ax, lower: np.ndarray, upper: np.ndarray, values: np.ndarray,
             norm, cmap: str = 'viridis') -> PolyCollection:
    vertices = [[lo, (hi[0], lo[1]), hi, (lo[0], hi[1])] for lo, hi in zip(lower, upper)]
    colors = plt.get_cmap(cmap).copy()
    colors.set_bad('white')
    colors.set_under('white')
    image = PolyCollection(vertices, array=np.ma.masked_invalid(values), cmap=colors,
                           norm=norm, edgecolors='none')
    ax.add_collection(image)
    ax.set(xscale='log', xlim=(lower[:, 0].min(), upper[:, 0].max()),
           ylim=(0, upper[:, 1].max()))
    ax.set_xticks([2, 5, 10, 20, 40, 80])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks(np.arange(0, 1.21, .2))
    diagnostics.style(ax)
    return image


def fingerprint(paths: list[Path]) -> dict[str, str]:
    return {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def reconstruct(folder: Path, mode: str, output: Path, expected: int = 30,
                checkpoint_points: int | None = None) -> dict:
    """Push a saved pilot GP through its rate map, optionally at a selected count."""
    import posterior_convergence as study
    outcome = next(r for r in json.loads((folder/'outcomes.json').read_text())
                   if (r['exposure'], r['mode'], r['case'], r['strategy']) == (expected, mode, 0, 'bo'))
    if outcome['state'] != 'reference_verified_convergence':
        raise ValueError('Rate reconstruction requires a validated BO fit')
    if checkpoint_points is not None:
        outcome = {**outcome, 'points': checkpoint_points}
    ref = next(r for r in json.loads((folder/'references.json').read_text())
               if (r['exposure'], r['mode'], r['case']) == (expected, mode, 0) and r['numerical_pass'])
    checkpoint = folder/f'n{expected}_{mode}_case0_bo/checkpoint_{outcome["points"]}.npz'
    ratepath = folder/f'reference_n{expected}_p{ref["power"]}_0.npz'
    proposal = folder/f'proposal_n{expected}_p{ref["power"]}_0.npz'
    catalogue = folder/f'catalogue_{expected}_{mode}_0.npz'
    with np.load(checkpoint) as t, np.load(ratepath) as r, np.load(proposal) as q, np.load(catalogue) as c:
        x, y, phi, rates, logq = t['x'], t['lnl'], r['phi'], r['rates'], q['logq']
        duration, events, coeff = float(c['duration']), len(c['latent']), c['coeff']
    core = study.core
    scaler = core.AdaptiveRobustScaler(compression='sqrt', soft_clipping=False,
                                       lower_clip_value=float(y.min()), max_scale=1e9)
    scaler.initialize_with_data(y)
    model = core.fit_exact_gp(core.JaxTrainingData(x, -scaler.transform(y)[:, None]),
        np.array([core.LOW[core.SHAPE], (core.LOW+core.WIDTH)[core.SHAPE]]),
        config=core.JaxGPConfig(optimisation_steps=250), seed=17)
    predicted = scaler.inverse_transform(-study.reference_means(model, phi))
    exact = study.event_target(rates, coeff, duration)
    metrics = core.posterior_metrics(exact-logq, predicted-logq, phi)
    kl = {k: float(v) for k, v in metrics.items() if k.startswith('kl_')}
    comparison = np.mean(list(kl.values())) if checkpoint_points is not None else max(kl.values())
    if comparison >= .01:
        raise ValueError('Reconstructed GP fails the posterior KL target')
    draws = intensity_draws(rates, predicted-logq, duration, events)
    quantiles = np.quantile(draws, [.05, .5, .95], axis=0)
    # Independent halves check the Monte Carlo resolution of displayed summaries.
    halves = [np.quantile(d, [.05, .5, .95], axis=0) for d in np.array_split(draws, 2)]
    visible = quantiles[1] >= DISPLAY_MIN
    relative = np.abs(halves[0]-halves[1])[:, visible]/quantiles[1, visible]
    resolution = float(relative.max())
    if resolution > .1:
        raise ValueError(f'Rate quantiles unresolved between split draws: {resolution}')
    result = dict(expected=expected, mode=mode, events=events, duration=duration,
        training_points=outcome['points'], draws=len(draws), posterior_kl=kl,
        max_split_quantile_difference_over_median=resolution,
        total_expected_count_quantiles=np.quantile(draws.sum(axis=1), [.05,.5,.95]).tolist(),
        sources=fingerprint([checkpoint, ratepath, proposal, catalogue, folder/'grid.npz']),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        definition='Binwise posterior quantiles of T*r, including amplitude; no Poisson or measurement noise')
    np.savez_compressed(output/f'rate_posterior_{mode}.npz', quantiles=quantiles,
                        mean=draws.mean(axis=0), totals=draws.sum(axis=1))
    (output/f'rate_posterior_{mode}.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def plot_figures(folder: Path, output: Path) -> None:
    with np.load(folder/'grid.npz') as grid:
        lower, upper, truth = grid['lower'], grid['upper'], grid['truth_rates']
    weights, latent = [], []
    for mode in MODES:
        with np.load(folder/f'catalogue_30_{mode}_0.npz') as cat:
            weights.append(event_weights(cat['coeff']).sum(axis=0))
            latent.append(cat['latent'])
            duration = float(cat['duration'])
            rho = cat['rho']
    np.testing.assert_array_equal(*latent)
    np.testing.assert_allclose([w.sum() for w in weights], len(latent[0]))
    np.testing.assert_allclose(duration*truth.sum(), 30)
    rate_q = [np.load(output/f'rate_posterior_{m}.npz')['quantiles'] for m in MODES]
    vmax = 10**np.ceil(np.log10(max(np.max(weights), max(q[1].max() for q in rate_q))))
    norm = LogNorm(DISPLAY_MIN, vmax)
    areas = np.prod(upper-lower, axis=1)
    region = highest_density_bins(truth, areas)
    boundary = boundary_segments(lower, upper, region)
    chosen = np.argsort(rho)[np.linspace(0, len(rho)-1, 3).astype(int)]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharex=True, sharey=True, layout='constrained')
    for ax, mode, weight in zip(axes, MODES, weights):
        im = draw_map(ax, lower, upper, weight, norm)
        ax.scatter(*latent[0][chosen].T, marker='+', color='black', s=30, lw=.9,
                   label='Selected latent events')
        ax.legend(title=rf'$\mathbb{{E}}[N_{{\rm obs}}]=30$, {mode}',
                  title_fontsize=9, fontsize=8, frameon=False, loc='upper left')
        ax.set_xlabel(r'Source-frame chirp mass $\mathcal{M}_c\ [M_\odot]$')
    axes[0].set_ylabel('Redshift $z$')
    fig.colorbar(im, ax=axes, fraction=.035, pad=.02).set_label('Summed event weight per bin')
    diagnostics.save(fig, output/'paired_weights')

    widths = [np.divide(50*(q[2]-q[0]), q[1], out=np.full_like(q[1], np.nan),
                        where=q[1]>=DISPLAY_MIN) for q in rate_q]
    uncertainty_max = 10*np.ceil(max(np.nanmax(w) for w in widths)/10)
    fig, axes = plt.subplots(2, 2, figsize=(8, 5.5), sharex=True, sharey=True, layout='constrained')
    for row, (mode, q, width) in enumerate(zip(MODES, rate_q, widths)):
        im = draw_map(axes[row, 0], lower, upper, q[1], norm)
        uncertainty = draw_map(axes[row, 1], lower, upper, width, Normalize(0, uncertainty_max), 'cividis')
        for ax in axes[row]:
            ax.add_collection(LineCollection(boundary, colors='black', linewidths=1.1))
        axes[row, 0].legend(handles=[Line2D([], [], color='black', lw=1.1, label='Injected 90% region')],
            title=rf'$\mathbb{{E}}[N_{{\rm obs}}]=30$, {mode}', title_fontsize=9,
            fontsize=8, frameon=False, loc='upper left')
        axes[row, 0].set_ylabel('Redshift $z$')
    for ax in axes[-1]:
        ax.set_xlabel(r'Source-frame chirp mass $\mathcal{M}_c\ [M_\odot]$')
    fig.colorbar(im, ax=list(axes[:, 0]), fraction=.045, pad=.02).set_label('Median expected detections per bin')
    fig.colorbar(uncertainty, ax=list(axes[:, 1]), fraction=.045, pad=.02).set_label(
        'Relative 90% interval half-width [%]')
    diagnostics.save(fig, output/'paired_rate_posterior')
    report = dict(expected=30, events=len(latent[0]), weight_sums=[float(w.sum()) for w in weights],
        color_limits=[DISPLAY_MIN, float(vmax)], truth_region_mass=float(truth[region].sum()/truth.sum()),
        subthreshold_weight_sums=[float(w[w<DISPLAY_MIN].sum()) for w in weights],
        selected_event_indices=chosen.tolist(), uncertainty_color_limits=[0, float(uncertainty_max)],
        weight_definition='w_ib=C_ib/sum_c C_ic; sum_i w_ib; not population posterior bin membership',
        sources=fingerprint([folder/'grid.npz', *[folder/f'catalogue_30_{m}_0.npz' for m in MODES]]),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/'weight_rate_figures.json').write_text(json.dumps(report, indent=2)+'\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE/'weight_rate_review')
    parser.add_argument('--plots-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.plots_only:
        for mode in MODES:
            print(json.dumps(reconstruct(CAMPAIGN/'pilot', mode, args.output)), flush=True)
    plot_figures(CAMPAIGN/'pilot', args.output)


if __name__ == '__main__':
    main()
