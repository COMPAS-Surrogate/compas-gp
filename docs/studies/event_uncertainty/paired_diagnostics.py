"""Data illustrations and posterior predictive checks for the paired campaign."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import chi2, gamma
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE, ORANGE = '#226A91', '#D55E00'


def style(ax) -> None:
    ax.minorticks_on()
    ax.tick_params(which='both', direction='in', top=True, right=True)
    for spine in ax.spines.values():
        spine.set_visible(True)


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(path.with_suffix('.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_data(folder: Path, output: Path, expected: int = 30) -> None:
    """Show identical events and illustrative likelihood contours, not PE samples."""
    path = folder / f'catalogue_{expected}_uncertain_0.npz'
    with np.load(path) as cat:
        latent, measured = cat['latent'], np.exp(cat['y'])
        y, scales, rho = cat['y'], cat['scales'], cat['rho']
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharex=True, sharey=True)
    axes[0].scatter(*latent.T, s=18, color=BLUE, label='Exact event parameters')
    axes[1].scatter(*latent.T, s=24, marker='+', color='black', label='Latent events', zorder=4)
    axes[1].scatter(*measured.T, s=16, color=ORANGE, label='Measured centres', zorder=3)
    chosen = np.argsort(rho)[np.linspace(0, len(rho)-1, min(5, len(rho))).astype(int)]
    angle = np.linspace(0, 2*np.pi, 160)
    circle = np.column_stack([np.cos(angle), np.sin(angle)]) * np.sqrt(chi2.ppf(.9, 2))
    for k, i in enumerate(chosen):
        contour = np.exp(y[i] + circle * scales[i])
        axes[1].plot(*contour.T, color=ORANGE, lw=.8, alpha=.7,
                     label='Selected likelihood contours' if k == 0 else None)
        axes[1].plot([latent[i,0], measured[i,0]], [latent[i,1], measured[i,1]], color='.65', lw=.65)
    for ax, mode in zip(axes, ('perfect', 'uncertain')):
        ax.set_xlabel(r'Source-frame chirp mass $\mathcal{M}_c\ [M_\odot]$')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(frameon=False, fontsize=7, title=rf'$\mathbb{{E}}[N_{{\rm obs}}]={expected}$, {mode}', title_fontsize=8)
        style(ax)
    axes[0].set_ylabel('Redshift $z$')
    fig.tight_layout()
    save(fig, output / 'paired_data')
    metadata = dict(expected=expected, events=len(latent), selected_event_indices=chosen.tolist(),
                    source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    contour='Delta chi-square=chi2.ppf(0.9,2) in log coordinates; likelihood contours, not posterior credible regions')
    (output / 'paired_data.json').write_text(json.dumps(metadata, indent=2)+'\n')


def draw_amplitudes(beta: np.ndarray, events: int, rng: np.random.Generator) -> np.ndarray:
    """Exact conditional Gamma draws truncated to the scientific amplitude prior."""
    shape = events + 1
    lo, hi = gamma.cdf(.005, shape, scale=1/beta), gamma.cdf(.015, shape, scale=1/beta)
    sl, sh = gamma.sf(.005, shape, scale=1/beta), gamma.sf(.015, shape, scale=1/beta)
    u = rng.random(len(beta))
    a = np.where(lo < .5, gamma.ppf(lo+u*(hi-lo), shape, scale=1/beta),
                 gamma.isf(sl-u*(sl-sh), shape, scale=1/beta))
    if not np.isfinite(a).all() or np.any((a < .005) | (a > .015)):
        raise ValueError('Unresolved conditional amplitude draws')
    return a


def replicate(rates: np.ndarray, logweights: np.ndarray, events: int, duration: float,
              lower: np.ndarray, upper: np.ndarray, mode: str,
              draws: int = 1000, seed: int = 202609125) -> tuple[list[np.ndarray], np.ndarray]:
    """Integrate posterior uncertainty and Poisson/measurement variation."""
    if mode not in ('perfect', 'uncertain'):
        raise ValueError('Unknown measurement model')
    rng = np.random.default_rng(seed)
    probabilities = np.exp(logweights-logsumexp(logweights))
    indices = rng.choice(len(rates), draws, p=probabilities)
    total = rates.sum(axis=1)
    a = draw_amplitudes(duration * total[indices] / .012, events, rng)
    numbers = rng.poisson(duration * total[indices] * a / .012)
    catalogues = []
    for index, number in zip(indices, numbers):
        bins = rng.choice(len(lower), number, p=rates[index]/total[index])
        latent = rng.uniform(lower[bins], upper[bins])
        rho = 12*rng.random(number)**(-1/3)
        noise = np.array([.03, .3])[None, :] * (12/rho[:, None]) * rng.normal(size=(number, 2))
        catalogues.append(latent if mode == 'perfect' else np.exp(np.log(latent)+noise))
    return catalogues, numbers


def predictive(folder: Path, expected: int, mode: str, output: Path,
               checkpoint_points: int | None = None) -> dict:
    """Reconstruct a saved BO checkpoint, optionally at a comparison-selected count."""
    import posterior_convergence as study
    outcomes = json.loads((folder / 'outcomes.json').read_text())
    outcome = next(r for r in outcomes if (r['exposure'], r['mode'], r['case'], r['strategy']) == (expected, mode, 0, 'bo'))
    if outcome['state'] != 'reference_verified_convergence':
        raise ValueError('Predictive output requires a validated BO fit')
    if checkpoint_points is not None:
        outcome = {**outcome, 'points': checkpoint_points}
    checkpoint = folder / f'n{expected}_{mode}_case0_bo/checkpoint_{outcome["points"]}.npz'
    catpath = folder / f'catalogue_{expected}_{mode}_0.npz'
    references = json.loads((folder / 'references.json').read_text())
    ref = next(r for r in references if (r['exposure'], r['mode'], r['case']) == (expected, mode, 0) and r['numerical_pass'])
    power = ref['power']
    ratepath = folder / f'reference_n{expected}_p{power}_0.npz'
    proposalpath = folder / f'proposal_n{expected}_p{power}_0.npz'
    with np.load(ratepath) as r, np.load(proposalpath) as q, np.load(checkpoint) as training:
        phi, rates, logq = r['phi'], r['rates'], q['logq']
        x, y = training['x'], training['lnl']
    with np.load(catpath) as cat, np.load(folder / 'grid.npz') as grid:
        events, duration = len(cat['latent']), float(cat['duration'])
        observed = cat['latent'] if mode == 'perfect' else np.exp(cat['y'])
        coeff, lower, upper = cat['coeff'], grid['lower'], grid['upper']
    core = study.core
    scaler = core.AdaptiveRobustScaler(compression='sqrt', soft_clipping=False,
                                       lower_clip_value=float(y.min()), max_scale=1e9)
    scaler.initialize_with_data(y)
    model = core.fit_exact_gp(core.JaxTrainingData(x, -scaler.transform(y)[:,None]),
        np.array([core.LOW[core.SHAPE], (core.LOW+core.WIDTH)[core.SHAPE]]),
        config=core.JaxGPConfig(optimisation_steps=250), seed=17)
    predicted = scaler.inverse_transform(-study.reference_means(model, phi))
    exact = study.event_target(rates, coeff, duration)
    metrics = core.posterior_metrics(exact-logq, predicted-logq, phi)
    kls = [metrics['kl_direct_to_gp'], metrics['kl_gp_to_direct']]
    comparison = np.mean(kls) if checkpoint_points is not None else max(kls)
    if comparison >= .01:
        raise ValueError('Reconstructed predictive GP fails the posterior KL target')
    replicated, numbers = replicate(rates, predicted-logq, events, duration, lower, upper, mode)
    columns = []
    nonempty = [r for r in replicated if len(r)]
    for j in (0, 1):
        endpoint = max(observed[:,j].max(initial=0), max(r[:,j].max() for r in nonempty))
        axis = np.linspace(0, endpoint*1.01, 350)
        cdfs = [np.searchsorted(np.sort(r[:,j]), axis, side='right')/len(r) for r in nonempty]
        obs = np.searchsorted(np.sort(observed[:,j]), axis, side='right')/events
        columns.append(dict(x=axis.tolist(), observed=obs.tolist(), quantiles=np.quantile(cdfs, [.05,.5,.95], axis=0).tolist()))
    result = dict(expected=expected, mode=mode, events=events, training_points=outcome['points'],
                  predictive_draws=len(numbers), empty_replicates=int(np.sum(numbers==0)),
                  shape_condition='CDFs conditional on nonempty replicated catalogue',
                  replicate_counts=numbers.tolist(), columns=columns,
                  posterior_kl={k: float(v) for k,v in metrics.items() if k.startswith('kl_')},
                  source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (checkpoint,catpath,ratepath,proposalpath)})
    output.mkdir(parents=True, exist_ok=True)
    (output / f'predictive_{expected}_{mode}.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def plot_predictive(folder: Path, output: Path,
                    checkpoint_points: dict[tuple[int, str], int] | None = None) -> None:
    """Draw replicated catalogues at the supplied comparison checkpoints."""
    for expected in (30, 1200):
        fig, axes = plt.subplots(2, 3, figsize=(9, 5), sharex='col')
        for row, mode in enumerate(('perfect', 'uncertain')):
            count = None if checkpoint_points is None else checkpoint_points[expected, mode]
            p = predictive(folder, expected, mode, output, checkpoint_points=count)
            for j in (0, 1):
                ax = axes[row,j]
                column = p['columns'][j]
                x, q = np.array(column['x']), np.array(column['quantiles'])
                ax.fill_between(x, q[0], q[2], color=BLUE, alpha=.16, label='90% predictive band')
                ax.plot(x, q[1], color=BLUE, lw=1.3, label='Predictive median')
                ax.step(x, column['observed'], where='post', color='black', lw=1, label='Observed catalogue')
                ax.set_ylim(0,1)
                if j == 0:
                    ax.set_ylabel('Cumulative fraction')
                    ax.legend(frameon=False, fontsize=7, title=f'{mode}', title_fontsize=8)
            axes[row,2].hist(p['replicate_counts'], bins=20, color=BLUE, alpha=.6, density=True)
            axes[row,2].axvline(p['events'], color='black', lw=1.2)
            axes[row,2].set_ylabel('Predictive density')
        for ax in axes.ravel():
            style(ax)
        for ax, label in zip(axes[-1], (r'Chirp mass $\mathcal{M}_c\ [M_\odot]$', 'Redshift $z$', r'Detected count $N_{\rm obs}$')):
            ax.set_xlabel(label)
        fig.tight_layout()
        save(fig, output / f'posterior_predictive_{expected}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-only', action='store_true')
    args = parser.parse_args()
    plot_data(args.folder, args.output)
    if not args.data_only:
        plot_predictive(args.folder, args.output)
