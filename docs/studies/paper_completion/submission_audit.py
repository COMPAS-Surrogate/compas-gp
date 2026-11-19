"""Bounded, read-only reanalysis of saved paper checkpoints and exact labels.

Run from the repository root. No acquisition, fitted parameters, or original
results are changed. The two holdout designs are numerical replicates, not
independent catalogues or GP realizations.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import simpson
from scipy.special import log_ndtr, logsumexp, ndtri
from scipy.spatial import cKDTree
from scipy.stats import gamma, spearmanr

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = HERE / 'submission_audit_20260916'


def expected_log_likelihood(mean: np.ndarray, variance: np.ndarray,
                            scale: float, reference: float) -> np.ndarray:
    """Log E[L] for ln L = reference - max(scale F, 0)^2, F Gaussian.

    This defines an expected-likelihood target, not the expectation of
    separately normalized random posteriors. It uses latent predictive variance.
    """
    m, v = np.asarray(mean)*scale, np.asarray(variance)*scale**2
    if np.any(v < 0):
        raise ValueError('Negative predictive variance')
    safe = np.maximum(v, 1e-300)
    denominator = 1 + 2*safe
    result = np.logaddexp(log_ndtr(-m/np.sqrt(safe)),
        -.5*np.log(denominator)-m*m/denominator
        + log_ndtr(m/np.sqrt(safe*denominator)))
    return reference + np.where(v == 0, -np.maximum(m, 0)**2, result)


def residual_metrics(exact: np.ndarray, predicted: np.ndarray,
                     logq: np.ndarray) -> dict:
    """Separate raw label error from the constant-offset-invariant error."""
    logw = exact-logq
    w = np.exp(logw-logsumexp(logw))
    residual = predicted-exact
    centered = residual-w@residual
    lp = logw-logsumexp(logw)
    lq = predicted-logq-logsumexp(predicted-logq)
    return dict(raw_rmse=float(np.sqrt(w@residual**2)),
        centered_rmse=float(np.sqrt(w@centered**2)),
        offset=float(w@residual), ess=float(1/(w@w)), max_weight=float(w.max()),
        symmetric_kl=float(.5*(w@(lp-lq)+np.exp(lq)@(lq-lp))))


def amplitude_check(number: int, expected: float, injection: float = .01) -> dict:
    """Compare the implemented explicit likelihood to independent Gamma moments."""
    from cosmic_integration.observation.amplitude_validation import joint_amplitude_log_likelihood
    a = np.linspace(.005, .015, 32769)
    beta = expected/injection
    # One event integral per detection; constants cancel from this conditional.
    labels = joint_amplitude_log_likelihood(np.ones((len(a), number)),
        np.full(len(a), beta*.012), a, 1.)
    density = np.exp(labels-labels.max())
    density /= simpson(density, x=a)
    mean = simpson(a*density, x=a)
    variance = simpson((a-mean)**2*density, x=a)
    k = number+1
    probability = gamma.cdf(.015, k, scale=1/beta)-gamma.cdf(.005, k, scale=1/beta)
    moments = [np.prod(np.arange(k, k+j))/beta**j
        *(gamma.cdf(.015, k+j, scale=1/beta)-gamma.cdf(.005, k+j, scale=1/beta))
        /probability for j in (1, 2)]
    sd = np.sqrt(moments[1]-moments[0]**2)
    return dict(observed=number, expected=expected, mean=float(mean), sd=float(np.sqrt(variance)),
        mean_error=float(abs(mean-moments[0])), sd_error=float(abs(np.sqrt(variance)-sd)),
        fractional_sd=float(np.sqrt(variance)/mean), unbounded_fractional_sd=float(1/np.sqrt(k)))


def main() -> None:
    sys.path.insert(0, str(HERE.parent/'operational_stopping/gpry_compas'))
    import run_experiment as base
    import score_comparison as scoring
    import jax
    import jax.numpy as jnp
    from cosmic_integration.lnl_surrogate.gpry_rules import latent_prediction, prepare_fantasies

    OUT.mkdir(exist_ok=True)
    root = HERE.parent/'operational_stopping/gpry_compas/count_history_20260912'
    rows, hashes, arrays = [], {}, []
    levels = np.array([.5, .68, .9, .95, .99])
    for count in (100, 1000):
        references = scoring.cached_references(0, count, hashes)
        for mode in ('perfect', 'uncertain'):
            name = f't0_n{count}_{mode}_c200_bo'
            folder = root/'training'/name
            outcome = json.loads((root/'offline'/name/'outcome.json').read_text())
            labels = outcome['stop']['selected']['training']
            checkpoint = folder/f'checkpoint_{labels}.npz'
            hashes[str(checkpoint.relative_to(REPO))] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            saved = json.loads((folder/'checkpoints.json').read_text())
            assert next(c['sha256'] for c in saved if c['points'] == labels) == hashes[str(checkpoint.relative_to(REPO))]
            cat = dict(np.load(folder/'catalogue.npz'))
            hashes[str((folder/'catalogue.npz').relative_to(REPO))] = hashlib.sha256((folder/'catalogue.npz').read_bytes()).hexdigest()
            state, training, _ = base.load_checkpoint(checkpoint)
            # Remove inactive padding, preserving every training label and parameter.
            state = {k: v[:labels, :labels] if k == 'inverse' else
                     v[:labels] if k in ('x', 'mask', 'alpha') else v for k,v in state.items()}
            fantasies = prepare_fantasies(state, jnp.zeros((1, 3)), jnp.zeros(1))
            predict = jax.jit(jax.vmap(latent_prediction, in_axes=(None, 0, None)))
            for scramble, (phi, rates, logq) in enumerate(references):
                indices = np.random.default_rng(20260916+scramble).choice(len(phi), 8192, replace=False)
                unit = (phi[indices]-base.LOW)/base.WIDTH
                assert cKDTree(training).query(unit)[0].min() > 1e-10
                exact = base.source_target.event_target(rates[indices], cat['coeff'], float(cat['duration']))
                pairs = [predict(state, jnp.asarray(part), fantasies) for part in np.split(unit, 32)]
                mean = np.concatenate([np.asarray(p[0]) for p in pairs])
                variance = np.concatenate([np.asarray(p[1]) for p in pairs])
                reference, scale = float(state['reference']), float(state['scale'])
                plug = reference-(scale*np.maximum(mean, 0))**2
                np.testing.assert_allclose(plug, base.predict(state, unit), atol=1e-9, rtol=0)
                q = logq[indices]
                w = np.exp(exact-q-logsumexp(exact-q))
                error = plug-exact
                coverage = []
                for level in levels:
                    z = ndtri((1+level)/2)
                    lower = reference-(scale*np.maximum(mean+z*np.sqrt(variance), 0))**2
                    upper = reference-(scale*np.maximum(mean-z*np.sqrt(variance), 0))**2
                    coverage.append(float(w@((exact >= lower) & (exact <= upper))))
                high = exact >= exact.max()-5
                marginal = expected_log_likelihood(mean, variance, scale, reference)
                raw_sd_proxy = 2*scale**2*np.maximum(mean, 0)*np.sqrt(variance)
                row = dict(name=name, expected=count, mode=mode, observed=len(cat['coeff']), labels=labels,
                    scramble=scramble, points=len(indices), **residual_metrics(exact, plug, q),
                    high_likelihood_points=int(high.sum()),
                    high_likelihood_rmse=float(np.sqrt(np.mean(error[high]**2))),
                    coverage=coverage, mass_above_gp_ceiling=float(w@(exact > reference)),
                    uncertainty_error_spearman=float(spearmanr(raw_sd_proxy[high], abs(error[high])).statistic),
                    expected_likelihood=residual_metrics(exact, marginal, q),
                    mean_to_expected_likelihood=residual_metrics(plug, marginal, q))
                rows.append(row)
                arrays.append(dict(name=name, scramble=scramble, indices=indices, exact=exact,
                    predicted=plug, mean=mean, variance=variance, logq=q, weights=w))
                print(name, scramble, row['centered_rmse'], row['coverage'], flush=True)
    for a in arrays:
        np.savez_compressed(OUT/f"{a['name']}_{a['scramble']}.npz", **a)
    amplitudes = [amplitude_check(n, n) for n in (10, 30, 100, 1000)]
    amplitudes += [amplitude_check(108, 100), amplitude_check(986, 1000)]
    hashes[str(Path(__file__).relative_to(REPO))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT/'summary.json').write_text(json.dumps(dict(rows=rows, amplitude=amplitudes,
        levels=levels.tolist(), source_sha256=hashes,
        scope='Four existing baseline checkpoints; reused withheld designs, no fresh catalogue ensemble. '
              '8192 points per design; self-normalized importance estimates. Predictive variance '
              'includes the existing numerical jitter, not observation noise. No calibration tuning.'), indent=2)+'\n')
    plot(rows, arrays, levels)


def plot(rows: list[dict], arrays: list[dict], levels: np.ndarray) -> None:
    """Posterior-weighted absolute errors and predictive-interval coverage."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), layout='constrained')
    colors = ['#0072B2', '#D55E00', '#009E73', '#9467B0']
    for color, name in zip(colors, dict.fromkeys(r['name'] for r in rows)):
        selected = [r for r in rows if r['name'] == name]
        item = selected[0]
        style = '-' if item['mode'] == 'perfect' else '--'
        label = f"{item['expected']}, {item['mode']}"
        for a in [a for a in arrays if a['name'] == name]:
            order = np.argsort(abs(a['predicted']-a['exact']))
            axes[0].plot(abs(a['predicted']-a['exact'])[order], np.cumsum(a['weights'][order]),
                color=color, ls=style, lw=1, alpha=.7)
        cover = np.array([r['coverage'] for r in selected])
        axes[1].plot(levels, cover.mean(0), color=color, ls=style, marker='o', ms=3, label=label)
        axes[1].fill_between(levels, cover.min(0), cover.max(0), color=color, alpha=.15)
    axes[0].set(xscale='log', xlim=(1e-4, 10), ylim=(0, 1),
        xlabel=r'$\epsilon$ [nats]', ylabel=r'$F_{|\delta\ell|}(\epsilon)$')
    axes[1].plot([0, 1], [0, 1], ':', color='.4', lw=1)
    axes[1].set(xlim=(.45, 1), ylim=(0, 1), xlabel=r'$p_{\rm nom}$',
        ylabel=r'$C(p_{\rm nom})$')
    axes[1].legend(frameon=False, fontsize=7, loc='upper left')
    for ax in axes:
        ax.minorticks_on()
        ax.tick_params(which='both', direction='in', top=True, right=True)
    for ext in ('pdf', 'png'):
        fig.savefig(OUT/f'gp_holdout.{ext}', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
