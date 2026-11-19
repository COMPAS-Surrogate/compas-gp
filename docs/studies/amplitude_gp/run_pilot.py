"""Bounded explicit-amplitude GP validation; run training before reference scoring."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import logsumexp
from scipy.stats import gamma, qmc

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent/'operational_stopping/gpry_compas'))
import run_experiment as base
import score_comparison as reference

from cosmic_integration.observation.amplitude_validation import (
    amplitude_proposal, joint_amplitude_log_likelihood,
)
from cosmic_integration.lnl_surrogate.gpry_rules import (
    state_from_gpjax, acquisition_values, prepare_fantasies,
)
from cosmic_integration.lnl_surrogate.stopping import posterior_summary, summary_change

LOW, WIDTH = base.core.LOW, base.core.WIDTH
SEEDS = (17, 29, 43)
SOURCE = HERE.parent/'operational_stopping/gpry_compas/count_history_20260912/training'


def fit(unit: np.ndarray, values: np.ndarray) -> dict:
    """Use existing repaired GP defaults; arbitrary input dimension in unit box."""
    scaler = base.core.AdaptiveRobustScaler(compression='sqrt', soft_clipping=False,
        lower_clip_value=float(values.min()), max_scale=1e9)
    scaler.initialize_with_data(values)
    dim = unit.shape[1]
    gp = base.core.fit_exact_gp(base.core.JaxTrainingData(unit, -scaler.transform(values)),
        np.array([np.zeros(dim), np.ones(dim)]),
        config=base.core.JaxGPConfig(optimisation_steps=250), seed=17)
    state = state_from_gpjax(gp, scaler, np.zeros(dim), np.ones(dim), len(unit))
    jax.block_until_ready(state)
    return state


def compare(a: dict, b: dict) -> dict:
    """All-parameter comparisons with the study's five quantiles."""
    result = summary_change(a, b)
    qa, qb = np.asarray(a['quantiles']), np.asarray(b['quantiles'])
    result['width95_fraction'] = float(np.max(abs((qb[4]-qb[0])/(qa[4]-qa[0])-1)))
    return result


def numerical(a: dict, b: dict) -> dict:
    change = compare(a, b)
    return dict(passed=bool(change['mean_sd'] < .05 and change['endpoint_sd'] < .05
                           and change['width_fraction'] < .03
                           and min(a['ess'], b['ess']) >= 100), **change)


def score(points: np.ndarray, exact: np.ndarray, predicted: np.ndarray,
          logq: np.ndarray) -> dict:
    p, q = exact-logq, predicted-logq
    p, q = p-logsumexp(p), q-logsumexp(q)
    weights = np.exp(p)
    direct, gp = posterior_summary(points, p), posterior_summary(points, q)
    change = compare(direct, gp)
    kls = [float(weights@(p-q)), float(np.exp(q)@(q-p))]
    residual = predicted-exact
    centered = residual-weights@residual
    def correlation(w: np.ndarray) -> np.ndarray:
        dx = points-w@points
        cov = dx.T@(w[:, None]*dx)
        sd = np.sqrt(np.diag(cov))
        return cov/(sd[:, None]*sd[None, :])
    passed = (max(kls) < .01 and change['mean_sd'] < .2 and change['endpoint_sd'] < .2
              and change['width_fraction'] < .1 and change['width95_fraction'] < .1)
    return dict(direct=direct, gp=gp, change=change, directed_kl=kls,
        centered_lnl_rmse=float(np.sqrt(weights@centered**2)),
        max_correlation_difference=float(np.max(abs(correlation(weights)-correlation(np.exp(q))))),
        accuracy_pass=bool(passed))


def conditional(out: Path) -> list[dict]:
    """Directly known count-amplitude scaling with the actual finite prior."""
    rows = []
    rng = np.random.default_rng(202609140)
    a = np.linspace(.005, .015, 32769)
    unit = ((a-.005)/.01)[:, None]
    for expected in (10, 100, 1000):
        number = int(rng.poisson(expected))
        beta = expected/.01
        exact = number*np.log(a)-beta*a
        shape = number+1
        analytic_cdf = ((gamma.cdf(a, shape, scale=1/beta)-gamma.cdf(a[0], shape, scale=1/beta))
            /(gamma.cdf(a[-1], shape, scale=1/beta)-gamma.cdf(a[0], shape, scale=1/beta)))
        # Trapezoidal CDF, distinct from the analytic Gamma implementation.
        density = np.exp(exact-exact.max())
        cdf = np.r_[0, np.cumsum((density[:-1]+density[1:])/2)]
        cdf /= cdf[-1]
        for seed in SEEDS:
            x = np.vstack([[[0.], [1.]], qmc.Sobol(1, scramble=True, seed=seed).random_base2(5)[:31]])
            ax = .005+.01*x[:, 0]
            values = number*np.log(ax)-beta*ax
            state = fit(x, values)
            predicted = base.predict(state, unit)
            result = score(a[:, None], exact, predicted, np.zeros(len(a)))
            result.update(expected=expected, observed=number, seed=seed, labels=len(x),
                exact_cdf_error=float(np.max(abs(cdf-analytic_cdf))),
                unbounded_fractional_sd=float(1/np.sqrt(shape)),
                bounded_fractional_sd=result['direct']['sd'][0]/result['direct']['mean'][0])
            rows.append(result)
            np.savez_compressed(out/f'conditional_n{expected}_s{seed}.npz', amplitude=a,
                exact=exact, gp=predicted, training_unit=x, training_lnl=values)
            base.write_json(out/'conditional.json', rows)
            logging.info('Conditional N=%d seed=%d KL=%.4g pass=%s', number, seed,
                         max(result['directed_kl']), result['accuracy_pass'])
    return rows


def acquire(state: dict, rng: np.random.Generator, strategy: str) -> np.ndarray:
    """Frozen new batch policy: 25% IID exploration and 75% GPry-score sampling."""
    if strategy == 'random':
        return rng.random((128, 4))
    pool = qmc.Sobol(4, scramble=True, seed=int(rng.integers(2**31))).random_base2(12)
    fantasies = prepare_fantasies(state, jnp.zeros((1, 4)), jnp.zeros(1))
    scores = np.asarray(acquisition_values(state, jnp.asarray(pool), fantasies, 4**(-.85)))
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite acquisition score')
    # Log-score sampling without replacement avoids underflow in tiny weights.
    order = np.argsort(scores+rng.gumbel(size=len(pool)))[-96:]
    return np.vstack([rng.random((32, 4)), pool[order]])


def train(out: Path, loader) -> list[dict]:
    rows = []
    boundary = np.array(list(itertools.product([0., .5, 1.], repeat=4)))
    boundary = boundary[np.any((boundary == 0) | (boundary == 1), axis=1)]
    for expected, mode in itertools.product((100, 1000), ('perfect', 'uncertain')):
        source = SOURCE/f't0_n{expected}_{mode}_c200_bo/catalogue.npz'
        cat = dict(np.load(source))
        np.savez_compressed(out/f'catalogue_n{expected}_{mode}.npz', **cat)
        def evaluate(unit: np.ndarray) -> np.ndarray:
            physical = LOW+WIDTH*unit
            rates = np.array([loader.fast(p[0], p[1], .012, p[3]).ravel()[loader.active] for p in physical])
            return joint_amplitude_log_likelihood(rates@cat['coeff'].T, rates.sum(axis=1),
                                                 physical[:, 2], float(cat['duration']))
        for seed in SEEDS:
            initial = np.vstack([qmc.Sobol(4, scramble=True, seed=seed).random_base2(6), boundary])
            initial_values = evaluate(initial)
            for strategy in ('bo', 'random'):
                tick = time.perf_counter()
                name = f'n{expected}_{mode}_s{seed}_{strategy}'
                folder = out/name
                folder.mkdir()
                unit, values = initial.copy(), initial_values.copy()
                rng = np.random.default_rng(seed+20260914)
                for checkpoint in (144, 272, 400):
                    if not np.isfinite(values).all():
                        raise ValueError('Nonfinite training label; no floor substituted')
                    state = fit(unit, values)
                    np.savez_compressed(folder/f'checkpoint_{checkpoint}.npz', unit=unit, lnl=values,
                                        **{f'state_{k}': np.asarray(v) for k, v in state.items()})
                    base.write_json(out/'status.json', dict(phase='training', job=name,
                        labels=len(unit), completed=len(rows), expected=24))
                    logging.info('Trained %s labels=%d elapsed=%.1fs', name, len(unit), time.perf_counter()-tick)
                    if checkpoint != 400:
                        new = acquire(state, rng, strategy)
                        unit, values = np.vstack([unit, new]), np.r_[values, evaluate(new)]
                row = dict(name=name, expected=expected, mode=mode, seed=seed, strategy=strategy,
                    labels=len(unit), seconds=time.perf_counter()-tick,
                    catalogue_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    checkpoint_sha256=hashlib.sha256((folder/'checkpoint_400.npz').read_bytes()).hexdigest())
                rows.append(row)
                base.write_json(out/'training.json', rows)
    return rows


def references(out: Path, expected: int, mode: str) -> tuple[list, dict]:
    """Direct reference plus explicit-amplitude quadrature; never used for training."""
    hashes = {}
    refs = reference.cached_references(0, expected, hashes)
    cat = dict(np.load(out/f'catalogue_n{expected}_{mode}.npz'))
    checked = reference.direct_checks(refs, cat, cat['truth'])
    output, parity = [], []
    for scramble, (phi, rates, logq) in enumerate(refs):
        beta = float(cat['duration'])*rates.sum(axis=1)/.012
        a, logqa = amplitude_proposal(beta, len(cat['coeff']),
                                     np.random.default_rng(2026091400+expected+scramble))
        points = np.column_stack([phi[:, :2], a, phi[:, 2]])
        values = np.concatenate([joint_amplitude_log_likelihood(
            r@cat['coeff'].T, r.sum(axis=1), aa, float(cat['duration']))
            for r, aa in zip(np.array_split(rates, int(np.ceil(len(rates)/512))),
                             np.array_split(a, int(np.ceil(len(rates)/512))))])
        weights = values-logq-logqa
        summary = posterior_summary(points, weights)
        analytic = dict(checked[2][scramble])
        analytic['quantiles'] = np.asarray(analytic['quantiles'])[[0, 2, 3, 4, 6]].tolist()
        analytic['ess'] = summary['ess']
        parity.append(numerical(analytic, summary))
        output.append((points, values, logq+logqa, summary))
    precision = numerical(output[0][3], output[1][3])
    meta = dict(analytic_reference=checked[-1], analytic_joint_parity=parity,
                joint_reference_precision=precision, source_sha256=hashes,
                passed=bool(checked[-1]['passed'] and precision['passed'] and all(p['passed'] for p in parity)))
    base.write_json(out/f'reference_n{expected}_{mode}.json', meta)
    return output, meta


def validate(out: Path, training: list[dict]) -> list[dict]:
    rows = []
    for expected, mode in itertools.product((100, 1000), ('perfect', 'uncertain')):
        refs, metadata = references(out, expected, mode)
        for job in [r for r in training if r['expected'] == expected and r['mode'] == mode]:
            path = out/job['name']/'checkpoint_400.npz'
            if hashlib.sha256(path.read_bytes()).hexdigest() != job['checkpoint_sha256']:
                raise ValueError('Training checkpoint changed before scoring')
            with np.load(path) as f:
                state = {k[6:]: jnp.asarray(f[k]) for k in f.files if k.startswith('state_')}
            results = []
            for scramble, (points, values, logq, _) in enumerate(refs):
                predicted = base.predict(state, (points-LOW)/WIDTH)
                results.append(score(points, values, predicted, logq))
                np.savez_compressed(out/job['name']/f'validation_{scramble}.npz',
                    points=points, exact=values, predicted=predicted, logq=logq)
            gp_precision = numerical(results[0]['gp'], results[1]['gp'])
            resolved = metadata['passed'] and gp_precision['passed']
            outcome = ('unresolved' if not resolved else 'accurate' if all(r['accuracy_pass'] for r in results) else 'inaccurate')
            row = dict(**job, outcome=outcome, scores=results, gp_precision=gp_precision,
                       reference_pass=metadata['passed'])
            rows.append(row)
            base.write_json(out/'results.json', rows)
            base.write_json(out/'status.json', dict(phase='scoring', job=job['name'], completed=len(rows), expected=len(training)))
            logging.info('Scored %s: %s max KL=%.5g', job['name'], outcome,
                         max(max(s['directed_kl']) for s in results))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--score-only', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    tick = time.perf_counter()
    if not args.score_only:
        out.mkdir(parents=True, exist_ok=False)
        sources = [Path(__file__), HERE/'PROTOCOL.md', *sorted((REPO/'src/cosmic_integration').rglob('*.py'))]
        base.write_json(out/'manifest.json', dict(source_sha256={str(p.relative_to(REPO)):
            hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, seeds=SEEDS,
            expected_counts=[100, 1000], checkpoints=[144, 272, 400], joint_jobs=24))
        conditional(out)
        loader = base.data_source.Experiment(out/'population', 'explicit_amplitude_no_reference', False)
        training = train(out, loader)
        base.write_json(out/'training_complete.json', dict(jobs=len(training), reference_accessed=False))
    else:
        training = json.loads((out/'training.json').read_text())
        if len(training) != 24 or not (out/'training_complete.json').exists():
            raise ValueError('Training must finish before reference scoring')
    rows = validate(out, training)
    base.write_json(out/'status.json', dict(phase='complete', completed=len(rows), expected=24,
        outcomes={s: sum(r['outcome'] == s for r in rows) for s in ('accurate', 'inaccurate', 'unresolved')},
        invocation_seconds=time.perf_counter()-tick))


if __name__ == '__main__':
    main()
