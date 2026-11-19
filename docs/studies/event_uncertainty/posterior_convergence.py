"""Posterior-convergence benchmark with selected-point-only training evaluations.

Independent numerical references verify convergence in this tractable benchmark.
This is explicitly not an oracle-free operational stopping rule or a new COMPAS run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import time

import numpy as np
import equinox as eqx
import jax.numpy as jnp
import jax.scipy as jsp
from scipy.special import logsumexp
from scipy.stats import qmc

import essential_runs as base

HERE, REPO, core = base.HERE, base.REPO, base.core
GROUPS = [(30, 'perfect'), (100, 'perfect'), (300, 'perfect'),
          (1000, 'perfect'), (50, 'uncertain'), (500, 'uncertain')]


@eqx.filter_jit
def mean_weights(posterior, x, y):
    """Factor the unchanged fitted GP once when only reference means are needed."""
    covariance = posterior.prior.kernel.gram(x).as_matrix()
    covariance += jnp.diag(posterior.likelihood.noise_vector(len(x))+posterior.jitter)
    return jsp.linalg.cho_solve((jnp.linalg.cholesky(covariance), True),
                               y-posterior.prior.mean_function(x))


@eqx.filter_jit
def mean_at(posterior, x, alpha, query):
    """Exact GP mean; avoids computing unused reference predictive variances."""
    return posterior.prior.mean_function(query)+posterior.prior.kernel.cross_covariance(x, query).T@alpha


def reference_means(model, phi: np.ndarray) -> np.ndarray:
    """Memory-bounded exact mean prediction, algebraically identical to GPJax."""
    x, y = jnp.asarray(model.data.query_points), jnp.asarray(model.data.observations)
    alpha = mean_weights(model.posterior, x, y)
    return np.concatenate([np.asarray(mean_at(model.posterior, x, alpha, jnp.asarray(phi[i:i+4096]))).ravel()
                           for i in range(0, len(phi), 4096)])


def event_target(rates: np.ndarray, coeff: np.ndarray, duration: float) -> np.ndarray:
    """Bound transient event-integral memory for large validation references."""
    return np.concatenate([base.log_marginal_events(r @ coeff.T, r.sum(axis=1), duration)
                           for r in (rates[i:i+1024] for i in range(0, len(rates), 1024))])


def stable_posterior(previous: list[np.ndarray] | None, current: list[np.ndarray],
                     previous_summary: list[dict] | None, current_summary: list[dict]) -> bool:
    """Require changes smaller than the independent-reference precision tolerances."""
    if previous is None or previous_summary is None:
        return False
    for old, new, a, b in zip(previous, current, previous_summary, current_summary):
        old, new = old-logsumexp(old), new-logsumexp(new)
        p, q = np.exp(old), np.exp(new)
        if max(float(p@(old-new)), float(q@(new-old))) >= .001:
            return False
        if not base.precision(a, b)[0]:
            return False
    return True


def stopping_streak(streak: int, accurate: bool, numerical: bool, stable: bool) -> int:
    """An unresolved/unstable checkpoint resets, rather than hides, the evidence."""
    return streak + 1 if accurate and numerical and stable else 0


def seed_design(seed: int, random_seed: int, max_points: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return coordinates only. No rates or likelihoods are evaluated here."""
    lo, width = core.LOW[core.SHAPE], core.WIDTH[core.SHAPE]
    unit = np.array(np.meshgrid(*([np.linspace(0, 1, 5)]*3), indexing='ij')).reshape(3, -1).T
    initial = lo + np.vstack([qmc.Sobol(3, scramble=True, seed=seed).random_base2(7),
                              unit[np.any((unit == 0) | (unit == 1), axis=1)]])*width
    candidate_power = max(12, int(np.ceil(np.log2(max_points-len(initial)))))
    candidates = lo + qmc.Sobol(3, scramble=True, seed=seed+100).random_base2(candidate_power)*width
    random = np.random.default_rng(random_seed).uniform(lo, lo+width, (max_points-len(initial), 3))
    return initial, candidates, random


class ConvergenceExperiment(base.Experiment):
    """Independent references plus measured, selected-only training calls."""

    def __init__(self, out: Path, catalogues: int, max_points: int, smoke: bool,
                 groups: list[tuple[int, str]], pilot: bool):
        super().__init__(out, 'posterior_convergence', smoke)
        self.catalogues, self.max_points, self.groups = catalogues, max_points, groups
        self.pilot, self.outcomes = pilot, []
        self.budgets = list(range(226, max_points+1, 128))
        self.reference_cache = {}
        manifest = json.loads((out/'manifest.json').read_text())
        manifest.update(catalogues_per_group=catalogues, groups=groups, budgets=self.budgets,
            catalogue_seed=202609120, training_seed=17, pilot_uses_original_case_zero=pilot,
            truth=core.TRUTH.tolist(), reference_seeds='202609300 + 2*expected_events + scramble',
            stopping='two consecutive stable and reference-accurate checkpoints; exact thresholds in CONVERGENCE_PLAN.md',
            candidate_evaluations='coordinates and GP predictions only; direct training rates evaluated after selection',
            maximum_points=max_points, resource_limit_is_not_convergence=True,
            scope='fixed-injection repetitions, not prior-drawn SBC; reference-assisted benchmark stop, not operational stop')
        for irrelevant in ['width_multipliers', 'robustness_truths', 'robustness_seeds']:
            manifest.pop(irrelevant, None)
        for source in [Path(__file__), HERE/'CONVERGENCE_PLAN.md']:
            relative = source.relative_to(REPO)
            target = out/'source_snapshot'/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
            manifest['source_sha256'][str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
        base.write_json(out/'manifest.json', manifest)

    def prepare_catalogues(self) -> None:
        """Save the full 100-catalogue ensemble before any method is scored."""
        rate = self.fast(*core.TRUTH).ravel()[self.active]
        for n, mode in self.groups:
            for case in range(self.catalogues):
                rng = np.random.default_rng(np.random.SeedSequence([202609120, n, case]))
                number = rng.poisson(n)
                bins = rng.choice(len(rate), number, p=rate/rate.sum())
                latent = rng.uniform(self.lower[bins], self.upper[bins])
                rho = 12*rng.random(number)**(-1/3)
                scales = np.array([.03, .3])[None, :]*12/rho[:, None]
                y = np.log(latent)+scales*rng.normal(size=(number, 2))
                if self.pilot:
                    if mode == 'uncertain':
                        old = dict(np.load(base.PILOT/f'catalogue_{n}_0.npz'))
                        bins, latent, rho, scales, y = [old[k] for k in ['bins', 'latent', 'rho', 'scales', 'y']]
                    else:
                        old = np.load(HERE.parent/'event_count_bo/results/catalogues.npz')
                        counts = old['counts'][np.flatnonzero(old['expected_events'] == n)[0]][self.active]
                        bins = np.repeat(np.arange(len(rate)), counts)
                        latent = rng.uniform(self.lower[bins], self.upper[bins])
                        rho = np.full(len(bins), 12.)
                        scales = np.broadcast_to([.03, .3], latent.shape).copy()
                        y = np.log(latent)
                if mode == 'perfect':
                    coeff = np.eye(len(rate))[bins]
                else:
                    _, coeff, _ = base.measurement({'latent': latent, 'scales': scales, 'y': y},
                                                   1., self.lower, self.upper)
                np.savez(self.out/f'catalogue_{n}_{mode}_{case}.npz', bins=bins, latent=latent,
                         rho=rho, scales=scales, y=y, coeff=coeff, duration=n/self.fiducial_total,
                         truth=core.TRUTH)
        base.write_json(self.out/'generation.json', {'state': 'complete',
            'catalogues': len(self.groups)*self.catalogues, 'groups': self.groups,
            'catalogues_per_group': self.catalogues})

    def references_for(self, n: int, power: int) -> list:
        """Reuse reference rates within each exposure; never expose them to acquisition."""
        if (n, power) not in self.reference_cache:
            duration = n/self.fiducial_total
            fisher = core.fisher_matrix(lambda t: self.fast(*t), core.TRUTH, core.WIDTH, duration)
            references = []
            for scramble in [0, 1]:
                seed = 202609300 + 2*n + scramble
                phi, logq = base.reference_design(core.TRUTH, fisher, seed, power=power)
                rates = self.rates(phi, f'reference_n{n}_p{power}_{scramble}')
                np.savez(self.out/f'proposal_n{n}_p{power}_{scramble}.npz', logq=logq, duration=duration)
                references.append((phi, logq, rates))
                logging.info('Reference N=%s power%s scramble%s ready', n, power, scramble)
            self.reference_cache[n, power] = references
        return self.reference_cache[n, power]

    def validation_for(self, coeff: np.ndarray, n: int, case: int, mode: str) -> tuple[list, bool]:
        """Refine numerical precision independently of GP fits, keeping every check."""
        duration = n/self.fiducial_total
        validation, passed = [], False
        for power in [16, 17, 18]:
            validation = []
            for phi, logq, rates in self.references_for(n, power):
                exact = event_target(rates, coeff, duration)
                ds = self.summarize(phi, rates, duration, len(coeff), core.TRUTH, exact-logq)
                validation.append((phi, logq, rates, exact, ds))
            passed, check = base.precision(validation[0][-1], validation[1][-1])
            self.references.append({'exposure': n, 'case': case, 'mode': mode, 'power': power,
                                    'numerical_pass': passed, 'check': check})
            self.progress()
            if passed:
                break
        return validation, passed

    def continuation_seed(self, mode: str) -> int:
        """Historical random designs; paired campaigns override this seed policy."""
        return 20260910 if mode == 'perfect' else 20260922

    def train(self, coeff: np.ndarray, duration: float, validation: list, refpass: bool,
              key: dict, strategy: str) -> None:
        folder = self.out/f"n{key['exposure']}_{key['mode']}_case{key['case']}_{strategy}"
        folder.mkdir()
        random_seed = self.continuation_seed(key['mode'])
        x, candidates, random = seed_design(17, random_seed, self.max_points)
        training_calls = 0
        timings = {'training_rate_seconds': 0., 'fit_seconds': 0., 'acquisition_seconds': 0.,
                   'reference_scoring_seconds': 0.}
        def acquire_labels(points):
            nonlocal training_calls
            tick = time.perf_counter()
            rates = np.array([self.fast(p[0], p[1], .012, p[2]).ravel()[self.active] for p in points])
            value = event_target(rates, coeff, duration)
            timings['training_rate_seconds'] += time.perf_counter()-tick
            training_calls += len(points)
            return value
        y = acquire_labels(x)
        available = np.ones(len(candidates), bool)
        rng = np.random.default_rng(17)
        streak, previous, previous_summary = 0, None, None
        state = 'resource_limit'
        for step, budget in enumerate(self.budgets):
            tick = time.perf_counter()
            scaler = core.AdaptiveRobustScaler(compression='sqrt', soft_clipping=False,
                                               lower_clip_value=float(y.min()), max_scale=1e9)
            scaler.initialize_with_data(y)
            model = core.fit_exact_gp(core.JaxTrainingData(x, -scaler.transform(y)[:, None]),
                np.array([core.LOW[core.SHAPE], (core.LOW+core.WIDTH)[core.SHAPE]]),
                config=core.JaxGPConfig(optimisation_steps=250), seed=17)
            timings['fit_seconds'] += time.perf_counter()-tick
            tick = time.perf_counter()
            metrics, summaries, weights = [], [], []
            for phi, logq, rates, exact, ds in validation:
                # Predict in bounded batches to limit GP cross-covariance memory.
                pred = scaler.inverse_transform(-reference_means(model, phi))
                gs = self.summarize(phi, rates, duration, len(coeff), core.TRUTH, pred-logq)
                metrics.append({**core.compare(ds, gs), **{k: v for k, v in
                    core.posterior_metrics(exact-logq, pred-logq, phi).items() if k.startswith('kl_')}})
                summaries.append(gs)
                weights.append(pred-logq)
            numerical = bool(refpass and base.precision(*summaries)[0])
            accurate = base.accuracy(metrics)
            stable = stable_posterior(previous, weights, previous_summary, summaries)
            streak = stopping_streak(streak, accurate, numerical, stable)
            timings['reference_scoring_seconds'] += time.perf_counter()-tick
            row = {**key, 'strategy': strategy, 'points': budget, 'training_rate_calls': training_calls,
                   'accuracy_pass': accurate, 'numerical_pass': numerical, 'stable': stable,
                   'streak': streak, 'metrics': metrics, 'summaries': summaries, 'timing': dict(timings)}
            self.records.append(row)
            np.savez(folder/f'checkpoint_{budget}.npz', x=x, lnl=y)
            self.progress()
            logging.info('%s %s points=%s accuracy=%s numerical=%s stable=%s streak=%s',
                         key, strategy, budget, accurate, numerical, stable, streak)
            if streak >= 2:
                state = 'reference_verified_convergence'
                break
            if budget == self.budgets[-1]:
                break
            previous, previous_summary = weights, summaries
            tick = time.perf_counter()
            if strategy == 'bo':
                mu, var = model.predict_f(candidates)
                one = core.sqrt_log_expected_likelihood(mu[:, 0], var[:, 0], scaler)
                two = core.sqrt_log_expected_likelihood(mu[:, 0], var[:, 0], scaler, power=2.)
                with np.errstate(divide='ignore'):
                    score = two+np.log(-np.expm1(np.minimum(2*one-two, 0.)))
                ix = np.flatnonzero(available)
                norm = logsumexp(score[ix])
                p = .9*np.exp(score[ix]-norm)+.1/len(ix) if np.isfinite(norm) else np.full(len(ix), 1/len(ix))
                chosen = rng.choice(ix, 128, replace=False, p=p)
                available[chosen] = False
                newx = candidates[chosen]
            else:
                newx = random[step*128:(step+1)*128]
            timings['acquisition_seconds'] += time.perf_counter()-tick
            x, y = np.vstack([x, newx]), np.r_[y, acquire_labels(newx)]
        self.outcomes.append({**key, 'strategy': strategy, 'state': state, 'points': len(x),
                              'training_rate_calls': training_calls, 'timing': timings})
        base.write_json(self.out/'outcomes.json', self.outcomes)

    def run(self, generate: bool = True) -> None:
        if generate:
            self.prepare_catalogues()
        for n, mode in self.groups:
            for case in range(self.catalogues):
                cat = np.load(self.out/f'catalogue_{n}_{mode}_{case}.npz')
                coeff, duration = cat['coeff'], float(cat['duration'])
                validation, refpass = self.validation_for(coeff, n, case, mode)
                key = {'exposure': n, 'mode': mode, 'case': case, 'events': len(coeff)}
                if not refpass:
                    for strategy in ['bo', 'random']:
                        self.outcomes.append({**key, 'strategy': strategy, 'state': 'reference_unresolved', 'points': 0})
                    base.write_json(self.out/'outcomes.json', self.outcomes)
                    continue
                for strategy in ['bo', 'random']:
                    self.train(coeff, duration, validation, refpass, key, strategy)
            # References are retained on disk; free old exposures before proceeding.
            self.reference_cache.clear()
        self.progress('complete')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--catalogues', type=int, default=100)
    parser.add_argument('--max-points', type=int, default=4322)
    parser.add_argument('--pilot', action='store_true', help='Original case zero at each selected exposure')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--group', choices=[f'{n}:{m}' for n, m in GROUPS], action='append')
    args = parser.parse_args()
    if args.catalogues < 1 or args.max_points < 354 or (args.max_points-226) % 128:
        parser.error('Use positive catalogue count and max_points=226+128*k, k>=1')
    if args.pilot and args.catalogues != 1:
        parser.error('Pilot reuses only case zero: set --catalogues 1')
    groups = [(int(s.split(':')[0]), s.split(':')[1]) for s in args.group] if args.group else GROUPS
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    run = ConvergenceExperiment(args.outdir, args.catalogues, args.max_points, args.smoke, groups, args.pilot)
    try:
        run.run()
    except Exception as exc:
        run.progress('failed')
        base.write_json(args.outdir/'failure.json', {'type': type(exc).__name__, 'message': str(exc)})
        raise


if __name__ == '__main__':
    main()
