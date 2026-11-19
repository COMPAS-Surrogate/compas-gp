"""Frozen follow-ups to the continuous-measurement pilot.

Runs are additive: never overwrite pilot results or use reference information in
acquisition. Candidate-rate caches are offline benchmark work, not online savings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
from scipy.sparse import load_npz
from scipy.special import logsumexp, ndtri
from scipy.stats import chi2, multivariate_t, qmc

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / 'event_count_bo'))
import run_study as core
from cosmic_integration.observation.log_measurement import bin_coefficients, log_marginal_events

PILOT = HERE / 'results'
BUDGETS = [226, 354, 482, 610, 738]
TRUTHS = [core.TRUTH.tolist(), [-.46, .15, .006, 4.35], [-.12, .52, .013, 5.05]]
SEEDS = [17, 29, 43]


def write_json(path: Path, value: object) -> None:
    """Atomically update progress so an interrupted run remains inspectable."""
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def reference_design(truth: np.ndarray, fisher: np.ndarray, seed: int,
                     power: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Full-support mixture, centered on truth ONLY for independent validation."""
    covariance = np.linalg.inv(fisher)[np.ix_(core.SHAPE, core.SHAPE)]
    eigen, vectors = np.linalg.eigh(4 * covariance)
    covariance = (vectors * np.clip(eigen, 1e-6, .25)) @ vectors.T
    center = ((truth - core.LOW) / core.WIDTH)[core.SHAPE]
    proposal = multivariate_t(loc=center, shape=covariance, df=5)
    u = qmc.Sobol(4, scramble=True, seed=seed + 1000).random_base2(power)[:3 * 2**(power-2)]
    normal = ndtri(u[:, :3]) @ np.linalg.cholesky(covariance).T
    local = center + normal / np.sqrt(chi2.ppf(u[:, 3], df=5) / 5)[:, None]
    unit = np.vstack([qmc.Sobol(3, scramble=True, seed=seed).random_base2(power-2), local])
    unit = unit[np.all((unit >= 0) & (unit <= 1), axis=1)]
    logq = np.logaddexp(np.log(.25), np.log(.75) + proposal.logpdf(unit))
    return core.LOW[core.SHAPE] + unit * core.WIDTH[core.SHAPE], logq


def measurement(cat: dict, multiplier: float, lower: np.ndarray,
                upper: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pair width variants using the saved standardized measurement residuals."""
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError('Measurement multiplier must be positive and finite')
    y = np.log(cat['latent']) + multiplier * (cat['y'] - np.log(cat['latent']))
    scales = multiplier * cat['scales']
    coeff = bin_coefficients(y, scales, lower, upper)
    maxima = coeff.max(axis=1)
    if np.any(maxima <= 0):
        raise ValueError('Unresolved event integral; do not clip observed data')
    return y, coeff / maxima[:, None], np.log(maxima)


def exact_center_coefficients(y: np.ndarray, lower: np.ndarray,
                              upper: np.ndarray) -> tuple[np.ndarray, int]:
    """Deliberately ignore errors; retain unsupported centres as a model failure."""
    x = np.exp(y)
    coeff = np.all((x[:, None, :] >= lower) & (x[:, None, :] < upper), axis=2).astype(float)
    return coeff, int(np.sum(coeff.sum(axis=1) == 0))


def precision(a: dict, b: dict) -> tuple[bool, dict]:
    metrics = core.compare(a, b)
    passed = (metrics['max_mean_shift_sd'] < .05 and metrics['max_endpoint_shift_sd'] < .05
              and np.max(abs(np.array(metrics['width68_ratio']) - 1)) < .03)
    return bool(passed), metrics


def accuracy(metrics: list[dict]) -> bool:
    return all(m['max_mean_shift_sd'] < .2 and m['max_endpoint_shift_sd'] < .2
               and np.max(abs(np.array(m['width68_ratio']) - 1)) < .1
               and np.max(abs(np.array(m['width95_ratio']) - 1)) < .1
               and m['kl_direct_to_gp'] < .01 and m['kl_gp_to_direct'] < .01 for m in metrics)


class Experiment:
    """One immutable run directory with explicit cache and trajectory accounting."""

    def __init__(self, out: Path, phase: str, smoke: bool):
        self.out = out
        out.mkdir(parents=True, exist_ok=False)
        self.started = time.perf_counter()
        self.records, self.references = [], []
        self.cache_seconds, self.cache_calls = 0., 0
        self.budgets = BUDGETS[:2] if smoke else BUDGETS
        self.smoke = smoke
        sources = [Path(__file__), HERE / 'run_study.py', HERE / 'ESSENTIAL_RUNS.md',
                   HERE.parent / 'event_count_bo/run_study.py',
                   *sorted((REPO / 'src/cosmic_integration').rglob('*.py')),
                   *sorted((HERE.parent / 'joint_pilot').glob('*.py')),
                   *sorted((HERE.parent / 'population_stability').glob('*.py'))]
        hashes = {}
        for source in sources:
            relative = source.relative_to(REPO)
            target = out / 'source_snapshot' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
            hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
        write_json(out / 'manifest.json', {
            'phase': phase, 'smoke': smoke, 'budgets': self.budgets,
            'width_multipliers': [.5, 2.], 'robustness_truths': TRUTHS,
            'robustness_seeds': SEEDS, 'catalogue_seed': 202609101,
            'reference_power': 15 if phase == 'sweep' else 16,
            'reference_seeds': 'pilot saved references' if phase == 'sweep' else '202609200 + 10*truth_index + 2*exposure_index + scramble',
            'population': 'fixed blocks0-7, matched generation/inference',
            'scope': 'Width sweep and fixed-injection robustness, not SBC, PE Monte Carlo validation, or online timing',
            'baseline_manifest_sha256': hashlib.sha256((PILOT / 'manifest.json').read_bytes()).hexdigest(),
            'source_sha256': hashes,
        })
        ci = core.BinnedCosmicIntegrator.from_compas_fpath(str(REPO / 'tests/large_test_data/h5out_32M_reduced.h5'))
        mass, z, _ = core.configurations()['mass10']
        self.fast = core.FixedPopulationRates(ci, mass, z)
        block = HERE.parent / 'population_stability/block_results'
        meta = json.loads((block / 'population_meta.json').read_text())
        ci.compas.nSystems = 256000000
        ci.compas.zamsZmin, ci.compas.zamsZmax = meta['zmin'], meta['zmax']
        self.fast.p_draw = 1 / np.log(meta['zmax'] / meta['zmin'])
        self.fast.operator = sum((load_npz(block / f'block_{i:02d}.npz') for i in range(1, 8)), load_npz(block / 'block_00.npz'))
        grid = np.load(PILOT / 'grid.npz')
        self.active, self.lower, self.upper = grid['active'], grid['lower'], grid['upper']
        np.testing.assert_array_equal(self.active, np.flatnonzero(self.fast.operator.getnnz(axis=1)))
        np.testing.assert_allclose(self.fast(*core.TRUTH).ravel()[self.active], grid['truth_rates'], rtol=1e-12)
        self.fiducial_total = float(grid['truth_rates'].sum())
        self.designs = {}

    def rates(self, phi: np.ndarray, name: str) -> np.ndarray:
        tick = time.perf_counter()
        rates = np.array([self.fast(p[0], p[1], .012, p[2]).ravel()[self.active] for p in phi])
        self.cache_seconds += time.perf_counter() - tick
        self.cache_calls += len(phi)
        np.savez(self.out / f'{name}.npz', phi=phi, rates=rates)
        return rates

    def design(self, seed: int) -> dict:
        if seed not in self.designs:
            lo, width = core.LOW[core.SHAPE], core.WIDTH[core.SHAPE]
            initial = qmc.Sobol(3, scramble=True, seed=seed).random_base2(7)
            unit = np.array(np.meshgrid(*([np.linspace(0, 1, 5)] * 3), indexing='ij')).reshape(3, -1).T
            initial = lo + np.vstack([initial, unit[np.any((unit == 0) | (unit == 1), axis=1)]]) * width
            candidates = lo + qmc.Sobol(3, scramble=True, seed=seed + 100).random_base2(12) * width
            random_seed = 20260922 if seed == 17 else 20260922 + seed
            random = np.random.default_rng(random_seed).uniform(lo, lo + width, (512, 3))
            sobol = lo + qmc.Sobol(3, scramble=True, seed=seed + 200).random_base2(9) * width
            self.designs[seed] = {key: (x, self.rates(x, f'design_{seed}_{key}'))
                                  for key, x in [('initial', initial), ('bo', candidates), ('random', random), ('sobol', sobol)]}
        return self.designs[seed]

    def reference(self, truth: np.ndarray, duration: float, ti: int, ni: int) -> list:
        fisher = core.fisher_matrix(lambda t: self.fast(*t), truth, core.WIDTH, duration)
        refs = []
        for scramble in [0, 1]:
            seed = 202609200 + 10 * ti + 2 * ni + scramble
            phi, logq = reference_design(truth, fisher, seed)
            name = f'reference_t{ti}_n{ni}_{scramble}'
            rates = self.rates(phi, name)
            np.savez(self.out / f'{name}_proposal.npz', logq=logq, truth=truth, duration=duration)
            refs.append((phi, logq, rates))
            logging.info('Reference t%s exposure%s scramble%s: %s points', ti, ni, scramble, len(phi))
        return refs

    @staticmethod
    def summarize(phi: np.ndarray, rates: np.ndarray, duration: float, number: int,
                  truth: np.ndarray, logweights: np.ndarray) -> dict:
        total = rates.sum(axis=1) * duration / .1
        return core.direct_summary(np.column_stack([total, total]), np.array([number]), phi,
                                   truth, log_target=logweights)

    def validate(self, refs: list, coeff: np.ndarray, duration: float,
                 truth: np.ndarray, key: dict) -> tuple[list, bool]:
        validation = []
        for phi, logq, rates in refs:
            exact = log_marginal_events(rates @ coeff.T, rates.sum(axis=1), duration)
            ds = self.summarize(phi, rates, duration, len(coeff), truth, exact - logq)
            validation.append((phi, logq, rates, exact, ds))
        passed, check = precision(validation[0][-1], validation[1][-1])
        self.references.append({**key, 'events': len(coeff), 'summaries': [v[-1] for v in validation],
                                'numerical_pass': passed, 'numerical_check': check})
        return validation, passed

    def trajectory(self, coeff: np.ndarray, duration: float, truth: np.ndarray,
                   validation: list, refpass: bool, key: dict, strategy: str,
                   seed: int, saved: Path | None = None) -> None:
        folder = self.out / ('_'.join(f'{k}{v}' for k, v in key.items()) + f'_{strategy}_seed{seed}')
        folder.mkdir()
        timers = {'fit_seconds': 0., 'acquisition_seconds': 0., 'label_contraction_seconds': 0.,
                  'validation_seconds': 0., 'supplied_training_labels': 0}
        def evaluate(rates):
            tick = time.perf_counter()
            y = log_marginal_events(rates @ coeff.T, rates.sum(axis=1), duration)
            timers['label_contraction_seconds'] += time.perf_counter() - tick
            return y
        if saved is None:
            design = self.design(seed)
            x, ir = design['initial']
            x, y = x.copy(), evaluate(ir)
            candidates, cr = design[strategy]
            available = np.ones(len(candidates), bool)
        rng = np.random.default_rng(seed)
        for step, budget in enumerate(self.budgets):
            if saved is not None:
                train_path = saved / f'checkpoint_{budget}.npz'
                with np.load(train_path) as train:
                    x, y = train['x'], train['lnl']
                write_json(folder / f'original_{budget}.json', {'path': str(train_path),
                           'sha256': hashlib.sha256(train_path.read_bytes()).hexdigest()})
            tick = time.perf_counter()
            scaler = core.AdaptiveRobustScaler(compression='sqrt', soft_clipping=False,
                                               lower_clip_value=float(y.min()), max_scale=1e9)
            scaler.initialize_with_data(y)
            model = core.fit_exact_gp(core.JaxTrainingData(x, -scaler.transform(y)[:, None]),
                np.array([core.LOW[core.SHAPE], (core.LOW + core.WIDTH)[core.SHAPE]]),
                config=core.JaxGPConfig(optimisation_steps=250), seed=seed)
            timers['fit_seconds'] += time.perf_counter() - tick
            tick = time.perf_counter()
            metrics, summaries = [], []
            for phi, logq, rates, exact, ds in validation:
                pred = scaler.inverse_transform(-model.predict_f(phi)[0][:, 0])
                gs = self.summarize(phi, rates, duration, len(coeff), truth, pred - logq)
                metrics.append({**core.compare(ds, gs), **{k: v for k, v in
                    core.posterior_metrics(exact-logq, pred-logq, phi).items() if k.startswith('kl_')}})
                summaries.append(gs)
            gp_precision, check = precision(*summaries)
            numerical, accurate = bool(refpass and gp_precision), accuracy(metrics)
            timers['validation_seconds'] += time.perf_counter() - tick
            timers['supplied_training_labels'] = len(x)
            self.records.append({**key, 'strategy': strategy, 'training_seed': seed, 'points': budget,
                'metrics': metrics, 'summaries': summaries, 'accuracy_pass': accurate,
                'numerical_pass': numerical, 'validated_pass': accurate and numerical,
                'numerical_check': check, 'timing': dict(timers)})
            np.savez(folder / f'checkpoint_{budget}.npz', x=x, lnl=y)
            self.progress()
            if budget == self.budgets[-1]:
                break
            if saved is not None:
                continue
            tick = time.perf_counter()
            if strategy == 'bo':
                mu, var = model.predict_f(candidates)
                one = core.sqrt_log_expected_likelihood(mu[:, 0], var[:, 0], scaler)
                two = core.sqrt_log_expected_likelihood(mu[:, 0], var[:, 0], scaler, power=2.)
                with np.errstate(divide='ignore'):
                    score = two + np.log(-np.expm1(np.minimum(2*one-two, 0.)))
                ix = np.flatnonzero(available)
                norm = logsumexp(score[ix])
                p = .9*np.exp(score[ix]-norm) + .1/len(ix) if np.isfinite(norm) else np.full(len(ix), 1/len(ix))
                chosen = rng.choice(ix, 128, replace=False, p=p)
                available[chosen] = False
            else:
                chosen = np.arange(step*128, (step+1)*128)
            timers['acquisition_seconds'] += time.perf_counter() - tick
            x = np.vstack([x, candidates[chosen]])
            y = np.r_[y, evaluate(cr[chosen])]
        logging.info('%s %s seed%s: accuracy=%s numerical=%s', key, strategy, seed, accurate, numerical)

    def progress(self, state: str = 'running') -> None:
        write_json(self.out / 'records.json', self.records)
        write_json(self.out / 'references.json', self.references)
        write_json(self.out / 'status.json', {'state': state, 'checkpoints': len(self.records),
            'reference_checks': len(self.references), 'seconds': time.perf_counter()-self.started,
            'offline_rate_cache_seconds': self.cache_seconds, 'offline_rate_cache_calls': self.cache_calls})

    def refine(self) -> None:
        n, case = 500, 8
        duration = n/self.fiducial_total
        refs = self.reference(core.TRUTH, duration, 0, 1)
        cat = dict(np.load(PILOT / f'catalogue_{n}_{case}.npz'))
        coeff = cat['coeff']
        key = {'exposure': n, 'case': case, 'scale': 1.}
        validation, passed = self.validate(refs, coeff, duration, core.TRUTH, key)
        for strategy in ['bo', 'random']:
            self.trajectory(coeff, duration, core.TRUTH, validation, passed, key, strategy, 17,
                            saved=PILOT / f'n{n}_case{case}_uncertain_{strategy}')

    def sweep(self) -> None:
        for ni, n in enumerate([50, 500][:1] if self.smoke else [50, 500]):
            # Reuse the pilot proposals; the unresolved 500-event case remains
            # visible unless separately resolved by reference-only refinement.
            refs = []
            for scramble in [0, 1]:
                with np.load(PILOT / f'reference_{n}_{scramble}.npz') as data:
                    refs.append((data['phi'], np.load(PILOT / f'proposal_{n}_{scramble}.npz')['logq'], data['rates']))
            duration = n/self.fiducial_total
            for case in range(1 if self.smoke else 10):
                cat = dict(np.load(PILOT / f'catalogue_{n}_{case}.npz'))
                for scale in ([.5] if self.smoke else [.5, 2.]):
                    y, coeff, offsets = measurement(cat, scale, self.lower, self.upper)
                    key = {'exposure': n, 'case': case, 'scale': scale}
                    np.savez(self.out / f'catalogue_{n}_{case}_scale{scale}.npz', **cat,
                             varied_y=y, varied_scales=cat['scales']*scale,
                             varied_coeff=coeff, varied_log_coefficient_scale=offsets)
                    validation, passed = self.validate(refs, coeff, duration, core.TRUTH, key)
                    naive, unsupported = exact_center_coefficients(y, self.lower, self.upper)
                    if unsupported:
                        self.references.append({**key, 'control': 'ignore_uncertainty',
                            'unsupported_events': unsupported, 'status': 'zero_likelihood_everywhere'})
                    else:
                        self.validate(refs, naive, duration, core.TRUTH, {**key, 'control': 'ignore_uncertainty'})
                    for strategy in ['bo', 'random']:
                        self.trajectory(coeff, duration, core.TRUTH, validation, passed, key, strategy, 17)

    def robustness(self) -> None:
        for ti, t in enumerate(TRUTHS[:1] if self.smoke else TRUTHS):
            truth = np.array(t)
            rate = self.fast(*truth).ravel()[self.active]
            for ni, n in enumerate([50, 500][:1] if self.smoke else [50, 500]):
                duration = n/self.fiducial_total
                refs = self.reference(truth, duration, ti, ni)
                for case in range(1 if self.smoke else 2):
                    rng = np.random.default_rng(np.random.SeedSequence([202609101, ti, ni, case]))
                    number = rng.poisson(duration*rate.sum())
                    bins = rng.choice(len(rate), number, p=rate/rate.sum())
                    latent = rng.uniform(self.lower[bins], self.upper[bins])
                    rho = 12*rng.random(number)**(-1/3)
                    scales = np.array([.03, .3])[None, :]*12/rho[:, None]
                    y = np.log(latent)+rng.normal(size=(number, 2))*scales
                    cat = {'latent': latent, 'scales': scales, 'y': y}
                    _, coeff, offsets = measurement(cat, 1., self.lower, self.upper)
                    key = {'truth': ti, 'exposure': n, 'case': case, 'scale': 1.}
                    np.savez(self.out / f'catalogue_t{ti}_{n}_{case}.npz', **cat, rho=rho, bins=bins,
                             coeff=coeff, log_coefficient_scale=offsets, truth=truth,
                             duration=duration, expected_events=duration*rate.sum())
                    validation, passed = self.validate(refs, coeff, duration, truth, key)
                    for seed in (SEEDS[:1] if self.smoke else SEEDS):
                        for strategy in ['bo', 'random', 'sobol']:
                            self.trajectory(coeff, duration, truth, validation, passed, key, strategy, seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['refine', 'sweep', 'robustness'])
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true', help='One catalogue, two budgets; not a study result')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    experiment = Experiment(args.outdir, args.phase, args.smoke)
    try:
        getattr(experiment, args.phase)()
    except Exception as exc:
        experiment.progress('failed')
        write_json(args.outdir / 'failure.json', {'type': type(exc).__name__, 'message': str(exc)})
        raise
    experiment.progress('complete')


if __name__ == '__main__':
    main()
