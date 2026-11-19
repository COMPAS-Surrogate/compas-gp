"""Frozen 482-label GP validation with a correlated categorical measurement model.

Pilot: --catalogues 3 --seed 20261030 --paired --outdir uncertain_pilot_results
Confirmation: --catalogues 100 --seed 20261101 --outdir uncertain_100_results
The response is a controlled post-detection sensor, not LVK PE samples.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import logging
import pickle
import time
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz, save_npz
from scipy.special import logsumexp
from scipy.stats import qmc, gamma, binomtest

from confirm_482 import compare
from bo_uncertainty_curve import (
    HERE, REPO, configurations, FixedPopulationRates, BinnedCosmicIntegrator,
    AdaptiveRobustScaler, JaxTrainingData, JaxGPConfig, fit_exact_gp,
    vector_marginal, sqrt_log_expected_likelihood, posterior_metrics,
)
from check_ensemble_quadrature import direct_summary
from cosmic_integration.observation.binned_response import correlated_bin_response, measure_counts

LOW = np.array([-.5, .1, 4.2])
WIDTH = np.array([.499, .5, 1.])


def fit_482(initial: np.ndarray, candidates: np.ndarray, initial_y: np.ndarray,
            label_function, folder: Path) -> tuple:
    """Acquire labels only after selection; retain all points and fixed budget."""
    x, y = initial.copy(), initial_y.copy()
    available = np.ones(len(candidates), dtype=bool)
    rng = np.random.default_rng(17)
    acquired = []
    tick = time.perf_counter()
    for batch in range(3):
        if not np.isfinite(y).all():
            raise ValueError('Nonfinite training labels')
        scaler = AdaptiveRobustScaler(compression='sqrt', soft_clipping=False,
                                      lower_clip_value=float(y.min()), max_scale=1e9)
        scaler.initialize_with_data(y)
        model = fit_exact_gp(JaxTrainingData(x, -scaler.transform(y)[:, None]),
                             np.array([LOW, LOW+WIDTH]),
                             config=JaxGPConfig(optimisation_steps=250), seed=17)
        if batch == 2:
            break
        mu, var = model.predict_f(candidates)
        first = sqrt_log_expected_likelihood(mu[:, 0], var[:, 0], scaler)
        second = sqrt_log_expected_likelihood(mu[:, 0], var[:, 0], scaler, power=2.)
        with np.errstate(divide='ignore'):
            score = second + np.log(-np.expm1(np.minimum(2*first-second, 0.)))
        idx = np.flatnonzero(available)
        s = score[idx]
        p = (.9*np.exp(s-logsumexp(s))+.1/len(idx) if np.isfinite(s).any()
             else np.full(len(idx), 1/len(idx)))
        chosen = rng.choice(idx, 128, replace=False, p=p)
        available[chosen] = False
        acquired.extend(chosen.tolist())
        x = np.vstack([x, candidates[chosen]])
        y = np.r_[y, label_function(chosen)]
    assert len(x) == len(np.unique(x, axis=0)) == 482
    elapsed = time.perf_counter()-tick
    folder.mkdir()
    with (folder/'model.pkl').open('wb') as stream:
        pickle.dump(model, stream)
    scaler.save(str(folder))
    np.savez(folder/'training.npz', x=x, lnl=y, acquired_indices=acquired)
    return model, scaler, elapsed


def posterior_predictive(rates: np.ndarray, counts: np.ndarray, log_target: np.ndarray,
                         rng: np.random.Generator, shape: tuple[int, int], draws: int = 1000) -> dict:
    """Replicate measured counts, including amplitude and Poisson uncertainty."""
    total = rates.sum(axis=1)
    w = np.exp(log_target-logsumexp(log_target))
    selected = rng.choice(len(rates), draws, p=w)
    beta = .1*total[selected]/.012
    k = counts.sum()+1
    lo, hi = gamma.cdf(.005, k, scale=1/beta), gamma.cdf(.015, k, scale=1/beta)
    slo, shi = gamma.sf(.005, k, scale=1/beta), gamma.sf(.015, k, scale=1/beta)
    u = rng.random(draws)
    amplitude = np.where(lo < .5, gamma.ppf(lo+u*(hi-lo), k, scale=1/beta),
                         gamma.isf(slo-u*(slo-shi), k, scale=1/beta))
    if not np.isfinite(amplitude).all():
        raise ValueError('Unresolved posterior predictive amplitude draw')
    expected = .1*rates[selected]*amplitude[:, None]/.012
    replicate = rng.poisson(expected).reshape(draws, *shape)
    observed = counts.reshape(shape)
    def statistics(c):
        n = c.sum(axis=(-2, -1))
        # Means in bin coordinates are intentional: no finite center is assigned
        # to the unbounded mass guard bin. Empty-catalogue means are defined as 0.
        denominator = np.maximum(n, 1)
        m = (c.sum(axis=-1)*np.arange(shape[0])).sum(axis=-1)/denominator
        z = (c.sum(axis=-2)*np.arange(shape[1])).sum(axis=-1)/denominator
        return np.stack([n, m, z], axis=-1)
    observed_stats = statistics(observed)
    rep_stats = statistics(replicate)
    return {'draws': draws, 'statistic_names': ['event_count', 'mean_measured_mass_bin', 'mean_measured_redshift_bin'],
            'observed_statistics': observed_stats.tolist(),
            'replicate_statistics': rep_stats.tolist(),
            'tail_probability_ge_observed': (rep_stats >= observed_stats).mean(axis=0).tolist(),
            'mass_count_quantiles': np.quantile(replicate.sum(axis=2), [.05, .5, .95], axis=0).tolist(),
            'z_count_quantiles': np.quantile(replicate.sum(axis=1), [.05, .5, .95], axis=0).tolist(),
            'observed_mass_counts': observed.sum(axis=1).tolist(),
            'observed_z_counts': observed.sum(axis=0).tolist()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalogues', type=int, default=3)
    parser.add_argument('--seed', type=int, default=20261030)
    parser.add_argument('--paired', action='store_true')
    parser.add_argument('--outdir', default='uncertain_pilot_results')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    out = HERE/args.outdir
    out.mkdir(exist_ok=False)  # Preserve earlier results rather than overwrite.
    start = time.perf_counter()
    manifest = {'catalogues': args.catalogues, 'seed': args.seed, 'paired': args.paired,
                'validation_seeds': [args.seed+1, args.seed+2], 'validation_points': 16384,
                'training_points': 482, 'training_seed': 17, 'duration_years': .1,
                'response': {'sigma_bins': 1., 'rho': .6, 'radius': 3, 'boundary': 'saturation'},
                'population': 'blocks0-7 for both generation and inference',
                'scope': 'Matched-model categorical post-detection uncertainty; not LVK PE or physical Gaussian errors',
                'source_sha256': {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in [Path(__file__), HERE/'confirm_482.py', HERE/'check_ensemble_quadrature.py',
                                            HERE/'bo_uncertainty_curve.py', HERE.parent/'population_stability/bootstrap_blocks.py',
                                            *sorted((REPO/'src/cosmic_integration').rglob('*.py'))]}}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    ci = BinnedCosmicIntegrator.from_compas_fpath(str(REPO/'tests/large_test_data/h5out_32M_reduced.h5'))
    mass, z, _ = configurations()['mass10']
    fast = FixedPopulationRates(ci, mass, z)
    block_dir = HERE.parent/'population_stability/block_results'
    meta = json.loads((block_dir/'population_meta.json').read_text())
    ci.compas.nSystems = 256000000
    ci.compas.zamsZmin, ci.compas.zamsZmax = meta['zmin'], meta['zmax']
    fast.p_draw = 1/np.log(meta['zmax']/meta['zmin'])
    fast.operator = sum((load_npz(block_dir/f'block_{i:02d}.npz') for i in range(1, 8)),
                        load_npz(block_dir/'block_00.npz'))
    response = correlated_bin_response(fast.shape)
    save_npz(out/'response.npz', response)
    rng = np.random.default_rng(args.seed)
    truths = rng.uniform([-.5, .1, .005, 4.2], [-.001, .6, .015, 5.2], (args.catalogues, 4))
    latent = np.array([rng.poisson(.1*fast(*t)) for t in truths])
    observed = np.array([measure_counts(c, response, rng) for c in latent])
    np.savez(out/'catalogues.npz', truths=truths, latent=latent, observed=observed, mass_edges=mass, z_groups=z)
    initial = LOW+qmc.Sobol(3, scramble=True, seed=17).random_base2(7)*WIDTH
    unit = np.array(np.meshgrid(*([np.linspace(0, 1, 5)]*3), indexing='ij')).reshape(3, -1).T
    initial = np.vstack([initial, LOW+unit[np.any((unit == 0)|(unit == 1), axis=1)]*WIDTH])
    candidates = LOW+qmc.Sobol(3, scramble=True, seed=117).random_base2(12)*WIDTH
    validation = [LOW+qmc.Sobol(3, scramble=True, seed=s).random_base2(14)*WIDTH
                  for s in manifest['validation_seeds']]
    def evaluate(points):
        return np.array([fast(t[0], t[1], .012, t[2]).ravel() for t in points])
    grids = [initial, candidates, *validation]
    raw = []
    for i, points in enumerate(grids):
        raw.append(evaluate(points))
        logging.info('Rate cache %d/4 ready (%d evaluations)', i+1, len(points))
    caches = {'uncertain': [np.asarray((response @ r.T).T) for r in raw]}
    if args.paired:
        caches['perfect'] = raw
    cache_seconds = time.perf_counter()-start
    records = []
    for case in range(args.catalogues):
        for mode, cache in caches.items():
            counts = (observed if mode == 'uncertain' else latent)[case].ravel()
            occupied = counts > 0
            cnt = counts[occupied]
            compact = [np.column_stack([r[:, occupied], r.sum(axis=1)]) for r in cache]
            row = {'case': case, 'mode': mode, 'events': int(counts.sum()), 'truth': truths[case].tolist()}
            try:
                model, scaler, fit_time = fit_482(initial, candidates, vector_marginal(compact[0], cnt)[0],
                    lambda chosen: vector_marginal(compact[1][chosen], cnt)[0], out/f'{mode}_{case:03d}')
                summaries, metrics, targets = [], [], []
                for phi, rr in zip(validation, compact[2:]):
                    exact = vector_marginal(rr, cnt)[0]
                    pred = scaler.inverse_transform(-model.predict_f(phi)[0][:, 0])
                    direct = direct_summary(rr, cnt, phi, truths[case])
                    gp = direct_summary(rr, cnt, phi, truths[case], log_target=pred)
                    m = {**compare(direct, gp), **{k: v for k, v in posterior_metrics(exact, pred, phi).items() if k.startswith('kl_')}}
                    m['posterior_pass'] = bool(m['max_mean_shift_sd'] < .2 and m['max_endpoint_shift_sd'] < .2
                        and np.max(abs(np.array(m['width68_ratio'])-1)) < .1 and np.max(abs(np.array(m['width95_ratio'])-1)) < .1)
                    m['kl_pass'] = bool(m['kl_direct_to_gp'] < .01 and m['kl_gp_to_direct'] < .01)
                    summaries.append({'direct': direct, 'gp': gp})
                    metrics.append(m)
                    targets.append({'direct': exact, 'gp': pred})
                checks = {name: compare(summaries[0][name], summaries[1][name]) for name in ['direct', 'gp']}
                numerical = all(m['max_mean_shift_sd'] < .05 and m['max_endpoint_shift_sd'] < .05
                                and np.max(abs(np.array(m['width68_ratio'])-1)) < .03 for m in checks.values())
                row.update(status='evaluated', summaries=summaries, metrics=metrics, numerical_checks=checks,
                           numerical_pass=bool(numerical), posterior_pass=all(m['posterior_pass'] for m in metrics),
                           kl_pass=all(m['kl_pass'] for m in metrics), fit_acquire_seconds=fit_time)
                if case < 3:
                    # Predictive checks are illustrative, not calibrated p-values.
                    ppc = {name: posterior_predictive(cache[2], counts, target,
                            np.random.default_rng(args.seed+100+case), fast.shape)
                           for name, target in targets[0].items()}
                    (out/f'{mode}_{case:03d}'/'predictive.json').write_text(json.dumps(ppc, indent=2)+'\n')
            except (ValueError, FloatingPointError) as exc:
                row.update(status='failed', reason=str(exc), posterior_pass=False, kl_pass=False, numerical_pass=False)
            records.append(row)
            (out/'cases.json').write_text(json.dumps(records, indent=2)+'\n')
            logging.info('Case %d/%d %s N=%d posterior=%s KL=%s numerical=%s', case+1, args.catalogues,
                         mode, row['events'], row['posterior_pass'], row['kl_pass'], row['numerical_pass'])
    summary = {'seconds': time.perf_counter()-start, 'cache_seconds': cache_seconds, 'modes': {}}
    for mode in caches:
        cases = [r for r in records if r['mode'] == mode]
        valid = [r for r in cases if r['status'] == 'evaluated']
        result = {'attempted': len(cases), 'evaluated': len(valid), 'failed': len(cases)-len(valid),
                  **{key: sum(r[key] for r in cases) for key in ['posterior_pass', 'kl_pass', 'numerical_pass']}}
        result['coverage'] = {}
        for name in ['direct', 'gp']:
            for level in [.68, .9, .95]:
                ranks = np.array([r['summaries'][0][name]['ranks'] for r in valid])
                hits = ((ranks >= (1-level)/2)&(ranks <= (1+level)/2)).sum(axis=0)
                result['coverage'][f'{name}_{level}'] = {'counts': hits.tolist(),
                    'denominator_evaluated': len(valid),
                    'individual_exact95': [list(binomtest(int(h), len(valid)).proportion_ci()) for h in hits]} if valid else {}
        summary['modes'][mode] = result
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    logging.info('Completed in %.1f seconds', summary['seconds'])


if __name__ == '__main__':
    main()
