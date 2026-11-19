"""Fresh paired COMPAS acquisition trajectories, with reference-free monitors."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import logging
from pathlib import Path
import time

import jax
import numpy as np
from scipy.special import logsumexp
from scipy.stats import qmc

import run_experiment as base
from check_precision import designs
from cosmic_integration.lnl_surrogate.stopping import audit_check, StoppingTolerances

HERE = Path(__file__).resolve().parent
TRUTHS = np.asarray(base.data_source.TRUTHS)
AUDIT_TOL = StoppingTolerances(audit_rms=.07, audit_q95=.15,
                             audit_weight_ess_fraction=.99)
RULES = ["original", "precision_refined", "guarded"]


def gp_designs(state: dict, unit: np.ndarray, lnl: np.ndarray,
               seed: int, power: int) -> list:
    """Two full-support importance designs using only the fitted GP."""
    weights = np.exp((lnl-lnl.max())/2); weights /= weights.sum()
    centre = weights@unit
    cov = (unit-centre).T@(weights[:, None]*(unit-centre))+np.eye(3)*.005**2
    points, logq = designs(centre, cov, seed, 14)[0]
    logw = base.predict(state, points)-logq
    weights = np.exp(logw-logsumexp(logw))
    return designs(weights@points, base.covariance(points, logw), seed+1000, power)


def stability_on_designs(previous: dict | None, state: dict, proposal: list) -> dict:
    current = [base.predict(state, p)-q for p, q in proposal]
    summaries = [base.posterior_summary(p, w) for (p, _), w in zip(proposal, current)]
    numerical, check = base.summaries_agree(*summaries, base.TOL)
    result = dict(numerical=numerical, numerical_current=check,
                  ess=[s['ess'] for s in summaries], gaussian_pass=False, joint_pass=False)
    if previous is None:
        return result
    old = [base.predict(previous, p)-q for p, q in proposal]
    old_summaries = [base.posterior_summary(p, w) for (p, _), w in zip(proposal, old)]
    previous_ok, previous_check = base.summaries_agree(*old_summaries, base.TOL)
    numerical &= previous_ok
    joint = max(base.symmetric_kl(a, b) for a, b in zip(old, current))
    gkl = max(base.gaussian_kl(np.asarray(a['mean']), base.covariance(p, wa),
                             np.asarray(b['mean']), base.covariance(p, wb))
              for (p, _), a, b, wa, wb in zip(proposal, old_summaries, summaries, old, current))
    result.update(numerical=bool(numerical), numerical_previous=previous_check,
        joint_kl=joint, gaussian_kl=gkl, gaussian_pass=bool(numerical and gkl < .02),
        joint_pass=bool(numerical and joint < .001 and all(
            base.summaries_agree(a, b, base.TOL)[0] for a, b in zip(old_summaries, summaries))))
    return result


def adaptive_stability(previous: dict | None, state: dict, unit: np.ndarray,
                       lnl: np.ndarray, seed: int) -> tuple[dict, list]:
    attempts = []
    for power in [14, 15, 16]:
        proposal = gp_designs(state, unit, lnl, seed, power)
        result = stability_on_designs(previous, state, proposal)
        attempts.append(dict(power=power, **result))
        if result['numerical']:
            break
    return dict(attempts=attempts, **result), proposal


def audit_points(state: dict, proposal: list, rng: np.random.Generator) -> np.ndarray:
    """Sample GP posterior/tempered posterior and prior with an independent RNG."""
    points = np.vstack([p for p, _ in proposal]); logq = np.concatenate([q for _, q in proposal])
    prediction = base.predict(state, points)
    selected = []
    for temperature, count in [(1., 64), (2., 16)]:
        logw = prediction/temperature-logq
        weights = np.exp(logw-logsumexp(logw))
        selected.append(points[rng.choice(len(points), count, p=weights)])
    return np.vstack([*selected, rng.uniform(size=(16, 3))])


def catalogue(loader, ti: int, n: int, case: int) -> dict:
    rng = np.random.default_rng(np.random.SeedSequence([202609112, ti, n, case]))
    rate = loader.fast(*TRUTHS[ti]).ravel()[loader.active]
    number = rng.poisson(n)
    bins = rng.choice(len(rate), number, p=rate/rate.sum())
    latent = rng.uniform(loader.lower[bins], loader.upper[bins])
    rho = 12*rng.random(number)**(-1/3)
    scales = np.array([.03, .3])[None, :]*12/rho[:, None]
    y = np.log(latent)+scales*rng.normal(size=(number, 2))
    return dict(latent=latent, scales=scales, y=y, rho=rho, bins=bins,
                duration=n/rate.sum(), truth=TRUTHS[ti])


def run_case(out: Path, loader, ti: int, n: int, mode: str, case: int,
             strategy: str, maximum: int) -> dict:
    name = f't{ti}_n{n}_{mode}_c{case}_{strategy}'
    folder = out/name; folder.mkdir()
    cat = catalogue(loader, ti, n, case)
    if mode == 'perfect':
        coeff = np.eye(len(loader.active))[cat['bins']]
    else:
        _, coeff, _ = base.data_source.measurement(cat, 1., loader.lower, loader.upper)
    np.savez(folder/'catalogue.npz', coeff=coeff, **cat)
    def evaluate(unit):
        points = base.LOW+base.WIDTH*unit
        rates = np.array([loader.fast(p[0], p[1], .012, p[2]).ravel()[loader.active] for p in points])
        values = base.source_target.event_target(rates, coeff, float(cat['duration']))
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite COMPAS likelihood inside the frozen prior')
        return values
    seed = 6011+case
    grid = np.array(np.meshgrid(*([np.linspace(0, 1, 3)]*3), indexing='ij')).reshape(3, -1).T
    unit = np.vstack([qmc.Sobol(3, scramble=True, seed=seed).random_base2(5),
                      grid[np.any((grid == 0) | (grid == 1), axis=1)]])
    capacity = int(np.ceil(maximum/32))*32
    started = time.perf_counter(); lnl = evaluate(unit)
    timing = dict(labels=time.perf_counter()-started, fit=0., acquisition=0., stability=0., audit=0.)
    tick = time.perf_counter(); state = base.fit(unit, lnl, capacity)
    jax.block_until_ready(state); timing['fit'] += time.perf_counter()-tick
    counter, streaks = base.CorrectCounter(3), Counter()
    rng = np.random.default_rng(seed+10000)
    audit_rng = np.random.default_rng(np.random.SeedSequence([202609113, ti, n, case, int(mode == 'uncertain'), int(strategy == 'random')]))
    rules = {r: None for r in RULES}; previous = None
    checkpoints, batches, audits = [], [], []
    for batch in range((maximum-58)//3+1):
        count = len(unit)
        if batch % 10 == 0:
            tick = time.perf_counter()
            checked, proposal = adaptive_stability(previous, state, unit, lnl, seed+20000)
            timing['stability'] += time.perf_counter()-tick
            original = checked['attempts'][0]
            for key, passed in [('original', original['gaussian_pass']),
                                ('gaussian', checked['gaussian_pass']), ('joint', checked['joint_pass'])]:
                streaks[key] = streaks[key]+1 if passed else 0
            predicted_ok = counter.streak >= counter.required
            for rule, key in [('original', 'original'), ('precision_refined', 'gaussian')]:
                if rules[rule] is None and predicted_ok and streaks[key] >= 2:
                    rules[rule] = dict(training=count, audit=0, total=count)
            eligible = (predicted_ok and streaks['gaussian'] >= 2 and streaks['joint'] >= 2
                        and rules['guarded'] is None and len(audits) < 4
                        and (not audits or count-audits[-1]['training'] >= 60))
            if eligible:
                tick = time.perf_counter(); points = audit_points(state, proposal, audit_rng)
                predicted = base.predict(state, points); actual = evaluate(points)
                result = audit_check(predicted, actual, 64, float(lnl.max()), AUDIT_TOL)
                audits.append(dict(training=count, points=points.tolist(), predicted=predicted.tolist(),
                                   actual=actual.tolist(), best_before=float(lnl.max()), result=result))
                timing['audit'] += time.perf_counter()-tick
                base.write_json(folder/'audits.json', audits)
                if result['passed']:
                    rules['guarded'] = dict(training=count, audit=96*len(audits), total=count+96*len(audits))
            sha = base.save_checkpoint(folder, state, unit, lnl)
            checkpoints.append(dict(points=count, sha256=sha, counter=counter.streak,
                stability=checked, streaks=dict(streaks), rules=dict(rules), audit_calls=96*len(audits),
                posterior_evaluation=len(checkpoints)+1, posterior_comparisons=len(checkpoints),
                bo_batches=batch, elapsed_seconds=time.perf_counter()-started,
                recorded_unix=time.time(), timing=dict(timing)))
            base.write_json(folder/'checkpoints.json', checkpoints)
            base.write_json(out/'status.json', dict(state='running', case=name, points=count))
            logging.info('%s n=%s counter=%s numerical=%s power=%s audits=%s stops=%s elapsed=%.1fs',
                name, count, counter.streak, checked['numerical'], checked['attempts'][-1]['power'],
                len(audits), [r for r, v in rules.items() if v], time.perf_counter()-started)
            previous = state
        if count == maximum:
            break
        tick = time.perf_counter()
        if strategy == 'bo':
            selected, diagnostics = base.acquire(state, unit, lnl, rng)
        else:
            selected, diagnostics = rng.uniform(size=(3, 3)), []
        before, best = base.predict(state, selected), float(lnl.max())
        timing['acquisition'] += time.perf_counter()-tick
        tick = time.perf_counter(); actual = evaluate(selected)
        timing['labels'] += time.perf_counter()-tick
        result = counter.update(before, actual, best)
        batches.append(dict(before_points=count, unit=selected.tolist(), prediction=before.tolist(),
            actual=actual.tolist(), best_before=best, counter=result, acquisition=diagnostics))
        base.write_json(folder/'batches.json', batches)
        unit, lnl = np.vstack([unit, selected]), np.r_[lnl, actual]
        tick = time.perf_counter()
        state = base.fit(unit, lnl, capacity) if (batch+1) % 10 == 0 else base.condition_on_data(state, unit, lnl)
        jax.block_until_ready(state); timing['fit'] += time.perf_counter()-tick
    base.write_json(folder/'audits.json', audits)
    result = dict(name=name, truth_index=ti, expected_events=n, mode=mode, case=case,
        strategy=strategy, seed=seed, actual_events=len(coeff), rules=rules,
        training_labels=len(unit), audit_labels=96*len(audits), reference_accessed=False,
        complete=True, timing=timing, wall_seconds=time.perf_counter()-started)
    base.write_json(folder/'decision.json', result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--maximum', type=int, default=658)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.maximum < 88 or (args.maximum-58) % 30:
        parser.error('maximum must be 58 plus a positive multiple of 30')
    args.outdir.mkdir(parents=True, exist_ok=args.resume)
    sources = [Path(__file__), HERE/'EXTENSION_PROTOCOL.md', HERE/'run_experiment.py',
               HERE/'check_precision.py', base.REPO/'src/cosmic_integration/lnl_surrogate/gpry_rules.py',
               base.REPO/'src/cosmic_integration/lnl_surrogate/stopping.py']
    hashes = {str(p.relative_to(base.REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    manifest = dict(source_sha256=hashes, truths=TRUTHS.tolist(), maximum=args.maximum,
                    cases=2, catalogue_seed=202609112, smoke=args.smoke)
    if args.resume:
        assert json.loads((args.outdir/'manifest.json').read_text()) == manifest
    else:
        base.write_json(args.outdir/'manifest.json', manifest)
        for p in sources:
            target = args.outdir/'source_snapshot'/p.relative_to(base.REPO)
            target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(p.read_bytes())
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    # A resumed run gets a new population-provenance folder; no model is resumed
    # halfway through a trajectory, avoiding hidden RNG/checkpoint assumptions.
    population = args.outdir/f'population_{time.time_ns()}'
    loader = base.data_source.Experiment(population, 'COMPAS_bo_random_stopping', False)
    groups = [(0, 100, 'perfect', 91, 'random')] if args.smoke else [
        (ti, n, mode, case, strategy) for ti in range(3) for n in [100, 1000]
        for mode in ['perfect', 'uncertain'] for case in range(2) for strategy in ['bo', 'random']]
    decisions = {}
    for ti, n, mode, case, strategy in groups:
        name = f't{ti}_n{n}_{mode}_c{case}_{strategy}'
        path = args.outdir/name/'decision.json'
        if args.resume and path.exists():
            result = json.loads(path.read_text()); assert result['complete']
        else:
            result = run_case(args.outdir, loader, ti, n, mode, case, strategy, args.maximum)
        decisions[name] = result
        base.write_json(args.outdir/'decisions.json', decisions)
    frozen = hashlib.sha256((args.outdir/'decisions.json').read_bytes()).hexdigest()
    base.write_json(args.outdir/'status.json', dict(state='complete', cases=len(decisions), decision_sha256=frozen))


if __name__ == '__main__':
    main()
