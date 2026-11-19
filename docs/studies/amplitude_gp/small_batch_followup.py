"""Run and independently score the frozen four-input small-batch follow-up."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import logging
from pathlib import Path
import shutil
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('amplitude_initial_pilot', HERE/'run_pilot.py')
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)
from cosmic_integration.lnl_surrogate.gpry_rules import condition_on_data
from cosmic_integration.lnl_surrogate.small_batch_acquisition import acquire_small_batch
from stratified_reference import amplitude_design

BASELINE = HERE/'pilot_20260914'
CAPACITY = 768
CHECKPOINTS = (144, 414, 744)


def jobs() -> list[dict]:
    """Paired methods share catalogue, initial data and random seed."""
    return [dict(name=f'n{n}_{mode}_s{seed}_{strategy}', expected=n, mode=mode, seed=seed, strategy=strategy)
            for n, mode, seed, strategy in itertools.product((100, 1000), ('perfect', 'uncertain'),
                                                            (17, 29, 43), ('bo', 'random'))]


def initialize(root: Path) -> dict:
    source = [HERE/'small_batch_followup.py', HERE/'stratified_reference.py',
              HERE/'SMALL_BATCH_PROTOCOL.md', HERE/'run_pilot.py',
              *sorted((pilot.REPO/'src/cosmic_integration').rglob('*.py')),
              *sorted((HERE.parent/'operational_stopping/gpry_compas').glob('*.py')),
              *sorted((HERE.parent/'event_uncertainty').glob('*.py')),
              *sorted((HERE.parent/'event_count_bo').glob('*.py')),
              *sorted((HERE.parent/'joint_pilot').glob('*.py')),
              *sorted((HERE.parent/'population_stability').glob('*.py'))]
    digest = {str(p.relative_to(pilot.REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source}
    metadata = dict(source_sha256=digest, jobs=jobs(), checkpoints=CHECKPOINTS, conditioning_capacity=CAPACITY)
    if (root/'manifest.json').exists():
        old = json.loads((root/'manifest.json').read_text())
        if old['source_sha256'] != digest:
            raise ValueError('Frozen source changed; use a new run directory')
        return old
    root.mkdir(parents=True, exist_ok=False)
    for p in source:
        target = root/'source_snapshot'/p.relative_to(pilot.REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
    pilot.base.write_json(root/'manifest.json', metadata)
    return metadata


def load_state(path: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    with np.load(path) as f:
        state = {k[6:]: jnp.asarray(f[k]) for k in f.files if k.startswith('state_')}
        return state, f['unit'], f['lnl']


def padded(state: dict, unit: np.ndarray, values: np.ndarray) -> dict:
    template = {k: v for k, v in state.items() if k not in ('x', 'mask', 'alpha', 'inverse')}
    template.update(x=jnp.zeros((CAPACITY, 4)), mask=jnp.zeros(CAPACITY))
    result = condition_on_data(template, unit, values)
    jax.block_until_ready(result)
    return result


def train(root: Path, indices: list[int]) -> None:
    loader = None
    for index in indices:
        job = jobs()[index]
        folder = root/job['name']
        if (folder/'complete.json').exists():
            logging.info('Already complete: %s', job['name'])
            continue
        if folder.exists():
            raise ValueError(f'Incomplete job retained at {folder}; use a new run or explicit recovery')
        if loader is None:
            loader = pilot.base.data_source.Experiment(root/f'population_{time.time_ns()}', 'four_input_small_batch', False)
        folder.mkdir()
        source = BASELINE/job['name']/'checkpoint_144.npz'
        state, unit, values = load_state(source)
        cat_path = BASELINE/f'catalogue_n{job["expected"]}_{job["mode"]}.npz'
        cat = dict(np.load(cat_path))
        state = padded(state, unit, values)
        started = time.perf_counter()
        timing = dict(labels=0., acquisition=0., fit=0., conditioning=0.)
        history, batches = [], []
        rng = np.random.default_rng(job['seed']+20260914)
        for batch in range(201):
            if batch % 10 == 0:
                pilot.base.write_json(root/'training_status.json', dict(phase='training', job=job['name'],
                    labels=len(unit), completed=sum((root/j['name']/'complete.json').exists() for j in jobs()),
                    total=24, elapsed_seconds=time.perf_counter()-started))
                logging.info('%s labels=%d elapsed=%.1fs', job['name'], len(unit), time.perf_counter()-started)
            if len(unit) in CHECKPOINTS:
                path = folder/f'checkpoint_{len(unit)}.npz'
                np.savez_compressed(path, unit=unit, lnl=values,
                                    **{f'state_{k}': np.asarray(v) for k, v in state.items()})
                history.append(dict(labels=len(unit), seconds=time.perf_counter()-started,
                    timing=timing.copy(), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                pilot.base.write_json(folder/'history.json', history)
            if len(unit) == 744:
                break
            tick = time.perf_counter()
            if job['strategy'] == 'bo':
                selected, diagnostics = acquire_small_batch(state, unit, values, rng)
            else:
                selected, diagnostics = rng.random((3, 4)), []
            before = pilot.base.predict(state, selected)
            timing['acquisition'] += time.perf_counter()-tick
            tick = time.perf_counter()
            physical = pilot.LOW+pilot.WIDTH*selected
            rates = np.array([loader.fast(p[0], p[1], .012, p[3]).ravel()[loader.active] for p in physical])
            actual = pilot.joint_amplitude_log_likelihood(rates@cat['coeff'].T, rates.sum(axis=1),
                                                         physical[:, 2], float(cat['duration']))
            timing['labels'] += time.perf_counter()-tick
            batches.append(dict(before_labels=len(unit), unit=selected.tolist(), prediction=before.tolist(),
                                actual=actual.tolist(), acquisition=diagnostics))
            unit, values = np.vstack([unit, selected]), np.r_[values, actual]
            tick = time.perf_counter()
            if (batch+1) % 10 == 0:
                state = padded(pilot.fit(unit, values), unit, values)
                timing['fit'] += time.perf_counter()-tick
                pilot.base.write_json(folder/'batches.json', batches)
            else:
                state = condition_on_data(state, unit, values)
                jax.block_until_ready(state)
                timing['conditioning'] += time.perf_counter()-tick
        if len(np.unique(unit, axis=0)) != 744:
            raise ValueError('Duplicate training coordinates')
        pilot.base.write_json(folder/'complete.json', dict(**job, labels=len(unit),
            seconds=time.perf_counter()-started, timing=timing, reference_accessed=False,
            initial_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            catalogue_sha256=hashlib.sha256(cat_path.read_bytes()).hexdigest()))
    completed = sum((root/j['name']/'complete.json').exists() for j in jobs())
    pilot.base.write_json(root/'training_status.json', dict(phase='complete' if completed == 24 else 'partial',
                                                         completed=completed, total=24))


def prepare_references(job: dict) -> tuple[list, dict]:
    hashes = {}
    cat = dict(np.load(BASELINE/f'catalogue_n{job["expected"]}_{job["mode"]}.npz'))
    refs = pilot.reference.cached_references(0, job['expected'], hashes)
    checked = pilot.reference.direct_checks(refs, cat, cat['truth'])
    output = []
    for scramble, (phi, rates, logq) in enumerate(refs):
        constant = np.concatenate([np.log(float(cat['duration'])*(r@cat['coeff'].T)).sum(axis=1)
                                   for r in np.array_split(rates, int(np.ceil(len(rates)/512)))])
        beta = float(cat['duration'])*rates.sum(axis=1)/.012
        analytic = dict(checked[2][scramble])
        analytic['quantiles'] = np.asarray(analytic['quantiles'])[[0, 2, 3, 4, 6]].tolist()
        output.append((phi, logq, beta, constant, analytic))
    return output, dict(analytic_reference=checked[-1], source_sha256=hashes, events=len(cat['coeff']))


def score_once(state: dict, refs: list, metadata: dict, replicas: int, seed: int) -> dict:
    scores, parity = [], []
    for scramble, (phi, logq, beta, constant, analytic) in enumerate(refs):
        a, qa = amplitude_design(beta, metadata['events'], replicas, seed+scramble)
        shape = np.repeat(phi, replicas, axis=0)
        points = np.column_stack([shape[:, :2], a.ravel(), shape[:, 2]])
        exact = (constant[:, None]+metadata['events']*np.log(a/.012)-beta[:, None]*a).ravel()
        q = np.repeat(logq, replicas)+qa.ravel()
        predicted = pilot.base.predict(state, (points-pilot.LOW)/pilot.WIDTH)
        result = pilot.score(points, exact, predicted, q)
        scores.append(result)
        target = dict(analytic, ess=result['direct']['ess'])
        parity.append(pilot.numerical(target, result['direct']))
    direct_precision = pilot.numerical(scores[0]['direct'], scores[1]['direct'])
    gp_precision = pilot.numerical(scores[0]['gp'], scores[1]['gp'])
    kls = np.asarray([s['directed_kl'] for s in scores])
    delta = abs(kls[0]-kls[1])
    kl_precision = bool(np.all(delta < np.maximum(.001, .1*np.max(abs(kls), axis=0))))
    resolved = (metadata['analytic_reference']['passed'] and all(p['passed'] for p in parity)
                and direct_precision['passed'] and gp_precision['passed'] and kl_precision)
    outcome = ('unresolved' if not resolved else 'accurate' if all(s['accuracy_pass'] for s in scores) else 'inaccurate')
    return dict(replicas=replicas, outcome=outcome, scores=scores, parity=parity,
        direct_precision=direct_precision, gp_precision=gp_precision,
        kl_precision=kl_precision, kl_scramble_difference=delta.tolist())


def score_jobs(root: Path, indices: list[int]) -> None:
    cached, key = None, None
    for index in indices:
        job = jobs()[index]
        folder = root/job['name']
        if not (folder/'complete.json').exists():
            continue
        if (folder/'scores.json').exists():
            continue
        current = (job['expected'], job['mode'])
        if current != key:
            cached = prepare_references(job)
            key = current
        refs, metadata = cached
        results = []
        history = json.loads((folder/'history.json').read_text())
        for count in (414, 744):
            tick = time.perf_counter()
            checkpoint = folder/f'checkpoint_{count}.npz'
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            assert digest == next(h['sha256'] for h in history if h['labels'] == count)
            state, _, _ = load_state(checkpoint)
            attempts = []
            for replicas in (16, 64):
                pilot.base.write_json(root/'scoring_status.json', dict(phase='scoring', job=job['name'],
                    labels=count, amplitude_replicas=replicas))
                result = score_once(state, refs, metadata, replicas, 2026091500+job['expected'])
                attempts.append(result)
                pilot.base.write_json(folder/f'score_{count}_attempts.json', attempts)
                logging.info('Score %s labels=%d replicas=%d %s KL=%.5g elapsed=%.1fs', job['name'],
                    count, replicas, result['outcome'], max(max(s['directed_kl']) for s in result['scores']), time.perf_counter()-tick)
                if result['outcome'] != 'unresolved':
                    break
            results.append(dict(labels=count, checkpoint_sha256=digest, attempts=attempts,
                                seconds=time.perf_counter()-tick, **attempts[-1]))
        pilot.base.write_json(folder/'reference.json', metadata)
        pilot.base.write_json(folder/'scores.json', results)
    summary = []
    for job in jobs():
        path = root/job['name']/'scores.json'
        if path.exists():
            summary.append(dict(**job, checkpoints=json.loads(path.read_text())))
    pilot.base.write_json(root/'results.json', summary)
    pilot.base.write_json(root/'scoring_status.json', dict(phase='complete' if len(summary) == 24 else 'partial',
                                                        completed=len(summary), total=24))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=['train', 'score'], required=True)
    parser.add_argument('--job', type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    root = args.output.resolve()
    initialize(root)
    indices = list(range(24)) if args.job is None else [args.job]
    if args.stage == 'train':
        train(root, indices)
    else:
        score_jobs(root, indices)


if __name__ == '__main__':
    main()
