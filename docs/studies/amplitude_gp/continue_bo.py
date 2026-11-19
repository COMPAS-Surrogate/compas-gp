"""Extend frozen four-input BO fits without accessing validation references.

All twelve BO fits continue from 744 to 1494 labels. The GP representation,
three-point acquisition and 30-label refits are unchanged. A new deterministic
RNG stream is used because the original runs did not save generator state.
Training completes for every job before the independent scoring stage starts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import shutil
import time

import jax
import numpy as np

import small_batch_followup as study

SOURCE = study.HERE / 'small_batch_20260914'
TARGET = 1494
CAPACITY = 1536


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    jobs = [job for job in study.jobs() if job['strategy'] == 'bo']
    write = study.pilot.base.write_json
    paths = [Path(__file__), study.HERE/'small_batch_followup.py',
             study.HERE/'run_pilot.py', study.HERE/'stratified_reference.py',
             *sorted((study.pilot.REPO/'src/cosmic_integration').rglob('*.py')),
             *sorted((study.HERE.parent/'operational_stopping/gpry_compas').glob('*.py')),
             *sorted((study.HERE.parent/'event_uncertainty').glob('*.py')),
             *sorted((study.HERE.parent/'event_count_bo').glob('*.py')),
             *sorted((study.HERE.parent/'joint_pilot').glob('*.py')),
             *sorted((study.HERE.parent/'population_stability').glob('*.py'))]
    for path in paths:
        destination = root/'source_snapshot'/path.relative_to(study.pilot.REPO)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    write(root/'manifest.json', dict(jobs=jobs, start=744, target=TARGET,
          capacity=CAPACITY, rng='seed + 2026091600; fresh continuation stream',
          source_sha256={str(p.relative_to(study.pilot.REPO)): digest(p) for p in paths},
          scope=__doc__))
    study.CAPACITY = CAPACITY
    loader = study.pilot.base.data_source.Experiment(root/'population', 'four_input_continuation', False)
    for job_index, job in enumerate(jobs):
        folder = root/job['name']
        folder.mkdir()
        source = SOURCE/job['name']/'checkpoint_744.npz'
        state, unit, values = study.load_state(source)
        initial_unit, initial_values = unit.copy(), values.copy()
        state = study.padded(state, unit, values)
        cat_path = study.BASELINE/f'catalogue_n{job["expected"]}_{job["mode"]}.npz'
        cat = dict(np.load(cat_path))
        rng = np.random.default_rng(job['seed']+2026091600)
        batches = []
        timing = dict(labels=0., acquisition=0., fit=0., conditioning=0.)
        started = time.perf_counter()
        for batch in range((TARGET-744)//3):
            tick = time.perf_counter()
            selected, diagnostics = study.acquire_small_batch(state, unit, values, rng)
            before = study.pilot.base.predict(state, selected)
            timing['acquisition'] += time.perf_counter()-tick
            tick = time.perf_counter()
            physical = study.pilot.LOW+study.pilot.WIDTH*selected
            rates = np.array([loader.fast(p[0], p[1], .012, p[3]).ravel()[loader.active] for p in physical])
            actual = study.pilot.joint_amplitude_log_likelihood(
                rates@cat['coeff'].T, rates.sum(axis=1), physical[:, 2], float(cat['duration']))
            if not np.isfinite(actual).all():
                raise ValueError('Nonfinite physical labels')
            timing['labels'] += time.perf_counter()-tick
            batches.append(dict(before_labels=len(unit), unit=selected.tolist(),
                prediction=before.tolist(), actual=actual.tolist(), acquisition=diagnostics))
            unit, values = np.vstack([unit, selected]), np.r_[values, actual]
            tick = time.perf_counter()
            if (batch+1) % 10 == 0:
                state = study.padded(study.pilot.fit(unit, values), unit, values)
                timing['fit'] += time.perf_counter()-tick
                write(folder/'batches.json', batches)
                write(root/'campaign_status.json', dict(phase='training', job=job['name'],
                    labels=len(unit), completed_training=job_index, total=len(jobs)))
            else:
                state = study.condition_on_data(state, unit, values)
                jax.block_until_ready(state)
                timing['conditioning'] += time.perf_counter()-tick
        assert np.array_equal(unit[:744], initial_unit)
        assert np.array_equal(values[:744], initial_values)
        assert len(np.unique(unit, axis=0)) == TARGET
        checkpoint = folder/f'checkpoint_{TARGET}.npz'
        np.savez_compressed(checkpoint, unit=unit, lnl=values,
                            **{f'state_{k}': np.asarray(v) for k, v in state.items()})
        write(folder/'complete.json', dict(**job, labels=TARGET, timing=timing,
            seconds=time.perf_counter()-started, reference_accessed=False,
            initial_sha256=digest(source), checkpoint_sha256=digest(checkpoint),
            catalogue_sha256=digest(cat_path), rng_state=rng.bit_generator.state))
        logging.info('Trained %s to %d in %.1fs', job['name'], TARGET, time.perf_counter()-started)
    results = []
    cached, key = None, None
    for job in jobs:
        current = job['expected'], job['mode']
        if current != key:
            cached = study.prepare_references(job)
            key = current
        refs, metadata = cached
        folder = root/job['name']
        state, _, _ = study.load_state(folder/f'checkpoint_{TARGET}.npz')
        attempts = []
        for replicas in (16, 64):
            write(root/'campaign_status.json', dict(phase='scoring', job=job['name'],
                completed_training=len(jobs), completed_scoring=len(results), total=len(jobs)))
            result = study.score_once(state, refs, metadata, replicas, 2026091500+job['expected'])
            attempts.append(result)
            write(folder/'score_attempts.json', attempts)
            if result['outcome'] != 'unresolved':
                break
        write(folder/'reference.json', metadata)
        row = dict(**job, labels=TARGET, **attempts[-1])
        results.append(row)
        write(root/'results.json', results)
        logging.info('Scored %s: %s maxKL %.5g', job['name'], result['outcome'],
                     max(max(s['directed_kl']) for s in result['scores']))
    write(root/'campaign_status.json', dict(phase='complete', completed_training=len(jobs),
        completed_scoring=len(results), total=len(jobs)))


if __name__ == '__main__':
    main()
