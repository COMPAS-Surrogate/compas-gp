"""Precompute one unchanged scoring reference in parallel, publishing atomically."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np
import score_comparison as scoring


def initialize(folder: str) -> None:
    global loader
    loader = scoring.base.data_source.Experiment(Path(folder)/f'worker_{os.getpid()}',
                                                 'reference_prefetch_only', False)


def evaluate(points: np.ndarray) -> np.ndarray:
    return np.array([loader.fast(p[0], p[1], .012, p[2]).ravel()[loader.active] for p in points])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--truth-index', type=int, required=True)
    parser.add_argument('--expected-events', type=int, required=True)
    parser.add_argument('--case', type=int, required=True)
    parser.add_argument('--scramble', type=int, required=True)
    parser.add_argument('--power', type=int, default=17)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args(); root = args.input.resolve()
    ti, n, case, scramble = args.truth_index, args.expected_events, args.case, args.scramble
    target = root/'offline'/f'refined_t{ti}_n{n}_c{case}_p{args.power}_{scramble}.npz'
    assert not target.exists()
    work = root/'offline'/f'prefetch_{time.time_ns()}'; work.mkdir()
    before = {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
              for name in ['decisions.json', 'mass_audit/decisions.json']}
    initialize(str(work))
    truth = scoring.run.TRUTHS[ti]
    cat = dict(np.load(root/f't{ti}_n{n}_perfect_c{case}_bo/catalogue.npz'))
    fisher = scoring.base.core.fisher_matrix(lambda t: loader.fast(*t), truth,
        scoring.base.core.WIDTH, float(cat['duration']))
    seed = 202609114+100*ti+n+10*case+scramble
    phi, logq = scoring.base.data_source.reference_design(truth, fisher, seed, args.power)
    started = time.perf_counter()
    print(f'Computing {len(phi)} exact rates with {args.workers} workers', flush=True)
    chunks = [phi[i:i+256] for i in range(0, len(phi), 256)]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize, initargs=(str(work),)) as pool:
        rates = np.concatenate(list(pool.map(evaluate, chunks)))
    indices = np.unique(np.r_[0, 255, 256, np.linspace(0, len(phi)-1, 7, dtype=int)])
    np.testing.assert_array_equal(rates[indices], evaluate(phi[indices]))
    for name, digest in before.items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest
    temporary = work/'reference.npz'
    np.savez(temporary, phi=phi, rates=rates, logq=logq)
    try:
        os.link(temporary, target)  # Fully written file, no overwrite or partial cache.
        published = True
    except FileExistsError:
        published = False
    manifest = dict(target=str(target), published=published, points=len(phi), seed=seed,
        workers=args.workers, checked_indices=indices.tolist(), wall_seconds=time.perf_counter()-started,
        sha256=hashlib.sha256(temporary.read_bytes()).hexdigest(), decision_sha256=before,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (work/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
