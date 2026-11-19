"""Fixed-label optimizer ablation; no acquisition, new labels or default changes.

Predeclared comparison: saved 744-label BO fits at both counts, perfect view,
seed 17; 250 versus 1500 versus 3000 Adam steps from identical initial conditions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import time

import jax
import numpy as np

import small_batch_followup as study


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected', type=int, choices=[100, 1000], required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    job = next(j for j in study.jobs() if j['expected'] == args.expected and j['mode'] == 'perfect'
               and j['seed'] == 17 and j['strategy'] == 'bo')
    source = args.input.resolve()/job['name']/'checkpoint_744.npz'
    _, unit, values = study.load_state(source)
    pilot = study.pilot
    scaler = pilot.base.core.AdaptiveRobustScaler(compression='sqrt', soft_clipping=False,
        lower_clip_value=float(values.min()), max_scale=1e9)
    scaler.initialize_with_data(values)
    data = pilot.base.core.JaxTrainingData(unit, -scaler.transform(values))
    rows = []
    # Fit and freeze every variant before accessing reference likelihoods.
    for steps in (250, 1500, 3000):
        tick = time.perf_counter()
        model = pilot.base.core.fit_exact_gp(data, np.array([np.zeros(4), np.ones(4)]),
            config=pilot.base.core.JaxGPConfig(optimisation_steps=steps), seed=17)
        state = pilot.state_from_gpjax(model, scaler, np.zeros(4), np.ones(4), len(unit))
        jax.block_until_ready(state)
        np.savez_compressed(folder/f'fit_{steps}.npz', unit=unit, lnl=values,
            objective=model.objective_history, **{f'state_{k}': np.asarray(v) for k, v in state.items()})
        row = dict(steps=steps, fit_seconds=time.perf_counter()-tick,
            initial_objective=float(model.objective_history[0]), final_objective=float(model.objective_history[-1]),
            last_100_improvement=float(model.objective_history[-100]-model.objective_history[-1]),
            lengthscale=np.asarray(state['lengthscale']).tolist(), amplitude=float(state['amplitude']),
            scale=float(state['scale']))
        rows.append(row)
        pilot.base.write_json(folder/'fits.json', rows)
        logging.info('Optimizer %s steps=%d objective=%.6g elapsed=%.1fs', job['name'], steps,
                     row['final_objective'], row['fit_seconds'])
    refs, metadata = study.prepare_references(job)
    for row in rows:
        state, _, _ = study.load_state(folder/f'fit_{row["steps"]}.npz')
        attempts = []
        for replicas in (16, 64):
            result = study.score_once(state, refs, metadata, replicas, 2026091500+args.expected)
            attempts.append(result)
            if result['outcome'] != 'unresolved':
                break
        row.update(attempts=attempts, result=attempts[-1])
        pilot.base.write_json(folder/'results.json', rows)
        logging.info('Optimizer score %d: %s KL %.6g', row['steps'], row['result']['outcome'],
                     max(max(s['directed_kl']) for s in row['result']['scores']))
    pilot.base.write_json(folder/'manifest.json', dict(job=job, training_labels=744,
        checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reference=metadata, complete=True,
        scope='Fixed labels and transform, optimizer iterations only; no new acquisition'))


if __name__ == '__main__':
    main()
