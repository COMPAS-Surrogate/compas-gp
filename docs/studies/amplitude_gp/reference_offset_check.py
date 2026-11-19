"""Fixed-label sqrt-reference offsets: smooth the transform near the peak.

Frozen exploratory ablation: all twelve 744-label BO fits, offsets +5 and +20
lnL above the best training label, otherwise unchanged scaler/noise/250 steps.
The offset uses training values only, not a reference maximum or injection.
Every model is fitted before direct-reference scoring. No defaults change.
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


def shifted_scaler(values: np.ndarray, offset: float):
    """Change only the transform reference; retain its original scale and floor."""
    if not np.isfinite(offset) or offset <= 0:
        raise ValueError('A positive finite reference offset is required')
    scaler = study.pilot.base.core.AdaptiveRobustScaler(compression='sqrt', soft_clipping=False,
        lower_clip_value=float(values.min()), max_scale=1e9)
    scaler.initialize_with_data(values)
    scaler.reference_value += offset
    return scaler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source_root, root = args.input.resolve(), args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    pilot = study.pilot
    jobs = [j for j in study.jobs() if j['strategy'] == 'bo']
    rows = []
    for job in jobs:
        source = source_root/job['name']/'checkpoint_744.npz'
        _, unit, values = study.load_state(source)
        for offset in (5., 20.):
            tick = time.perf_counter()
            scaler = shifted_scaler(values, offset)
            model = pilot.base.core.fit_exact_gp(
                pilot.base.core.JaxTrainingData(unit, -scaler.transform(values)),
                np.array([np.zeros(4), np.ones(4)]),
                config=pilot.base.core.JaxGPConfig(optimisation_steps=250), seed=17)
            state = pilot.state_from_gpjax(model, scaler, np.zeros(4), np.ones(4), len(unit))
            jax.block_until_ready(state)
            name = f'{job["name"]}_offset{int(offset)}'
            np.savez_compressed(root/f'{name}.npz', unit=unit, lnl=values,
                objective=model.objective_history, **{f'state_{k}': np.asarray(v) for k, v in state.items()})
            rows.append(dict(**job, fit_name=name, offset=offset, fit_seconds=time.perf_counter()-tick,
                checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                final_objective=float(model.objective_history[-1])))
            pilot.base.write_json(root/'fits.json', rows)
            logging.info('Offset fit %s %.1fs', name, time.perf_counter()-tick)
    pilot.base.write_json(root/'training_complete.json', dict(models=24, reference_accessed=False))
    key, cached = None, None
    amplitude = np.linspace(.005, .015, 32769)
    for row in rows:
        current = (row['expected'], row['mode'])
        if current != key:
            cached, key = study.prepare_references(row), current
        refs, metadata = cached
        state, _, _ = study.load_state(root/f'{row["fit_name"]}.npz')
        attempts = []
        for replicas in (16, 64):
            result = study.score_once(state, refs, metadata, replicas, 2026091500+row['expected'])
            attempts.append(result)
            if result['outcome'] != 'unresolved':
                break
        cat = dict(np.load(study.BASELINE/f'catalogue_n{row["expected"]}_{row["mode"]}.npz'))
        points = np.tile(cat['truth'], (len(amplitude), 1))
        points[:, 2] = amplitude
        exact = len(cat['coeff'])*np.log(amplitude)-row['expected']/cat['truth'][2]*amplitude
        prediction = pilot.base.predict(state, (points-pilot.LOW)/pilot.WIDTH)
        conditional = pilot.score(amplitude[:, None], exact, prediction, np.zeros(len(amplitude)))
        row.update(attempts=attempts, result=attempts[-1], conditional=conditional)
        pilot.base.write_json(root/'results.json', rows)
        np.savez_compressed(root/f'{row["fit_name"]}_conditional.npz', amplitude=amplitude, exact=exact, gp=prediction)
        logging.info('Offset score %s: %s KL %.6g; conditional %s', row['fit_name'], row['result']['outcome'],
                     max(max(s['directed_kl']) for s in row['result']['scores']), conditional['accuracy_pass'])
    pilot.base.write_json(root/'manifest.json', dict(complete=True, models=24, offsets=[5., 20.],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Exploratory fixed-label refit after failed small-batch pilot; not prospective acquisition validation'))


if __name__ == '__main__':
    main()
