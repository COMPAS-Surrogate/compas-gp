"""Audit source preservation, matched designs, labels, and scoring completeness."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

import small_batch_followup as study


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    args = parser.parse_args()
    root = args.input.resolve()
    manifest = json.loads((root/'manifest.json').read_text())
    for name, digest in manifest['source_sha256'].items():
        assert hashlib.sha256((study.pilot.REPO/name).read_bytes()).hexdigest() == digest, name
        assert hashlib.sha256((root/'source_snapshot'/name).read_bytes()).hexdigest() == digest, name
    loader = study.pilot.base.data_source.Experiment(root/f'verification_population_{time.time_ns()}',
                                                    'four_input_followup_parity', False)
    maximum_error, checked_labels, minimum_ess_bound = 0., 0, np.inf
    for job in study.jobs():
        folder = root/job['name']
        complete = json.loads((folder/'complete.json').read_text())
        assert complete['labels'] == 744 and not complete['reference_accessed']
        history = json.loads((folder/'history.json').read_text())
        batches = json.loads((folder/'batches.json').read_text())
        assert [r['labels'] for r in history] == [144, 414, 744]
        assert len(batches) == 200
        for row in history:
            path = folder/f'checkpoint_{row["labels"]}.npz'
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
        _, unit, values = study.load_state(folder/'checkpoint_744.npz')
        assert unit.shape == (744, 4) and len(np.unique(unit, axis=0)) == 744
        _, middle, ml = study.load_state(folder/'checkpoint_414.npz')
        np.testing.assert_array_equal(unit[:414], middle)
        np.testing.assert_array_equal(values[:414], ml)
        original = study.BASELINE/job['name']/'checkpoint_144.npz'
        assert hashlib.sha256(original.read_bytes()).hexdigest() == complete['initial_sha256']
        _, initial, il = study.load_state(original)
        np.testing.assert_array_equal(unit[:144], initial)
        np.testing.assert_array_equal(values[:144], il)
        np.testing.assert_array_equal(unit[144:], np.vstack([b['unit'] for b in batches]))
        np.testing.assert_array_equal(values[144:], np.concatenate([b['actual'] for b in batches]))
        assert [b['before_labels'] for b in batches] == list(range(144, 744, 3))
        assert np.isfinite(np.array([b['prediction'] for b in batches])).all()
        if job['strategy'] == 'bo':
            other = root/job['name'].replace('_bo', '_random')/'checkpoint_144.npz'
            _, ru, ry = study.load_state(other)
            np.testing.assert_array_equal(initial, ru)
            np.testing.assert_array_equal(il, ry)
        cat_path = study.BASELINE/f'catalogue_n{job["expected"]}_{job["mode"]}.npz'
        assert hashlib.sha256(cat_path.read_bytes()).hexdigest() == complete['catalogue_sha256']
        cat = dict(np.load(cat_path))
        rate = loader.fast(*cat['truth']).ravel()[loader.active]
        np.testing.assert_allclose(float(cat['duration'])*rate.sum(), job['expected'], rtol=1e-12)
        indices = np.linspace(144, 743, 8, dtype=int)
        for i in indices:
            physical = study.pilot.LOW+study.pilot.WIDTH*unit[i]
            rates = loader.fast(*physical).ravel()[loader.active]
            actual = np.log(float(cat['duration'])*(rates@cat['coeff'].T)).sum()-float(cat['duration'])*rates.sum()
            maximum_error = max(maximum_error, abs(actual-values[i]))
            np.testing.assert_allclose(actual, values[i], rtol=0, atol=1e-8)
            checked_labels += 1
        scores = json.loads((folder/'scores.json').read_text())
        assert [s['labels'] for s in scores] == [414, 744]
        for score in scores:
            assert score['checkpoint_sha256'] == next(h['sha256'] for h in history if h['labels'] == score['labels'])
            bound = min(s[t]['ess']/score['replicas'] for s in score['scores'] for t in ['direct', 'gp'])
            minimum_ess_bound = min(minimum_ess_bound, bound)
            assert bound >= 100
    result = dict(passed=True, trajectories=24, scored_checkpoints=48, pre_training_batches=4800,
        checked_physical_labels=checked_labels, maximum_absolute_lnl_error=maximum_error,
        minimum_conservative_shape_ess=minimum_ess_bound, source_snapshots_match=True,
        paired_initial_designs_match=True, conditional_count_scaling_verified=True)
    study.pilot.base.write_json(root/'verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
