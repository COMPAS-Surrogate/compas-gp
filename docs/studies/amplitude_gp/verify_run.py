"""Check saved pilot provenance, paired designs, and direct physical-rate parity."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

spec = importlib.util.spec_from_file_location('explicit_amplitude_pilot', Path(__file__).with_name('run_pilot.py'))
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    args = parser.parse_args()
    root = args.input.resolve()
    manifest = json.loads((root/'manifest.json').read_text())
    for name, digest in manifest['source_sha256'].items():
        assert hashlib.sha256((pilot.REPO/name).read_bytes()).hexdigest() == digest, name
    rows = json.loads((root/'training.json').read_text())
    assert len(rows) == 24
    loader = pilot.base.data_source.Experiment(root/f'verification_population_{time.time_ns()}',
                                               'explicit_amplitude_parity', False)
    worst, labels = 0., 0
    for job in rows:
        folder = root/job['name']
        checkpoint = folder/'checkpoint_400.npz'
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == job['checkpoint_sha256']
        with np.load(checkpoint) as data:
            unit, saved = data['unit'], data['lnl']
        assert unit.shape == (400, 4) and len(np.unique(unit, axis=0)) == 400
        with np.load(folder/'checkpoint_144.npz') as initial:
            np.testing.assert_array_equal(unit[:144], initial['unit'])
            np.testing.assert_array_equal(saved[:144], initial['lnl'])
        if job['strategy'] == 'bo':
            other = root/job['name'].replace('_bo', '_random')/'checkpoint_144.npz'
            with np.load(other) as random:
                np.testing.assert_array_equal(unit[:144], random['unit'])
                np.testing.assert_array_equal(saved[:144], random['lnl'])
        cat = dict(np.load(root/f'catalogue_n{job["expected"]}_{job["mode"]}.npz'))
        # Independently evaluate actual A through the physical rate operator.
        # Do not reuse the reference-amplitude scaling helper under test.
        indices = np.linspace(0, 399, 8, dtype=int)
        points = pilot.LOW+pilot.WIDTH*unit[indices]
        for i, point in zip(indices, points):
            rates = loader.fast(*point).ravel()[loader.active]
            actual = np.log(float(cat['duration'])*(rates@cat['coeff'].T)).sum()-float(cat['duration'])*rates.sum()
            worst = max(worst, abs(actual-saved[i]))
            np.testing.assert_allclose(actual, saved[i], rtol=0, atol=1e-8)
            labels += 1
    for n in [100, 1000]:
        with np.load(root/f'catalogue_n{n}_perfect.npz') as perfect, np.load(root/f'catalogue_n{n}_uncertain.npz') as uncertain:
            for key in ['latent', 'bins', 'duration', 'truth', 'scales', 'y']:
                np.testing.assert_array_equal(perfect[key], uncertain[key])
    result = dict(passed=True, checked_jobs=24, checked_physical_labels=labels,
                  maximum_absolute_lnl_difference=worst, source_hashes_passed=True,
                  initial_design_pairing_passed=True, catalogue_pairing_passed=True)
    pilot.base.write_json(root/'verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
