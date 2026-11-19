"""Run matched perfect/uncertain catalogues with a gated pilot and durable status.

The historical campaign remains unchanged. References are shared within this
campaign only; reference information never reaches acquisition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import posterior_convergence as study

COUNTS = (30, 100, 300, 900, 1200)
GROUPS = [(n, mode) for n in COUNTS for mode in ('perfect', 'uncertain')]
HERE, REPO = study.HERE, study.REPO


def verify_pairs(folder: Path, catalogues: int) -> list[dict]:
    """Require exact latent pairing and the same exposure, count and truth."""
    rows = []
    for n in COUNTS:
        for case in range(catalogues):
            with np.load(folder / f'catalogue_{n}_perfect_{case}.npz') as p, \
                    np.load(folder / f'catalogue_{n}_uncertain_{case}.npz') as u:
                for key in ('bins', 'latent', 'rho', 'scales', 'y', 'duration', 'truth'):
                    np.testing.assert_array_equal(p[key], u[key], err_msg=f'{n}/{case}/{key}')
                if len(p['coeff']) != len(u['coeff']):
                    raise ValueError('Paired likelihoods have different event counts')
                if not np.isfinite(u['coeff']).all() or np.any(u['coeff'].sum(axis=1) <= 0):
                    raise ValueError('Unresolved uncertain event coefficients')
                rows.append(dict(expected=n, case=case, events=len(p['latent'])))
    return rows


def pilot_gate(outcomes: list[dict]) -> dict:
    """All BO cases must validate; random resource limits remain explicit."""
    expected = {(n, m, s) for n, m in GROUPS for s in ('bo', 'random')}
    actual = {(r['exposure'], r['mode'], r['strategy']) for r in outcomes}
    if len(outcomes) != len(expected) or actual != expected:
        raise ValueError('Pilot must contain exactly 20 paired strategy outcomes')
    failures = [r for r in outcomes if r['state'] != 'reference_verified_convergence'
                and not (r['strategy'] == 'random' and r['state'] == 'resource_limit')]
    return dict(passed=not failures, failures=failures,
                random_resource_limits=sum(r['strategy'] == 'random' and
                                          r['state'] == 'resource_limit' for r in outcomes))


class PairedExperiment(study.ConvergenceExperiment):
    """Same measurement generator and numerical tolerances as the legacy study."""

    def continuation_seed(self, mode: str) -> int:
        return 20260910

    def rates(self, phi: np.ndarray, name: str) -> np.ndarray:
        """Share immutable reference rates across pilot and ensemble, with checks."""
        shared = self.out.parent / 'reference_cache'
        shared.mkdir(exist_ok=True)
        path = shared / f'{name}.npz'
        tick = time.perf_counter()
        if path.exists():
            with np.load(path) as saved:
                np.testing.assert_array_equal(saved['phi'], phi)
                rates = saved['rates']
        else:
            needed = len(phi) * len(self.active) * 8
            if shutil.disk_usage(shared).free < needed + 5 * 2**30:
                raise RuntimeError('Reference cache would leave less than 5 GiB free')
            rates = np.array([self.fast(p[0], p[1], .012, p[2]).ravel()[self.active] for p in phi])
            temporary = shared / f'{name}.tmp.npz'
            np.savez(temporary, phi=phi, rates=rates)
            temporary.replace(path)
            self.cache_calls += len(phi)
        self.cache_seconds += time.perf_counter() - tick
        target = self.out / path.name
        if not target.exists():
            os.link(path, target)
        return rates


def worker(root: Path, phase: str, catalogues: int) -> None:
    """Write all paired data before fitting; keep a run manifest and grid."""
    count = 1 if phase == 'pilot' else catalogues
    run = PairedExperiment(root / phase, count, 4322, False, GROUPS, False)
    manifest = json.loads((run.out / 'manifest.json').read_text())
    manifest.update(expected_counts=COUNTS, paired_measurements=True,
                    random_continuation_seed=20260910, acquisition_seed=17,
                    independent_latent_catalogues=len(COUNTS) * count,
                    likelihood_cases=len(GROUPS) * count,
                    pilot_overlap='Pilot case zero is also ensemble case zero; never pool them')
    study.base.write_json(run.out / 'manifest.json', manifest)
    np.savez(run.out / 'grid.npz', lower=run.lower, upper=run.upper, active=run.active,
             truth_rates=run.fast(*study.core.TRUTH).ravel()[run.active])
    try:
        run.prepare_catalogues()
        study.base.write_json(run.out / 'pairing.json', verify_pairs(run.out, count))
        if phase == 'pilot':
            import paired_diagnostics
            paired_diagnostics.plot_data(run.out, root / 'figures')
        run.run(generate=False)
    except Exception as exc:
        run.progress('failed')
        study.base.write_json(run.out / 'failure.json', dict(type=type(exc).__name__, message=str(exc)))
        raise


def source_hashes() -> dict[str, str]:
    paths = [Path(__file__), HERE / 'paired_diagnostics.py', HERE / 'PAIRED_CAMPAIGN.md',
             *HERE.glob('*.py'), *sorted((REPO / 'src/cosmic_integration').rglob('*.py')),
             *sorted((HERE.parent / 'joint_pilot').glob('*.py')),
             *sorted((HERE.parent / 'population_stability').glob('*.py')),
             *sorted((HERE.parent / 'event_count_bo').glob('*.py'))]
    return {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def campaign(root: Path, catalogues: int) -> None:
    root.mkdir(parents=True, exist_ok=False)
    hashes = source_hashes()
    for name in hashes:
        target = root / 'source_snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / name, target)
    status = dict(state='running', phase='pilot', pid=os.getpid(), started_unix=time.time(),
                  catalogues_per_count=catalogues, expected_counts=COUNTS)
    study.base.write_json(root / 'manifest.json', dict(source_sha256=hashes, **status))
    def save() -> None:
        study.base.write_json(root / 'campaign_status.json', status)
    save()
    try:
        for phase in ('pilot', 'ensemble'):
            if source_hashes() != hashes:
                raise RuntimeError('Campaign sources changed; recorded pilot cannot gate modified code')
            status.update(phase=phase)
            command = [sys.executable, str(Path(__file__).resolve()), '--root', str(root),
                       '--worker', phase, '--catalogues', str(catalogues)]
            with (root / f'{phase}.log').open('x') as log:
                child = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
                status.update(child_pid=child.pid)
                save()
                code = child.wait()
            if code:
                raise RuntimeError(f'{phase} failed with exit {code}; see {phase}.log')
            if phase == 'pilot':
                gate = pilot_gate(json.loads((root / phase / 'outcomes.json').read_text()))
                study.base.write_json(root / 'pilot_gate.json', gate)
                if not gate['passed']:
                    raise RuntimeError('Pilot scientific gate failed; ensemble was not launched')
                status.update(phase='pilot_predictive_checks', child_pid=None)
                save()
                with (root / 'predictive.log').open('x') as log:
                    subprocess.run([sys.executable, str(HERE / 'paired_diagnostics.py'),
                                    '--folder', str(root / 'pilot'), '--output', str(root / 'figures')],
                                   cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
        status.update(state='complete', phase=None, child_pid=None, finished_unix=time.time())
        save()
    except Exception as exc:
        status.update(state='failed', error=str(exc), finished_unix=time.time())
        save()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--catalogues', type=int, default=100)
    parser.add_argument('--worker', choices=('pilot', 'ensemble'))
    args = parser.parse_args()
    if args.catalogues < 1:
        parser.error('Use at least one catalogue per expected count')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    if args.worker:
        worker(args.root.resolve(), args.worker, args.catalogues)
    else:
        campaign(args.root.resolve(), args.catalogues)


if __name__ == '__main__':
    main()
