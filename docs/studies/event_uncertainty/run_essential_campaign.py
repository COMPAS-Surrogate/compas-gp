"""Run the authorized essential phases serially, with persistent status and logs."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ROOT = HERE / 'essential_20260910'


def main() -> None:
    phases = ['refine', 'sweep', 'robustness']
    if any((ROOT / phase).exists() for phase in phases):
        raise FileExistsError('Campaign output already exists; preserve it and choose a new campaign')
    smoke = json.loads((ROOT / 'smoke/status.json').read_text())
    if smoke['state'] != 'complete' or smoke['checkpoints'] != 4:
        raise RuntimeError('Complete the isolated two-strategy smoke test first')
    sources = [HERE / 'essential_runs.py', HERE / 'ESSENTIAL_RUNS.md', Path(__file__),
               *sorted((REPO / 'src/cosmic_integration').rglob('*.py')),
               *sorted((HERE.parent / 'joint_pilot').glob('*.py')),
               *sorted((HERE.parent / 'population_stability').glob('*.py')),
               HERE.parent / 'event_count_bo/run_study.py']
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    manifest = {'pid': os.getpid(), 'started_unix': time.time(), 'phases': phases,
                'python': sys.executable, 'source_sha256': hashes}
    with (ROOT / 'campaign_manifest.json').open('x') as stream:
        json.dump(manifest, stream, indent=2)
    status = {'state': 'running', 'pid': os.getpid(), 'completed': [], 'active_phase': None}
    def save() -> None:
        temporary = ROOT / 'campaign_status.tmp'
        temporary.write_text(json.dumps(status, indent=2)+'\n')
        temporary.replace(ROOT / 'campaign_status.json')
    env = {**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
           'MPLCONFIGDIR': '/tmp/compas-essential', 'PYTHONUNBUFFERED': '1'}
    for phase in phases:
        changed = [name for name, digest in hashes.items()
                   if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest]
        if changed:
            status.update(state='failed', error='Source changed between phases', changed_sources=changed)
            save()
            raise RuntimeError('Source changed; do not silently mix implementations')
        status['active_phase'] = phase
        save()
        command = [sys.executable, str(HERE / 'essential_runs.py'), phase, '--outdir', str(ROOT / phase)]
        with (ROOT / f'{phase}.log').open('x') as log:
            child = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
            status['child_pid'] = child.pid
            save()
            returncode = child.wait()
        if returncode:
            status.update(state='failed', returncode=returncode)
            save()
            raise SystemExit(returncode)
        status['completed'].append(phase)
    status.update(state='complete', active_phase=None, child_pid=None, completed_unix=time.time())
    save()


if __name__ == '__main__':
    main()
