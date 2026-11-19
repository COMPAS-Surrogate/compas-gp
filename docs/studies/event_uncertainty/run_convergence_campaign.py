"""Generate 600 fresh catalogues, run an extension pilot, then the frozen ensemble."""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time

import posterior_convergence as study

ROOT = study.HERE/'convergence_20260910'


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    ROOT.mkdir(exist_ok=False)
    smoke = json.loads((study.HERE/'convergence_smoke_20260910/status.json').read_text())
    if smoke['state'] != 'complete' or smoke['checkpoints'] != 4:
        raise RuntimeError('A completed selected-only smoke run is required')
    sources = [study.HERE/'posterior_convergence.py', study.HERE/'essential_runs.py',
               study.HERE/'CONVERGENCE_PLAN.md', Path(__file__),
               *sorted((study.REPO/'src/cosmic_integration').rglob('*.py')),
               *sorted((study.HERE.parent/'joint_pilot').glob('*.py')),
               *sorted((study.HERE.parent/'population_stability').glob('*.py')),
               study.HERE.parent/'event_count_bo/run_study.py']
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    study.base.write_json(ROOT/'manifest.json', {'pid': os.getpid(), 'started_unix': time.time(),
        'source_sha256': hashes, 'catalogues_per_group': 100, 'fresh_catalogues': 600,
        'groups': study.GROUPS, 'pilot_groups': [[100, 'perfect'], [1000, 'perfect'], [500, 'uncertain']]})
    status = {'state': 'running', 'phase': 'generate_600', 'pid': os.getpid()}
    def save():
        study.base.write_json(ROOT/'campaign_status.json', status)
    def verify_sources():
        changed = [name for name, digest in hashes.items() if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest]
        if changed:
            raise RuntimeError(f'Source changed between stages: {changed}')
    save()
    try:
        ensemble = study.ConvergenceExperiment(ROOT/'ensemble', 100, 4322, False, study.GROUPS, False)
        ensemble.prepare_catalogues()
        ensemble.progress('awaiting_extension_pilot')
        status.update(phase='extension_pilot', generated_catalogues=600)
        save()
        command = [sys.executable, str(study.HERE/'posterior_convergence.py'), '--outdir', str(ROOT/'pilot'),
                   '--catalogues', '1', '--pilot', '--group', '100:perfect', '--group', '1000:perfect',
                   '--group', '500:uncertain']
        with (ROOT/'pilot.log').open('x') as log:
            child = subprocess.Popen(command, cwd=study.REPO, stdout=log, stderr=subprocess.STDOUT)
            status['child_pid'] = child.pid
            save()
            code = child.wait()
        if code:
            raise RuntimeError(f'Extension pilot execution failed, exit {code}; inspect pilot.log')
        verify_sources()
        status.update(phase='ensemble', child_pid=None)
        save()
        ensemble.run(generate=False)
        status.update(state='complete', phase=None, finished_unix=time.time())
        save()
    except Exception as exc:
        status.update(state='failed', error=str(exc))
        save()
        raise


if __name__ == '__main__':
    main()
