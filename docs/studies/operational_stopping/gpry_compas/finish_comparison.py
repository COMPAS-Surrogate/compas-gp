"""Run audit, scoring, verification and plotting after training has completed."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--resume-scoring', action='store_true', help='Resume scoring after frozen training and audits')
    args = parser.parse_args(); here = Path(__file__).resolve().parent
    state_path = args.input/'pipeline_status.json'
    def state(phase, **extra):
        temporary = state_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(phase=phase, **extra), indent=2)+'\n')
        temporary.replace(state_path)
    state('waiting_for_training')
    while True:
        path = args.input/'status.json'
        status = json.loads(path.read_text())
        if status['state'] == 'complete':
            break
        if status['state'] == 'failed' or time.time()-path.stat().st_mtime > 1800:
            state('failed', reason='Training failed or made no progress for 30 minutes')
            raise RuntimeError('Training did not complete')
        time.sleep(5)
    scripts = ['score_comparison.py', 'verify_comparison.py', 'publish_comparison.py']
    if not args.resume_scoring:
        scripts.insert(0, 'run_mass_audit.py')
    for script in scripts:
        state('running', script=script)
        command = [sys.executable, str(here/script), '--input', str(args.input)]
        if args.resume_scoring and script == 'score_comparison.py':
            command.append('--resume')
        if script == 'publish_comparison.py':
            command += ['--outdir', str(args.outdir)]
        suffix = 'resume_v2' if args.resume_scoring else 'v2'
        with (here/f'{Path(script).stem}_{suffix}.log').open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
        if result.returncode:
            state('failed', script=script, returncode=result.returncode)
            raise RuntimeError(f'{script} failed; inspect its log')
    state('complete')


if __name__ == '__main__':
    main()
