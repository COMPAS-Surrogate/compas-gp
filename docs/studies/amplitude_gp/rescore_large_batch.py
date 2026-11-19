"""Apply refined amplitude integration to the unchanged historical 400-label GPs."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import small_batch_followup as study


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    old = {r['name']: r for r in json.loads((study.BASELINE/'training.json').read_text())}
    rows, cached, key = [], None, None
    for job in study.jobs():
        source = study.BASELINE/job['name']/'checkpoint_400.npz'
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        assert digest == old[job['name']]['checkpoint_sha256']
        current = (job['expected'], job['mode'])
        if current != key:
            cached, key = study.prepare_references(job), current
        refs, metadata = cached
        state, _, _ = study.load_state(source)
        attempts = []
        for replicas in (16, 64):
            result = study.score_once(state, refs, metadata, replicas, 2026091500+job['expected'])
            attempts.append(result)
            if result['outcome'] != 'unresolved':
                break
        row = dict(**job, labels=400, checkpoint_sha256=digest, attempts=attempts, **attempts[-1])
        rows.append(row)
        study.pilot.base.write_json(root/'results.json', rows)
        logging.info('Rescored %s: %s KL %.5g', job['name'], row['outcome'],
                     max(max(s['directed_kl']) for s in row['scores']))
    study.pilot.base.write_json(root/'status.json', dict(complete=True, trajectories=24,
        original_checkpoints_unchanged=True, new_training_labels=0,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))


if __name__ == '__main__':
    main()
