"""Versioned, bounded four-input manuscript campaign; no reference-based acquisition."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def write(path: Path, value: dict) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def build_tasks(jobs: list[dict], parent_tasks: list[dict]) -> list[dict]:
    """16 high-count continuations, eight high-count pilots, 68 lower-count runs."""
    existing = [t['job'] for t in parent_tasks if t['kind'] == 'trajectory']
    names = {j['name'] for j in existing}
    tasks = [dict(job=j, maximum=1884, parent=True, group='high_count')
             for j in existing if j['name'].startswith('count_n500_')]
    for job in jobs:
        if job['group'] != 'count_history' or job['name'] in names:
            continue
        count = int(job['name'].split('_')[1][1:])
        if count == 1000 and any(f'_c{c}_' in job['name'] for c in (200, 201)):
            tasks.append(dict(job=job, maximum=1884, parent=False, group='high_count'))
    for job in jobs:
        if job['group'] != 'count_history' or job['name'] in names:
            continue
        count = int(job['name'].split('_')[1][1:])
        if count <= 200:
            tasks.append(dict(job=job, maximum=1284, parent=False, group='lower_count'))
    return tasks


def build_remaining_tasks(jobs: list[dict], existing_names: set[str]) -> list[dict]:
    """Complete the primary matrix and retain the four original predictive examples."""
    return [dict(job=j, maximum=1284 if j['name'].startswith('predictive_n30_') else 1884,
                 group=j['group']) for j in jobs
            if j['group'] in ('count_history', 'predictive') and j['name'] not in existing_names]


def prepare_remaining(root: Path, original: Path, parent: Path, previous: Path) -> None:
    if (root / 'manifest.json').exists():
        raise ValueError('Campaign already frozen')
    for folder in (parent, previous):
        verify(folder, read(folder / 'checksums.json'))
    old = read(previous / 'manifest.json')
    if old['source_campaign'] != str(original) or old['parent_campaign'] != str(parent):
        raise ValueError('Campaign lineage mismatch')
    verify(original / 'code', read(original / 'code/paper_source_manifest.json'))
    verify(original / 'inputs', old['rate_files'])
    verify(original / 'inputs', old['reference_files'])
    if digest(original / 'inputs/manifest.json') != old['input_manifest_sha256']:
        raise ValueError('Input manifest changed')
    for name in ('warm_validation.py', 'score_checkpoint_history.py'):
        if digest(root / name) != digest(previous / name):
            raise ValueError('Training/scoring policy changed')
    existing = {}
    for t in read(parent / 'manifest.json')['tasks']:
        if t['kind'] == 'trajectory':
            existing[t['job']['name']] = (t['job'], parent / 'trajectories/runs' / t['job']['name'])
    for t in old['tasks']:
        existing[t['job']['name']] = (t['job'], previous / 'production/runs' / t['job']['name'])
    for job, folder in existing.values():
        complete = read(folder / 'complete.json')
        if read(folder / 'status.json')['state'] != 'complete':
            raise ValueError('Previous scoring incomplete')
        if digest(folder / f"checkpoint_{complete['labels']}.npz") != complete['checkpoint_sha256']:
            raise ValueError('Completed checkpoint changed')
    tasks = build_remaining_tasks(read(original / 'inputs/manifest.json')['jobs'], set(existing))
    if len(tasks) != 44 or sum(t['group']=='count_history' for t in tasks) != 40:
        raise ValueError('Unexpected remaining campaign size')
    diagnostics = []
    for n in (10,50,100,200,500,1000):
        for view in ('perfect','uncertain'):
            job, folder = existing[f'count_n{n}_{view}_c200_s17_bo']
            complete = read(folder / 'complete.json')
            labels = complete['labels']
            diagnostics.append(dict(job=job, folder=str(folder), labels=labels,
                checkpoint_sha256=complete['checkpoint_sha256'],
                score_sha256=digest(folder / f'score_{labels}.json'),
                holdout=n in (100,1000)))
    manifest = {k:old[k] for k in ('source_campaign','input_manifest_sha256',
                'source_manifest_sha256','rate_files','reference_files','score_checkpoints')}
    manifest.update(phase='remaining', previous_campaign=str(previous), parent_campaign=str(parent),
        tasks=tasks, diagnostics=diagnostics,
        scope='40 missing primary trajectories and four predictive examples; unchanged acceptance criteria')
    write(root / 'manifest.json', manifest)
    files = ['campaign.py','warm_validation.py','score_checkpoint_history.py','diagnostics.py',
             'run.slurm','submit.sh','manifest.json','README.md']
    write(root / 'checksums.json', {name:digest(root / name) for name in files})


def verify(root: Path, hashes: dict) -> None:
    for relative, expected in hashes.items():
        if digest(root / relative) != expected:
            raise ValueError(f'Checksum mismatch: {root / relative}')


def parent_record(folder: Path, job: dict, inputs_hash: str) -> dict:
    identity = read(folder / 'run.json')
    complete = read(folder / 'complete.json')
    if identity['job'] != job or identity['input_manifest_sha256'] != inputs_hash:
        raise ValueError('Parent job or input identity mismatch')
    if identity['maximum'] != 1284 or complete['labels'] != 1284:
        raise ValueError('Parent must be a completed 1284-label trajectory')
    if digest(folder / 'checkpoint_1284.npz') != complete['checkpoint_sha256']:
        raise ValueError('Parent final checkpoint mismatch')
    files = ['run.json', 'complete.json']
    for n in range(144, 1285, 30):
        files += [f'checkpoint_{n}.npz', f'timing_{n}.json']
    return dict(folder=str(folder), files={name: digest(folder / name) for name in files})


def seed_parent(folder: Path, parent: dict) -> None:
    """Copy immutable parent states, never old completion or run identity markers."""
    source = Path(parent['folder'])
    verify(source, parent['files'])
    folder.mkdir(parents=True, exist_ok=True)
    for name, expected in parent['files'].items():
        if not name.startswith(('checkpoint_', 'timing_')):
            continue
        target = folder / name
        if target.exists():
            if digest(target) != expected:
                raise ValueError(f'Existing child checkpoint changed: {target}')
        else:
            temporary = target.with_suffix('.copying')
            shutil.copyfile(source / name, temporary)
            temporary.replace(target)


def prepare(root: Path, original: Path, parent: Path) -> None:
    if (root / 'manifest.json').exists():
        raise ValueError('Campaign already frozen; choose a new output directory')
    verify(parent, read(parent / 'checksums.json'))
    verify(original / 'code', read(original / 'code/paper_source_manifest.json'))
    for name in ('warm_validation.py', 'score_checkpoint_history.py'):
        if digest(root / name) != digest(parent / name):
            raise ValueError(f'Training/scoring policy differs from parent: {name}')
    if read(parent / 'manifest.json')['source_campaign'] != str(original):
        raise ValueError('Parent campaign uses a different source')
    inputs_hash = digest(original / 'inputs/manifest.json')
    tasks = build_tasks(read(original / 'inputs/manifest.json')['jobs'], read(parent / 'manifest.json')['tasks'])
    if len(tasks) != 92:
        raise ValueError(f'Unexpected stage-one task count: {len(tasks)}')
    for task in tasks:
        if task.pop('parent'):
            task['parent'] = parent_record(parent / 'trajectories/runs' / task['job']['name'], task['job'], inputs_hash)
    manifest = dict(source_campaign=str(original), parent_campaign=str(parent),
                    input_manifest_sha256=inputs_hash, tasks=tasks,
                    source_manifest_sha256=digest(original / 'code/paper_source_manifest.json'),
                    rate_files={name:digest(original / 'inputs' / name) for name in ('operator.npz','redshifts.npy','rates.json')},
                    reference_files={f'references/t0_n{n}_{seed}.npz':digest(original / 'inputs/references' / f't0_n{n}_{seed}.npz') for n in (100,1000) for seed in (0,1)},
                    score_checkpoints=[504,534,744,774,1254,1284,1314,1554,1584,1854,1884],
                    scope='Stage one only; first tested confirmation, not an operational stopping rule')
    write(root / 'manifest.json', manifest)
    files = ['campaign.py','warm_validation.py','score_checkpoint_history.py','run.slurm','submit.sh','manifest.json','README.md']
    write(root / 'checksums.json', {name:digest(root / name) for name in files})


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(root: Path, index: int, pilot: bool) -> None:
    verify(root, read(root / 'checksums.json'))
    manifest = read(root / 'manifest.json')
    original = Path(manifest['source_campaign'])
    inputs = original / 'inputs'
    if digest(inputs / 'manifest.json') != manifest['input_manifest_sha256']:
        raise ValueError('Input manifest changed')
    verify(inputs, manifest['rate_files'])
    verify(inputs, manifest['reference_files'])
    if digest(original / 'code/paper_source_manifest.json') != manifest['source_manifest_sha256']:
        raise ValueError('Source manifest changed')
    verify(original / 'code', read(original / 'code/paper_source_manifest.json'))
    sys.path.insert(0, str(original / 'code/docs/studies/amplitude_gp'))
    import paper_campaign as campaign
    import numpy as np
    warm = load_module('frozen_warm', root / 'warm_validation.py')
    scorer = load_module('frozen_scoring', root / 'score_checkpoint_history.py')
    task = manifest['tasks'][index]
    maximum = (1314 if 'parent' in task else 204) if pilot else task['maximum']
    output = root / ('pilot' if pilot else 'production')
    folder = output / 'runs' / task['job']['name']
    identity = dict(job=task['job'], maximum=maximum, manifest_sha256=digest(root / 'manifest.json'),
                    checksums_sha256=digest(root / 'checksums.json'),
                    input_manifest_sha256=manifest['input_manifest_sha256'], parent=task.get('parent'))
    if (folder / 'run.json').exists() and read(folder / 'run.json') != identity:
        raise ValueError('Child identity changed')
    if 'parent' in task:
        seed_parent(folder, task['parent'])
    elif pilot:
        warm.train(campaign, scorer, output, inputs, task['job'], maximum, identity, stop_after=174)
    if pilot:
        initial_labels = 1284 if 'parent' in task else 174
        path = folder / f'checkpoint_{initial_labels}.npz'
        capacity = 32 * ((task['maximum'] + 31) // 32)
        state, x, y, rng = campaign.restore(path, capacity)
        active, _, _, _ = campaign.restore(path, initial_labels)
        query = np.random.default_rng(17).random((64,4))
        np.testing.assert_allclose(campaign.study.pilot.base.predict(state, query),
                                   campaign.study.pilot.base.predict(active, query), rtol=1e-8, atol=1e-6)
        chosen, _ = campaign.study.acquire_small_batch(state, x, y, rng)
        restored, xx, yy, rr = campaign.restore(path, capacity)
        repeated, _ = campaign.study.acquire_small_batch(restored, xx, yy, rr)
        np.testing.assert_allclose(chosen, repeated, rtol=0, atol=1e-10)
    folder = warm.train(campaign, scorer, output, inputs, task['job'], maximum, identity)
    if pilot:
        warm.train(campaign, scorer, output, inputs, task['job'], maximum, identity)
        write(root / 'pilot_success.json', dict(parent_labels=initial_labels, labels=maximum,
              padded_prediction_parity=True, restored_acquisition_replay=True,
              checkpoint_sha256=digest(folder / f'checkpoint_{maximum}.npz'),
              scope='Continuation implementation check, not scientific acceptance'))
    else:
        checkpoints = [n for n in manifest['score_checkpoints'] if n <= maximum]
        warm.score(campaign, scorer, folder, inputs, task['job'], checkpoints)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare','pilot','run'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--original', type=Path)
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--previous', type=Path, help='Complete the remaining matrix after this stage-one campaign')
    args = parser.parse_args()
    if args.stage == 'prepare':
        if args.original is None or args.parent is None:
            parser.error('prepare requires --original and --parent')
        if args.previous is not None:
            prepare_remaining(args.root.resolve(), args.original.resolve(), args.parent.resolve(), args.previous.resolve())
        else:
            prepare(args.root.resolve(), args.original.resolve(), args.parent.resolve())
    else:
        run(args.root.resolve(), args.index, args.stage == 'pilot')


if __name__ == '__main__':
    main()
