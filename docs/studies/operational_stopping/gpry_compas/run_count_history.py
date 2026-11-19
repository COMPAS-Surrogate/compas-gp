"""Small paired count iterations with frozen stops and complete posterior histories.

Training and blind audit replay finish before reference scoring for each job.
The fixed-budget continuation is diagnostic: it does not change the selected stop.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
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
from scipy.special import logsumexp

import compare_random as train
import run_stable_scheduled_audit as audit
import score_scheduled_audit as validation
from cosmic_integration.lnl_surrogate.scheduled_audit import AuditPolicy

BASE = train.base
HERE = Path(__file__).resolve().parent
COUNTS = (10, 50, 100, 200, 500, 700, 1000)
MODES = ('perfect', 'uncertain')


def jobs(cases: int) -> list[dict]:
    """Visit all counts and both models before adding the next paired catalogue."""
    return [dict(name=f't0_n{n}_{mode}_c{case}_bo', truth_index=0,
                 expected_events=n, mode=mode, case=case, strategy='bo', iteration=i+1)
            for i, case in enumerate(range(200, 200+cases)) for n in COUNTS for mode in MODES]


def hashes(frozen: dict[str, str] | None = None) -> dict[str, str]:
    """Hash launch inputs; unrelated files added later are not campaign inputs."""
    if frozen is not None:
        return {name: hashlib.sha256((BASE.REPO/name).read_bytes()).hexdigest()
                for name in frozen}
    sources = [*HERE.glob('*.py'), HERE/'COUNT_HISTORY.md',
               *sorted((BASE.REPO/'src/cosmic_integration').rglob('*.py'))]
    for folder in ('event_uncertainty', 'event_count_bo', 'joint_pilot', 'population_stability'):
        sources.extend((HERE.parents[1]/folder).glob('*.py'))
    return {str(p.relative_to(BASE.REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources}


def stop_record(decision: dict, checkpoints: list[dict], maximum: int) -> dict:
    """Keep the selected stopping cost distinct from the diagnostic endpoint."""
    selected = decision['rules']['fixed96']
    by_count = {c['points']: c for c in checkpoints}
    cp = by_count[selected['training']] if selected else None
    return dict(state='stopped' if selected else 'resource_limit', selected=selected,
                posterior_evaluation=cp['posterior_evaluation'] if cp else None,
                posterior_comparisons=cp['posterior_comparisons'] if cp else None,
                bo_batches=cp['bo_batches'] if cp else None,
                trajectory_elapsed_seconds=cp['elapsed_seconds'] if cp else None,
                trajectory_checkpoint_unix=cp['recorded_unix'] if cp else None,
                audit_replay_seconds=decision['wall_seconds'],
                decision_recorded_unix=time.time(), diagnostic_endpoint=maximum,
                post_stop_labels=maximum-selected['training'] if selected else None,
                stop_is_shadow_replay=True, reference_accessed=False)


def cached_references(n: int, cat: dict) -> tuple[list, tuple, dict]:
    """Reuse existing full-support designs, testing precision for this new target.

The design's old expected count controls its proposal only; catalogue coefficients
and duration always come from the current case. No likelihood interpolation.
"""
    attempts, provenance = [], {}
    choices = (100, 1000) if n <= 200 else (1000, 100)
    for source_count in choices:
        refs = validation.score.cached_references(0, source_count, provenance)
        checked = validation.score.direct_checks(refs, cat, cat['truth'])
        attempts.append(dict(proposal_expected_events=source_count, check=checked[-1]))
        if checked[-1]['passed']:
            break
    return refs, checked, dict(attempts=attempts, source_sha256=provenance,
                               proposal_expected_events=source_count,
                               precision_pass=checked[-1]['passed'])


def score_history(root: Path, job: dict, maximum: int, frozen: bytes) -> dict:
    """Save every GP posterior and both KL directions against direct and late GP."""
    folder = root/'training'/job['name']
    out = root/'offline'/job['name']; out.mkdir(parents=True, exist_ok=True)
    cat = dict(np.load(folder/'catalogue.npz'))
    refs, checked, provenance = cached_references(job['expected_events'], cat)
    BASE.write_json(out/'reference.json', provenance)
    last = folder/f'checkpoint_{maximum}.npz'
    final_score = validation.score_checkpoint(last, refs, cat, cat['truth'], checked,
                                               dict(job, points=maximum))
    accepted = bool(final_score['accuracy_pass'] and final_score['maximum_directed_kl'] < .001)
    BASE.write_json(out/'late_reference.json', dict(points=maximum, accepted=accepted,
        checkpoint_sha256=hashlib.sha256(last.read_bytes()).hexdigest(), score=final_score))
    state, _, _ = BASE.load_checkpoint(last)
    late = [BASE.predict(state, (p-BASE.LOW)/BASE.WIDTH)-q for p, _, q in refs]
    np.savez_compressed(out/'reference_weights.npz',
                        **{f'direct_{s}': w-logsumexp(w) for s, w in enumerate(checked[0])},
                        **{f'late_gp_{s}': w-logsumexp(w) for s, w in enumerate(late)})
    checkpoints = json.loads((folder/'checkpoints.json').read_text())
    rows = []
    for cp in checkpoints:
        count = cp['points']; path = folder/f'checkpoint_{count}.npz'
        if hashlib.sha256(path.read_bytes()).hexdigest() != cp['sha256']:
            raise ValueError(f'Checkpoint hash mismatch: {path}')
        row = final_score.copy() if count == maximum else validation.score_checkpoint(
            path, refs, cat, cat['truth'], checked, dict(job, points=count))
        state, _, _ = BASE.load_checkpoint(path)
        weights = [BASE.predict(state, (p-BASE.LOW)/BASE.WIDTH)-q for p, _, q in refs]
        kls = [validation.score.directed_kls(w, f) for w, f in zip(weights, late)]
        # Average of directed KLs is used for the plotted symmetric KL. Keep
        # the conservative maximum and each scramble/direction explicitly too.
        row.update(posterior_evaluation=cp['posterior_evaluation'],
            posterior_comparisons=cp['posterior_comparisons'], bo_batches=cp['bo_batches'],
            elapsed_seconds=cp['elapsed_seconds'], checkpoint_sha256=cp['sha256'],
            kl_to_late_gp=[dict(gp_to_late=k[0], late_to_gp=k[1]) for k in kls],
            symmetric_kl_to_late_gp=float(np.mean(kls)),
            maximum_directed_kl_to_late_gp=float(np.max(kls)),
            late_reference_accepted=accepted, labels_to_late_reference=maximum-count,
            late_reference_is_self=(count == maximum))
        np.savez_compressed(out/f'posterior_{count}.npz',
                            **{f'logweights_{s}': w-logsumexp(w) for s, w in enumerate(weights)})
        rows.append(row)
        BASE.write_json(out/'history.json', rows)
        BASE.write_json(root/'job_status.json', dict(state='scoring', **job,
                       points=count, checkpoints_scored=len(rows), updated_unix=time.time()))
    if (root/'stops'/f'{job["name"]}.json').read_bytes() != frozen:
        raise ValueError('Offline scoring changed the frozen stop')
    stop = json.loads(frozen)
    selected = stop['selected']
    endpoint = next(r for r in rows if r['points'] == (selected['training'] if selected else maximum))
    resolved = endpoint['reference_numerical'] and endpoint['gp_numerical'] and endpoint['four_numerical']
    outcome = ('no_stop' if selected is None else 'unresolved' if not resolved else
               'accurate' if endpoint['accuracy_pass'] else 'premature')
    result = dict(job, outcome=outcome, stop=stop, late_reference_accepted=accepted,
                  checkpoints=len(rows), maximum_directed_kl=endpoint['maximum_directed_kl'],
                  symmetric_kl_to_late_gp=endpoint['symmetric_kl_to_late_gp'])
    BASE.write_json(out/'outcome.json', result)
    return result


def worker(root: Path, index: int) -> None:
    manifest = json.loads((root/'manifest.json').read_text())
    if hashes(manifest['source_sha256']) != manifest['source_sha256']:
        raise RuntimeError('Sources changed since launch; refusing an unfrozen continuation')
    job = manifest['jobs'][index]; maximum = manifest['maximum']
    name = job['name']
    BASE.write_json(root/'job_status.json', dict(state='training', pid=os.getpid(), **job,
                   started_unix=time.time()))
    loader = BASE.data_source.Experiment(root/f'population_{name}_{time.time_ns()}',
                                        'count_history_no_reference', False)
    tr = root/'training'; tr.mkdir(exist_ok=True)
    decision = train.run_case(tr, loader, 0, job['expected_events'], job['mode'], job['case'], 'bo', maximum)
    # This is a shadow stopping experiment: replay only training-time GP state
    # and fresh exact audits, and freeze the answer before opening any reference.
    audit.POLICY = AuditPolicy(sizes=(96,))
    ar = root/'audits'; ar.mkdir(exist_ok=True)
    audited = audit.run_case(tr, ar, loader, name, decision)
    checkpoints = json.loads((tr/name/'checkpoints.json').read_text())
    stop = stop_record(audited, checkpoints, maximum)
    stops = root/'stops'; stops.mkdir(exist_ok=True)
    BASE.write_json(stops/f'{name}.json', stop)
    frozen = (stops/f'{name}.json').read_bytes()
    # Measurement pairs use the same latent catalogue seed. Check actual files,
    # not merely configuration, before declaring the second member complete.
    if job['mode'] == 'uncertain':
        other = tr/name.replace('_uncertain_', '_perfect_')/'catalogue.npz'
        with np.load(other) as p, np.load(tr/name/'catalogue.npz') as u:
            for key in ('latent', 'scales', 'y', 'rho', 'bins', 'duration', 'truth'):
                np.testing.assert_array_equal(p[key], u[key], err_msg=f'Pair mismatch: {key}')
        BASE.write_json(tr/name/'pairing.json', dict(passed=True, perfect=str(other),
                       actual_events=decision['actual_events']))
    score_history(root, job, maximum, frozen)
    BASE.write_json(root/'job_status.json', dict(state='complete', **job, finished_unix=time.time()))


def campaign(root: Path, cases: int, maximum: int, resume: bool = False) -> None:
    root.mkdir(parents=True, exist_ok=resume)
    previous = json.loads((root/'manifest.json').read_text()) if resume else None
    source = hashes(previous['source_sha256'] if previous else None); matrix = jobs(cases)
    manifest = dict(expected_counts=COUNTS, cases_per_count=cases, jobs=matrix, maximum=maximum,
        source_sha256=source, policy=asdict(AuditPolicy(sizes=(96,))),
        checkpoint_interval=30, acquisition_batch=3, initial_labels=58,
        scope='Fixed fiducial injection, paired noisy catalogues; shadow operational stops',
        reference='Actual final continued GP at maximum; independently checked with cached direct grids',
        reference_precision_failure='Retain unresolved; never silently call convergence',
        cost='Training, legacy monitor audits, fixed96 audit replay and offline reference scoring are separate')
    manifest = json.loads(json.dumps(manifest))
    if resume:
        if json.loads((root/'manifest.json').read_text()) != manifest:
            raise ValueError('Resume configuration/source mismatch')
    else:
        BASE.write_json(root/'manifest.json', manifest)
        for name in source:
            dest = root/'source_snapshot'/name; dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(BASE.REPO/name, dest)
    status = dict(state='running', pid=os.getpid(), total_jobs=len(matrix),
                  completed=0, failed=0, started_unix=time.time(), expected_counts=COUNTS)
    results = []
    for index, job in enumerate(matrix):
        result_path = root/'offline'/job['name']/'outcome.json'
        if not result_path.exists():
            if shutil.disk_usage(root).free < 5*2**30:
                raise RuntimeError('Less than 5 GiB free; campaign stopped before the next job')
            # Keep interrupted files intact; explicit operator action is required
            # before retrying a partially trained case, so no evidence is erased.
            if (root/'training'/job['name']).exists():
                raise RuntimeError(f'Incomplete existing job: {job["name"]}')
            status.update(current_job=job, updated_unix=time.time())
            BASE.write_json(root/'campaign_status.json', status)
            log_path = root/f'{job["name"]}.log'
            with log_path.open('x') as log:
                child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                    '--root', str(root), '--worker', str(index)], cwd=BASE.REPO,
                    stdout=log, stderr=subprocess.STDOUT)
                status.update(child_pid=child.pid)
                BASE.write_json(root/'campaign_status.json', status)
                code = child.wait()
            if code:
                status.update(state='failed', failed=status['failed']+1, returncode=code,
                              finished_unix=time.time())
                BASE.write_json(root/'campaign_status.json', status)
                raise RuntimeError(f'{job["name"]} failed; see {log_path}')
        results.append(json.loads(result_path.read_text()))
        status.update(completed=len(results), child_pid=None)
        BASE.write_json(root/'outcomes.json', results)
        BASE.write_json(root/'campaign_status.json', status)
    status.update(state='complete', current_job=None, finished_unix=time.time())
    BASE.write_json(root/'campaign_status.json', status)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--cases', type=int, default=5)
    parser.add_argument('--maximum', type=int, default=1258)
    parser.add_argument('--worker', type=int)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.cases < 1 or args.maximum < 88 or (args.maximum-58) % 30:
        parser.error('Use positive cases and maximum = 58 + a positive multiple of 30')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    if args.worker is None:
        campaign(args.root.resolve(), args.cases, args.maximum, args.resume)
    else:
        worker(args.root.resolve(), args.worker)


if __name__ == '__main__':
    main()
