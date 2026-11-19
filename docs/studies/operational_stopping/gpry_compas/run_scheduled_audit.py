"""Replay fixed and adaptive periodic audits without loading reference scores."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import logging
from pathlib import Path
import time

import numpy as np
from scipy.special import logsumexp

import compare_random as run
from cosmic_integration.lnl_surrogate.scheduled_audit import AuditPolicy, posterior_audit

base = run.base
HERE = Path(__file__).resolve().parent
POLICY = AuditPolicy()
PRIORITY = ['t2_n100_uncertain_c0_bo', 't2_n1000_perfect_c0_bo', 't2_n1000_uncertain_c0_bo']


def prepare_proposal(state: dict, unit: np.ndarray, labels: np.ndarray,
                     seed: int) -> tuple[dict | None, list[dict]]:
    """Require resolved GP-only integration; posterior stability is not a gate."""
    checks = []
    for power in [14, 15, 16]:
        proposal = run.gp_designs(state, unit, labels, seed, power)
        predictions = [base.predict(state, p) for p, _ in proposal]
        normalizers = np.array([[logsumexp(pred/t-q)-np.log(2**power) for t in [1., 2.]]
                                for pred, (_, q) in zip(predictions, proposal)])
        summaries = [base.posterior_summary(p, pred-q) for (p, q), pred in zip(proposal, predictions)]
        precision, detail = base.summaries_agree(*summaries, base.TOL)
        difference = float(np.max(abs(normalizers[0]-normalizers[1])))
        numerical = bool(precision and difference < .02)
        checks.append(dict(power=power, numerical=numerical, posterior_precision=detail,
            maximum_log_normalizer_difference=difference, ess=[s['ess'] for s in summaries]))
        if numerical:
            return dict(points=np.vstack([p for p, _ in proposal]),
                prediction=np.concatenate(predictions), logq=np.concatenate([q for _, q in proposal]),
                normalizers=logsumexp(normalizers, axis=0)-np.log(2.)), checks
    return None, checks


def sample_proposal(state: dict, proposal: dict, size: int,
                    rng: np.random.Generator) -> dict:
    """Independent increment, with density of the whole 4:1:1 mixture."""
    if size < 6 or size % 6:
        raise ValueError('Increment must preserve 4:1:1 allocation')
    points, groups = [], []
    for group, (t, count) in enumerate([(1., 4*size//6), (2., size//6)]):
        logw = proposal['prediction']/t-proposal['logq']
        idx = rng.choice(len(logw), count, p=np.exp(logw-logsumexp(logw)))
        points.append(proposal['points'][idx]); groups.extend([group]*count)
    points.append(rng.uniform(size=(size//6, 3))); groups.extend([2]*(size//6))
    points = np.vstack(points)
    pred = base.predict(state, points)
    norm = proposal['normalizers']
    density = logsumexp(np.vstack([np.log(2/3)+pred-norm[0],
        np.log(1/6)+pred/2-norm[1], np.full(size, np.log(1/6))]), axis=0)
    return dict(unit=points, predicted=pred, log_proposal=density, groups=np.array(groups))


def run_case(root: Path, out: Path, loader, name: str, decision: dict) -> dict:
    folder = out/name
    folder.mkdir(exist_ok=True)
    source = root/name
    cat = dict(np.load(source/'catalogue.npz'))
    checkpoints = json.loads((source/'checkpoints.json').read_text())
    rules = {'fixed96': None, 'adaptive384': None}
    costs = dict.fromkeys(rules, 0)
    history = []
    started = time.perf_counter()
    for checkpoint in checkpoints:
        count = checkpoint['points']
        if not POLICY.due(count, decision['training_labels']) or all(rules.values()):
            continue
        path = source/f'checkpoint_{count}.npz'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == checkpoint['sha256']
        state, unit, labels = base.load_checkpoint(path)
        proposal, checks = prepare_proposal(state, unit, labels, decision['seed']+20000)
        record = dict(training=count, checkpoint_sha256=checkpoint['sha256'], numerical_checks=checks,
            diagnostics=dict(counter=checkpoint['counter'], streaks=checkpoint['streaks']), stages=[])
        history.append(record)
        if proposal is None:
            record['action'] = 'numerical_unresolved'
            base.write_json(folder/'progress.json', dict(rules=rules, audit_labels=costs, history=history))
            logging.info('%s n=%d GP numerical unresolved', name, count)
            continue
        rng = np.random.default_rng(np.random.SeedSequence([202609121, decision['truth_index'],
            decision['expected_events'], decision['case'], int(decision['mode']=='uncertain'),
            int(decision['strategy']=='random'), count]))
        pooled = None
        for size in POLICY.sizes:
            previous = 0 if pooled is None else len(pooled['actual'])
            batch = sample_proposal(state, proposal, size-previous, rng)
            # Save the blind query before exact evaluation. No reference access.
            stem = f'audit_{count}_{size}'
            np.savez(folder/f'{stem}_query.npz', **batch)
            physical = base.LOW+base.WIDTH*batch['unit']
            rates = np.array([loader.fast(p[0], p[1], .012, p[2]).ravel()[loader.active] for p in physical])
            batch['actual'] = base.source_target.event_target(rates, cat['coeff'], float(cat['duration']))
            np.savez(folder/f'{stem}_rates.npz', phi=physical, rates=rates)
            pooled = batch if pooled is None else {k: np.concatenate([pooled[k], batch[k]]) for k in batch}
            np.savez(folder/f'{stem}.npz', **pooled)
            result = posterior_audit(pooled['predicted'], pooled['actual'], pooled['log_proposal'],
                pooled['groups'], 202609122+count, POLICY)
            record['stages'].append(dict(size=size, increment=size-previous, result=result,
                path=f'{name}/{stem}.npz'))
            for rule in rules:
                if rules[rule] is None and (rule == 'adaptive384' or size == POLICY.sizes[0]):
                    costs[rule] += size-previous
                    if result['passed']:
                        rules[rule] = dict(training=count, audit=costs[rule], total=count+costs[rule])
            logging.info('%s n=%d audit=%d action=%s KL=%.5g upper=%.5g costs=%s elapsed=%.1fs',
                name, count, size, result['action'], result['maximum_directed_kl'],
                result['bootstrap_95_kl'], costs, time.perf_counter()-started)
            base.write_json(folder/'progress.json', dict(rules=rules, audit_labels=costs, history=history))
            if rules['adaptive384'] is not None or result['action'] != 'enlarge_audit':
                break
    result = {k: decision[k] for k in ['truth_index', 'expected_events', 'case', 'mode', 'strategy']}
    result.update(name=name, rules=rules, audit_labels=costs, training_labels=decision['training_labels'],
        history=history, reference_accessed=False, cohort='development_replay',
        wall_seconds=time.perf_counter()-started)
    base.write_json(out/f'{name}.json', result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    root, out = args.input.resolve(), args.outdir.resolve()
    source_bytes = (root/'decisions.json').read_bytes()
    decisions = json.loads(source_bytes)
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    assert json.loads((root/'status.json').read_text())['decision_sha256'] == source_sha
    sources = [Path(__file__), HERE/'SCHEDULED_AUDIT_PROTOCOL.md', HERE/'compare_random.py',
        HERE/'run_experiment.py', HERE/'check_precision.py',
        base.REPO/'src/cosmic_integration/lnl_surrogate/scheduled_audit.py']
    hashes = {str(p.relative_to(base.REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    policy = json.loads(json.dumps(asdict(POLICY)))
    if args.resume:
        manifest = json.loads((out/'manifest.json').read_text())
        assert manifest['source_sha256'] == hashes and manifest['policy'] == policy
        assert manifest['training_decision_sha256'] == source_sha
    else:
        out.mkdir(parents=True, exist_ok=False)
        for p in sources:
            target = out/'source_snapshot'/p.relative_to(base.REPO)
            target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(p.read_bytes())
        base.write_json(out/'manifest.json', dict(source_sha256=hashes, policy=policy,
            training_root=str(root), training_decision_sha256=source_sha,
            scope='Known-catalogue development replay; new independent audit draws'))
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    order = PRIORITY + sorted((n for n in decisions if n not in PRIORITY),
                             key=lambda n: (decisions[n]['strategy']=='random', n))
    if args.limit:
        order = order[:args.limit]
    completed = {p.stem: json.loads(p.read_text()) for p in out.glob('t*_*.json')}
    loader = None
    for name in order:
        if name in completed:
            continue
        if loader is None:
            loader = base.data_source.Experiment(out/f'population_{time.time_ns()}', 'scheduled_audit_no_reference', False)
        completed[name] = run_case(root, out, loader, name, decisions[name])
        base.write_json(out/'decisions.json', completed)
        base.write_json(out/'status.json', dict(state='running', completed=len(completed), last_case=name))
    assert (root/'decisions.json').read_bytes() == source_bytes
    sha = hashlib.sha256((out/'decisions.json').read_bytes()).hexdigest()
    base.write_json(out/'status.json', dict(state='complete' if len(completed)==len(decisions) else 'pilot_complete',
        completed=len(completed), expected=len(decisions), decision_sha256=sha))


if __name__ == '__main__':
    main()
