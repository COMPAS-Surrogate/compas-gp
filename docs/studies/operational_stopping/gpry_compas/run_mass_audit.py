"""Mass-weighted stopping on frozen trajectories, before reference scoring."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import time

import numpy as np
from scipy.special import logsumexp

import compare_random as run
from cosmic_integration.lnl_surrogate.posterior_audit import importance_audit_check

HERE = Path(__file__).resolve().parent
base = run.base


def proposal_sample(state: dict, proposal: list, power: int,
                    rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Sample the mixture and freeze its GP-based density before exact calls."""
    predictions = [base.predict(state, p) for p, _ in proposal]
    normalizers = np.array([[logsumexp(pred/t-q)-np.log(2**power) for t in [1., 2.]]
                            for pred, (_, q) in zip(predictions, proposal)])
    delta = float(np.max(abs(normalizers[0]-normalizers[1])))
    info = dict(power=power, log_normalizers=normalizers.tolist(), numerical=delta < .02,
                maximum_log_normalizer_difference=delta)
    if not info['numerical']:
        return np.empty((0, 3)), np.empty(0), np.empty(0), info
    norm = logsumexp(normalizers, axis=0)-np.log(2.)
    points = np.vstack([p for p, _ in proposal])
    pred = np.concatenate(predictions); logq = np.concatenate([q for _, q in proposal])
    selected = []
    for t, count in [(1., 64), (2., 16)]:
        logw = pred/t-logq
        selected.append(points[rng.choice(len(points), count, p=np.exp(logw-logsumexp(logw)))])
    selected = np.vstack([*selected, rng.uniform(size=(16, 3))])
    prediction = base.predict(state, selected)
    log_density = logsumexp(np.vstack([np.log(2/3)+prediction-norm[0],
        np.log(1/6)+prediction/2-norm[1], np.full(96, np.log(1/6))]), axis=0)
    return selected, prediction, log_density, info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    args = parser.parse_args(); root = args.input
    assert json.loads((root/'status.json').read_text())['state'] == 'complete'
    assert not (root/'offline').exists(), 'Freeze all audit decisions before reference scoring'
    frozen = (root/'decisions.json').read_bytes(); training_sha = hashlib.sha256(frozen).hexdigest()
    assert json.loads((root/'status.json').read_text())['decision_sha256'] == training_sha
    decisions = json.loads(frozen); out = root/'mass_audit'; out.mkdir(exist_ok=False)
    freeze = json.loads((HERE/'mass_audit_freeze.json').read_text())
    for path, sha in freeze['source_sha256'].items():
        assert hashlib.sha256((base.REPO/path).read_bytes()).hexdigest() == sha
    sources = [Path(__file__), HERE/'MASS_AUDIT_PROTOCOL.md', HERE/'mass_audit_freeze.json',
               base.REPO/'src/cosmic_integration/lnl_surrogate/posterior_audit.py']
    hashes = {}
    for path in sources:
        rel = path.relative_to(base.REPO); target = out/'source_snapshot'/rel
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(path.read_bytes())
        hashes[str(rel)] = hashlib.sha256(path.read_bytes()).hexdigest()
    base.write_json(out/'manifest.json', dict(training_decision_sha256=training_sha,
                                            source_sha256=hashes, freeze=freeze))
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    loader = base.data_source.Experiment(out/'population', 'mass_audit_no_references', False)
    result = {}; started = time.perf_counter()
    for name, decision in decisions.items():
        folder = root/name; cat = dict(np.load(folder/'catalogue.npz'))
        rng = np.random.default_rng(np.random.SeedSequence([202609115, decision['truth_index'],
            decision['expected_events'], decision['case'], int(decision['mode'] == 'uncertain'),
            int(decision['strategy'] == 'random')]))
        audits, numerical_checks, selected_stop = [], [], None
        for checkpoint in json.loads((folder/'checkpoints.json').read_text()):
            count = checkpoint['points']; streak = checkpoint['streaks']
            if (checkpoint['counter'] < 4 or streak['gaussian'] < 2 or streak['joint'] < 2
                    or (audits and count-audits[-1]['training'] < 60)):
                continue
            state, unit, labels = base.load_checkpoint(folder/f'checkpoint_{count}.npz')
            start_power = checkpoint['stability']['attempts'][-1]['power']
            for power in range(start_power, 17):
                proposal = run.gp_designs(state, unit, labels, decision['seed']+20000, power)
                points, predicted, density, check = proposal_sample(state, proposal, power, rng)
                numerical_checks.append(dict(training=count, **check))
                if check['numerical']:
                    break
            if not check['numerical']:
                continue
            # points/predictions/density above cannot access any of these labels.
            physical = base.LOW+base.WIDTH*points
            rates = np.array([loader.fast(p[0], p[1], .012, p[2]).ravel()[loader.active] for p in physical])
            actual = base.source_target.event_target(rates, cat['coeff'], float(cat['duration']))
            metric = importance_audit_check(predicted, actual, density, 202609116+count)
            audits.append(dict(training=count, points=points.tolist(), predicted=predicted.tolist(),
                actual=actual.tolist(), log_proposal=density.tolist(), result=metric,
                checkpoint_sha256=checkpoint['sha256']))
            logging.info('%s n=%s audit=%s pass=%s RMS=%.4g KL=%.4g upper=%.4g', name, count,
                len(audits), metric['passed'], metric['posterior_rms'], metric['maximum_directed_kl'], metric['bootstrap_95_kl'])
            if metric['passed']:
                selected_stop = dict(training=count, audit=96*len(audits), total=count+96*len(audits))
                break
            if len(audits) == 4:
                break
        result[name] = dict(name=name, selected=selected_stop, audit_labels=96*len(audits),
            training_labels=decision['training_labels'], reference_accessed=False,
            cohort='development' if decision['truth_index'] == 0 else 'confirmation')
        base.write_json(out/f'{name}.json', dict(audits=audits, numerical_checks=numerical_checks, **result[name]))
        base.write_json(out/'decisions.json', result)
        base.write_json(out/'status.json', dict(state='running', case=name, completed=len(result)))
    sha = hashlib.sha256((out/'decisions.json').read_bytes()).hexdigest()
    assert frozen == (root/'decisions.json').read_bytes()
    base.write_json(out/'status.json', dict(state='complete', cases=len(result), decision_sha256=sha,
                                          wall_seconds=time.perf_counter()-started))


if __name__ == '__main__':
    main()
