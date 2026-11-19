"""Replay counters, audit decisions, exact-call accounting, and pairing."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

import compare_random as run
from cosmic_integration.lnl_surrogate.posterior_audit import importance_audit_check


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    args = parser.parse_args()
    root = args.input; base = run.base
    manifest = json.loads((root/'manifest.json').read_text())
    for relative, sha in manifest['source_sha256'].items():
        assert hashlib.sha256((root/'source_snapshot'/relative).read_bytes()).hexdigest() == sha
        assert hashlib.sha256((base.REPO/relative).read_bytes()).hexdigest() == sha
    frozen = (root/'decisions.json').read_bytes(); sha = hashlib.sha256(frozen).hexdigest()
    assert json.loads((root/'status.json').read_text())['decision_sha256'] == sha
    decisions = json.loads(frozen)
    mass_root = root/'mass_audit'
    mass_manifest = json.loads((mass_root/'manifest.json').read_text())
    assert mass_manifest['training_decision_sha256'] == sha
    for relative, source_sha in mass_manifest['source_sha256'].items():
        assert hashlib.sha256((mass_root/'source_snapshot'/relative).read_bytes()).hexdigest() == source_sha
        assert hashlib.sha256((base.REPO/relative).read_bytes()).hexdigest() == source_sha
    mass_frozen = (mass_root/'decisions.json').read_bytes()
    mass_sha = hashlib.sha256(mass_frozen).hexdigest()
    assert json.loads((mass_root/'status.json').read_text())['decision_sha256'] == mass_sha
    mass_decisions = json.loads(mass_frozen)
    totals = Counter(); maximum = 0.
    paired = {}
    for name, d in decisions.items():
        folder = root/name
        checkpoints = json.loads((folder/'checkpoints.json').read_text())
        batches = json.loads((folder/'batches.json').read_text())
        audits = json.loads((folder/'audits.json').read_text())
        with np.load(folder/'checkpoint_58.npz') as data:
            unit, labels = data['unit'], data['lnl']
        cat = dict(np.load(folder/'catalogue.npz'))
        key = (d['truth_index'], d['expected_events'], d['mode'], d['case'])
        if key in paired:
            a, b, c = paired[key]
            np.testing.assert_array_equal(a, unit); np.testing.assert_array_equal(b, labels)
            for k in cat:
                np.testing.assert_array_equal(c[k], cat[k])
            totals['matched_pairs'] += 1
        else:
            paired[key] = (unit.copy(), labels.copy(), cat)
        counter = base.CorrectCounter(3); streaks_at = {58: 0}
        for batch in batches:
            assert batch['before_points'] == len(unit)
            assert batch['best_before'] == float(labels.max())
            got = counter.update(np.asarray(batch['prediction']), np.asarray(batch['actual']), batch['best_before'])
            assert got == batch['counter']
            unit = np.vstack([unit, batch['unit']]); labels = np.r_[labels, batch['actual']]
            assert np.all((unit[-3:] >= 0) & (unit[-3:] <= 1))
            streaks_at[len(unit)] = counter.streak
            totals['batches'] += 1
        next_batch = {b['before_points']: b for b in batches}
        audit_map = {a['training']: a for a in audits}
        streaks = Counter(); rules = {r: None for r in run.RULES}; audit_count = 0
        for checkpoint in checkpoints:
            n = checkpoint['points']; path = folder/f'checkpoint_{n}.npz'
            assert hashlib.sha256(path.read_bytes()).hexdigest() == checkpoint['sha256']
            state, x, y = base.load_checkpoint(path)
            np.testing.assert_array_equal(x, unit[:n]); np.testing.assert_array_equal(y, labels[:n])
            assert checkpoint['counter'] == streaks_at[n]
            if n in next_batch:
                b = next_batch[n]
                pred = base.predict(state, np.asarray(b['unit']))
                maximum = max(maximum, float(np.max(abs(pred-b['prediction']))))
                np.testing.assert_allclose(pred, b['prediction'], atol=1e-7, rtol=0)
                totals['predictions_checked'] += len(pred)
            check = checkpoint['stability']; original = check['attempts'][0]
            for k, passed in [('original', original['gaussian_pass']),
                              ('gaussian', check['gaussian_pass']), ('joint', check['joint_pass'])]:
                streaks[k] = streaks[k]+1 if passed else 0
            assert dict(streaks) == checkpoint['streaks']
            for rule, k in [('original', 'original'), ('precision_refined', 'gaussian')]:
                if rules[rule] is None and counter.required <= streaks_at[n] and streaks[k] >= 2:
                    rules[rule] = dict(training=n, audit=0, total=n)
            if n in audit_map:
                a = audit_map[n]
                assert rules['guarded'] is None and streaks['gaussian'] >= 2 and streaks['joint'] >= 2
                assert streaks_at[n] >= counter.required
                pred = base.predict(state, np.asarray(a['points']))
                np.testing.assert_allclose(pred, a['predicted'], atol=1e-7, rtol=0)
                got = run.audit_check(np.asarray(a['predicted']), np.asarray(a['actual']),
                                      64, float(y.max()), run.AUDIT_TOL)
                assert got == a['result']
                if audit_count:
                    assert n-audits[audit_count-1]['training'] >= 60
                audit_count += 1
                if got['passed']:
                    rules['guarded'] = dict(training=n, audit=96*audit_count, total=n+96*audit_count)
            assert rules == checkpoint['rules']
            assert checkpoint['audit_calls'] == 96*audit_count
            totals['checkpoints'] += 1
        assert rules == d['rules'] and audit_count == len(audits) <= 4
        assert len(unit) == d['training_labels'] and 96*audit_count == d['audit_labels']
        totals['training_labels'] += len(unit); totals['audit_labels'] += 96*audit_count
        mass = json.loads((mass_root/f'{name}.json').read_text())
        mass_stop = None
        by_n = {c['points']: c for c in checkpoints}
        for index, audit in enumerate(mass['audits']):
            n = audit['training']; c = by_n[n]
            assert mass_stop is None and c['counter'] >= 4
            assert c['streaks']['gaussian'] >= 2 and c['streaks']['joint'] >= 2
            assert c['sha256'] == audit['checkpoint_sha256']
            if index:
                assert n-mass['audits'][index-1]['training'] >= 60
            state, _, _ = base.load_checkpoint(folder/f'checkpoint_{n}.npz')
            pred = base.predict(state, np.asarray(audit['points']))
            np.testing.assert_allclose(pred, audit['predicted'], rtol=0, atol=1e-7)
            normalizer_check = [v for v in mass['numerical_checks'] if v['training'] == n and v['numerical']][-1]
            normalizers = np.asarray(normalizer_check['log_normalizers'])
            norm = np.logaddexp(normalizers[0], normalizers[1])-np.log(2.)
            density = np.logaddexp.reduce(np.vstack([np.log(2/3)+pred-norm[0],
                np.log(1/6)+pred/2-norm[1], np.full(len(pred), np.log(1/6))]), axis=0)
            np.testing.assert_allclose(density, audit['log_proposal'], rtol=0, atol=1e-7)
            metric = importance_audit_check(np.asarray(audit['predicted']), np.asarray(audit['actual']),
                                           np.asarray(audit['log_proposal']), 202609116+n)
            assert metric == audit['result']
            if metric['passed']:
                mass_stop = dict(training=n, audit=96*(index+1), total=n+96*(index+1))
        assert mass_stop == mass_decisions[name]['selected'] == mass['selected']
        assert len(mass['audits']) <= 4
        assert mass['audit_labels'] == mass_decisions[name]['audit_labels'] == 96*len(mass['audits'])
        totals['mass_audit_labels'] += mass['audit_labels']
    base.write_json(root/'verification.json', dict(state='passed', decision_sha256=sha,
        mass_decision_sha256=mass_sha,
        source_files=len(manifest['source_sha256']), max_prediction_difference=maximum, **totals))


if __name__ == '__main__':
    main()
