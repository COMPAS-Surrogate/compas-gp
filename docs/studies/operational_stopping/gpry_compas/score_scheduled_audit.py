"""Validate frozen periodic-audit stops using cached COMPAS reference rates."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import logging
from pathlib import Path

import numpy as np

import score_comparison as score
from cosmic_integration.lnl_surrogate.scheduled_audit import AuditPolicy, posterior_audit
from cosmic_integration.lnl_surrogate.stability_audit import stable_audit_acceptance

base = score.base


def references(root: Path, ti: int, n: int, hashes: dict) -> list:
    """Reuse the finest completed refinement; never generate another rate grid."""
    if ti == 2:
        for power in [18, 17]:
            paths = [root/'offline'/f'refined_t{ti}_n{n}_c1_p{power}_{s}.npz' for s in [0, 1]]
            if all(p.exists() for p in paths):
                result = []
                for p in paths:
                    hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
                    with np.load(p) as d:
                        result.append((d['phi'], d['rates'], d['logq']))
                return result
    return score.cached_references(ti, n, hashes)


def score_checkpoint(path: Path, refs: list, cat: dict, truth: np.ndarray,
                     direct_info: tuple, original: dict) -> dict:
    exact, direct, direct_four, refcheck = direct_info
    state, _, _ = base.load_checkpoint(path)
    predicted = [base.predict(state, (p-base.LOW)/base.WIDTH)-q for p, _, q in refs]
    gp = [base.posterior_summary((p-base.LOW)/base.WIDTH, w) for (p, _, _), w in zip(refs, predicted)]
    gp_ok, gp_check = base.summaries_agree(*gp, base.TOL)
    kls = [score.directed_kls(a, b) for a, b in zip(exact, predicted)]
    metrics = [score.agreement(a, b, k) for a, b, k in zip(direct, gp, kls)]
    gpf = [score.four_summary(p, r, cat, truth, w) for (p, r, _), w in zip(refs, predicted)]
    four_ok, four_check = base.data_source.precision(*gpf)
    fm = [score.four_accuracy(a, b) for a, b in zip(direct_four, gpf)]
    result = {k: original[k] for k in ['name', 'truth_index', 'expected_events', 'mode', 'case', 'strategy', 'points']}
    result.update(reference_numerical=refcheck['passed'], reference_check=refcheck,
        gp_numerical=gp_ok, gp_check=gp_check, four_numerical=four_ok, four_check=four_check,
        four_checked=True, four_metrics=fm, four_summaries={'direct': direct_four, 'gp': gpf},
        metrics=metrics, symmetric_kl=float(np.mean(kls)), maximum_directed_kl=float(np.max(kls)),
        accuracy_pass=bool(refcheck['passed'] and gp_ok and four_ok
                           and all(m['passed'] for m in metrics) and all(m['passed'] for m in fm)),
        scoring_source='New four-parameter validation at a frozen selected checkpoint')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    args = parser.parse_args()
    folder = args.input.resolve()
    manifest = json.loads((folder/'manifest.json').read_text())
    root = Path(manifest['training_root'])
    frozen = (folder/'decisions.json').read_bytes()
    sha = hashlib.sha256(frozen).hexdigest()
    status = json.loads((folder/'status.json').read_text())
    assert status['state'] == 'complete' and status['decision_sha256'] == sha
    decisions = json.loads(frozen)
    assert all(not d['reference_accessed'] for d in decisions.values())
    out = folder/'offline'; out.mkdir(exist_ok=True)
    old = json.loads((root/'offline/records.json').read_text())
    original = {(r['name'], r['points']): r for r in old}
    policy = AuditPolicy(**{**manifest['policy'], 'sizes': tuple(manifest['policy']['sizes'])})
    # Reconstruct every audit from persisted observations and verify costs.
    verified_stages = 0
    for name, decision in decisions.items():
        accumulated = {'fixed96': 0, 'adaptive384': 0}
        accepted = dict.fromkeys(accumulated)
        for checkpoint in decision['history']:
            for stage in checkpoint['stages']:
                with np.load(folder/stage['path']) as d:
                    metric = posterior_audit(d['predicted'], d['actual'], d['log_proposal'],
                                             d['groups'], 202609122+checkpoint['training'], policy)
                assert metric == stage['result']
                for rule in accumulated:
                    if accepted[rule] is None and (rule == 'adaptive384' or stage['size'] == 96):
                        accumulated[rule] += stage['increment']
                        accepted_here = (stable_audit_acceptance(metric, checkpoint['diagnostics']['streaks']['joint'])
                                         if manifest.get('require_joint_stability') else metric['passed'])
                        if accepted_here:
                            n = checkpoint['training']
                            accepted[rule] = dict(training=n, audit=accumulated[rule], total=n+accumulated[rule])
                verified_stages += 1
        assert accumulated == decision['audit_labels'] and accepted == decision['rules']
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    current_key, refs, direct_cache = None, None, {}
    previous_manifest = out/'manifest.json'
    hashes = json.loads(previous_manifest.read_text())['reference_sha256'] if previous_manifest.exists() else {}
    scored, outcomes = [], []
    for name, decision in sorted(decisions.items(), key=lambda item: (item[1]['truth_index'], item[1]['expected_events'], item[0])):
        counts = sorted({s['training'] for s in decision['rules'].values() if s})
        selected_rows = {}
        for count in counts:
            original_row = original[name, count]
            target = out/f'{name}_{count}.json'
            if target.exists():
                row = json.loads(target.read_text())
            elif original_row.get('four_checked') and 'accuracy_pass' in original_row:
                row = dict(original_row, scoring_source='Existing complete four-parameter checkpoint score')
                base.write_json(target, row)
            else:
                key = (decision['truth_index'], decision['expected_events'])
                if key != current_key:
                    refs = references(root, *key, hashes)
                    current_key, direct_cache = key, {}
                catkey = (decision['mode'], decision['case'])
                cat = dict(np.load(root/name/'catalogue.npz'))
                truth = score.run.TRUTHS[decision['truth_index']]
                if catkey not in direct_cache:
                    direct_cache[catkey] = score.direct_checks(refs, cat, truth)
                row = score_checkpoint(root/name/f'checkpoint_{count}.npz', refs, cat, truth,
                                       direct_cache[catkey], original_row)
                base.write_json(target, row)
            selected_rows[count] = row
            scored.append(row)
        for rule, selected in decision['rules'].items():
            outcome = dict(name=name, strategy=decision['strategy'], rule=rule, selected=selected,
                total_spent=selected['total'] if selected else decision['training_labels']+decision['audit_labels'][rule])
            if selected is None:
                outcome['outcome'] = 'no_stop'
            else:
                row = selected_rows[selected['training']]
                numerical = row['reference_numerical'] and row['gp_numerical'] and row['four_numerical']
                outcome.update(outcome='unresolved' if not numerical else 'accurate' if row['accuracy_pass'] else 'premature',
                    symmetric_kl=row['symmetric_kl'], maximum_directed_kl=row['maximum_directed_kl'])
            outcomes.append(outcome)
        logging.info('%s %s', name, [(o['rule'], o['outcome'], o['selected']) for o in outcomes[-2:]])
        base.write_json(out/'outcomes.json', outcomes)
        base.write_json(out/'status.json', dict(state='running', cases=len(outcomes)//2))
    summary = {}
    for strategy in ['bo', 'random']:
        summary[strategy] = {}
        for rule in ['fixed96', 'adaptive384']:
            subset = [r for r in outcomes if (r['strategy'], r['rule']) == (strategy, rule)]
            stops = [r['selected'] for r in subset if r['selected']]
            summary[strategy][rule] = dict(outcomes=dict(Counter(r['outcome'] for r in subset)),
                median_training=float(np.median([s['training'] for s in stops])) if stops else None,
                median_total_accepted=float(np.median([s['total'] for s in stops])) if stops else None,
                median_total_spent=float(np.median([r['total_spent'] for r in subset])))
    assert (folder/'decisions.json').read_bytes() == frozen
    base.write_json(out/'records.json', scored)
    base.write_json(out/'summary.json', summary)
    base.write_json(out/'manifest.json', dict(decision_sha256=sha, audit_stages_verified=verified_stages,
        reference_sha256=hashes, original_records_sha256=hashlib.sha256((root/'offline/records.json').read_bytes()).hexdigest(),
        scorer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    base.write_json(out/'status.json', dict(state='complete', cases=len(decisions)))


if __name__ == '__main__':
    main()
