"""Independent post-decision scoring, including all four physical parameters."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import logging
from pathlib import Path
import time

import numpy as np

import compare_random as run
from score_experiment import directed_kls, agreement

base = run.base
HERE = Path(__file__).resolve().parent


def four_summary(phi: np.ndarray, rates: np.ndarray, cat: dict,
                 truth: np.ndarray, logweights: np.ndarray) -> dict:
    return base.data_source.Experiment.summarize(phi, rates, float(cat['duration']),
                                                len(cat['coeff']), truth, logweights)


def four_accuracy(a: dict, b: dict) -> dict:
    change = base.core.compare(a, b)
    passed = (change['max_mean_shift_sd'] < .2 and change['max_endpoint_shift_sd'] < .2
              and np.max(abs(np.asarray(change['width68_ratio'])-1)) < .1
              and np.max(abs(np.asarray(change['width95_ratio'])-1)) < .1)
    return dict(passed=bool(passed), **change)


def cached_references(ti: int, n: int, hashes: dict) -> list:
    root = HERE.parents[1]/'event_uncertainty'
    output = []
    for scramble in [0, 1]:
        if ti == 0:
            folder = root/'convergence_20260910/ensemble'
            power = 16 if n == 100 else 17
            rates_path = folder/f'reference_n{n}_p{power}_{scramble}.npz'
            proposal_path = folder/f'proposal_n{n}_p{power}_{scramble}.npz'
        else:
            folder = root/'essential_20260910/robustness'
            ni = int(n == 1000)
            rates_path = folder/f'reference_t{ti}_n{ni}_{scramble}.npz'
            proposal_path = folder/f'reference_t{ti}_n{ni}_{scramble}_proposal.npz'
        for path in [rates_path, proposal_path]:
            hashes[str(path.relative_to(base.REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(rates_path) as data:
            phi, rates = data['phi'], data['rates']
        with np.load(proposal_path) as data:
            logq = data['logq']
        output.append((phi, rates, logq))
    return output


def direct_checks(references: list, cat: dict, truth: np.ndarray) -> tuple:
    exact = [base.source_target.event_target(r, cat['coeff'], float(cat['duration']))-q
             for _, r, q in references]
    summaries = [base.posterior_summary((p-base.LOW)/base.WIDTH, w)
                 for (p, _, _), w in zip(references, exact)]
    four = [four_summary(p, r, cat, truth, w) for (p, r, _), w in zip(references, exact)]
    shape_ok, shape_check = base.summaries_agree(*summaries, base.TOL)
    four_ok, four_check = base.data_source.precision(*four)
    return exact, summaries, four, dict(passed=bool(shape_ok and four_ok),
                                       shape=shape_check, four=four_check)


def case_outcomes(decision: dict, case_rules: dict, case_rows: list[dict]) -> list[dict]:
    """Classify frozen decisions, also when restoring completed case scores."""
    by_count = {r['points']: r for r in case_rows}
    outcomes = []
    for rule, selected in case_rules.items():
        record = {k: decision[k] for k in ['name', 'truth_index', 'expected_events', 'mode', 'case', 'strategy']}
        record.update(rule=rule, selected=selected,
                      cohort='development' if decision['truth_index'] == 0 else 'confirmation')
        if selected is None:
            record['outcome'] = 'no_stop'
        else:
            score = by_count[selected['training']]
            numerical = score['reference_numerical'] and score['gp_numerical'] and score['four_numerical']
            record.update(outcome='unresolved' if not numerical else 'accurate' if score['accuracy_pass'] else 'premature',
                symmetric_kl=score['symmetric_kl'], maximum_directed_kl=score['maximum_directed_kl'])
        outcomes.append(record)
    return outcomes


def completed_case(path: Path, decision: dict, checkpoints: list[dict]) -> list[dict]:
    """Restore only a complete matching case; checkpoint hashes are checked separately."""
    rows = json.loads(path.read_text())
    assert [r['points'] for r in rows] == [c['points'] for c in checkpoints]
    for row in rows:
        for key in ['name', 'truth_index', 'expected_events', 'mode', 'case', 'strategy']:
            assert row[key] == decision[key]
    assert rows[-1]['four_checked'] and 'accuracy_pass' in rows[-1]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--resume', action='store_true', help='Keep completed case scores and exact-rate caches')
    args = parser.parse_args()
    args.input = args.input.resolve()
    frozen = (args.input/'decisions.json').read_bytes()
    status = json.loads((args.input/'status.json').read_text())
    assert status['state'] == 'complete'
    assert status['decision_sha256'] == hashlib.sha256(frozen).hexdigest()
    decisions = json.loads(frozen)
    assert all(d['complete'] and not d['reference_accessed'] for d in decisions.values())
    mass_bytes = (args.input/'mass_audit/decisions.json').read_bytes()
    mass_sha = hashlib.sha256(mass_bytes).hexdigest()
    mass_status = json.loads((args.input/'mass_audit/status.json').read_text())
    assert mass_status['state'] == 'complete' and mass_status['decision_sha256'] == mass_sha
    mass_decisions = json.loads(mass_bytes)
    assert set(mass_decisions) == set(decisions)
    assert all(not d['reference_accessed'] for d in mass_decisions.values())
    out = args.input/'offline'; out.mkdir(exist_ok=args.resume)
    assert not (out/'manifest.json').exists(), 'Scoring is already complete'
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    population_folder = out/(f'population_parity_resume_{time.time_ns()}' if args.resume else 'population_parity')
    loader = base.data_source.Experiment(population_folder, 'comparison_scoring_only', False)
    cached, hashes, parity, data_cache = {}, {}, [], {}
    rows, outcomes, restored = [], [], {}
    started = time.perf_counter()
    for name, decision in decisions.items():
        folder = args.input/name
        assert json.loads((folder/'decision.json').read_text()) == decision
        case_rules = dict(decision['rules'], mass_guarded=mass_decisions[name]['selected'])
        checkpoints = json.loads((folder/'checkpoints.json').read_text())
        for checkpoint in checkpoints:
            path = folder/f'checkpoint_{checkpoint["points"]}.npz'
            assert hashlib.sha256(path.read_bytes()).hexdigest() == checkpoint['sha256']
        ti, n = decision['truth_index'], decision['expected_events']
        truth = run.TRUTHS[ti]
        key = (ti, n)
        if key not in cached:
            cached[key] = cached_references(ti, n, hashes)
            for scramble, (phi, rates, _) in enumerate(cached[key]):
                idx = np.unique(np.linspace(0, len(phi)-1, 3, dtype=int))
                actual = np.array([loader.fast(p[0], p[1], .012, p[2]).ravel()[loader.active] for p in phi[idx]])
                np.testing.assert_allclose(actual, rates[idx], atol=1e-13, rtol=2e-11)
                parity.append(dict(truth_index=ti, expected_events=n, scramble=scramble,
                                   maximum_absolute_difference=float(np.max(abs(actual-rates[idx])))))
        saved = out/f'{name}.json'
        if args.resume and saved.exists():
            case_rows = completed_case(saved, decision, checkpoints)
            outcomes.extend(case_outcomes(decision, case_rules, case_rows)); rows.extend(case_rows)
            restored[name] = hashlib.sha256(saved.read_bytes()).hexdigest()
            logging.info('Restored completed case %s', name)
            continue
        cat = dict(np.load(folder/'catalogue.npz'))
        catalogue_key = (ti, n, decision['mode'], decision['case'])
        if catalogue_key not in data_cache:
            refs = cached[key]
            checked = direct_checks(refs, cat, truth)
            # Reference-only refinement is triggered solely by direct numerical
            # disagreement, never by a failed GP accuracy threshold.
            if not checked[-1]['passed']:
                fisher = base.core.fisher_matrix(lambda t: loader.fast(*t), truth,
                                                 base.core.WIDTH, float(cat['duration']))
                for power in [17, 18]:
                    refs = []
                    for scramble in [0, 1]:
                        seed = 202609114+100*ti+n+10*decision['case']+scramble
                        phi, logq = base.data_source.reference_design(truth, fisher, seed, power)
                        path = out/f'refined_t{ti}_n{n}_c{decision["case"]}_p{power}_{scramble}.npz'
                        if path.exists():
                            with np.load(path) as d:
                                np.testing.assert_array_equal(phi, d['phi']); rates = d['rates']
                        else:
                            logging.info('Refining reference %s: %d exact-rate coordinates', path.name, len(phi))
                            rates = np.array([loader.fast(p[0], p[1], .012, p[2]).ravel()[loader.active] for p in phi])
                            np.savez(path, phi=phi, rates=rates, logq=logq)
                        hashes[str(path.relative_to(base.REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
                        refs.append((phi, rates, logq))
                    checked = direct_checks(refs, cat, truth)
                    if checked[-1]['passed']:
                        break
                # Rates do not depend on the catalogue or sensor. Reuse this
                # finer exact design for subsequent cases at this injection and
                # exposure; every new posterior gets its own precision check.
                cached[key] = refs
            data_cache[catalogue_key] = (refs, *checked)
        references, exact, direct, direct_four, refcheck = data_cache[catalogue_key]
        special = {d['training'] for d in case_rules.values() if d is not None}
        special.add(decision['training_labels'])
        case_rows = []
        for checkpoint in checkpoints:
            count = checkpoint['points']; path = folder/f'checkpoint_{count}.npz'
            assert hashlib.sha256(path.read_bytes()).hexdigest() == checkpoint['sha256']
            state, _, _ = base.load_checkpoint(path)
            pred = [base.predict(state, (p-base.LOW)/base.WIDTH)-q for p, _, q in references]
            gp = [base.posterior_summary((p-base.LOW)/base.WIDTH, w) for (p, _, _), w in zip(references, pred)]
            gp_ok, gpcheck = base.summaries_agree(*gp, base.TOL)
            kls = [directed_kls(a, b) for a, b in zip(exact, pred)]
            metrics = [agreement(a, b, k) for a, b, k in zip(direct, gp, kls)]
            row = {k: decision[k] for k in ['name', 'truth_index', 'expected_events', 'mode', 'case', 'strategy']}
            row.update(points=count, symmetric_kl=float(np.mean(kls)), maximum_directed_kl=float(np.max(kls)),
                reference_numerical=refcheck['passed'], reference_check=refcheck,
                gp_numerical=gp_ok, gp_check=gpcheck, metrics=metrics,
                shape_accuracy=bool(refcheck['passed'] and gp_ok and all(m['passed'] for m in metrics)),
                four_checked=count in special)
            if count in special:
                try:
                    gpf = [four_summary(p, r, cat, truth, w) for (p, r, _), w in zip(references, pred)]
                    four_ok, four_check = base.data_source.precision(*gpf)
                    fm = [four_accuracy(a, b) for a, b in zip(direct_four, gpf)]
                    row.update(four_numerical=four_ok, four_check=four_check, four_metrics=fm,
                        four_summaries={'direct': direct_four, 'gp': gpf},
                        accuracy_pass=bool(row['shape_accuracy'] and four_ok and all(m['passed'] for m in fm)))
                except ValueError as exc:
                    row.update(four_numerical=False, four_error=str(exc), accuracy_pass=False)
            case_rows.append(row)
        outcomes.extend(case_outcomes(decision, case_rules, case_rows))
        rows.extend(case_rows)
        base.write_json(out/f'{name}.json', case_rows)
        base.write_json(out/'status.json', dict(state='running', case=name, checkpoints=len(rows)))
        logging.info('%s reference=%s final KL=%.4g four_accuracy=%s', name, refcheck['passed'],
                     case_rows[-1]['symmetric_kl'], case_rows[-1]['accuracy_pass'])
    summary = {s: {r: dict(Counter(o['outcome'] for o in outcomes if o['rule'] == r and o['strategy'] == s))
                   for r in [*run.RULES, 'mass_guarded']} for s in ['bo', 'random']}
    base.write_json(out/'records.json', rows); base.write_json(out/'outcomes.json', outcomes)
    base.write_json(out/'summary.json', summary)
    assert frozen == (args.input/'decisions.json').read_bytes()
    assert mass_bytes == (args.input/'mass_audit/decisions.json').read_bytes()
    base.write_json(out/'manifest.json', dict(decision_sha256=hashlib.sha256(frozen).hexdigest(),
        mass_decision_sha256=mass_sha,
        reference_sha256=hashes, rate_parity=parity, wall_seconds=time.perf_counter()-started,
        wall_seconds_scope='this scoring invocation', restored_case_sha256=restored,
        scorer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        joint_kl='equal to shape KL under the shared exact amplitude conditional'))
    base.write_json(out/'status.json', dict(state='complete', trajectories=len(decisions), checkpoints=len(rows)))


if __name__ == '__main__':
    main()
