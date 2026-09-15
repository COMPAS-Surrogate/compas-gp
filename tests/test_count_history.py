"""Count-history contracts: pairing, cost axes and reuse of numerical proposals."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def history(monkeypatch):
    folder = Path(__file__).resolve().parents[1]/'docs/studies/operational_stopping/gpry_compas'
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location('count_history_test', folder/'run_count_history.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_requested_matrix_is_paired_and_round_robin(history):
    matrix = history.jobs(5)
    assert len(matrix) == 70
    assert len({row['name'] for row in matrix}) == 70
    assert {row['expected_events'] for row in matrix} == {10, 50, 100, 200, 500, 700, 1000}
    for iteration in range(1, 6):
        part = matrix[(iteration-1)*14:iteration*14]
        assert {r['iteration'] for r in part} == {iteration}
        for perfect, uncertain in zip(part[::2], part[1::2]):
            assert (perfect['mode'], uncertain['mode']) == ('perfect', 'uncertain')
            assert perfect['case'] == uncertain['case']
            assert perfect['expected_events'] == uncertain['expected_events']


def test_frozen_hashes_ignore_additions_but_detect_changed_or_missing_inputs(history, monkeypatch, tmp_path):
    monkeypatch.setattr(history.BASE, 'REPO', tmp_path)
    source = tmp_path/'training.py'
    source.write_text('original')
    frozen = history.hashes({'training.py': ''})
    (tmp_path/'unrelated.py').write_text('new unrelated work')
    assert history.hashes(frozen) == frozen
    source.write_text('modified')
    assert history.hashes(frozen) != frozen
    source.unlink()
    with pytest.raises(FileNotFoundError):
        history.hashes(frozen)


def test_stop_axes_and_diagnostic_cost_are_distinct(history):
    cp = dict(points=298, posterior_evaluation=9, posterior_comparisons=8,
              bo_batches=80, elapsed_seconds=31., recorded_unix=1000.)
    decision = dict(rules={'fixed96': dict(training=298, audit=192, total=490)}, wall_seconds=4.)
    stop = history.stop_record(decision, [cp], 1258)
    assert stop['selected']['total'] == 490
    assert stop['posterior_comparisons'] == 8
    assert stop['post_stop_labels'] == 960
    assert stop['stop_is_shadow_replay'] and not stop['reference_accessed']
    decision['rules']['fixed96'] = None
    limited = history.stop_record(decision, [cp], 1258)
    assert limited['state'] == 'resource_limit'
    assert limited['posterior_comparisons'] is None


def test_cached_proposals_use_current_catalogue_and_retain_failed_precision(history, monkeypatch):
    cat = dict(duration=.4, coeff=np.eye(2), truth=np.ones(4))
    seen = []
    monkeypatch.setattr(history.validation.score, 'cached_references', lambda ti, n, hashes: [n])
    def check(refs, actual_cat, truth):
        assert actual_cat is cat
        np.testing.assert_array_equal(truth, cat['truth'])
        seen.append(refs[0])
        return None, None, None, {'passed': refs[0] == 1000}
    monkeypatch.setattr(history.validation.score, 'direct_checks', check)
    refs, checked, provenance = history.cached_references(50, cat)
    assert seen == [100, 1000] and refs == [1000]
    assert not provenance['attempts'][0]['check']['passed']
    assert checked[-1]['passed'] and provenance['precision_pass']
    monkeypatch.setattr(history.validation.score, 'direct_checks',
                        lambda *args: (None, None, None, {'passed': False}))
    _, _, unresolved = history.cached_references(700, cat)
    assert not unresolved['precision_pass'] and len(unresolved['attempts']) == 2


def test_checkpoint_metadata_has_unambiguous_counts(history, monkeypatch, tmp_path):
    # Exercise the actual training loop with inexpensive deterministic numerical
    # stand-ins, checking the initial and first comparison records on disk.
    import json
    train, base = history.train, history.BASE
    cat = dict(bins=np.array([0]), duration=1., latent=np.ones((1, 2)))
    monkeypatch.setattr(train, 'catalogue', lambda *args: cat)
    monkeypatch.setattr(base.source_target, 'event_target', lambda rates, coeff, duration: np.zeros(len(rates)))
    monkeypatch.setattr(base, 'fit', lambda *args: {})
    monkeypatch.setattr(base, 'condition_on_data', lambda *args: {})
    monkeypatch.setattr(base, 'predict', lambda state, unit: np.zeros(len(unit)))
    monkeypatch.setattr(base, 'save_checkpoint', lambda *args: 'test-hash')
    monkeypatch.setattr(train.jax, 'block_until_ready', lambda state: state)
    check = dict(numerical=True, gaussian_pass=False, joint_pass=False, power=14)
    monkeypatch.setattr(train, 'adaptive_stability', lambda *args: (dict(check, attempts=[check]), []))
    class Loader:
        active = np.array([0])
        @staticmethod
        def fast(*args):
            return np.ones(1)
    train.run_case(tmp_path, Loader(), 0, 10, 'perfect', 200, 'random', 88)
    rows = json.loads((tmp_path/'t0_n10_perfect_c200_random/checkpoints.json').read_text())
    assert [(r['points'], r['posterior_evaluation'], r['posterior_comparisons'], r['bo_batches'])
            for r in rows] == [(58, 1, 0, 0), (88, 2, 1, 10)]
    assert rows[1]['recorded_unix'] >= rows[0]['recorded_unix']
    assert rows[1]['elapsed_seconds'] >= rows[0]['elapsed_seconds']
