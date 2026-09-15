"""Scientific invariants of the paired stopping comparison."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.special import rel_entr


@pytest.fixture
def experiment(monkeypatch):
    folder = Path(__file__).resolve().parents[1]/'docs/studies/operational_stopping/gpry_compas'
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location('compas_stopping_extension_test', folder/'compare_random.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_precision_refinement_never_evaluates_likelihood(experiment, monkeypatch):
    powers = []
    def design(state, unit, labels, seed, power):
        powers.append(power)
        return power
    monkeypatch.setattr(experiment, 'gp_designs', design)
    monkeypatch.setattr(experiment, 'stability_on_designs',
                        lambda previous, current, proposal: {'numerical': proposal >= 15})
    result, proposal = experiment.adaptive_stability(None, {}, np.zeros((1, 3)), np.zeros(1), 5)
    assert powers == [14, 15] and proposal == 15 and result['numerical']
    monkeypatch.setattr(experiment, 'stability_on_designs',
                        lambda previous, current, proposal: {'numerical': False})
    result, _ = experiment.adaptive_stability(None, {}, np.zeros((1, 3)), np.zeros(1), 5)
    assert len(result['attempts']) == 3 and not result['numerical']


def test_guarded_audit_ignores_offsets_and_rejects_residual_variation(experiment):
    actual = np.r_[np.linspace(-3, 0, 64), np.full(32, -30.)]
    assert experiment.audit_check(actual+73, actual, 64, 0, experiment.AUDIT_TOL)['passed']
    error = np.r_[np.tile([-.11, .11], 32), np.zeros(32)]
    assert not experiment.audit_check(actual+error, actual, 64, 0, experiment.AUDIT_TOL)['passed']
    missed = actual.copy(); missed[-1] = 0
    assert not experiment.audit_check(actual, missed, 64, 0, experiment.AUDIT_TOL)['passed']


def test_shared_exact_conditional_preserves_joint_kl():
    # Different shape-dependent amplitude conditionals; neither is independent.
    p = np.array([.1, .3, .6]); q = np.array([.2, .25, .55])
    conditional = np.array([[.8, .15, .05], [.3, .4, .3], [.05, .25, .7]])
    joint_p, joint_q = p[:, None]*conditional, q[:, None]*conditional
    for a, b, ja, jb in [(p, q, joint_p, joint_q), (q, p, joint_q, joint_p)]:
        assert rel_entr(a, b).sum() == pytest.approx(rel_entr(ja, jb).sum())
    assert rel_entr(joint_p.sum(axis=0), joint_q.sum(axis=0)).sum() <= rel_entr(p, q).sum()


def test_catalogue_is_paired_and_exposure_scales_with_injected_amplitude(experiment):
    class Loader:
        active = np.arange(2)
        lower = np.array([[10., .1], [20., .2]])
        upper = lower + [2., .05]
        @staticmethod
        def fast(alpha, sigma, amplitude, d):
            return amplitude*np.array([1., 2.])
    a = experiment.catalogue(Loader(), 0, 100, 1)
    b = experiment.catalogue(Loader(), 0, 100, 1)
    np.testing.assert_array_equal(a['y'], b['y'])
    assert a['duration']*Loader.fast(*a['truth']).sum() == pytest.approx(100)


def test_audit_mixture_density_uses_gp_normalizers(experiment, monkeypatch):
    folder = Path(experiment.__file__).parent
    spec = importlib.util.spec_from_file_location('mass_audit_proposal_test', folder/'run_mass_audit.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    monkeypatch.setattr(module.base, 'predict', lambda state, points: np.full(len(points), 5.))
    points = np.random.default_rng(2).uniform(size=(8, 3))
    proposal = [(points, np.zeros(8)), (points[::-1], np.zeros(8))]
    selected, predicted, density, info = module.proposal_sample({}, proposal, 3, np.random.default_rng(17))
    assert info['numerical'] and selected.shape == (96, 3)
    np.testing.assert_allclose(info['log_normalizers'], [[5., 2.5], [5., 2.5]])
    np.testing.assert_allclose(predicted, 5.)
    np.testing.assert_allclose(density, 0., atol=1e-12)
    # The evidence divisor is generated count, not retained inside-prior count.
    short = [(points[:4], np.zeros(4)), (points[4:], np.zeros(4))]
    _, _, _, info = module.proposal_sample({}, short, 3, np.random.default_rng(17))
    np.testing.assert_allclose(info['log_normalizers'], np.array([[5., 2.5], [5., 2.5]])-np.log(2))


def test_scoring_resume_preserves_failures_and_rejects_incomplete_cases(experiment, tmp_path):
    import json
    folder = Path(experiment.__file__).parent
    spec = importlib.util.spec_from_file_location('resume_scoring_test', folder/'score_comparison.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    decision = dict(name='paired_case', truth_index=1, expected_events=100,
                    mode='perfect', case=0, strategy='bo')
    rows = [dict(decision, points=n, reference_numerical=True, gp_numerical=True,
                 four_numerical=(n != 118), four_checked=True, accuracy_pass=(n == 58),
                 symmetric_kl=.005, maximum_directed_kl=.006) for n in [58, 88, 118]]
    path = tmp_path/'case.json'; path.write_text(json.dumps(rows))
    checkpoints = [dict(points=r['points']) for r in rows]
    restored = module.completed_case(path, decision, checkpoints)
    rules = {str(n): dict(training=n, total=n+96, audit=96) for n in [58, 88, 118]}
    rules['never'] = None
    result = module.case_outcomes(decision, rules, restored)
    assert [r['outcome'] for r in result] == ['accurate', 'premature', 'unresolved', 'no_stop']
    assert all(r['cohort'] == 'confirmation' for r in result)
    with pytest.raises(AssertionError):
        module.completed_case(path, dict(decision, case=1), checkpoints)
    with pytest.raises(AssertionError):
        module.completed_case(path, decision, checkpoints[:-1])


def test_parallel_reference_cannot_overwrite_a_competing_cache(experiment, tmp_path):
    spec = importlib.util.spec_from_file_location('reference_prefetch_test',
        Path(experiment.__file__).parent/'prefetch_reference.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    temporary, target = tmp_path/'temporary.npz', tmp_path/'reference.npz'
    temporary.write_bytes(b'complete prefetch')
    target.write_bytes(b'original reference')
    assert not module.publish_cache(temporary, target)
    assert target.read_bytes() == b'original reference'
    target.unlink()
    assert module.publish_cache(temporary, target)
    temporary.unlink()
    assert target.read_bytes() == b'complete prefetch'
