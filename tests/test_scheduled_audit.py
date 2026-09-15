"""Independent numerical and operational invariants of periodic audits."""
import numpy as np
import pytest
from scipy.special import rel_entr

from cosmic_integration.lnl_surrogate.scheduled_audit import AuditPolicy, posterior_audit
from cosmic_integration.lnl_surrogate.stability_audit import stable_audit_acceptance


def test_final_audit_is_always_due_and_schedule_is_bounded():
    policy = AuditPolicy()
    assert policy.due(58, 58)
    assert [n for n in range(58, 659, 30) if policy.due(n, 658)] == list(range(238, 659, 60))
    assert policy.due(628, 628)
    with pytest.raises(ValueError):
        AuditPolicy(sizes=(96, 95))


def test_audit_matches_discrete_kl_and_ignores_constant_offsets():
    pred = np.linspace(-.03, .03, 96)
    actual = pred[::-1]
    groups = np.repeat([0, 1, 2], [64, 16, 16])
    a = posterior_audit(pred, actual, np.zeros(96), groups, 13)
    p, q = np.exp(pred), np.exp(actual); p /= p.sum(); q /= q.sum()
    assert a['kl_gp_to_direct'] == pytest.approx(rel_entr(p, q).sum())
    b = posterior_audit(pred+99, actual-27, np.full(96, 7.), groups, 13)
    assert b['maximum_directed_kl'] == pytest.approx(a['maximum_directed_kl'], abs=1e-12)
    assert a['passed'] and b['passed']


def test_low_ess_enlarges_then_exhausts_audit_without_false_acceptance():
    for size in [96, 192, 384]:
        groups = np.repeat([0, 1, 2], [4*size//6, size//6, size//6])
        pred = np.full(size, -100.); pred[0] = 0.
        result = posterior_audit(pred, pred, np.zeros(size), groups, 8)
        assert not result['passed']
        assert result['action'] == ('enlarge_audit' if size < 384 else 'continue_training')


def test_stratification_accepts_increment_order_and_rejects_wrong_mix():
    groups = np.tile(np.repeat([0, 1, 2], [64, 16, 16]), 2)
    values = np.zeros(192)
    assert posterior_audit(values, values, values, groups, 8)['passed']
    groups[0] = 1
    with pytest.raises(ValueError):
        posterior_audit(values, values, values, groups, 8)


def test_material_shape_error_and_nonfinite_values_do_not_stop():
    groups = np.repeat([0, 1, 2], [64, 16, 16])
    actual = np.tile([-.5, .5], 48)
    result = posterior_audit(np.zeros(96), actual, np.zeros(96), groups, 4)
    assert result['action'] == 'continue_training'
    actual[0] = np.nan
    assert not posterior_audit(np.zeros(96), actual, np.zeros(96), groups, 4)['passed']


def test_passing_audit_cannot_accept_a_changing_posterior():
    assert not stable_audit_acceptance({'passed': True}, 0)
    assert not stable_audit_acceptance({'passed': True}, 1)
    assert stable_audit_acceptance({'passed': True}, 2)
    assert not stable_audit_acceptance({'passed': False}, 20)
