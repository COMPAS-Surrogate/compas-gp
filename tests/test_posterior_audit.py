import numpy as np
import pytest

from cosmic_integration.lnl_surrogate.posterior_audit import importance_audit_check


def test_mass_audit_is_offset_invariant_and_rejects_shape_error():
    pred = np.zeros(96); actual = np.r_[np.tile([-.02, .02], 32), np.zeros(32)]
    a = importance_audit_check(pred, actual, np.zeros(96), 77)
    b = importance_audit_check(pred+100, actual-30, np.full(96, 20.), 77)
    assert a['passed'] and b['passed']
    assert a['maximum_directed_kl'] == pytest.approx(b['maximum_directed_kl'], abs=1e-12)
    actual[:64] *= 10
    assert not importance_audit_check(pred, actual, np.zeros(96), 77)['passed']


def test_mass_audit_distinguishes_negligible_tail_error_from_missed_mass():
    pred = np.r_[np.zeros(64), np.full(32, -30.)]; actual = pred.copy()
    actual[-1] = -20
    assert importance_audit_check(pred, actual, np.zeros(96), 77)['passed']
    actual[-1] = -4
    proposal = np.zeros(96); proposal[-1] = -10
    result = importance_audit_check(pred, actual, proposal, 77)
    assert not result['passed'] and result['maximum_directed_kl'] > .01


def test_mass_audit_rejects_invalid_or_nonfinite_inputs():
    with pytest.raises(ValueError):
        importance_audit_check(np.zeros(95), np.zeros(95), np.zeros(95), 1)
    actual = np.zeros(96); actual[3] = np.nan
    assert not importance_audit_check(np.zeros(96), actual, np.zeros(96), 1)['passed']
