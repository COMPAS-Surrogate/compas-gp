"""Independent exact-amplitude and proposal-density controls."""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import logsumexp
from scipy.stats import gamma

from cosmic_integration.observation.amplitude_validation import (
    amplitude_proposal, joint_amplitude_log_likelihood,
)
from cosmic_integration.observation.log_measurement import log_marginal_events


def test_joint_integrates_to_existing_marginal_and_scales_linearly():
    events = np.array([[2., 3., 1.], [1., 4., 2.]])
    total = np.array([8., 10.])
    duration = .7
    marginal = log_marginal_events(events, total, duration)
    for i in range(2):
        f = lambda a: float(joint_amplitude_log_likelihood(events[i:i+1], total[i:i+1], np.array([a]), duration)[0])
        integrated = np.log(quad(lambda a: np.exp(f(a)), .005, .015, epsabs=1e-12)[0]/.01)
        assert integrated == pytest.approx(marginal[i], abs=1e-10)
        expected = 3*np.log(2)-duration*total[i]
        assert f(.024)-f(.012) == pytest.approx(expected)


def test_proposal_recovers_truncated_gamma_moments_and_has_full_support():
    beta = np.full(100000, 100/.01)
    a, logq = amplitude_proposal(beta, 100, np.random.default_rng(31))
    logp = gamma.logpdf(a, 101, scale=1/beta)
    w = np.exp(logp-logq-logsumexp(logp-logq))
    exact_mean = gamma.expect(lambda a: a, args=(101,), scale=1/beta[0],
                              lb=.005, ub=.015, conditional=True)
    assert w@a == pytest.approx(exact_mean, abs=1e-5)
    assert np.sum(a < .006) > 4000
    assert np.all(logq >= np.log(.5))


def test_underflow_and_upper_tail_proposals_remain_finite():
    beta = np.array([1e-6, 1e6, 2e4, 10000.])
    a, logq = amplitude_proposal(beta, 100, np.random.default_rng(2))
    assert np.all((a >= .005) & (a <= .015))
    assert np.isfinite(logq).all()
    assert logq[0] == logq[1] == 0


def test_zero_support_and_invalid_inputs_are_not_floored():
    assert np.isneginf(joint_amplitude_log_likelihood(np.array([[0.]]), np.ones(1), np.array([.01]), 1.)[0])
    with pytest.raises(ValueError):
        amplitude_proposal(np.array([-1.]), 2, np.random.default_rng(1))
    with pytest.raises(ValueError):
        joint_amplitude_log_likelihood(np.ones((1, 1)), np.ones(1), np.array([0.]), 1.)
