"""Data-free safeguards for the paired follow-up design and validation references."""
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def study(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'docs/studies/event_uncertainty'))
    import essential_runs
    return essential_runs


def test_width_pairing_preserves_latents_and_standardized_residuals(study):
    latent = np.array([[1.2, .3], [2.4, .6]])
    scales = np.array([[.03, .3], [.02, .2]])
    residual = np.array([[.4, -.8], [-1., .7]])
    cat = {'latent': latent.copy(), 'scales': scales, 'y': np.log(latent) + scales*residual}
    lo, hi = np.array([[1., .1], [2., .4]]), np.array([[2., .4], [3., .9]])
    for multiplier in [.5, 1., 2.]:
        y, coeff, offsets = study.measurement(cat, multiplier, lo, hi)
        np.testing.assert_allclose((y-np.log(latent))/(multiplier*scales), residual)
        from cosmic_integration.observation.log_measurement import bin_coefficients
        np.testing.assert_allclose(coeff*np.exp(offsets[:, None]), bin_coefficients(y, multiplier*scales, lo, hi))
        np.testing.assert_array_equal(cat['latent'], latent)
    with pytest.raises(ValueError):
        study.measurement(cat, 0., lo, hi)


def test_ignored_uncertainty_retains_unsupported_events(study):
    lo, hi = np.array([[1., .1], [3., .1]]), np.array([[2., .9], [4., .9]])
    y = np.log([[1.5, .4], [2.5, .4], [7., .4]])
    coeff, unsupported = study.exact_center_coefficients(y, lo, hi)
    assert unsupported == 2
    assert coeff.shape == (3, 2)
    np.testing.assert_array_equal(coeff, [[1, 0], [0, 0], [0, 0]])


def test_reference_matches_pilot_and_covers_prior_at_boundary(study):
    fisher = np.eye(4)*100
    phi, logq = study.reference_design(study.core.TRUTH, fisher, 35, power=14)
    old_phi, old_logq, _ = study.core.reference_design(100, fisher, 35, power=14)
    np.testing.assert_array_equal(phi, old_phi)
    np.testing.assert_array_equal(logq, old_logq)
    phi, logq = study.reference_design(np.array(study.TRUTHS[1]), fisher, 35, power=15)
    unit = (phi-study.core.LOW[study.core.SHAPE])/study.core.WIDTH[study.core.SHAPE]
    weights = np.exp(-logq)
    weights /= weights.sum()
    np.testing.assert_allclose(weights@unit, .5, atol=.008)
    np.testing.assert_allclose(weights@(unit*unit), 1/3, atol=.008)


def test_rank_summary_uses_actual_injection(study):
    phi = np.array([[-.4, .2, 4.4], [-.2, .4, 4.8]])
    rates = np.array([[10., 20.], [10., 20.]])
    low = np.array([-.49, .11, .006, 4.21])
    high = np.array([-.01, .59, .014, 5.19])
    a = study.Experiment.summarize(phi, rates, .1, 3, low, np.zeros(2))
    b = study.Experiment.summarize(phi, rates, .1, 3, high, np.zeros(2))
    np.testing.assert_allclose(a['mean'], b['mean'])
    np.testing.assert_array_equal(np.array(a['ranks'])[[0, 1, 3]], 0.)
    np.testing.assert_array_equal(np.array(b['ranks'])[[0, 1, 3]], 1.)
