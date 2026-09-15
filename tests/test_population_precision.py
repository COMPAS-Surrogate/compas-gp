"""Independent Poisson identities and coherent block resampling."""
import numpy as np
import pytest
from cosmic_integration.observation.population_precision import (
    poisson_loss, bootstrap_coefficients, unsupported_fraction,
    log_bin_coefficients,
)


def test_loss_includes_candidate_only_support_and_missing_support():
    ref = np.array([2., 0., 3.])
    assert poisson_loss(ref, [2., 4., 3.]) == pytest.approx(4 / 5)
    assert poisson_loss(ref, ref) == 0
    assert np.isinf(poisson_loss(ref, [0., 0., 3.]))
    assert unsupported_fraction(ref, [0., 0., 3.]) == pytest.approx(.4)


def test_expected_loss_matches_explicit_poisson_catalogue_average():
    from scipy.stats import poisson
    from scipy.special import xlogy
    ref, other = np.array([2., 3.]), np.array([1., 4.])
    n = np.arange(60)
    expected = sum(np.sum(poisson.pmf(n, a) *
                         (xlogy(n, a) - a - xlogy(n, b) + b))
                   for a, b in zip(ref, other))
    assert ref.sum() * poisson_loss(ref, other) == pytest.approx(expected)
    assert poisson_loss(3 * ref, 3 * other) == pytest.approx(poisson_loss(ref, other))


def test_bootstrap_keeps_population_normalization_and_bin_dependence():
    c = bootstrap_coefficients(4, 2, 100, 10)
    np.testing.assert_allclose(c.sum(axis=1), 1)
    np.testing.assert_array_equal(c * 2, np.round(c * 2))
    rates = np.array([[1., 2.], [2., 4.], [3., 6.], [4., 8.]])
    samples = c @ rates
    np.testing.assert_allclose(samples[:, 1], 2 * samples[:, 0])
    np.testing.assert_array_equal(c, bootstrap_coefficients(4, 2, 100, 10))
    with pytest.raises(ValueError):
        bootstrap_coefficients(4, 0, 100, 10)


def test_coarsening_cannot_increase_binned_poisson_discrepancy():
    ref = np.array([2., 4., 3., 1.])
    other = np.array([1., 3., 5., 2.])
    assert poisson_loss(ref.reshape(2, 2).sum(1), other.reshape(2, 2).sum(1)) <= poisson_loss(ref, other)


def test_log_sensor_matches_linear_sensor_and_resolves_extreme_tail():
    from cosmic_integration.observation.log_measurement import bin_coefficients
    low = np.array([[1., 0.], [2., .1]])
    high = np.array([[2., .1], [3., .2]])
    y = np.log([[1.5, .08]])
    scale = np.array([[.03, .3]])
    np.testing.assert_allclose(np.exp(log_bin_coefficients(y, scale, low, high)),
                               bin_coefficients(y, scale, low, high), rtol=1e-12)
    distant = np.log([[100., 10.]])
    assert np.all(np.isfinite(log_bin_coefficients(distant, scale, low, high)))
    assert np.all(bin_coefficients(distant, scale, low, high) == 0)
