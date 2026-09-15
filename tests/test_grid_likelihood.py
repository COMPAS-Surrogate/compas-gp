import numpy as np
import pytest

from cosmic_integration.lnl_computer import core_ln_likelihood
from cosmic_integration.observation.grid_likelihood import (
    aggregate_nested_grid, counts_log_likelihood, posterior_grid_coefficients,
)


def test_count_likelihood_matches_event_form():
    counts = np.array([[2, 0], [1, 3]])
    weights = np.repeat(np.eye(4), counts.ravel(), axis=0).reshape(-1, 2, 2)
    differences = []
    for rates in [np.array([[2., 3.], [4., 1.]]), np.array([[6., 1.], [2., 8.]])]:
        differences.append(counts_log_likelihood(rates, counts, .2)
                           - core_ln_likelihood(rates, .2, weights))
    np.testing.assert_allclose(differences, 0., atol=1e-12)


def test_zero_support_is_not_clipped():
    assert counts_log_likelihood(np.zeros((1, 1)), np.zeros((1, 1)), 1.) == 0
    assert counts_log_likelihood(np.zeros((1, 1)), np.ones((1, 1)), 1.) == -np.inf
    with pytest.raises(ValueError):
        counts_log_likelihood(np.ones((1, 1)), np.array([[.5]]), 1.)


def test_unequal_areas_and_prior_correction_preserve_flat_event():
    # Posterior == uniform PE prior: the event supplies no shape information.
    samples = np.array([[.5, .5], [1.5, .5], [2.5, .5]])
    fine = posterior_grid_coefficients(samples, np.full(3, 1/3), [0, 1, 3], [0, 1])
    coarse = posterior_grid_coefficients(samples, np.full(3, 1/3), [0, 3], [0, 1])
    np.testing.assert_allclose(fine, 1.)
    np.testing.assert_allclose(coarse, 1.)
    for rates in [np.array([[1.], [9.]]), np.array([[7.], [3.]])]:
        assert np.sum(rates*fine) == pytest.approx(rates.sum()*coarse.item())


def test_outside_samples_remain_in_denominator_not_boundary_bins():
    result = posterior_grid_coefficients(
        np.array([[.5, .5], [2., .5]]), np.ones(2), [0, 1], [0, 1])
    assert result.item() == .5


def test_nested_count_aggregation_and_invalid_partial_pixel():
    matrix = np.arange(12).reshape(4, 3)
    actual = aggregate_nested_grid(matrix, [0, 1, 2, 3, np.inf], [0, 1, 2, 3],
                                   [0, 2, np.inf], [0, 2, 3])
    np.testing.assert_array_equal(actual, [[8, 7], [32, 19]])
    assert actual.sum() == matrix.sum()
    with pytest.raises(ValueError):
        aggregate_nested_grid(matrix, [0, 1, 2, 3, np.inf], [0, 1, 2, 3],
                              [0, 1.5, np.inf], [0, 3])


def test_amplitude_marginal_matches_numerical_integration():
    from scipy.integrate import quad
    from cosmic_integration.observation.grid_likelihood import amplitude_marginal_log_likelihood
    rates = np.array([[30., 20.], [0., 5.]])
    counts = np.array([[3, 2], [0, 1]])
    for factor in [0.01, 1., 100.]:
        r = rates*factor
        value, beta = amplitude_marginal_log_likelihood(r,counts,.1)
        center = counts_log_likelihood(r,counts,.1)
        integral,_ = quad(lambda a: np.exp(counts_log_likelihood(r*a/.012,counts,.1)-center),.005,.015,epsabs=1e-11)
        np.testing.assert_allclose(value, center+np.log(integral/.01), atol=1e-9)
        assert beta == .1*r.sum()/.012


def test_amplitude_marginal_zero_support():
    from cosmic_integration.observation.grid_likelihood import amplitude_marginal_log_likelihood
    zero=np.zeros((1,1))
    assert amplitude_marginal_log_likelihood(zero,zero,1.) == (0.,0.)
    assert amplitude_marginal_log_likelihood(zero,np.ones((1,1)),1.)[0] == -np.inf


def test_amplitude_marginal_extreme_tail_is_finite():
    from cosmic_integration.observation.grid_likelihood import amplitude_marginal_log_likelihood
    value,beta=amplitude_marginal_log_likelihood(np.array([[120000.]]),np.zeros((1,1)),.1)
    expected=-beta*.005+np.log(-np.expm1(-beta*.01))-np.log(beta*.01)
    np.testing.assert_allclose(value,expected,atol=1e-8)
