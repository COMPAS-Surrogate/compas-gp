import numpy as np
import pytest

from cosmic_integration.ratesSampler.grid_diagnostics import (
    effective_size, nested_mass_edges, shape_js, system_grid_moments,
)


def test_redshift_covariance_and_mass_boundaries():
    # One system contributes to both z cells: merging cannot create two draws.
    rates = np.array([[2., 3.], [4., 1.], [0., 2.]])
    edges = np.array([0., 1., 2., np.inf])
    total, square = system_grid_moments(rates, np.array([.5, 1., 3.]), edges, [0, 2])
    np.testing.assert_allclose(total[:, 0], [5., 5., 2.])
    np.testing.assert_allclose(square[:, 0], [25., 25., 4.])
    np.testing.assert_allclose(effective_size(total, square), 1.)
    assert total.sum() == rates.sum()


def test_merged_mass_size_and_empty_bins():
    total, square = system_grid_moments(
        np.array([[1., 1.], [1., 1.]]), [.5, .8], [0., 1., np.inf], [0, 2])
    np.testing.assert_allclose(effective_size(total, square)[:, 0], [2., 0.])


def test_nested_edges_preserve_guards():
    np.testing.assert_array_equal(nested_mass_edges([.5, 1., 2., 4.], 2),
                                  [0., .5, 2., 4., np.inf])
    with pytest.raises(ValueError):
        nested_mass_edges([.5, 1.], 0)
    with pytest.raises(ValueError):
        system_grid_moments(np.ones((1, 2)), [.5], [0., 1.], [0, 1])


def test_js_is_amplitude_invariant_and_contracts_on_merging():
    p, q = np.array([1., 4., 2., 1.]), np.array([4., 1., 1., 2.])
    assert shape_js(p, 7 * p) == pytest.approx(0., abs=1e-14)
    assert shape_js(p.reshape(2, 2).sum(1), q.reshape(2, 2).sum(1)) <= shape_js(p, q)
