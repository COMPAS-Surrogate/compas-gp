"""Scientific contracts for event-weight displays and posterior intensity maps."""
from pathlib import Path
import numpy as np
import pytest


@pytest.fixture
def plots(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'docs/studies/paper_completion'))
    import weight_rate_figures
    return weight_rate_figures


def test_weights_preserve_event_count_and_ignore_event_normalization(plots):
    coeff = np.array([[1., 3., 0.], [0., 1., 0.]])
    weights = plots.event_weights(coeff)
    np.testing.assert_allclose(weights, [[.25,.75,0], [0,1,0]])
    np.testing.assert_allclose(plots.event_weights(coeff*np.array([[100.],[.001]])), weights)
    assert weights.sum() == 2
    with pytest.raises(ValueError):
        plots.event_weights(np.zeros((1,3)))


def test_intensity_retains_amplitude_uncertainty_without_poisson_noise(plots):
    rates = np.array([[30., 90.], [60., 60.]])
    draws = plots.intensity_draws(rates, np.array([0., -np.inf]), 1., 99, draws=20000, seed=12)
    # Gamma(100, beta=10000) amplitude is nearly untruncated. Intensity totals
    # have mean=100, variance=100; adding Poisson draws would double variance.
    totals = draws.sum(axis=1)
    assert abs(totals.mean()-100) < .5
    assert abs(totals.var()-100) < 5
    np.testing.assert_allclose(draws[:,1], 3*draws[:,0])


def test_truth_region_uses_density_and_boundary_removes_internal_edges(plots):
    # Large first bin has most mass but lower density than the second bin.
    selected = plots.highest_density_bins(np.array([6.,4.]), np.array([10.,1.]), .3)
    np.testing.assert_array_equal(selected, [False,True])
    lo, hi = np.array([[0.,0.],[1.,0.]]), np.array([[1.,1.],[2.,1.]])
    edges = plots.boundary_segments(lo, hi, np.array([True,True]))
    assert len(edges) == 6
    assert not any(np.allclose(edge, [(1.,0.),(1.,1.)]) for edge in edges)
