from pathlib import Path
import numpy as np
import pytest


def test_expected_poisson_gap_and_fisher(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'docs/studies/event_count_bo'))
    from run_study import poisson_expected_gap, fisher_matrix
    mean=np.array([2.,0.,5.])
    assert poisson_expected_gap(mean,mean)==0.
    assert poisson_expected_gap(mean,np.array([2.,1.,5.]))==pytest.approx(-1)
    assert poisson_expected_gap(mean,np.array([0.,0.,5.]))==-np.inf
    for other in np.random.default_rng(17).uniform(.1,10,(20,3)):
        assert poisson_expected_gap(mean,other)<0
    weights=np.array([1.,2.,3.,4.])
    def rate(t):return weights*np.exp(t)
    theta=np.array([.1,.2,.3,.4])
    actual=fisher_matrix(rate,theta,np.ones(4),3.)
    np.testing.assert_allclose(actual,np.diag(3*rate(theta)),rtol=1e-8)


def test_reference_importance_weights_recover_uniform_prior(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'docs/studies/event_count_bo'))
    from run_study import reference_design, LOW, WIDTH, SHAPE
    phi, logq, _ = reference_design(100, np.eye(4)*100, seed=35)
    unit = (phi-LOW[SHAPE])/WIDTH[SHAPE]
    weights = np.exp(-logq)
    weights /= weights.sum()
    np.testing.assert_allclose(weights@unit, .5, atol=.008)
    np.testing.assert_allclose(weights@(unit*unit), 1/3, atol=.008)
