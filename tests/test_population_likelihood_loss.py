"""Check the new diagnostic against elementary Poisson and sensor models."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest


def module(monkeypatch):
    folder=Path(__file__).resolve().parents[1]/'docs/studies/paper_completion'
    monkeypatch.syspath_prepend(str(folder))
    spec=importlib.util.spec_from_file_location('loss_diagnostic',folder/'population_likelihood_loss.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def test_integrated_loss_matches_expected_poisson_log_likelihood(monkeypatch):
    m=module(monkeypatch)
    baseline=np.array([2.,5.,3.]);candidate=np.array([1.,7.,2.])
    expected=100
    p=baseline/baseline.sum()
    integral=expected*(p@m.poisson_loss_integrand(candidate/baseline))
    # Compute E[log L_baseline - log L_candidate] independently from the
    # expected bin counts, including the Poisson total-rate term.
    t=expected/baseline.sum()
    mean_counts=t*baseline
    exact=np.sum(mean_counts*np.log(baseline/candidate)+t*(candidate-baseline))
    assert integral==pytest.approx(exact)
    np.testing.assert_array_equal(m.poisson_loss_integrand(np.ones(3)),0)
    assert np.isinf(m.poisson_loss_integrand(np.array([0.]))[0])


def test_normalized_measurement_smearing_cannot_increase_expected_loss(monkeypatch):
    m=module(monkeypatch)
    baseline=np.array([2.,5.,3.]);candidate=np.array([1.,7.,2.])
    # Columns are normalized probabilities of measured categories per true bin.
    response=np.array([[.8,.1,.1],[.1,.8,.2],[.1,.1,.7]])
    observed=response@baseline;other=response@candidate
    original=baseline@m.poisson_loss_integrand(candidate/baseline)
    smeared=observed@m.poisson_loss_integrand(other/observed)
    assert 0 < smeared < original


def test_nearly_identical_populations_keep_small_positive_loss(monkeypatch):
    m=module(monkeypatch)
    ratio=1+np.array([1e-8,-1e-8,1e-6])
    loss=m.poisson_loss_integrand(ratio)
    np.testing.assert_allclose(loss,.5*(ratio-1)**2,rtol=1e-6,atol=0)
    with pytest.raises(ValueError):m.poisson_loss_integrand(np.array([-1.]))
