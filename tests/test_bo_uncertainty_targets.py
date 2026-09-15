"""Check the experimental uncertainty target independently of GP fitting."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm
from cosmic_integration.lnl_surrogate.adaptive_robust_scalar import AdaptiveRobustScaler


@pytest.fixture
def study(monkeypatch):
    path=Path(__file__).resolve().parents[1]/'docs/studies/joint_pilot'
    monkeypatch.syspath_prepend(str(path))
    spec=importlib.util.spec_from_file_location('bo_curve_test',path/'bo_uncertainty_curve.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('power',[1.,2.])
@pytest.mark.parametrize('mean,variance',[(-2.,.3),(0.,1.),(2.,.3),(1.,4.),(5.,.01)])
def test_expected_likelihood_through_sqrt_matches_integral(study,mean,variance,power):
    scaler=AdaptiveRobustScaler(compression='sqrt',soft_clipping=False)
    scaler.initialize_with_data(np.array([-9.,-4.,-1.]));scaler.scale=1.7
    actual=study.sqrt_log_expected_likelihood(np.array([mean]),np.array([variance]),scaler,power=power)[0]
    m=mean*scaler.scale;s=np.sqrt(variance)*scaler.scale
    # Rescale the tiny integrand so adaptive quadrature cannot stop at its
    # absolute tolerance without resolving a narrow, far-from-zero peak.
    upper=max(m+12*s,12*s)
    probe=np.linspace(0,upper,1000)
    shift=np.max(-power*probe**2+norm.logpdf(probe,loc=m,scale=s))
    integral=quad(lambda q:np.exp(-power*q*q+norm.logpdf(q,loc=m,scale=s)-shift),0,upper,epsabs=1e-12)[0]
    expected=norm.cdf(0,loc=m,scale=s)+np.exp(shift)*integral
    np.testing.assert_allclose(actual,power*scaler.reference_value+np.log(expected),atol=1e-9)


def test_zero_variance_recovers_plugin_and_kl_ignores_offset(study):
    scaler=AdaptiveRobustScaler(compression='sqrt',soft_clipping=False)
    scaler.initialize_with_data(np.array([-9.,-4.,-1.]))
    means=np.array([-2.,0.,1.,3.])
    np.testing.assert_allclose(study.sqrt_log_expected_likelihood(means,np.zeros(4),scaler),scaler.inverse_transform(-means))
    phi=np.random.default_rng(5).uniform(size=(100,3));lnl=-np.sum(phi**2,axis=1)
    result=study.posterior_metrics(lnl,lnl+43.,phi)
    assert abs(result['kl_direct_to_gp'])<1e-12
    assert abs(result['kl_gp_to_direct'])<1e-12
