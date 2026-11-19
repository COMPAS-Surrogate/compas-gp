"""Independent known-answer controls for the bounded submission diagnostics."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

spec = importlib.util.spec_from_file_location('submission_audit',
    Path(__file__).parents[1]/'docs/studies/paper_completion/submission_audit.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.mark.parametrize('mean,variance', [(0., 1.), (-2., .3), (2., .2), (3., 0.)])
def test_expected_likelihood_matches_independent_normal_integral(mean, variance):
    if variance:
        integral = quad(lambda z: np.exp(-max(.7*z, 0)**2)
            * norm.pdf(z, mean, np.sqrt(variance)), -np.inf, np.inf,
            epsabs=1e-11)[0]
    else:
        integral = np.exp(-max(.7*mean, 0)**2)
    actual = audit.expected_log_likelihood(np.array([mean]), np.array([variance]), .7, 10.)
    assert actual[0] == pytest.approx(10.+np.log(integral), abs=1e-9)


def test_constant_likelihood_offset_cancels_from_posterior_error():
    exact = -np.linspace(-3, 3, 100)**2
    result = audit.residual_metrics(exact, exact+7., np.zeros(100))
    assert result['raw_rmse'] == pytest.approx(7.)
    assert result['centered_rmse'] < 1e-12
    assert abs(result['symmetric_kl']) < 1e-12


@pytest.mark.parametrize('number,expected', [(0, 1), (10, 10), (108, 100), (986, 1000)])
def test_explicit_amplitude_likelihood_matches_truncated_gamma(number, expected):
    result = audit.amplitude_check(number, expected)
    assert result['mean_error'] < 1e-12
    assert result['sd_error'] < 1e-12
