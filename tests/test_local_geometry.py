"""Arithmetic and autodiff checks for the COMPAS geometry diagnostics."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cosmic_integration.lnl_surrogate.local_geometry import (
    derivative_stencil, derivative_audit, geometry_change, cached_log_likelihood,
)
from cosmic_integration.lnl_surrogate.jax_active_learner import fit_exact_gp, JaxGPConfig, JaxTrainingData
from cosmic_integration.lnl_surrogate.adaptive_robust_scalar import AdaptiveRobustScaler
from cosmic_integration.lnl_surrogate.nuts_sampler import make_log_likelihood


def test_stencil_recovers_full_gradient_and_mixed_hessian():
    x, g, h = derivative_stencil(3)
    assert len(x) == 37
    gradient = np.array([.1, -.2, .3])
    matrix = np.array([[2., .3, -.2], [.3, 3., .7], [-.2, .7, 4.]])
    values = 12+x@gradient+.5*np.einsum("ni,ij,nj->n", x, matrix, x)
    np.testing.assert_allclose(np.einsum("sdn,n->sd", g, values), [gradient, gradient], atol=1e-12)
    np.testing.assert_allclose(np.einsum("sijn,n->sij", h, values), [matrix, matrix], atol=1e-11)


def test_audit_ignores_offset_but_detects_slope_and_cross_curvature():
    x, g, h = derivative_stencil(3)
    predicted = -.5*np.sum(x*x, axis=1)
    assert derivative_audit(predicted, predicted+100, g, h)["passed"]
    assert not derivative_audit(predicted, predicted+.2*x[:, 0], g, h)["passed"]
    assert not derivative_audit(predicted, predicted+.3*x[:, 0]*x[:, 1], g, h)["passed"]
    assert not derivative_audit(predicted, predicted*np.nan, g, h)["passed"]


def test_geometry_is_invariant_under_linear_unit_changes():
    cov = np.diag([.04, .09])
    old, new = np.zeros((3, 2)), np.full((3, 2), .1)
    h0, h1 = np.eye(2), np.diag([1.2, 1.3])
    a = geometry_change(old, new, h0, h1, cov)
    scale = np.diag([2., 3.]); inverse = np.linalg.inv(scale)
    b = geometry_change(old@inverse, new@inverse, inverse@h0@inverse,
                        inverse@h1@inverse, scale@cov@scale)
    assert a["score_rms"] == pytest.approx(b["score_rms"])
    assert a["curvature_change"] == pytest.approx(b["curvature_change"])


def test_cached_mean_and_derivatives_match_existing_transformed_target():
    # Small deterministic fixture checks the implementation, not scientific adequacy.
    x = np.linspace(-.5, .5, 12)[:, None]
    y = -10*(x[:, 0]-.08)**2
    scaler = AdaptiveRobustScaler(compression="sqrt", soft_clipping=False, lower_clip_value=float(y.min()))
    scaler.initialize_with_data(y)
    model = fit_exact_gp(JaxTrainingData(x, -scaler.transform(y)[:, None]), np.array([[-.5], [.5]]),
                         config=JaxGPConfig(optimisation_steps=5), seed=7)
    cached = cached_log_likelihood(model, scaler, np.array([-.5]), np.ones(1))
    original = make_log_likelihood(model, scaler)
    u = jnp.array([.7])
    assert float(cached(u)) == pytest.approx(float(original(u-.5)), abs=1e-10)
    np.testing.assert_allclose(jax.grad(cached)(u), jax.grad(original)(u-.5), rtol=1e-9, atol=1e-9)
    step = 1e-4
    finite = (float(cached(u+step))-2*float(cached(u))+float(cached(u-step)))/step**2
    assert float(jax.hessian(cached)(u)[0, 0]) == pytest.approx(finite, rel=1e-4, abs=1e-4)
