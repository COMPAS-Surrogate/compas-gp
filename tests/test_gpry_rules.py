"""Numerical parity and stopping semantics for the COMPAS GPry adaptation."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from cosmic_integration.lnl_surrogate.gpry_rules import (
    CorrectCounter, state_from_gpjax, condition_on_data, latent_prediction,
    prepare_fantasies, predict_lnl, sqrt_lnl_standard_deviation,
    log_acquisition, acquisition_value_gradient,
)
from cosmic_integration.lnl_surrogate.jax_active_learner import (
    FittedJaxGP, JaxTrainingData, JaxGPConfig, fit_exact_gp,
)
from cosmic_integration.lnl_surrogate.adaptive_robust_scalar import AdaptiveRobustScaler


def test_counter_matches_paper_scaling_and_resets_after_any_failure():
    counter = CorrectCounter(3)
    assert counter.absolute == pytest.approx(.03526740380261718)
    assert not counter.update(np.zeros(3), np.zeros(3), 0)["passed"]
    assert counter.update(np.zeros(1), np.zeros(1), 0)["passed"]
    # A pass early in a batch must not hide a later failure.
    result = counter.update(np.zeros(3), [0, 1, 0], 0)
    assert not result["passed"] and result["streak"] == 1
    # Equation 16 scales with the frozen prediction, not the revealed label.
    result = counter.update(np.array([-100.]), np.array([-100.5]), 0)
    assert result["tolerance"] == pytest.approx([1.0352674038026172])


def test_counter_invariance_and_nonfinite_rejection():
    a, b = CorrectCounter(3), CorrectCounter(3)
    pred, actual = np.array([-4., -2., 0.]), np.array([-4.02, -2.5, .01])
    assert a.update(pred, actual, 0) == b.update(pred+50, actual+50, 50)
    a.update(np.zeros(4), np.zeros(4), 0)
    assert not a.update(np.array([np.nan]), np.array([0.]), 0)["passed"]
    with pytest.raises(ValueError): CorrectCounter(0)


@pytest.fixture(scope="module")
def fitted():
    unit = np.array([[.05, .15], [.2, .8], [.35, .35], [.6, .7], [.85, .15], [.95, .95]])
    low, width = np.array([-3., 4.]), np.array([2., 7.])
    y = -15*np.sum((unit-[.48, .52])**2, axis=1)
    scaler = AdaptiveRobustScaler(compression="sqrt", soft_clipping=False, lower_clip_value=y.min())
    scaler.initialize_with_data(y)
    model = fit_exact_gp(JaxTrainingData(low+width*unit, -scaler.transform(y)),
        np.array([low, low+width]), config=JaxGPConfig(optimisation_steps=8), seed=29)
    return model, scaler, low, width, unit, y


def test_padded_prediction_and_conditioning_match_gpjax(fitted):
    model, scaler, low, width, unit, y = fitted
    state = state_from_gpjax(model, scaler, low, width, 16)
    test = jnp.array([[.45, .61], [.8, .5], [.1, .1]])
    empty = prepare_fantasies(state, jnp.zeros((3, 2)), jnp.zeros(3))
    got = np.array([latent_prediction(state, p, empty) for p in test])
    mean, var = model.predict_f(low+width*np.asarray(test))
    np.testing.assert_allclose(got[:, 0], mean[:, 0], atol=2e-9)
    np.testing.assert_allclose(got[:, 1], var[:, 0], atol=2e-9)
    np.testing.assert_allclose(predict_lnl(state, test), scaler.inverse_transform(-mean[:, 0]), atol=1e-8)
    enlarged_x = np.vstack([unit, np.asarray(test[:1])]); enlarged_y = np.r_[y, -.8]
    enlarged = condition_on_data(state, enlarged_x, enlarged_y)
    direct = FittedJaxGP(model.posterior,
        JaxTrainingData(low+width*enlarged_x, -scaler.transform(enlarged_y)), model.objective_history)
    mean, _ = direct.predict_f(low+width*np.asarray(test))
    np.testing.assert_allclose(predict_lnl(enlarged, test), scaler.inverse_transform(-mean[:, 0]), atol=1e-8)


def test_kriging_believer_matches_explicit_mean_label_conditioning(fitted):
    model, scaler, low, width, unit, y = fitted
    state = state_from_gpjax(model, scaler, low, width, 16)
    point = np.array([.48, .62]); query = jnp.array([.5, .5])
    mean, _ = model.predict_f((low+width*point)[None, :])
    augmented = FittedJaxGP(model.posterior, JaxTrainingData(
        np.vstack([model.data.query_points, low+width*point]),
        np.r_[model.data.observations[:, 0], mean[0, 0]]), model.objective_history)
    pending = jnp.array([point, [0., 0.], [0., 0.]])
    fantasies = prepare_fantasies(state, pending, jnp.array([1., 0., 0.]))
    got = latent_prediction(state, query, fantasies)
    expected = augmented.predict_f((low+width*np.asarray(query))[None, :])
    np.testing.assert_allclose(got, [float(expected[0][0, 0]), float(expected[1][0, 0])], atol=1e-8)


@pytest.mark.parametrize("mean", [-3., -.3, 0., 1., 10.])
def test_sqrt_uncertainty_matches_integrated_transformed_distribution(mean):
    variance, scale = .49, 2.3
    m2 = quad(lambda z: z**2*norm.pdf(z, mean, np.sqrt(variance)), 0, np.inf, epsabs=1e-11)[0]
    m4 = quad(lambda z: z**4*norm.pdf(z, mean, np.sqrt(variance)), 0, np.inf, epsabs=1e-11)[0]
    expected = scale**2*np.sqrt(m4-m2*m2)
    assert float(sqrt_lnl_standard_deviation(mean, variance, scale)) == pytest.approx(expected, rel=2e-7, abs=1e-9)


def test_acquisition_autodiff_matches_finite_differences(fitted):
    model, scaler, low, width, _, _ = fitted
    state = state_from_gpjax(model, scaler, low, width, 16)
    empty = prepare_fantasies(state, jnp.zeros((3, 2)), jnp.zeros(3))
    point = jnp.array([.41, .58]); zeta = 2**(-.85)
    value, gradient = acquisition_value_gradient(state, point, empty, zeta)
    step = 1e-5
    numerical = [(log_acquisition(state, point+step*e, empty, zeta)
                 -log_acquisition(state, point-step*e, empty, zeta))/(2*step) for e in np.eye(2)]
    assert np.isfinite(value)
    np.testing.assert_allclose(gradient, numerical, rtol=1e-4, atol=1e-4)
