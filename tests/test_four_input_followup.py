"""Small-batch dimension parity and independent amplitude-integration checks."""
import importlib.util
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import gamma, qmc

from cosmic_integration.lnl_surrogate.gpry_rules import condition_on_data
from cosmic_integration.lnl_surrogate.small_batch_acquisition import acquire_small_batch


def state_for_dimension(dimension):
    unit = qmc.Sobol(dimension, scramble=True, seed=11).random_base2(4)
    values = -10*np.sum((unit-.4)**2, axis=1)
    template = dict(x=jnp.zeros((32, dimension)), mask=jnp.zeros(32),
        lengthscale=jnp.full(dimension, .4), amplitude=jnp.array(1.),
        constant=jnp.array(2.), noise=jnp.array(1e-5), jitter=jnp.array(1e-6),
        reference=jnp.array(0.), scale=jnp.array(1.), floor=jnp.array(-100.))
    return condition_on_data(template, unit, values), unit, values


def test_generic_acquisition_reproduces_historical_three_dimensional_result(monkeypatch):
    folder = Path(__file__).resolve().parents[1]/'docs/studies/operational_stopping/gpry_compas'
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location('original_acquisition_parity', folder/'run_experiment.py')
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    state, unit, values = state_for_dimension(3)
    a, ad = original.acquire(state, unit, values, np.random.default_rng(19))
    b, bd = acquire_small_batch(state, unit, values, np.random.default_rng(19))
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)
    assert ad == bd


def test_four_dimensional_batch_is_distinct_in_prior_and_reproducible():
    state, unit, values = state_for_dimension(4)
    a, diagnostics = acquire_small_batch(state, unit, values, np.random.default_rng(3))
    b, _ = acquire_small_batch(state, unit, values, np.random.default_rng(3))
    assert a.shape == (3, 4) and np.all((a >= 0) & (a <= 1))
    np.testing.assert_array_equal(a, b)
    combined = np.vstack([unit, a])
    assert len(np.unique(combined, axis=0)) == len(combined)
    assert all(np.isfinite(row['log_acquisition']) for row in diagnostics)
    with pytest.raises(ValueError):
        acquire_small_batch(state, unit*2, values, np.random.default_rng(3))


def test_four_input_acquisition_gradient_and_fantasy_variance():
    """All four coordinates affect acquisition; pending labels reduce variance."""
    from cosmic_integration.lnl_surrogate.gpry_rules import (
        prepare_fantasies, acquisition_value_gradient, log_acquisition,
        latent_prediction,
    )
    state, _, _ = state_for_dimension(4)
    point = jnp.array([.31, .44, .56, .67])
    empty = prepare_fantasies(state, jnp.zeros((3, 4)), jnp.zeros(3))
    _, gradient = acquisition_value_gradient(state, point, empty, 4**(-.85))
    h = 1e-5
    numerical = [(log_acquisition(state, point+h*e, empty, 4**(-.85))
                  -log_acquisition(state, point-h*e, empty, 4**(-.85)))/(2*h)
                 for e in np.eye(4)]
    np.testing.assert_allclose(gradient, numerical, rtol=1e-4, atol=1e-4)
    assert np.all(np.abs(gradient) > 1e-6)
    pending = jnp.zeros((3, 4)).at[0].set(point)
    fantasy = prepare_fantasies(state, pending, jnp.array([1., 0., 0.]))
    mean0, var0 = latent_prediction(state, point, empty)
    mean1, var1 = latent_prediction(state, point, fantasy)
    np.testing.assert_allclose(mean0, mean1, atol=1e-10)
    assert var1 < .01*var0


@pytest.fixture
def design():
    folder = Path(__file__).resolve().parents[1]/'docs/studies/amplitude_gp'
    spec = importlib.util.spec_from_file_location('stratified_amplitude_test', folder/'stratified_reference.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.amplitude_design


def test_stratified_balance_weights_integrate_prior_and_gamma(design):
    beta = np.full(8192, 10000.)
    amplitude, logq = design(beta, 100, 16, 7)
    assert amplitude.shape == (8192, 16)
    # This tests the mixture density including its unit-coordinate Jacobian.
    assert np.mean(np.exp(-logq)) == pytest.approx(1., abs=.005)
    assert np.mean((amplitude-.005)/.01*np.exp(-logq)) == pytest.approx(.5, abs=.003)
    w = np.exp(gamma.logpdf(amplitude, 101, scale=.0001)-logq)
    mean = np.sum(amplitude*w)/np.sum(w)
    exact = gamma.expect(lambda a: a, args=(101,), scale=.0001, lb=.005, ub=.015, conditional=True)
    assert mean == pytest.approx(exact, abs=3e-6)


def test_stratified_upper_tail_underflow_and_invalid_replicas(design):
    beta = np.array([1e-6, 1e7, 30000., 10000.])
    a, logq = design(beta, 100, 16, 8)
    assert np.isfinite(a).all() and np.isfinite(logq).all()
    assert np.all((a >= .005) & (a <= .015))
    np.testing.assert_array_equal(logq[:2], 0.)
    with pytest.raises(ValueError):
        design(beta, 100, 3, 8)


def test_reference_offset_preserves_labels_floor_scale_and_smooths_peak(monkeypatch):
    folder = Path(__file__).resolve().parents[1]/'docs/studies/amplitude_gp'
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location('reference_offset_unit_test', folder/'reference_offset_check.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    values = -np.linspace(0, 100, 40)**2
    a, b = module.shifted_scaler(values, 5.), module.shifted_scaler(values, 20.)
    assert a.scale == b.scale
    assert a.lower_clip_value == b.lower_clip_value == values.min()
    assert a.reference_value == values.max()+5
    np.testing.assert_allclose(a.inverse_transform(a.transform(values)), values, atol=1e-9)
    step = 1e-5
    derivative = (a.transform(values.max()+step)-a.transform(values.max()-step))/(2*step)
    assert derivative == pytest.approx(1/(2*a.scale*np.sqrt(5.)), rel=1e-7)
    with pytest.raises(ValueError):
        module.shifted_scaler(values, 0.)
