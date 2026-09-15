"""Regression checks for the experimental frozen-kernel append benchmark."""
import importlib.util
from pathlib import Path
import jax.numpy as jnp
import numpy as np
import pytest
from cosmic_integration.lnl_surrogate.gpry_rules import condition_on_data, predict_lnl, prepare_fantasies, acquisition_values

spec = importlib.util.spec_from_file_location('runtime_profile', Path(__file__).resolve().parents[1]/'docs/studies/amplitude_gp/profile_runtime.py')
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


def test_repeated_append_preserves_predictions_and_acquisition():
    rng = np.random.default_rng(9)
    unit = rng.random((39, 4)); values = -30*np.sum((unit-.4)**2, axis=1)
    template = dict(x=jnp.zeros((48,4)), mask=jnp.zeros(48), lengthscale=jnp.ones(4)*.3,
        amplitude=jnp.array(2.), constant=jnp.array(.2), noise=jnp.array(1e-5),
        jitter=jnp.array(1e-6), reference=jnp.array(0.), floor=jnp.array(-10.), scale=jnp.array(2.))
    state = condition_on_data(template, unit[:9], values[:9])
    query = jnp.asarray(rng.random((20,4)))
    for n in range(9, 39, 3):
        state = profile.append_frozen(state, unit[n:n+3], values[n:n+3])
        expected = condition_on_data(template, unit[:n+3], values[:n+3])
        np.testing.assert_allclose(predict_lnl(state,query),predict_lnl(expected,query),rtol=1e-9,atol=1e-9)
        a = prepare_fantasies(state,jnp.zeros((3,4)),jnp.zeros(3))
        b = prepare_fantasies(expected,jnp.zeros((3,4)),jnp.zeros(3))
        np.testing.assert_allclose(acquisition_values(state,query,a,.3),acquisition_values(expected,query,b,.3),rtol=1e-9,atol=1e-9)
    with pytest.raises(ValueError):profile.append_frozen(state,unit[:12],values[:12])
    with pytest.raises(ValueError):profile.append_frozen(state,unit[:3],values[:2])
