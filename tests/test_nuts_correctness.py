from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import expit

from cosmic_integration.lnl_surrogate.adaptive_robust_scalar import AdaptiveRobustScaler
from cosmic_integration.lnl_surrogate.nuts_sampler import (
    _inverse_transform_jax, _unconstrained_starts, chain_diagnostics, make_log_likelihood,
)


@pytest.mark.parametrize("compression", ["none", "sqrt", "log", "softlog"])
@pytest.mark.parametrize("soft", [False, True])
def test_numpy_jax_inverse_parity(compression, soft):
    scaler = AdaptiveRobustScaler(compression=compression, soft_clipping=soft)
    scaler.initialize_with_data(np.array([-100., -30., -10., -2., 0.]))
    transformed = np.array([-2.9, -1., -.1, 0., .1])
    np.testing.assert_allclose(_inverse_transform_jax(jnp.asarray(transformed), scaler),
                               scaler.inverse_transform(transformed), rtol=1e-10, atol=1e-10)


def test_physical_starts_are_unconstrained_and_single_chain_shape():
    low, high = jnp.array([-.5, .005]), jnp.array([-.001, .015])
    points = np.array([[-.325, .012], [-.1, .007]])
    state = _unconstrained_starts(lambda x: 0., low, high, points)["theta"]
    np.testing.assert_allclose(low+(high-low)*expit(state), points)
    assert _unconstrained_starts(lambda x: 0., low, high, points[:1])["theta"].shape == (2,)
    with pytest.raises(ValueError):
        _unconstrained_starts(lambda x: 0., low, high, [low])


def test_invalid_marginal_is_rejected():
    scaler = AdaptiveRobustScaler(compression="sqrt", soft_clipping=False)
    with pytest.raises(ValueError, match="nonlinear"):
        make_log_likelihood(SimpleNamespace(), scaler, target="marginal")


def test_diagnostics_detect_drift_scale_failure_and_stuck_chains():
    rng = np.random.default_rng(7)
    iid = rng.normal(size=(4, 2000, 1))
    r, bulk, tail = chain_diagnostics(iid)
    assert r[0] < 1.01 and bulk[0] > 400 and tail[0] > 400
    drift = iid.copy()
    drift[:, 1000:] += 3
    assert chain_diagnostics(drift)[0][0] > 1.1
    scale = iid.copy()
    scale[0] *= 10
    assert chain_diagnostics(scale)[0][0] > 1.1
    assert np.isnan(chain_diagnostics(np.zeros((4, 100, 1)))[0][0])
    assert np.isnan(chain_diagnostics(iid[:1])[0][0])
