"""Cheap noisy-toy calibration controls, GP interpolation and NUTS checks.

Run after the fast tests. This does not establish COMPAS or full-workflow GP SBC.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer import MCMC, NUTS
from scipy.integrate import cumulative_trapezoid
from scipy.stats import truncnorm, kstest

from cosmic_integration.lnl_surrogate.adaptive_robust_scalar import AdaptiveRobustScaler
from cosmic_integration.lnl_surrogate.jax_active_learner import (
    JaxGPConfig, JaxTrainingData, fit_exact_gp,
)
from cosmic_integration.lnl_surrogate.lnl_surrogate import BOUNDS
from cosmic_integration.lnl_surrogate.nuts_sampler import (
    _numpyro_model, _unconstrained_starts, chain_diagnostics, make_log_likelihood,
)


def main() -> None:
    started = time.perf_counter()
    rng = np.random.default_rng(20260909)
    low, high = BOUNDS
    span = high-low
    sd = .05*span
    # Proper generative experiment: parameters from prior, then noisy data.
    truth = rng.uniform(low, high, size=(1000, 4))
    observed = truth+rng.normal(size=truth.shape)*sd
    posterior = truncnorm((low-observed)/sd, (high-observed)/sd, loc=observed, scale=sd)
    median = posterior.ppf(.5)
    ranks = {
        "exact": posterior.cdf(truth),
        "intervals_halved": posterior.cdf(median+(truth-median)/.5),
        "shifted_one_sigma": posterior.cdf(truth-sd),
    }
    controls = {}
    for label, rank in ranks.items():
        controls[label] = {"coverage68": np.mean(np.abs(rank-.5) <= .34, axis=0).tolist(),
                           "coverage95": np.mean(np.abs(rank-.5) <= .475, axis=0).tolist(),
                           "ks_statistics": [float(kstest(rank[:, i], "uniform").statistic) for i in range(4)]}
    assert np.max(np.abs(np.array(controls["exact"]["coverage68"])-.68)) < .06
    assert np.max(controls["intervals_halved"]["coverage68"]) < .5
    assert np.max(controls["shifted_one_sigma"]["coverage68"]) < .58

    # Actual production model/initialization path, with an exactly known target.
    obs = low+.55*span
    def log_likelihood(theta):
        return -.5*jnp.sum(((theta-jnp.asarray(obs))/jnp.asarray(sd))**2)
    initial = low+np.array([.2, .4, .6, .8])[:, None]*span
    mcmc = MCMC(NUTS(_numpyro_model, target_accept_prob=.9), num_warmup=500,
                num_samples=1500, num_chains=4, chain_method="sequential", progress_bar=False)
    mcmc.run(jax.random.PRNGKey(42), log_likelihood, jnp.asarray(low), jnp.asarray(high),
             init_params=_unconstrained_starts(log_likelihood, low, high, initial),
             extra_fields=("diverging",))
    draws = np.asarray(mcmc.get_samples(group_by_chain=True)["theta"])
    rh, bulk, tail = chain_diagnostics(draws)
    divergences = int(np.sum(np.asarray(mcmc.get_extra_fields()["diverging"])))
    error = (draws.mean(axis=(0, 1))-obs)/sd
    width = draws.std(axis=(0, 1))/sd
    assert np.max(np.abs(error)) < .1 and np.max(np.abs(width-1)) < .1
    assert np.max(rh) < 1.01 and min(bulk.min(), tail.min()) > 400 and divergences == 0

    # One dimensional GP baseline: known noisy observation; independent grid scoring.
    x = np.linspace(-3, 3, 61)[:, None]
    observed_1d, sigma_1d = .37, .7
    lnl = -.5*((x[:, 0]-observed_1d)/sigma_1d)**2
    scaler = AdaptiveRobustScaler(compression="sqrt", soft_clipping=False, max_scale=1e9)
    scaler.initialize_with_data(lnl)
    config = JaxGPConfig(kernel="matern52", optimisation_steps=100, candidate_count=128)
    model = fit_exact_gp(JaxTrainingData(x, -scaler.transform(lnl)[:, None]),
                         np.array([[-3.], [3.]]), config=config, seed=17)
    grid = np.linspace(-3, 3, 2001)
    target = make_log_likelihood(model, scaler, target="mean")
    predicted = np.asarray(jax.jit(jax.vmap(lambda v: target(v.reshape(1))))(jnp.asarray(grid)))
    density = np.exp(predicted-predicted.max())
    cdf = np.r_[0., cumulative_trapezoid(density, grid)]
    cdf /= cdf[-1]
    exact = truncnorm((-3-observed_1d)/sigma_1d, (3-observed_1d)/sigma_1d,
                      loc=observed_1d, scale=sigma_1d)
    probabilities = np.array([.025, .16, .5, .84, .975])
    delta_q = (np.interp(probabilities, cdf, grid)-exact.ppf(probabilities))/sigma_1d
    cdf_error = float(np.max(np.abs(cdf-exact.cdf(grid))))
    assert cdf_error < .02 and np.max(np.abs(delta_q)) < .1
    result = {"scope": "Calibration controls, exact-target NUTS, 1D GP; not full GP SBC",
              "calibration_simulations": 1000, "controls": controls,
              "nuts": {"rhat": rh.tolist(), "bulk_ess": bulk.tolist(), "tail_ess": tail.tolist(),
                       "mean_error_sigma": error.tolist(), "width_ratio": width.tolist(),
                       "divergences": divergences},
              "gp_1d": {"max_cdf_error": cdf_error, "quantile_error_sigma": delta_q.tolist(),
                        "training_points": 61, "config": config.__dict__, "target": "mean"},
              "seconds": time.perf_counter()-started}
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    (out / "sampler_baseline.json").write_text(json.dumps(result, indent=2)+"\n")
    np.save(out / "sampler_baseline_chains.npy", draws)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
