"""Experimental GPry-style acquisition and stopping for the COMPAS surrogate.

Equations 15/16 and Appendix A of arXiv:2211.02045 inspire the acquisition and
counter. This is an independent JAX implementation, not a GPry dependency or a
reproduction of its complete GP/SVM model. The existing sqrt target is retained.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import paramax
from scipy.stats import chi2

from .adaptive_robust_scalar import AdaptiveRobustScaler
from .jax_active_learner import FittedJaxGP


@dataclass
class CorrectCounter:
    """Count consecutive pre-training predictions satisfying paper Eq. 16.

    ``best_before`` and all predictions must be frozen before evaluating the
    batch. A failure anywhere resets the streak. Batch acceptance depends on
    the final streak; an early pass cannot hide a later failure in that batch.
    """
    dimension: int
    relative: float = .01
    absolute: float | None = None
    required: int | None = None
    streak: int = 0

    def __post_init__(self) -> None:
        if self.dimension < 1:
            raise ValueError("A positive dimension is required")
        if self.absolute is None:
            self.absolute = .01*float(chi2.ppf(math.erf(1/math.sqrt(2)), self.dimension))
        if self.required is None:
            self.required = max(4, math.ceil(self.dimension/2))
        if self.absolute <= 0 or self.relative < 0 or self.required < 1:
            raise ValueError("Invalid convergence tolerances")

    def update(self, predicted: np.ndarray, actual: np.ndarray, best_before: float) -> dict:
        predicted, actual = np.asarray(predicted), np.asarray(actual)
        if predicted.ndim != 1 or actual.shape != predicted.shape or not len(actual):
            raise ValueError("Matching nonempty label vectors are required")
        tolerance = self.absolute+self.relative*abs(best_before-predicted)
        correct = (np.isfinite(predicted) & np.isfinite(actual) & np.isfinite(best_before)
                   & (abs(predicted-actual) < tolerance))
        for ok in correct:
            self.streak = self.streak+1 if ok else 0
        return {"passed": self.streak >= self.required, "streak": self.streak,
                "correct": correct.tolist(), "tolerance": tolerance.tolist()}


def matern52(x: jax.Array, y: jax.Array, lengthscale: jax.Array,
             amplitude: jax.Array) -> jax.Array:
    """ARD Matérn-5/2 covariance; amplitude is the kernel variance."""
    distance2 = jnp.sum(((x[:, None, :]-y[None, :, :])/lengthscale)**2, axis=-1)
    r = jnp.sqrt(jnp.maximum(5*distance2, 1e-30))
    return amplitude*(1+r+r*r/3)*jnp.exp(-r)


@jax.jit
def _condition(state: dict, target: jax.Array) -> dict:
    mask = state["mask"]
    matrix = matern52(state["x"], state["x"], state["lengthscale"], state["amplitude"])
    matrix *= mask[:, None]*mask[None, :]
    matrix += jnp.diag(jnp.where(mask > 0, state["noise"], 1.))
    chol = jnp.linalg.cholesky(matrix)
    inverse = jsp.linalg.cho_solve((chol, True), jnp.eye(len(mask)))
    alpha = jsp.linalg.cho_solve((chol, True), (target-state["constant"])*mask)
    return {**state, "inverse": inverse, "alpha": alpha}


def condition_on_data(template: dict, unit: np.ndarray, lnl: np.ndarray) -> dict:
    """Condition all labels using frozen hyperparameters and target transform.

    Padding is algebraically inactive. Fixed capacity prevents recompilation
    merely because one small BO batch has enlarged the training set.
    """
    unit, lnl = np.asarray(unit), np.asarray(lnl)
    capacity = len(template["mask"])
    if unit.ndim != 2 or lnl.shape != (len(unit),) or not 0 < len(unit) <= capacity:
        raise ValueError("Training labels must fit the fixed capacity")
    if not np.isfinite(unit).all() or not np.isfinite(lnl).all():
        raise ValueError("Finite training coordinates and labels are required")
    x = np.zeros((capacity, unit.shape[1])); x[:len(unit)] = unit
    mask = np.zeros(capacity); mask[:len(unit)] = 1
    target = np.zeros(capacity)
    target[:len(unit)] = np.sqrt(np.maximum(float(template["reference"])
        -np.maximum(lnl, float(template["floor"])), 0))/float(template["scale"])
    state = {k: v for k, v in template.items() if k not in ["inverse", "alpha"]}
    state.update(x=jnp.asarray(x), mask=jnp.asarray(mask))
    return _condition(state, jnp.asarray(target))


def state_from_gpjax(model: FittedJaxGP, scaler: AdaptiveRobustScaler,
                     low: np.ndarray, width: np.ndarray, capacity: int) -> dict:
    """Cache an exact GPJax Matérn-5/2, constant-mean, sqrt-target prediction.

    Unsupported transforms/kernels are rejected rather than approximated.
    Physical-coordinate GP hyperparameters are converted to unit coordinates.
    """
    p = paramax.unwrap(model.posterior)
    if (type(p.prior.kernel).__name__ != "Matern52" or scaler.compression != "sqrt"
            or scaler.soft_clipping or scaler.median != 0):
        raise ValueError("Requires Matérn-5/2 and an unclipped-in-transformed-space sqrt target")
    width = np.asarray(width)
    if np.any(width <= 0) or capacity < model.data.n:
        raise ValueError("Positive widths and sufficient capacity are required")
    state = dict(x=jnp.zeros((capacity, len(width))), mask=jnp.zeros(capacity),
        lengthscale=jnp.asarray(p.prior.kernel.lengthscale)/width,
        amplitude=jnp.asarray(p.prior.kernel.variance),
        constant=jnp.asarray(p.prior.mean_function.constant).reshape(()),
        noise=jnp.asarray(p.likelihood.obs_stddev**2+p.jitter),
        jitter=jnp.asarray(p.jitter),
        reference=jnp.asarray(scaler.reference_value), scale=jnp.asarray(scaler.scale),
        floor=jnp.asarray(scaler.lower_clip_value if scaler.lower_clip_value is not None else -jnp.inf))
    # Recover raw labels from the existing training target. The caller replaces
    # these with raw labels when conditioning subsequently acquired points.
    raw = scaler.inverse_transform(-model.data.observations[:, 0])
    return condition_on_data(state, (model.data.query_points-low)/width, raw)


def latent_mean(state: dict, point: jax.Array) -> jax.Array:
    k = matern52(state["x"], point[None, :], state["lengthscale"], state["amplitude"])[:, 0]*state["mask"]
    return state["constant"]+k@state["alpha"]


def log_likelihood(state: dict, point: jax.Array) -> jax.Array:
    return state["reference"]-(state["scale"]*jnp.maximum(latent_mean(state, point), 0))**2


predict_lnl = jax.jit(jax.vmap(log_likelihood, in_axes=(None, 0)))


@jax.jit
def prepare_fantasies(state: dict, pending: jax.Array, mask: jax.Array) -> dict:
    """Condition covariance on mean-valued temporary labels (Kriging believer)."""
    cross = matern52(state["x"], pending, state["lengthscale"], state["amplitude"])*state["mask"][:, None]
    beta = state["inverse"]@cross
    cov = matern52(pending, pending, state["lengthscale"], state["amplitude"])-cross.T@beta
    cov *= mask[:, None]*mask[None, :]
    cov += jnp.diag(jnp.where(mask > 0, state["noise"], 1.))
    return {"points": pending, "mask": mask, "beta": beta, "precision": jnp.linalg.inv(cov)}


def latent_prediction(state: dict, point: jax.Array, fantasies: dict) -> tuple:
    k = matern52(state["x"], point[None, :], state["lengthscale"], state["amplitude"])[:, 0]*state["mask"]
    mean = state["constant"]+k@state["alpha"]
    variance = state["amplitude"]-k@state["inverse"]@k
    cross = (matern52(point[None, :], fantasies["points"], state["lengthscale"], state["amplitude"])[0]
             -k@fantasies["beta"])*fantasies["mask"]
    variance -= cross@fantasies["precision"]@cross
    # GPJax adds numerical jitter to the predictive diagonal as well. Preserve
    # that convention, without adding an observation-noise realization.
    return mean, jnp.maximum(variance+state["jitter"], 1e-18)


def sqrt_lnl_standard_deviation(mean: jax.Array, variance: jax.Array,
                                scale: jax.Array) -> jax.Array:
    """Std of reference - scale² max(Z,0)² for Gaussian transformed target Z.

    Compute truncated-normal second/fourth moments, retaining the peak clamp.
    This propagates the fitted GP uncertainty; it does not assert calibration.
    """
    s = jnp.sqrt(jnp.maximum(variance, 1e-18)); a = mean/s
    phi = jnp.exp(-.5*a*a)/jnp.sqrt(2*jnp.pi); cdf = jsp.special.ndtr(a)
    second = (mean*mean+variance)*cdf+mean*s*phi
    fourth = (mean**4+6*mean*mean*variance+3*variance**2)*cdf+(mean**3*s+5*mean*s**3)*phi
    var_square = jnp.where(a > 8, 4*mean*mean*variance+2*variance**2, fourth-second**2)
    return scale**2*jnp.sqrt(jnp.maximum(var_square, 1e-24))


def log_acquisition(state: dict, point: jax.Array, fantasies: dict, zeta: float) -> jax.Array:
    """GPry Eq. 15 form using uncertainty propagated to original LnL units.

    Only latent uncertainty is used; no new observation noise is added.
    Subtracting the reference is an x-independent stabilizing constant.
    """
    mean, variance = latent_prediction(state, point, fantasies)
    drop = -(state["scale"]*jnp.maximum(mean, 0))**2
    std = sqrt_lnl_standard_deviation(mean, variance, state["scale"])
    log_expm1 = std+jnp.log(-jnp.expm1(-std))
    return 2*zeta*drop+log_expm1


acquisition_value_gradient = jax.jit(jax.value_and_grad(log_acquisition, argnums=1))
acquisition_values = jax.jit(jax.vmap(log_acquisition, in_axes=(None, 0, None, None)))
