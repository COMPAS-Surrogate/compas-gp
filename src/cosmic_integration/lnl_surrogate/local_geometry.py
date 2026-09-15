"""Differentiable GP means and local checks in posterior-standardized units.

These diagnostics target the COMPAS LnL surrogate. They do not change its
training transform, prior, acquisition policy, or stopping defaults.
"""
from __future__ import annotations

from collections.abc import Callable
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from .jax_active_learner import FittedJaxGP
from .adaptive_robust_scalar import AdaptiveRobustScaler
from .nuts_sampler import _inverse_transform_jax


@eqx.filter_jit
def _mean_weights(posterior, x, y):
    covariance = posterior.prior.kernel.gram(x).as_matrix()
    covariance += jnp.diag(posterior.likelihood.noise_vector(len(x)) + posterior.jitter)
    return jsp.linalg.cho_solve((jnp.linalg.cholesky(covariance), True),
                                y-posterior.prior.mean_function(x))


def cached_log_likelihood(model: FittedJaxGP, scaler: AdaptiveRobustScaler,
                          low: np.ndarray, width: np.ndarray) -> Callable:
    """Return a JAX scalar LnL on unit coordinates, factoring the GP once.

    Differentiate the inverse-transformed mean, including the nonlinear target
    transform. This agrees with the existing deterministic GP plug-in target.
    Predictive variances and repeated training-matrix factorizations are avoided.
    """
    x, y = jnp.asarray(model.data.query_points), jnp.asarray(model.data.observations)
    alpha = _mean_weights(model.posterior, x, y)
    low, width = jnp.asarray(low), jnp.asarray(width)
    if low.shape != (x.shape[1],) or width.shape != low.shape or np.any(np.asarray(width) <= 0):
        raise ValueError("Positive coordinate widths matching the training dimension are required")
    def log_likelihood(unit):
        query = (low+width*unit)[None, :]
        mean = (model.posterior.prior.mean_function(query)
                + model.posterior.prior.kernel.cross_covariance(x, query).T @ alpha)[0, 0]
        return _inverse_transform_jax(-mean, scaler)
    return log_likelihood


def geometry_change(old_gradients: np.ndarray, new_gradients: np.ndarray,
                    old_hessian: np.ndarray, new_hessian: np.ndarray,
                    covariance: np.ndarray) -> dict:
    """Measure changes to GP scores/curvature in posterior SD coordinates."""
    root = np.linalg.cholesky(covariance)
    delta = (np.asarray(new_gradients)-old_gradients) @ root
    curvature = root.T @ (np.asarray(new_hessian)-old_hessian) @ root
    rms = float(np.sqrt(np.mean(np.sum(delta**2, axis=1))))
    curvature_norm = float(np.linalg.norm(curvature, ord=2))
    return {"score_rms": rms, "curvature_change": curvature_norm,
            "passed": bool(np.isfinite(rms+curvature_norm) and rms < .1 and curvature_norm < .1)}


def derivative_stencil(dimension: int, steps: tuple[float, ...] = (.25, .125)) -> tuple:
    """Return shared coordinates and central-difference operators for grad/H.

    The full mixed Hessian is checked. Two resolutions share their centre,
    requiring 37 points for three parameters. Operators act on function values.
    """
    if dimension < 1 or not steps or any(not np.isfinite(h) or h <= 0 for h in steps):
        raise ValueError("A positive dimension and finite positive steps are required")
    points = [np.zeros(dimension)]
    entries = []
    for h in steps:
        axis = []
        pairs = {}
        for i in range(dimension):
            ids = []
            for sign in [-1, 1]:
                x = np.zeros(dimension); x[i] = sign*h
                ids.append(len(points)); points.append(x)
            axis.append(ids)
        for i in range(dimension):
            for j in range(i+1, dimension):
                ids = []
                for si, sj in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                    x = np.zeros(dimension); x[i], x[j] = si*h, sj*h
                    ids.append(len(points)); points.append(x)
                pairs[i, j] = ids
        entries.append((h, axis, pairs))
    gradients, hessians = [], []
    for h, axis, pairs in entries:
        g = np.zeros((dimension, len(points)))
        hess = np.zeros((dimension, dimension, len(points)))
        for i, (minus, plus) in enumerate(axis):
            g[i, minus], g[i, plus] = -.5/h, .5/h
            hess[i, i, [minus, 0, plus]] = np.array([1., -2., 1.])/h**2
        for (i, j), ids in pairs.items():
            hess[i, j, ids] = np.array([1., -1., -1., 1.])/(4*h**2)
            hess[j, i] = hess[i, j]
        gradients.append(g); hessians.append(hess)
    return np.array(points), np.array(gradients), np.array(hessians)


def derivative_audit(predicted: np.ndarray, actual: np.ndarray,
                     gradients: np.ndarray, hessians: np.ndarray) -> dict:
    """Compare exact and GP local geometry using matched finite differences.

    Matched differences avoid mistaking ordinary non-quadratic structure for GP
    error. Refinement checks finite-difference resolution, using two step sizes.
    All units are set by the posterior covariance used to construct the stencil.
    """
    predicted, actual = np.asarray(predicted), np.asarray(actual)
    if (actual.ndim != 1 or predicted.shape != actual.shape or len(gradients) != 2
            or gradients.shape[-1] != len(actual) or hessians.shape[-1] != len(actual)):
        raise ValueError("Matching labels and two complete derivative stencils are required")
    if not np.isfinite(predicted).all() or not np.isfinite(actual).all():
        return {"passed": False, "reason": "nonfinite_labels"}
    residual = actual-predicted
    residual -= residual[0]
    score_error = np.einsum("sdn,n->sd", gradients, residual)
    curvature_error = np.einsum("sijn,n->sij", hessians, residual)
    score = float(max(np.linalg.norm(g) for g in score_error))
    curvature = float(max(np.linalg.norm(h, ord=2) for h in curvature_error))
    score_resolution = float(np.linalg.norm(score_error[0]-score_error[1]))
    curvature_resolution = float(np.linalg.norm(curvature_error[0]-curvature_error[1], ord=2))
    passed = score < .1 and curvature < .1 and score_resolution < .03 and curvature_resolution < .05
    return {"passed": bool(passed), "score_error": score, "curvature_error": curvature,
            "score_resolution": score_resolution, "curvature_resolution": curvature_resolution,
            "centred_residual_max": float(np.max(abs(residual)))}
