"""Dimension-independent form of the validated study's three-point acquisition.

This is an opt-in experimental helper; production entrypoints are unchanged.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

from .gpry_rules import prepare_fantasies, acquisition_values, acquisition_value_gradient


def acquire_small_batch(
    state: dict, unit: np.ndarray, lnl: np.ndarray, rng: np.random.Generator,
) -> tuple[np.ndarray, list[dict]]:
    """Select three novel coordinates with fantasy covariance and local refinement.

    Full-prior Sobol candidates and tempered-training local candidates reproduce
    the historical three-dimensional study, with dimension inferred from data.
    No direct likelihood or reference posterior is accepted by this function.
    """
    unit, lnl = np.asarray(unit), np.asarray(lnl)
    if (unit.ndim != 2 or not len(unit) or unit.shape[1] < 1
            or lnl.shape != (len(unit),) or not np.isfinite(unit).all()
            or not np.isfinite(lnl).all() or np.any((unit < 0) | (unit > 1))):
        raise ValueError('Finite training coordinates in the unit prior and matching labels required')
    dim = unit.shape[1]
    selected, diagnostics = [], []
    weights = np.exp((lnl-lnl.max())/2)
    weights /= weights.sum()
    for member in range(3):
        pending, mask = np.zeros((3, dim)), np.zeros(3)
        if member:
            pending[:member] = selected
        mask[:member] = 1
        fantasies = prepare_fantasies(state, jnp.asarray(pending), jnp.asarray(mask))
        broad = qmc.Sobol(dim, scramble=True, seed=int(rng.integers(2**31))).random_base2(9)
        indices = rng.choice(len(unit), (64, 4), p=weights)
        local = np.clip(unit[indices].mean(axis=1)+rng.normal(size=(64, dim))*.08, 0, 1)
        pool = np.vstack([broad, local])
        occupied = np.vstack([unit, pending[:member]])

        def novel(point: np.ndarray) -> bool:
            return bool(np.min(np.sum((occupied-point)**2, axis=1)) > 1e-8)

        scores = np.asarray(acquisition_values(state, jnp.asarray(pool), fantasies, dim**(-.85)))
        order = [i for i in np.argsort(scores)[::-1] if np.isfinite(scores[i]) and novel(pool[i])]
        if not order:
            raise RuntimeError('Acquisition failed to find a finite, distinct point')
        starts = []
        for i in order:
            if not starts or min(np.linalg.norm(pool[i]-p) for p in starts) > .05:
                starts.append(pool[i])
            if len(starts) == 3:
                break

        def objective(point: np.ndarray) -> tuple[float, np.ndarray]:
            value, gradient = acquisition_value_gradient(state, jnp.asarray(point), fantasies, dim**(-.85))
            return -float(value), -np.asarray(gradient)

        candidates = [(float(scores[order[0]]), pool[order[0]], 'pool')]
        evaluations, successes = 0, 0
        for start in starts:
            result = minimize(objective, start, jac=True, method='L-BFGS-B', bounds=[(0., 1.)]*dim,
                options={'maxiter': 35, 'maxls': 15, 'ftol': 1e-9, 'gtol': 1e-6})
            evaluations += result.nfev
            successes += int(result.success)
            if np.isfinite(result.fun) and np.isfinite(result.x).all() and novel(result.x):
                candidates.append((-float(result.fun), result.x, 'optimizer'))
        value, point, origin = max(candidates, key=lambda row: row[0])
        selected.append(point)
        diagnostics.append(dict(member=member, starts=len(starts), optimizer_successes=successes,
            function_evaluations=evaluations, origin=origin, log_acquisition=value))
    return np.asarray(selected), diagnostics
