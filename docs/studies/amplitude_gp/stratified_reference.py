"""Balanced stratified importance designs for an explicit amplitude coordinate."""
from __future__ import annotations

import numpy as np
from scipy.special import gammainc, gammaincc, gammaincinv, gammainccinv
from scipy.stats import gamma


def amplitude_design(beta: np.ndarray, number: int, replicas: int, seed: int,
                     low: float = .005, high: float = .015) -> tuple[np.ndarray, np.ndarray]:
    """Return row-major amplitudes and log proposal density in unit A.

    Half of each row's coordinates stratify the uniform prior and half stratify
    the conditional truncated-Gamma CDF. Averaging f/q over both components
    uses the balance-heuristic mixture density q=(q_uniform+q_gamma)/2.
    Underflowing Gamma intervals use uniform for both components and q=1.
    """
    beta = np.asarray(beta, float)
    if (beta.ndim != 1 or np.any(beta <= 0) or not np.isfinite(beta).all()
            or replicas < 2 or replicas % 2 or number < 0 or int(number) != number
            or not np.isfinite([low, high]).all() or not 0 < low < high):
        raise ValueError('Positive beta, integer count, even replicas and ordered bounds required')
    k, half = number+1, replicas//2
    rng = np.random.default_rng(seed)
    u = (np.tile(np.arange(half), 2)[None, :]+rng.random((len(beta), replicas)))/half
    amplitude = low+(high-low)*u
    left, right = beta*low, beta*high
    c0, c1 = gammainc(k, left), gammainc(k, right)
    s0, s1 = gammaincc(k, left), gammaincc(k, right)
    upper = left >= k
    probability = np.where(upper, s0-s1, c1-c0)
    valid = probability > 1e-250
    for indices, inverse, start in [(valid & ~upper, gammaincinv, c0), (valid & upper, gammainccinv, s1)]:
        amplitude[indices, half:] = inverse(k, start[indices, None]
            + u[indices, half:]*probability[indices, None])/beta[indices, None]
    if not np.isfinite(amplitude).all() or np.any((amplitude < low) | (amplitude > high)):
        raise FloatingPointError('Gamma inverse outside prior support')
    logq = np.zeros_like(amplitude)
    conditional = gamma.logpdf(amplitude[valid], k, scale=1/beta[valid, None])
    conditional += np.log(high-low)-np.log(probability[valid, None])
    logq[valid] = np.logaddexp(np.log(.5), np.log(.5)+conditional)
    return amplitude, logq
