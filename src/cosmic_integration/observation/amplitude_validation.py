"""Exact linear-amplitude controls for testing an explicit likelihood surrogate."""
from __future__ import annotations

import numpy as np
from scipy.special import gammainc, gammaincc, gammaincinv, gammainccinv
from scipy.stats import gamma


def joint_amplitude_log_likelihood(
    event_rates: np.ndarray, total_rates: np.ndarray, amplitude: np.ndarray,
    duration: float, reference: float = .012,
) -> np.ndarray:
    """Poisson-process log likelihood, omitting only data-dependent constants.

    Rates are evaluated at ``reference``. Each row has its own amplitude;
    columns are event integrals, including measurement coefficients if used.
    Zero event support produces minus infinity, not an artificial floor.
    """
    events = np.asarray(event_rates, float)
    total = np.asarray(total_rates, float)
    a = np.asarray(amplitude, float)
    if (events.ndim != 2 or total.shape != (len(events),) or a.shape != total.shape
            or not np.isfinite(events).all() or np.any(events < 0)
            or not np.isfinite(total).all() or np.any(total < 0)
            or not np.isfinite(a).all() or np.any(a <= 0)
            or not np.isfinite([duration, reference]).all()
            or duration <= 0 or reference <= 0):
        raise ValueError("Finite nonnegative rates and positive amplitudes/exposure required")
    ratio = a / reference
    with np.errstate(divide="ignore"):
        return (np.log(events).sum(axis=1) + events.shape[1]*np.log(duration*ratio)
                - duration*total*ratio)


def amplitude_proposal(
    beta: np.ndarray, number: int, rng: np.random.Generator,
    low: float = .005, high: float = .015,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw a full-support Gamma/uniform mixture; return density in unit A.

    The Gamma is the conditional posterior for a bounded uniform prior.
    Where its interval probability underflows, both mixture components become
    uniform. Survival functions avoid cancellation in the Gamma's upper tail.
    """
    beta = np.asarray(beta, float)
    if (beta.ndim != 1 or not np.isfinite(beta).all() or np.any(beta <= 0)
            or number < 0 or int(number) != number
            or not np.isfinite([low, high]).all() or not 0 < low < high):
        raise ValueError("Positive beta, ordered bounds and nonnegative integer count required")
    shape = number + 1
    left, right = beta*low, beta*high
    upper = left >= shape
    c0, c1 = gammainc(shape, left), gammainc(shape, right)
    s0, s1 = gammaincc(shape, left), gammaincc(shape, right)
    probability = np.where(upper, s0-s1, c1-c0)
    valid = probability > 1e-250
    u = rng.uniform(np.finfo(float).eps, 1-np.finfo(float).eps, len(beta))
    a = low + (high-low)*u
    choose = valid & (rng.random(len(beta)) < .5)
    lower_choice, upper_choice = choose & ~upper, choose & upper
    a[lower_choice] = gammaincinv(shape, c0[lower_choice] + u[lower_choice]*probability[lower_choice])/beta[lower_choice]
    a[upper_choice] = gammainccinv(shape, s1[upper_choice] + u[upper_choice]*probability[upper_choice])/beta[upper_choice]
    if np.any(~np.isfinite(a)) or np.any((a < low) | (a > high)):
        raise FloatingPointError("Conditional Gamma inverse outside amplitude support")
    logq = np.zeros(len(beta))
    conditional = (gamma.logpdf(a[valid], shape, scale=1/beta[valid])
                   - np.log(probability[valid]) + np.log(high-low))
    logq[valid] = np.logaddexp(np.log(.5), np.log(.5)+conditional)
    return a, logq
