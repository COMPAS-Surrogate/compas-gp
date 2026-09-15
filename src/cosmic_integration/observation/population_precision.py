"""Finite-population diagnostics for fixed, normalized detection-rate models.

The comparison population is finite. Bootstrap results are conditional on its
empirical blocks being exchangeable, not an estimate of missing physics.
"""
from __future__ import annotations

import numpy as np
from scipy.special import xlogy, log_ndtr


def poisson_loss(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Expected lnL loss per reference event, including the total-rate term.

The last axis contains mutually exclusive bins; leading axes broadcast.
Candidate-only support contributes its predicted rate. Missing reference
support gives infinity, including when other bins have zero reference rate.
"""
    reference, candidate = np.broadcast_arrays(
        np.asarray(reference, float), np.asarray(candidate, float))
    if (reference.ndim < 1 or np.any(reference < 0) or np.any(candidate < 0)
            or not np.isfinite(reference).all() or not np.isfinite(candidate).all()
            or np.any(reference.sum(axis=-1) <= 0)):
        raise ValueError("Finite nonnegative rates and positive reference totals required")
    with np.errstate(divide="ignore", invalid="ignore"):
        value = xlogy(reference, reference) - xlogy(reference, candidate)
    return np.sum(value + candidate - reference, axis=-1) / reference.sum(axis=-1)


def bootstrap_coefficients(
    blocks: int, draws: int, replicates: int, seed: int,
) -> np.ndarray:
    """Normalized block multiplicities; each row represents `draws` blocks.

Input block rates must each be normalized to one block's initial-system count.
One multiplicity vector is used for every bin and physical parameter point.
"""
    if any(int(v) != v or v < 1 for v in [blocks, draws, replicates]):
        raise ValueError("Positive integer blocks, draws and replicates required")
    rng = np.random.default_rng(seed)
    return rng.multinomial(draws, np.full(blocks, 1 / blocks), size=replicates) / draws


def unsupported_fraction(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Reference detection probability in pixels with zero candidate rate."""
    reference, candidate = np.broadcast_arrays(reference, candidate)
    return np.sum(np.where(candidate == 0, reference, 0), axis=-1) / reference.sum(axis=-1)


def log_bin_coefficients(
    y: np.ndarray, scales: np.ndarray, lower: np.ndarray, upper: np.ndarray,
) -> np.ndarray:
    """Log of the normalized log-Gaussian sensor integral, including far tails.

Same uniform-within-linear-bin sensor as log_measurement.bin_coefficients.
Use this fallback when ordinary probabilities underflow for sparse populations.
"""
    from .log_measurement import bin_coefficients
    # Validate with the established sensor API (only rare fallback rows enter).
    bin_coefficients(y, scales, lower, upper)
    y, scales, lower, upper = map(np.asarray, (y, scales, lower, upper))
    with np.errstate(divide="ignore"):
        a = (np.log(lower)[None] - y[:, None] - scales[:, None]**2) / scales[:, None]
        b = (np.log(upper)[None] - y[:, None] - scales[:, None]**2) / scales[:, None]
    # Subtract survival probabilities in the positive tail to avoid 1 - 1.
    big = np.where(a > 0, log_ndtr(-a), log_ndtr(b))
    small = np.where(a > 0, log_ndtr(-b), log_ndtr(a))
    with np.errstate(divide="ignore", invalid="ignore"):
        interval = big + np.log(-np.expm1(small - big))
    return np.sum(y[:, None] + scales[:, None]**2 / 2 + interval
                  - np.log(upper - lower)[None], axis=-1)
