"""Explicit grid likelihoods for validation of perfect and uncertain events.

These utilities do not reinterpret legacy observation weights. Rates are
expected detections per bin-year. Uncertain-event coefficients must correspond
to the same physical rate model and selection convention used in inference.
"""
from __future__ import annotations

import numpy as np
from scipy.special import xlogy


def counts_log_likelihood(
    rates: np.ndarray, counts: np.ndarray, duration: float,
) -> float:
    """Independent Poisson-bin lnL, omitting data-only factorial constants.

    This targets perfect binned observations. Empty unsupported bins contribute
    zero; an observed event in a zero-rate bin gives minus infinity, not a floor.
    """
    rates, counts = np.asarray(rates, float), np.asarray(counts, float)
    if (rates.shape != counts.shape or rates.ndim != 2
            or not np.all(np.isfinite(rates)) or np.any(rates < 0)
            or not np.all(np.isfinite(counts)) or np.any(counts < 0)
            or np.any(counts != np.floor(counts))):
        raise ValueError("matching 2D nonnegative rates and integer counts required")
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be positive and finite")
    expected = rates * duration
    return float(np.sum(xlogy(counts, expected) - expected))


def posterior_grid_coefficients(
    samples: np.ndarray, prior_density: np.ndarray,
    mass_edges: np.ndarray, z_edges: np.ndarray,
) -> np.ndarray:
    """Estimate event-integral coefficients from equally weighted PE samples.

    For a piecewise-uniform model density, the event integral (up to event
    evidence) is sum_bin rate_bin * coefficient_bin, with coefficients
    mean_samples[1_bin / PE_prior_density] / bin_area. Prior density must be in
    the SAME source-mass/redshift measure as the samples and grid. This helper
    does not perform cosmological transforms or selection corrections.

    Out-of-domain samples contribute zero but REMAIN in the denominator. They
    are never clipped to boundary bins or renormalized into the retained domain.
    Finite edges are required because this representation assumes finite areas.
    """
    samples, prior = np.asarray(samples, float), np.asarray(prior_density, float)
    mc, z = np.asarray(mass_edges, float), np.asarray(z_edges, float)
    if (samples.ndim != 2 or samples.shape[1] != 2 or not len(samples)
            or prior.shape != (len(samples),) or not np.all(np.isfinite(samples))
            or not np.all(np.isfinite(prior)) or np.any(prior <= 0)):
        raise ValueError("finite (n,2) samples and positive prior densities required")
    for edges in (mc, z):
        if (edges.ndim != 1 or len(edges) < 2 or not np.all(np.isfinite(edges))
                or np.any(np.diff(edges) <= 0)):
            raise ValueError("finite strictly increasing edges required")
    i = np.searchsorted(mc, samples[:, 0], side="right") - 1
    j = np.searchsorted(z, samples[:, 1], side="right") - 1
    inside = (i >= 0) & (i < len(mc)-1) & (j >= 0) & (j < len(z)-1)
    coefficient = np.zeros((len(mc)-1, len(z)-1))
    np.add.at(coefficient, (i[inside], j[inside]), 1 / prior[inside])
    return coefficient / len(samples) / np.outer(np.diff(mc), np.diff(z))


def aggregate_nested_grid(
    matrix: np.ndarray, source_mass: np.ndarray, source_z: np.ndarray,
    target_mass: np.ndarray, target_z: np.ndarray,
) -> np.ndarray:
    """Conserve counts or rates when merging complete source pixels.

    Edge arrays include both endpoints; infinite mass guards are supported.
    This is NOT a way to rebin posterior-grid coefficients: rebuild those from
    samples with the new bin areas instead.
    """
    result = np.asarray(matrix)
    if result.ndim != 2:
        raise ValueError("matrix must be two dimensional")
    for axis, (source, target) in enumerate(
        [(source_mass, target_mass), (source_z, target_z)]
    ):
        source, target = np.asarray(source, float), np.asarray(target, float)
        if (source.ndim != 1 or target.ndim != 1 or len(target) < 2
                or len(source) != result.shape[axis]+1
                or np.any(np.isnan(source)) or np.any(np.isnan(target))
                or np.any(np.diff(source) <= 0) or np.any(np.diff(target) <= 0)
                or source[0] != target[0] or source[-1] != target[-1]):
            raise ValueError("ascending edges must preserve the full source domain")
        indices = np.searchsorted(source, target)
        if np.any(indices >= len(source)) or not np.all(source[indices] == target):
            raise ValueError("target edges must coincide with source edges")
        result = np.add.reduceat(result, indices[:-1], axis=axis)
    return result


def amplitude_marginal_log_likelihood(
    rates: np.ndarray, counts: np.ndarray, duration: float,
    low: float = .005, high: float = .015, reference: float = .012,
) -> tuple[float, float]:
    """Integrate a linear rate amplitude with a bounded uniform prior.

    ``rates`` are evaluated at ``reference``. Return log marginal likelihood
    (omitting data factorials) and conditional Gamma rate beta. This only applies
    when amplitude scales every bin linearly and its prior is independent.
    """
    from scipy.integrate import quad
    from scipy.special import gammainc, gammaincc, gammaln

    counts_log_likelihood(rates, counts, duration)  # Validate inputs.
    if not (np.isfinite([low, high, reference]).all() and 0 < low < high and reference > 0):
        raise ValueError("positive reference and ordered positive amplitude bounds required")
    rates, counts = np.asarray(rates), np.asarray(counts)
    beta = float(duration*rates.sum()/reference)
    n = float(counts.sum())
    constant = float(np.sum(xlogy(counts, duration*rates/reference)))
    if not np.isfinite(constant):
        return -np.inf, beta
    if beta == 0:
        return 0., beta  # Only possible here with an empty catalogue.
    shape = n+1
    left, right = beta*low, beta*high
    probability = (gammainc(shape, right)-gammainc(shape, left) if left < shape
                   else gammaincc(shape, left)-gammaincc(shape, right))
    if probability > 1e-250:
        log_integral = gammaln(shape)-shape*np.log(beta)+np.log(probability)
    else:
        # Rare extreme tails: integrate after removing the maximum exponent.
        mode = np.clip(n/beta, low, high)
        peak = n*np.log(mode)-beta*mode
        # Rescale to resolve a narrow endpoint peak even when the original
        # interval is thousands of peak widths wide.
        scale = min(high-low, mode/np.sqrt(max(n, 1.)),
                    1/max(abs(n/mode-beta), np.finfo(float).tiny))
        left_u, right_u = (low-mode)/scale, (high-mode)/scale
        extent = max(abs(left_u), abs(right_u), 1.)
        breaks = 10.**np.arange(int(np.ceil(np.log10(extent)))+1)
        points = np.r_[-breaks, 0., breaks]
        points = points[(points > left_u) & (points < right_u)]
        integral, _ = quad(lambda u: np.exp(
            n*np.log(mode+scale*u)-beta*(mode+scale*u)-peak),
            left_u, right_u, epsabs=1e-11, epsrel=1e-10,
            points=np.sort(points), limit=200)
        integral *= scale
        if integral <= 0:
            raise FloatingPointError("Amplitude integral unresolved in an extreme tail")
        log_integral = peak+np.log(integral)
    return float(constant+log_integral-np.log(high-low)), beta
