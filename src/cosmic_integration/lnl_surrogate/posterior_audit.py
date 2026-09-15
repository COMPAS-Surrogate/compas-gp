"""Finite importance-sampling audit of a deterministic likelihood surrogate.

This is an experimental diagnostic, not a certificate of unseen-mode discovery.
The bootstrap measures observed audit variability and is not a sequentially
valid confidence bound on the full posterior error.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def importance_audit_check(predicted: np.ndarray, actual: np.ndarray,
                           log_proposal: np.ndarray, seed: int,
                           group_sizes: tuple[int, ...] = (64, 16, 16)) -> dict:
    """Check posterior-weighted errors on a held-out stratified mixture sample.

    The first group samples the numerical GP posterior. Remaining groups may
    be tempered/broad proposals. ``log_proposal`` is the log density of the
    complete mixture, including its actual group fractions. Predictive values
    and this density must be frozen before revealing ``actual``.
    """
    predicted, actual, log_proposal = [np.asarray(v, dtype=float) for v in
                                      (predicted, actual, log_proposal)]
    if (predicted.ndim != 1 or actual.shape != predicted.shape or log_proposal.shape != predicted.shape
            or sum(group_sizes) != len(actual) or min(group_sizes) < 2):
        raise ValueError('Matching audit vectors and nonempty sampling groups are required')
    if not all(np.isfinite(v).all() for v in (predicted, actual, log_proposal)):
        return dict(passed=False, reason='nonfinite_audit')
    count = group_sizes[0]
    residual = actual[:count]-predicted[:count]
    residual -= np.median(residual)
    rms = float(np.sqrt(np.mean(residual**2)))
    q95 = float(np.quantile(abs(residual), .95))
    correction = np.exp(residual-logsumexp(residual))
    correction_ess = float(1/(correction@correction)/count)
    ga, da = predicted-log_proposal, actual-log_proposal

    def metrics(gp, direct):
        gp, direct = gp-logsumexp(gp), direct-logsumexp(direct)
        return max(0., float(np.exp(gp)@(gp-direct)), float(np.exp(direct)@(direct-gp)))

    kl = metrics(ga, da)
    rng = np.random.default_rng(seed)
    ends = np.cumsum((0, *group_sizes))
    boot = []
    for _ in range(256):
        indices = np.concatenate([rng.integers(a, b, b-a) for a, b in zip(ends[:-1], ends[1:])])
        boot.append(metrics(ga[indices], da[indices]))
    upper = float(np.quantile(boot, .95))
    weight = np.exp(da-logsumexp(da)); ess = float(1/(weight@weight))
    passed = (rms <= .07 and q95 <= .15 and correction_ess >= .99
              and kl < .003 and upper < .005 and ess >= 32)
    return dict(passed=bool(passed), posterior_rms=rms, posterior_q95=q95,
                correction_ess_fraction=correction_ess, maximum_directed_kl=kl,
                bootstrap_95_kl=upper, corrected_ess=ess)
