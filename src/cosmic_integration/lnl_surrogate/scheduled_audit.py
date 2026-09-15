"""Reference-free periodic posterior audits with bounded sample refinement.

Bootstrap quantiles describe the observed importance sample. They are not
sequential confidence bounds and do not certify discovery of unobserved modes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp


@dataclass(frozen=True)
class AuditPolicy:
    """Frozen operational settings; counts include every exact audit label."""
    first_training: int = 238
    interval: int = 60
    sizes: tuple[int, ...] = (96, 192, 384)
    kl_limit: float = .003
    bootstrap_limit: float = .005
    minimum_ess: float = 32.
    bootstrap_samples: int = 256

    def __post_init__(self) -> None:
        if (self.first_training < 1 or self.interval < 1 or not self.sizes
                or any(n < 12 or n % 6 for n in self.sizes)
                or any(a >= b for a, b in zip(self.sizes, self.sizes[1:]))
                or min(self.kl_limit, self.bootstrap_limit, self.minimum_ess) <= 0
                or self.bootstrap_samples < 20):
            raise ValueError('Positive tolerances and increasing six-divisible audit sizes required')

    def due(self, training: int, maximum: int) -> bool:
        """Always audit the final checkpoint, independent of counter or stability."""
        return training == maximum or (training >= self.first_training
                and (training-self.first_training) % self.interval == 0)


def posterior_audit(predicted: np.ndarray, actual: np.ndarray,
                    log_proposal: np.ndarray, groups: np.ndarray, seed: int,
                    policy: AuditPolicy = AuditPolicy()) -> dict:
    """Estimate both posterior KLs using the complete stratified proposal density.

    Group labels preserve posterior/tempered/prior allocation when independent
    increments are pooled. Refinement keeps the GP and proposal density fixed.
    """
    predicted, actual, log_proposal = [np.asarray(x, dtype=float)
                                      for x in (predicted, actual, log_proposal)]
    groups = np.asarray(groups)
    if (predicted.ndim != 1 or actual.shape != predicted.shape
            or log_proposal.shape != predicted.shape or groups.shape != predicted.shape
            or len(actual) not in policy.sizes
            or not np.array_equal(np.unique(groups), [0, 1, 2])
            or not np.array_equal(np.bincount(groups.astype(int)),
                                  np.array([4, 1, 1])*len(actual)//6)):
        raise ValueError('Matching vectors and a 4:1:1 stratified audit are required')
    if not all(np.isfinite(x).all() for x in (predicted, actual, log_proposal)):
        return dict(action='continue_training', passed=False, reason='nonfinite', samples=len(actual))
    gp, direct = predicted-log_proposal, actual-log_proposal

    def kls(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
        a, b = a-logsumexp(a), b-logsumexp(b)
        return max(0., float(np.exp(a)@(a-b))), max(0., float(np.exp(b)@(b-a)))

    forward, reverse = kls(gp, direct)
    maximum = max(forward, reverse)
    rng = np.random.default_rng(seed)
    pools = [np.flatnonzero(groups == g) for g in [0, 1, 2]]
    boot = []
    for _ in range(policy.bootstrap_samples):
        idx = np.concatenate([rng.choice(pool, len(pool), replace=True) for pool in pools])
        boot.append(max(kls(gp[idx], direct[idx])))
    lower, upper = np.quantile(boot, [.05, .95])
    ess = [float(1/np.sum(np.exp(w-logsumexp(w))**2)) for w in (gp, direct)]
    adequate = min(ess) >= policy.minimum_ess
    passed = bool(adequate and maximum < policy.kl_limit and upper < policy.bootstrap_limit)
    if passed:
        action, reason = 'stop', 'posterior_change_small'
    elif len(actual) < max(policy.sizes) and (not adequate or lower < policy.kl_limit):
        action, reason = 'enlarge_audit', 'finite_audit_inconclusive'
    else:
        action = 'continue_training'
        reason = 'posterior_change_detected' if adequate and lower >= policy.kl_limit else 'audit_budget_inconclusive'
    residual = actual[groups == 0]-predicted[groups == 0]
    residual -= np.median(residual)
    return dict(action=action, reason=reason, passed=passed, samples=len(actual),
        kl_gp_to_direct=forward, kl_direct_to_gp=reverse, maximum_directed_kl=maximum,
        bootstrap_05_kl=float(lower), bootstrap_95_kl=float(upper),
        gp_ess=ess[0], direct_ess=ess[1],
        posterior_rms=float(np.sqrt(np.mean(residual**2))),
        posterior_q95=float(np.quantile(abs(residual), .95)))
