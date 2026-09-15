"""Candidate operational stopping diagnostics; no direct posterior is required.

These tolerances are experimental. Passing is not a guarantee of posterior
accuracy or mode discovery; validate the rule on independent problems first.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp


@dataclass(frozen=True)
class StoppingTolerances:
    prediction_abs: float = 0.1
    prediction_rel: float = 0.01
    prediction_q90: float = 1.0
    prediction_max: float = 3.0
    stability_kl: float = 0.001
    mean_sd: float = 0.05
    endpoint_sd: float = 0.05
    width_fraction: float = 0.03
    minimum_ess: float = 100.0
    consecutive: int = 2
    audit_rms: float = 0.1
    audit_q95: float = 0.2
    audit_weight_ess_fraction: float = 0.95


def normalized_logweights(logweights: np.ndarray) -> np.ndarray:
    """Normalize finite log weights without losing additive-offset invariance."""
    value = np.asarray(logweights, dtype=float)
    if value.ndim != 1 or not len(value) or not np.isfinite(value).all():
        raise ValueError('A nonempty finite vector of log weights is required')
    return value-logsumexp(value)


def symmetric_kl(a: np.ndarray, b: np.ndarray) -> float:
    """Maximum of the two directed KLs on identical quadrature coordinates."""
    a, b = normalized_logweights(a), normalized_logweights(b)
    if a.shape != b.shape:
        raise ValueError('KL weights must have identical support')
    return max(0., float(np.exp(a)@(a-b)), float(np.exp(b)@(b-a)))


def weighted_quantiles(x: np.ndarray, weights: np.ndarray,
                       probabilities: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    w = np.asarray(weights)[order]
    centres = (np.cumsum(w)-0.5*w)/np.sum(w)
    return np.interp(probabilities, centres, np.asarray(x)[order])


def posterior_summary(points: np.ndarray, logweights: np.ndarray) -> dict:
    """Moments and equal-tailed intervals of a GP-only weighted sample."""
    points = np.asarray(points)
    w = np.exp(normalized_logweights(logweights))
    mean = w@points
    sd = np.sqrt(w@((points-mean)**2))
    if points.ndim != 2 or len(points) != len(w) or np.any(sd <= 0):
        raise ValueError('Posterior sample must span each parameter dimension')
    quantiles = np.array([weighted_quantiles(points[:, i], w, [.025,.16,.5,.84,.975])
                          for i in range(points.shape[1])]).T
    return {'mean': mean.tolist(), 'sd': sd.tolist(), 'quantiles': quantiles.tolist(),
            'ess': float(1/(w@w))}


def summary_change(a: dict, b: dict) -> dict:
    """Compare posterior shape using the preceding posterior's SD as a scale."""
    scale = np.asarray(a['sd'])
    qa, qb = np.asarray(a['quantiles']), np.asarray(b['quantiles'])
    return {'mean_sd': float(np.max(abs(np.asarray(a['mean'])-b['mean'])/scale)),
            'endpoint_sd': float(np.max(abs(qa[[0,1,3,4]]-qb[[0,1,3,4]])/scale)),
            'width_fraction': float(np.max(abs((qb[3]-qb[1])/(qa[3]-qa[1])-1)))}


def summaries_agree(a: dict, b: dict, tolerances: StoppingTolerances) -> tuple[bool, dict]:
    values = summary_change(a,b)
    passed = all(values[k] < getattr(tolerances,k)
                 for k in ['mean_sd','endpoint_sd','width_fraction'])
    passed &= min(a['ess'],b['ess']) >= tolerances.minimum_ess
    return bool(passed), values


def prediction_check(predicted: np.ndarray, actual: np.ndarray,
                     tolerances: StoppingTolerances) -> dict:
    """Check new labels against predictions made before fitting those labels.

    Remove a posterior-irrelevant constant residual. Relative tolerance relaxes
    deep-tail errors without dropping any training points. The caller must keep
    these predictions frozen until evaluation and scoring have finished.
    """
    predicted, actual = np.asarray(predicted), np.asarray(actual)
    if predicted.shape != actual.shape or actual.ndim != 1 or not len(actual):
        raise ValueError('Matching nonempty prediction/label vectors required')
    if not np.isfinite(predicted).all() or not np.isfinite(actual).all():
        return {'passed': False, 'reason': 'nonfinite_prediction_or_label'}
    weights = np.exp(actual-actual.max())
    residual = actual-predicted
    offset = float(weighted_quantiles(residual, weights, np.array([.5]))[0])
    error = abs(residual-offset)
    tolerance = tolerances.prediction_abs+tolerances.prediction_rel*(actual.max()-actual)
    scaled = error/tolerance
    q90, maximum = float(np.quantile(scaled,.9)), float(scaled.max())
    return {'passed': q90 <= tolerances.prediction_q90 and maximum <= tolerances.prediction_max,
            'offset': offset, 'q90_scaled_error': q90, 'max_scaled_error': maximum}


def audit_check(predicted: np.ndarray, actual: np.ndarray, posterior_count: int,
                best_training: float, tolerances: StoppingTolerances) -> dict:
    """Score fresh posterior draws plus broad points before using them to train.

    Posterior rows come first. Broad checks reject appreciable likelihood that
    the GP falsely dismissed, and spurious high-likelihood regions predicted by
    the GP. A finite audit cannot certify absence of undiscovered modes.
    """
    predicted, actual = np.asarray(predicted), np.asarray(actual)
    if posterior_count < 2 or posterior_count > len(actual) or predicted.shape != actual.shape:
        raise ValueError('Audit requires matching labels and at least two posterior points')
    if not np.isfinite(predicted).all() or not np.isfinite(actual).all():
        return {'passed': False, 'reason': 'nonfinite_prediction_or_label'}
    residual = actual-predicted
    offset = float(np.median(residual[:posterior_count]))
    centred = residual-offset
    core = centred[:posterior_count]
    rms = float(np.sqrt(np.mean(core**2)))
    q95 = float(np.quantile(abs(core),.95))
    w = np.exp(normalized_logweights(core))
    ess_fraction = float(1/(w@w)/posterior_count)
    broad = slice(posterior_count,None)
    relevant = np.maximum(actual[broad],predicted[broad]+offset) > best_training-12
    broad_error = abs(centred[broad][relevant])
    broad_max = float(broad_error.max()) if len(broad_error) else 0.
    passed = (rms <= tolerances.audit_rms and q95 <= tolerances.audit_q95
              and ess_fraction >= tolerances.audit_weight_ess_fraction
              and broad_max <= 3*tolerances.prediction_abs)
    return {'passed': bool(passed), 'offset': offset, 'rms': rms, 'q95': q95,
            'weight_ess_fraction': ess_fraction, 'broad_max_relevant_error': broad_max}


def next_streak(streak: int, prediction_ok: bool, stability_ok: bool,
                numerical_ok: bool) -> int:
    return streak+1 if prediction_ok and stability_ok and numerical_ok else 0
