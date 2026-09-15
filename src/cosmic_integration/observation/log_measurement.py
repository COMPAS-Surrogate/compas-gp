"""Continuous post-detection mock sensor and exact bin integrals.

Latent mass/redshift are uniform within each population bin in linear coordinates.
Observed log coordinates have independent Gaussian errors with known event scales.
No observed measurements are truncated to the population domain. This is not an
LVK strain likelihood or a selection model.
"""
from __future__ import annotations
import numpy as np
from scipy.special import ndtr, logsumexp, gammainc, gammaincc, gammaln
from scipy.stats import truncnorm
from .grid_likelihood import amplitude_marginal_log_likelihood


def normal_interval(low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Stable Gaussian probability between ordered endpoints."""
    low, high = np.asarray(low), np.asarray(high)
    return np.maximum(np.where(low > 0, ndtr(-low)-ndtr(-high), ndtr(high)-ndtr(low)), 0.)


def bin_coefficients(y: np.ndarray, scales: np.ndarray, lower: np.ndarray,
                     upper: np.ndarray) -> np.ndarray:
    """Return p(y_i | bin_b), averaged uniformly over each finite linear bin.

    Arrays y/scales have shape (events,2); lower/upper have shape (bins,2).
    The analytic integral resolves arbitrarily narrow errors inside coarse bins.
    """
    y, scales, lower, upper = map(lambda x: np.asarray(x,float), (y,scales,lower,upper))
    if (y.ndim!=2 or y.shape[1]!=2 or scales.shape!=y.shape or lower.ndim!=2
            or lower.shape[1]!=2 or upper.shape!=lower.shape or np.any(lower<0)
            or np.any(upper<=lower) or np.any(scales<=0)
            or not all(np.isfinite(x).all() for x in [y,scales,lower,upper])):
        raise ValueError('Finite matching event arrays, positive scales and finite positive-width bins required')
    with np.errstate(divide='ignore'):
        a=(np.log(lower)[None,:,:]-y[:,None,:]-scales[:,None,:]**2)/scales[:,None,:]
        b=(np.log(upper)[None,:,:]-y[:,None,:]-scales[:,None,:]**2)/scales[:,None,:]
    one=np.exp(y[:,None,:]+scales[:,None,:]**2/2)*normal_interval(a,b)/(upper-lower)[None,:,:]
    return one.prod(axis=2)


def log_marginal_events(event_rates: np.ndarray, total_rates: np.ndarray,
                        duration: float, reference: float=.012) -> np.ndarray:
    """Log Poisson event likelihood integrated over a ~ Uniform(.005,.015).

    Event integrals need not sum to total rate. This distinction matters for
    overlapping measurement kernels and repeated events.
    """
    e,total=np.asarray(event_rates,float),np.asarray(total_rates,float)
    if (e.ndim!=2 or total.shape!=(len(e),) or np.any(e<=0) or np.any(total<=0)
            or not np.isfinite(e).all() or not np.isfinite(total).all() or duration<=0):
        raise ValueError('Positive finite event integrals and total rates required')
    n=e.shape[1];k=n+1;beta=duration*total/reference
    left,right=.005*beta,.015*beta
    prob=np.where(left<k,gammainc(k,right)-gammainc(k,left),gammaincc(k,left)-gammaincc(k,right))
    with np.errstate(divide='ignore'):
        logint=gammaln(k)-k*np.log(beta)+np.log(prob)-np.log(.01)
    value=np.log(duration*e/reference).sum(axis=1)+logint
    for i in np.flatnonzero(prob<=1e-250):
        # Use a real one-bin Poisson model for the stable amplitude integral.
        single,_=amplitude_marginal_log_likelihood(np.array([[total[i]]]),np.array([[n]]),duration)
        value[i]=single+np.log(e[i]/total[i]).sum()
    return value


def mock_pe_samples(y: np.ndarray, scales: np.ndarray, lower: np.ndarray,
                    upper: np.ndarray, powers: np.ndarray, size: int,
                    rng: np.random.Generator) -> tuple[np.ndarray,np.ndarray,float]:
    """Exact mock posterior samples for pi(x) proportional to product x_j**powers_j.

    Returns linear-coordinate samples, normalized PE prior densities, and evidence
    for the log-coordinate measurement. Completing the square in log x gives an
    independently truncated Normal posterior for each coordinate.
    """
    y,scales,lower,upper,powers=map(np.asarray,(y,scales,lower,upper,powers))
    if np.any(powers<=-1) or np.any(lower<0) or np.any(upper<=lower) or np.any(scales<=0):
        raise ValueError('Integrable power priors and ordered nonnegative bounds required')
    q=powers+1;mu=y+q*scales**2
    with np.errstate(divide='ignore'):
        a=(np.log(lower)-mu)/scales;b=(np.log(upper)-mu)/scales
    u=np.clip(rng.random((size,2)),np.finfo(float).eps,1-np.finfo(float).eps)
    x=np.exp(truncnorm.ppf(u,a,b,loc=mu,scale=scales))
    norm=(upper**q-lower**q)/q
    prior=np.prod(x**powers/norm,axis=1)
    evidence=float(np.prod(np.exp(q*y+.5*q*q*scales**2)*normal_interval(a,b)/norm))
    return x,prior,evidence
