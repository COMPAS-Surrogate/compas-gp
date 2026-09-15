"""Controlled categorical measurement errors after detection.

This is a synthetic observation model in bin-index coordinates, not an LVK
parameter-estimation model. A column-stochastic response maps latent detected
rates to measured-bin rates. Detection is already included in the input rates.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix


def correlated_bin_response(
    shape: tuple[int, int], sigma: float = 1., rho: float = .6,
    radius: int = 3,
) -> csr_matrix:
    """Return R[measured, latent] for correlated discrete bin offsets.

    Offset probabilities are proportional to a bivariate Gaussian evaluated on
    the integer lattice within ``radius``. Offsets beyond the output boundary
    are recorded in its edge category (saturation), explicitly part of this
    toy sensor. No events or mass-domain bins are removed. Sigma is in bin
    indices, so this does not imply constant physical measurement precision.
    Sigma zero gives perfect categorical measurements.
    """
    if (len(shape) != 2 or any(int(n) != n or n <= 0 for n in shape)
            or not np.isfinite([sigma, rho]).all() or sigma < 0
            or abs(rho) >= 1 or int(radius) != radius or radius < 0):
        raise ValueError("Positive grid shape, sigma >= 0, |rho| < 1 and integer radius required")
    nm, nz = map(int, shape)
    offsets = np.array([(i, j) for i in range(-radius, radius+1)
                        for j in range(-radius, radius+1)])
    if sigma == 0:
        offsets = np.zeros((1, 2), dtype=int)
        probabilities = np.ones(1)
    else:
        i, j = offsets.T
        probabilities = np.exp(-(i*i-2*rho*i*j+j*j)/(2*sigma*sigma*(1-rho*rho)))
        probabilities /= probabilities.sum()
    latent = np.arange(nm*nz)
    i, j = np.unravel_index(latent, (nm, nz))
    rows = [np.clip(i+di, 0, nm-1)*nz + np.clip(j+dj, 0, nz-1)
            for di, dj in offsets]
    return coo_matrix((np.repeat(probabilities, len(latent)),
                       (np.concatenate(rows), np.tile(latent, len(offsets)))),
                      shape=(len(latent), len(latent))).tocsr()


def measure_counts(counts: np.ndarray, response: csr_matrix,
                   rng: np.random.Generator) -> np.ndarray:
    """Measure each event once; preserve the paired latent catalogue's size."""
    counts = np.asarray(counts)
    if (counts.ndim != 2 or np.any(counts < 0) or not np.isfinite(counts).all()
            or np.any(counts != np.floor(counts))
            or response.shape != (counts.size, counts.size)
            or np.any(response.data < 0) or not np.isfinite(response.data).all()
            or not np.allclose(np.asarray(response.sum(axis=0)), 1.)):
        raise ValueError("Integer counts and a matching column-stochastic response required")
    result = np.zeros(counts.size, dtype=int)
    columns = response.tocsc()
    for k in np.flatnonzero(counts):
        start, stop = columns.indptr[k:k+2]
        result[columns.indices[start:stop]] += rng.multinomial(
            int(counts.ravel()[k]), columns.data[start:stop])
    return result.reshape(counts.shape)
