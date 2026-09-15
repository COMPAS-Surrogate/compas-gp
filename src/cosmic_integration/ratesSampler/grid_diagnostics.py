"""Data-free diagnostics for nested output grids, without changing integration."""
from __future__ import annotations

import numpy as np


def nested_mass_edges(right_edges: np.ndarray, stride: int) -> np.ndarray:
    """Merge log-spaced bins, preserving the low-mass bin and infinite overflow."""
    edges = np.asarray(right_edges, dtype=float)
    if stride < 1 or int(stride) != stride:
        raise ValueError("stride must be a positive integer")
    if edges.ndim != 1 or not len(edges) or np.any(np.diff(edges) <= 0):
        raise ValueError("right_edges must be a nonempty ascending vector")
    if not np.all(np.isfinite(edges)) or edges[0] <= 0:
        raise ValueError("right_edges must be positive and finite")
    return np.r_[0.0, np.unique(np.r_[edges[::stride], edges[-1]]), np.inf]


def system_grid_moments(
    rates: np.ndarray, masses: np.ndarray, mass_edges: np.ndarray,
    z_groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sum(w), sum(w²), merging redshift weights BEFORE squaring.

    Rows are independent systems; redshift columns are correlated deterministic
    contributions from each system. ``z_groups`` contains column boundaries,
    including zero and the input column count. Mass bins are left closed, with
    interior edge values assigned to the bin on the right.
    """
    rates, masses = np.asarray(rates, float), np.asarray(masses, float)
    edges, groups = np.asarray(mass_edges, float), np.asarray(z_groups)
    if rates.ndim != 2 or masses.shape != (rates.shape[0],):
        raise ValueError("rates must be systems x z, with one mass per system")
    if not np.all(np.isfinite(rates)) or np.any(rates < 0):
        raise ValueError("rates must be finite and nonnegative")
    if (groups.ndim != 1 or len(groups) < 2 or groups[0] != 0
            or groups[-1] != rates.shape[1] or np.any(np.diff(groups) <= 0)
            or not np.all(groups == groups.astype(int))):
        raise ValueError("z_groups must partition every input column")
    if (edges.ndim != 1 or len(edges) < 2 or np.any(np.isnan(edges))
            or np.any(np.diff(edges) <= 0) or not np.all(np.isfinite(masses))
            or np.any(masses < edges[0]) or np.any(masses >= edges[-1])):
        raise ValueError("mass_edges must be ascending and cover all masses")
    merged = np.add.reduceat(rates, groups[:-1].astype(int), axis=1)
    indices = np.searchsorted(edges, masses, side="right") - 1
    total = np.zeros((len(edges) - 1, len(groups) - 1))
    square = np.zeros_like(total)
    np.add.at(total, indices, merged)
    np.add.at(square, indices, merged**2)
    return total, square


def effective_size(total: np.ndarray, square: np.ndarray) -> np.ndarray:
    """Kish weight concentration diagnostic; zero for unsupported cells."""
    return np.divide(total**2, square, out=np.zeros_like(total), where=square > 0)


def shape_js(left: np.ndarray, right: np.ndarray) -> float:
    """Jensen–Shannon divergence (nats) between normalized rate matrices."""
    p, q = np.asarray(left, float), np.asarray(right, float)
    if p.shape != q.shape or p.sum() <= 0 or q.sum() <= 0:
        raise ValueError("matching positive-total rate matrices required")
    if np.any(p < 0) or np.any(q < 0):
        raise ValueError("rates must be nonnegative")
    p, q = p / p.sum(), q / q.sum()
    mid = (p + q) / 2
    def kl(a: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / mid[mask])))
    return (kl(p) + kl(q)) / 2
