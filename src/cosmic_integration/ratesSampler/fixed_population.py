"""Exact reusable contraction for a fixed population and detection/output grid.

Rebuild when the population, cosmology, selection, or integration grid changes.
Only alpha, sigma, SFR amplitude and SFR exponent may vary after construction.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d
from scipy.sparse import coo_matrix, csr_matrix


class FixedPopulationRates:
    """Precompute the legacy integrator's parameter-independent linear map."""

    def __init__(self, integrator, mass_edges: np.ndarray, z_groups: np.ndarray):
        self.ci = integrator
        ci = integrator
        ci.CalculateRedshiftRelatedParams()
        masses = (ci.compas.mass1 * ci.compas.mass2)**.6 / (ci.compas.mass1 + ci.compas.mass2)**.2
        eta = ci.compas.mass1 * ci.compas.mass2 / (ci.compas.mass1 + ci.compas.mass2)**2
        edges, groups = np.asarray(mass_edges), np.asarray(z_groups)
        if (np.any(np.diff(edges) <= 0) or np.any(masses < edges[0])
                or np.any(masses >= edges[-1]) or groups[0] != 0
                or groups[-1] != ci.nRedshiftsDetection or np.any(np.diff(groups) <= 0)
                or np.any(groups != groups.astype(int))):
            raise ValueError("Edges must cover and partition the fixed population/grid")
        self.shape = (len(edges)-1, len(groups)-1)
        _, metals, self.p_draw = ci.FindZdistribution(ci.redshifts, np.log(ci.compas.zamsZmin), np.log(ci.compas.zamsZmax))
        metal_index = np.digitize(ci.compas.Zsystems, metals)
        self.metal_count = len(metals)
        mass_bin = np.searchsorted(edges, masses, side="right")-1
        z_bin = np.searchsorted(groups, np.arange(ci.nRedshiftsDetection), side="right")-1
        convert = interp1d(ci.times, ci.redshifts)
        self.operator = csr_matrix((np.prod(self.shape), len(ci.redshifts)*self.metal_count))
        # Bound temporary detection matrices and Python lists for large files.
        for first in range(0, len(masses), 10000):
            last = min(first+10000, len(masses))
            probability = ci.FindDetectionProbability(masses[first:last], eta[first:last],
                ci.redshifts, ci.distances, ci.nRedshiftsDetection, last-first,
                ci.SE.SNRgridAt1Mpc, ci.SE.detectionProbabilityFromSNR)
            weights = probability * ci.shellVolumes[:ci.nRedshiftsDetection] / (1+ci.redshifts[:ci.nRedshiftsDetection])
            rows, cols, values = [], [], []
            for i in range(first, last):
                formation_time = ci.times-ci.compas.delayTime[i]
                stop = np.digitize(np.min(ci.times), formation_time)
                stop += int(stop == len(ci.times))
                count = min(max(stop-1, 0), ci.nRedshiftsDetection)
                index = np.ceil(convert(formation_time[:count]) / (ci.redshifts[1]-ci.redshifts[0])).astype(int)
                rows.extend(mass_bin[i]*self.shape[1]+z_bin[:count])
                cols.extend(index*self.metal_count+metal_index[i])
                values.extend(weights[i-first, :count])
            self.operator += coo_matrix((values, (rows, cols)), shape=self.operator.shape).tocsr()
        self.operator.eliminate_zeros()

    def __call__(self, alpha: float, sigma: float, sfr_a: float, sfr_d: float) -> np.ndarray:
        """Return rates with the same discretization as FindDetectionRate."""
        ci = self.ci
        density, _, _ = ci.FindZdistribution(ci.redshifts, np.log(ci.compas.zamsZmin),
            np.log(ci.compas.zamsZmax), p_Alpha=alpha, p_Sigma=sigma)
        formed = ci.CalculateSFR(ci.redshifts, sfr_a, sfr_d) / (ci.compas.massEvolvedPerBinary*ci.compas.nSystems)
        return (self.operator @ (formed[:, None]*density/self.p_draw).ravel()).reshape(self.shape)
