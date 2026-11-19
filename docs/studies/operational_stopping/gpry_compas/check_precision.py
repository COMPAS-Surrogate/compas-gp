"""Diagnose a final numerical non-stop with more GP-only integration points.

This does not alter any recorded stopping decision or use exact reference data.
It is a same-checkpoint numerical diagnostic after the frozen experiment.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import time

import numpy as np
from scipy.special import ndtri
from scipy.stats import chi2, multivariate_t, qmc

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("gpry_precision_runner", HERE/"run_experiment.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def designs(centre: np.ndarray, cov: np.ndarray, seed: int, power: int) -> list:
    shape = 4*cov+np.eye(3)*1e-8
    root = np.linalg.cholesky(shape); distribution = multivariate_t(centre, shape, df=5)
    result = []
    quarter = 2**(power-2)
    for scramble in [0, 1]:
        u = qmc.Sobol(4, scramble=True, seed=seed+scramble).random_base2(power)
        local = centre+(ndtri(u[quarter:, :3])@root.T)/np.sqrt(chi2.ppf(u[quarter:, 3], 5)/5)[:, None]
        points = np.vstack([u[:quarter, :3], local])
        points = points[np.all((points > 0) & (points < 1), axis=1)]
        logq = np.logaddexp(np.log(.25), np.log(.75)+distribution.logpdf(points))
        result.append((points, logq))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--points", type=int, default=658)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    previous, _, _ = runner.load_checkpoint(args.case/f"checkpoint_{args.points-30}.npz")
    current, unit, lnl = runner.load_checkpoint(args.case/f"checkpoint_{args.points}.npz")
    results = []
    for power in [14, 15, 16]:
        # Change only numerical integration resolution in the independent module.
        runner.designs = lambda centre, cov, seed: designs(centre, cov, seed, power)
        started = time.perf_counter()
        result = runner.stability(previous, current, unit, lnl, args.seed+20000)
        results.append({"power": power, "base_points_per_design": 2**power,
                        "seconds": time.perf_counter()-started, **result})
    runner.write_json(args.output, {"case": str(args.case), "points": args.points,
        "exact_likelihood_calls": 0, "stopping_decisions_changed": False, "checks": results})


if __name__ == "__main__":
    main()
