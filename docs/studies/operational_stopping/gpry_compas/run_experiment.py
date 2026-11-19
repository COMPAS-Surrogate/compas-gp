"""New COMPAS BO trajectories; no reference posterior is loaded by this runner."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, ndtri
from scipy.stats import chi2, multivariate_t, qmc

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE.parents[1]/"event_uncertainty"))
import essential_runs as data_source
import posterior_convergence as source_target

from cosmic_integration.lnl_surrogate.gpry_rules import (
    CorrectCounter, state_from_gpjax, condition_on_data, prepare_fantasies,
    acquisition_values, acquisition_value_gradient, predict_lnl,
)
from cosmic_integration.lnl_surrogate.stopping import (
    StoppingTolerances, posterior_summary, summaries_agree, symmetric_kl,
)

core = data_source.core
LOW, WIDTH = core.LOW[core.SHAPE], core.WIDTH[core.SHAPE]
TOL = StoppingTolerances()
RULES = ["correct_counter", "counter_and_gaussian", "joint_stability"]


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def predict(state: dict, unit: np.ndarray) -> np.ndarray:
    output = []
    for start in range(0, len(unit), 512):
        part = unit[start:start+512]
        padded = np.pad(part, ((0, 512-len(part)), (0, 0)), mode="edge")
        output.append(np.asarray(predict_lnl(state, jnp.asarray(padded)))[:len(part)])
    return np.concatenate(output)


def fit(unit: np.ndarray, lnl: np.ndarray, capacity: int) -> dict:
    scaler = core.AdaptiveRobustScaler(compression="sqrt", soft_clipping=False,
        lower_clip_value=float(lnl.min()), max_scale=1e9)
    scaler.initialize_with_data(lnl)
    gp = core.fit_exact_gp(core.JaxTrainingData(LOW+WIDTH*unit, -scaler.transform(lnl)),
        np.array([LOW, LOW+WIDTH]), config=core.JaxGPConfig(optimisation_steps=250), seed=17)
    return state_from_gpjax(gp, scaler, LOW, WIDTH, capacity)


def save_checkpoint(folder: Path, state: dict, unit: np.ndarray, lnl: np.ndarray) -> str:
    parameters = {f"param_{k}": np.asarray(v) for k, v in state.items()
                  if k not in ["x", "mask", "inverse", "alpha"]}
    path = folder/f"checkpoint_{len(unit)}.npz"
    if not path.exists():
        np.savez(path, unit=unit, lnl=lnl, capacity=len(state["mask"]), **parameters)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_checkpoint(path: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    with np.load(path) as data:
        unit, lnl, capacity = data["unit"], data["lnl"], int(data["capacity"])
        state = {k[6:]: jnp.asarray(data[k]) for k in data.files if k.startswith("param_")}
    state.update(x=jnp.zeros((capacity, 3)), mask=jnp.zeros(capacity))
    return condition_on_data(state, unit, lnl), unit, lnl


def acquire(state: dict, unit: np.ndarray, lnl: np.ndarray,
            rng: np.random.Generator) -> tuple[np.ndarray, list[dict]]:
    selected, diagnostics = [], []
    weights = np.exp((lnl-lnl.max())/2); weights /= weights.sum()
    for member in range(3):
        pending = np.zeros((3, 3)); mask = np.zeros(3)
        if member:
            pending[:member] = selected
        mask[:member] = 1
        fantasies = prepare_fantasies(state, jnp.asarray(pending), jnp.asarray(mask))
        broad = qmc.Sobol(3, scramble=True, seed=int(rng.integers(2**31))).random_base2(9)
        indices = rng.choice(len(unit), (64, 4), p=weights)
        local = np.clip(unit[indices].mean(axis=1)+rng.normal(size=(64, 3))*.08, 0, 1)
        pool = np.vstack([broad, local])
        occupied = np.vstack([unit, pending[:member]])
        def novel(p):
            return np.min(np.sum((occupied-p)**2, axis=1)) > 1e-8
        scores = np.asarray(acquisition_values(state, jnp.asarray(pool), fantasies, 3**(-.85)))
        order = [i for i in np.argsort(scores)[::-1] if np.isfinite(scores[i]) and novel(pool[i])]
        if not order:
            raise RuntimeError("Acquisition failed to find a finite, distinct point")
        starts = []
        for i in order:
            if not starts or min(np.linalg.norm(pool[i]-p) for p in starts) > .05:
                starts.append(pool[i])
            if len(starts) == 3:
                break
        def objective(point):
            value, gradient = acquisition_value_gradient(state, jnp.asarray(point), fantasies, 3**(-.85))
            return -float(value), -np.asarray(gradient)
        candidates = [(float(scores[order[0]]), pool[order[0]], "pool")]
        evaluations, successes = 0, 0
        for start in starts:
            result = minimize(objective, start, jac=True, method="L-BFGS-B", bounds=[(0., 1.)]*3,
                options={"maxiter": 35, "maxls": 15, "ftol": 1e-9, "gtol": 1e-6})
            evaluations += result.nfev; successes += int(result.success)
            if np.isfinite(result.fun) and np.isfinite(result.x).all() and novel(result.x):
                candidates.append((-float(result.fun), result.x, "optimizer"))
        value, point, origin = max(candidates, key=lambda row: row[0])
        selected.append(point)
        diagnostics.append({"member": member, "starts": len(starts), "optimizer_successes": successes,
                            "function_evaluations": evaluations, "origin": origin, "log_acquisition": value})
    return np.asarray(selected), diagnostics


def covariance(points: np.ndarray, logweights: np.ndarray) -> np.ndarray:
    weights = np.exp(logweights-logsumexp(logweights)); centre = weights@points
    return (points-centre).T@(weights[:, None]*(points-centre))


def designs(centre: np.ndarray, cov: np.ndarray, seed: int) -> list[tuple]:
    shape = 4*cov+np.eye(3)*1e-8
    root = np.linalg.cholesky(shape); distribution = multivariate_t(centre, shape, df=5)
    output = []
    for scramble in [0, 1]:
        u = qmc.Sobol(4, scramble=True, seed=seed+scramble).random_base2(14)
        local = centre+(ndtri(u[4096:, :3])@root.T)/np.sqrt(chi2.ppf(u[4096:, 3], 5)/5)[:, None]
        points = np.vstack([u[:4096, :3], local])
        points = points[np.all((points > 0) & (points < 1), axis=1)]
        logq = np.logaddexp(np.log(.25), np.log(.75)+distribution.logpdf(points))
        output.append((points, logq))
    return output


def gp_designs(state: dict, unit: np.ndarray, lnl: np.ndarray, seed: int) -> list[tuple]:
    weights = np.exp((lnl-lnl.max())/2); weights /= weights.sum()
    centre = weights@unit
    cov = (unit-centre).T@(weights[:, None]*(unit-centre))+np.eye(3)*.005**2
    points, logq = designs(centre, cov, seed)[0]
    logw = predict(state, points)-logq
    weights = np.exp(logw-logsumexp(logw))
    return designs(weights@points, covariance(points, logw), seed+1000)


def gaussian_kl(mean_a: np.ndarray, cov_a: np.ndarray, mean_b: np.ndarray, cov_b: np.ndarray) -> float:
    def directed(ma, ca, mb, cb):
        delta = ma-mb
        return .5*(np.trace(np.linalg.solve(cb, ca))+delta@np.linalg.solve(cb, delta)
                   -len(ma)+np.linalg.slogdet(cb)[1]-np.linalg.slogdet(ca)[1])
    return max(0., float(directed(mean_a, cov_a, mean_b, cov_b)),
               float(directed(mean_b, cov_b, mean_a, cov_a)))


def stability(previous: dict | None, state: dict, unit: np.ndarray, lnl: np.ndarray, seed: int) -> dict:
    proposal = gp_designs(state, unit, lnl, seed)
    current = [predict(state, p)-q for p, q in proposal]
    summaries = [posterior_summary(p, w) for (p, _), w in zip(proposal, current)]
    numerical, check = summaries_agree(*summaries, TOL)
    result = {"numerical": numerical, "numerical_current": check,
              "ess": [s["ess"] for s in summaries], "gaussian_pass": False, "joint_pass": False}
    if previous is None:
        return result
    old = [predict(previous, p)-q for p, q in proposal]
    old_summaries = [posterior_summary(p, w) for (p, _), w in zip(proposal, old)]
    previous_ok, previous_check = summaries_agree(*old_summaries, TOL)
    numerical &= previous_ok
    joint = max(symmetric_kl(a, b) for a, b in zip(old, current))
    gkl = max(gaussian_kl(np.asarray(a["mean"]), covariance(p, wa),
                         np.asarray(b["mean"]), covariance(p, wb))
              for (p, _), a, b, wa, wb in zip(proposal, old_summaries, summaries, old, current))
    result.update(numerical=numerical, numerical_previous=previous_check,
        joint_kl=joint, gaussian_kl=gkl, gaussian_pass=bool(numerical and gkl < .02),
        joint_pass=bool(numerical and joint < .001
                       and all(summaries_agree(a, b, TOL)[0] for a, b in zip(old_summaries, summaries))))
    return result


def generate_catalogue(loader, n: int, case: int, mode: str, path: Path) -> dict:
    rng = np.random.default_rng(np.random.SeedSequence([2026091101, n, case]))
    rate = loader.fast(*core.TRUTH).ravel()[loader.active]
    number = rng.poisson(n)
    bins = rng.choice(len(rate), number, p=rate/rate.sum())
    latent = rng.uniform(loader.lower[bins], loader.upper[bins])
    rho = 12*rng.random(number)**(-1/3)
    scales = np.array([.03, .3])[None, :]*12/rho[:, None]
    observed = np.log(latent)+scales*rng.normal(size=(number, 2))
    if mode == "perfect":
        coeff = np.eye(len(rate))[bins]
    else:
        _, coeff, _ = data_source.measurement({"latent": latent, "scales": scales, "y": observed},
                                             1., loader.lower, loader.upper)
    result = dict(coeff=coeff, duration=n/loader.fiducial_total, latent=latent, rho=rho,
                  scales=scales, y=observed, bins=bins)
    np.savez(path, **result)
    return result


def run_case(out: Path, loader, n: int, mode: str, case: int, seed: int, maximum: int) -> dict:
    folder = out/f"n{n}_{mode}_case{case}"
    folder.mkdir()
    cat = generate_catalogue(loader, n, case, mode, folder/"catalogue.npz")
    duration, coeff = cat["duration"], cat["coeff"]
    def evaluate(unit):
        points = LOW+WIDTH*unit
        rates = np.array([loader.fast(p[0], p[1], .012, p[2]).ravel()[loader.active] for p in points])
        labels = source_target.event_target(rates, coeff, duration)
        if not np.isfinite(labels).all():
            raise RuntimeError("Non-finite COMPAS labels encountered within the frozen prior")
        return labels
    grid = np.array(np.meshgrid(*([np.linspace(0, 1, 3)]*3), indexing="ij")).reshape(3, -1).T
    unit = np.vstack([qmc.Sobol(3, scramble=True, seed=seed).random_base2(5),
                      grid[np.any((grid == 0) | (grid == 1), axis=1)]])
    capacity = int(np.ceil(maximum/32))*32
    start = time.perf_counter(); lnl = evaluate(unit)
    timing = {"labels": time.perf_counter()-start, "fit": 0., "conditioning": 0., "acquisition": 0., "stability": 0.}
    tick = time.perf_counter(); state = fit(unit, lnl, capacity)
    jax.block_until_ready(state)
    timing["fit"] += time.perf_counter()-tick
    counter, streaks, rng = CorrectCounter(3), Counter(), np.random.default_rng(seed+10000)
    rules = {r: None for r in RULES}; checkpoints, batches = [], []
    previous = None
    for batch in range((maximum-58)//3+1):
        npoints = len(unit); checked = None
        if batch % 10 == 0:
            tick = time.perf_counter()
            checked = stability(previous, state, unit, lnl, seed+20000)
            timing["stability"] += time.perf_counter()-tick
            for key in ["gaussian", "joint"]:
                streaks[key] = streaks[key]+1 if checked[key+"_pass"] else 0
            if counter.streak >= counter.required and streaks["gaussian"] >= 2 and rules["counter_and_gaussian"] is None:
                rules["counter_and_gaussian"] = npoints
            if streaks["joint"] >= 2 and rules["joint_stability"] is None:
                rules["joint_stability"] = npoints
            previous = state
        if counter.streak >= counter.required and rules["correct_counter"] is None:
            rules["correct_counter"] = npoints
        if checked is not None or npoints in rules.values():
            sha = save_checkpoint(folder, state, unit, lnl)
            checkpoints.append({"points": npoints, "sha256": sha, "stability": checked,
                                "counter": counter.streak, "rules": dict(rules), "timing": dict(timing)})
            write_json(folder/"checkpoints.json", checkpoints)
            write_json(folder/"decision.json", {"rules": rules, "points_so_far": npoints,
                "reference_accessed": False, "complete": npoints == maximum})
        if checked is not None:
            logging.info("%s labels=%s counter=%s gaussian=%s joint=%s numerical=%s seconds=%.1f",
                folder.name, npoints, counter.streak, streaks["gaussian"], streaks["joint"], checked["numerical"],
                time.perf_counter()-start)
            write_json(out/"status.json", {"state": "running", "case": folder.name, "points": npoints})
        if npoints == maximum:
            break
        tick = time.perf_counter(); selected, diagnostics = acquire(state, unit, lnl, rng)
        before, best = predict(state, selected), float(lnl.max())
        timing["acquisition"] += time.perf_counter()-tick
        tick = time.perf_counter(); actual = evaluate(selected)
        timing["labels"] += time.perf_counter()-tick
        check = counter.update(before, actual, best)
        batches.append({"before_points": npoints, "unit": selected.tolist(), "prediction": before.tolist(),
                        "actual": actual.tolist(), "best_before": best, "counter": check, "acquisition": diagnostics})
        write_json(folder/"batches.json", batches)
        unit, lnl = np.vstack([unit, selected]), np.r_[lnl, actual]
        tick = time.perf_counter()
        refit = (batch+1) % 10 == 0
        state = fit(unit, lnl, capacity) if refit else condition_on_data(state, unit, lnl)
        jax.block_until_ready(state)
        timing["fit" if refit else "conditioning"] += time.perf_counter()-tick
    result = {"expected_events": n, "mode": mode, "case": case, "seed": seed,
        "actual_events": len(coeff), "rules": rules, "total_labels": len(unit), "timing": timing,
        "wall_seconds": time.perf_counter()-start, "reference_accessed": False, "complete": True}
    write_json(folder/"decision.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--maximum", type=int, default=658)
    args = parser.parse_args()
    if args.maximum < 88 or (args.maximum-58) % 30:
        parser.error("maximum must be 58 plus a positive multiple of 30")
    args.outdir.mkdir(parents=True, exist_ok=False)
    groups = [(100, "perfect", 91, 9011)] if args.smoke else [
        (n, mode, case, 5011+case) for n in [100, 1000] for mode in ["perfect", "uncertain"] for case in range(3)]
    files = [Path(__file__), HERE/"PROTOCOL.md", REPO/"src/cosmic_integration/lnl_surrogate/gpry_rules.py"]
    hashes = {}
    for path in files:
        relative = path.relative_to(REPO); destination = args.outdir/"source_snapshot"/relative
        destination.parent.mkdir(parents=True, exist_ok=True); destination.write_bytes(path.read_bytes())
        hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(args.outdir/"manifest.json", {"groups": groups, "maximum": args.maximum,
        "source_sha256": hashes, "created_unix": time.time(), "gpry_installed_for_experiment": False,
        "versions": {p: importlib.metadata.version(p) for p in ["jax", "gpjax", "numpy", "scipy"]}})
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    loader = data_source.Experiment(args.outdir/"population", "gpry_COMPAS_online", False)
    decisions = {}
    try:
        for n, mode, case, seed in groups:
            result = run_case(args.outdir, loader, n, mode, case, seed, args.maximum)
            decisions[f"n{n}_{mode}_case{case}"] = result
        write_json(args.outdir/"decisions.json", decisions)
        frozen = hashlib.sha256((args.outdir/"decisions.json").read_bytes()).hexdigest()
        write_json(args.outdir/"status.json", {"state": "complete", "cases": len(decisions), "decision_sha256": frozen})
    except Exception as exc:
        write_json(args.outdir/"status.json", {"state": "failed", "cases": len(decisions), "error": repr(exc)})
        raise


if __name__ == "__main__":
    main()
