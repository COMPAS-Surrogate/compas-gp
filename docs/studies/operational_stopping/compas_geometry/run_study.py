"""COMPAS-only, reference-free checkpoint screening using GP autodiff."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
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
UNCERTAINTY = HERE.parents[1] / "event_uncertainty"
sys.path.insert(0, str(UNCERTAINTY))
import posterior_convergence as base
import essential_runs
from cosmic_integration.lnl_surrogate.local_geometry import (
    cached_log_likelihood, derivative_stencil, derivative_audit, geometry_change,
)
from cosmic_integration.lnl_surrogate.stopping import (
    StoppingTolerances, posterior_summary, summaries_agree, symmetric_kl,
)

core = base.core
TOL = StoppingTolerances()
GROUPS = [("ensemble", n, "perfect", case) for n in [30, 100, 1000] for case in range(3)]
GROUPS += [("pilot", 500, "uncertain", 0)]


def write_json(path: Path, obj: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def fit(x: np.ndarray, y: np.ndarray) -> tuple:
    scaler = core.AdaptiveRobustScaler(compression="sqrt", soft_clipping=False,
                                      lower_clip_value=float(y.min()), max_scale=1e9)
    scaler.initialize_with_data(y)
    low, width = core.LOW[core.SHAPE], core.WIDTH[core.SHAPE]
    model = core.fit_exact_gp(core.JaxTrainingData(x, -scaler.transform(y)[:, None]),
                             np.array([low, low+width]), config=core.JaxGPConfig(optimisation_steps=250), seed=17)
    function = cached_log_likelihood(model, scaler, low, width)
    return {"model": model, "scaler": scaler, "function": function,
            "batch": jax.jit(jax.vmap(function)), "gradient": jax.jit(jax.vmap(jax.grad(function))),
            "hessian": jax.jit(jax.hessian(function)), "value_gradient": jax.jit(jax.value_and_grad(function))}


def predict(gp: dict, points: np.ndarray) -> np.ndarray:
    values = []
    for first in range(0, len(points), 512):
        block = points[first:first+512]
        padded = np.pad(block, ((0, 512-len(block)), (0, 0)), mode="edge")
        values.append(np.asarray(gp["batch"](jnp.asarray(padded)))[:len(block)])
    return np.concatenate(values)


def proposal(x: np.ndarray, y: np.ndarray, seed: int, *, centre=None, covariance_matrix=None) -> list[tuple]:
    """Two independent full-prior designs; only training labels define centering."""
    weights = np.exp((y-y.max())/2); weights /= weights.sum()
    if centre is None:
        centre = weights@x
        covariance_matrix = (x-centre).T@(weights[:, None]*(x-centre)) + np.eye(3)*.005**2
    shape = 4*covariance_matrix
    root = np.linalg.cholesky(shape)
    distribution = multivariate_t(centre, shape, df=5)
    result = []
    for scramble in [0, 1]:
        u = qmc.Sobol(4, scramble=True, seed=seed+scramble).random_base2(14)
        uniform = u[:4096, :3]
        local = centre+(ndtri(u[4096:, :3])@root.T)/np.sqrt(chi2.ppf(u[4096:, 3], 5)/5)[:, None]
        points = np.vstack([uniform, local])
        points = points[np.all((points > 0) & (points < 1), axis=1)]
        logq = np.logaddexp(np.log(.25), np.log(.75)+distribution.logpdf(points))
        result.append((points, logq))
    return result


def gp_proposals(gp: dict, x: np.ndarray, y: np.ndarray, seed: int) -> list[tuple]:
    """Refine a numerical integration proposal with cheap GP-only evaluations."""
    initial = proposal(x, y, seed)[0]
    points, logq = initial
    logw = predict(gp, points)-logq
    centre = np.exp(logw-logsumexp(logw))@points
    cov = covariance(points, logw)
    return proposal(x, y, seed+1000, centre=centre, covariance_matrix=cov)


def covariance(points: np.ndarray, logweights: np.ndarray) -> np.ndarray:
    weights = np.exp(logweights-logsumexp(logweights)); centre = weights@points
    return (points-centre).T@(weights[:, None]*(points-centre))


def gaussian_kl(mean_a: np.ndarray, cov_a: np.ndarray, mean_b: np.ndarray, cov_b: np.ndarray) -> float:
    """Maximum directed Gaussian KL, as a GPry-inspired moment-stability check."""
    def one(ma, ca, mb, cb):
        delta = ma-mb
        return .5*(np.trace(np.linalg.solve(cb, ca))+delta@np.linalg.solve(cb, delta)
                   -len(ma)+np.linalg.slogdet(cb)[1]-np.linalg.slogdet(ca)[1])
    return max(0., float(one(mean_a, cov_a, mean_b, cov_b)), float(one(mean_b, cov_b, mean_a, cov_a)))


def optimize_mode(gp: dict, points: np.ndarray, predicted: np.ndarray) -> dict:
    order = np.argsort(predicted)[::-1]
    starts = []
    for i in order:
        if not starts or min(np.linalg.norm(points[i]-s) for s in starts) > .03:
            starts.append(points[i])
        if len(starts) == 6:
            break
    def objective(unit):
        value, grad = gp["value_gradient"](jnp.asarray(unit))
        return -float(value), -np.asarray(grad)
    fits = [minimize(objective, start, jac=True, method="L-BFGS-B", bounds=[(0, 1)]*3,
                     options={"maxiter": 150, "gtol": 1e-7, "ftol": 1e-12}) for start in starts]
    finite = [f for f in fits if np.isfinite(f.fun) and np.isfinite(f.x).all()]
    if not finite:
        return {"success": False}
    best = min(finite, key=lambda f: f.fun)
    return {"success": bool(best.success), "unit": best.x.tolist(), "lnl": -float(best.fun),
            "restarts": len(fits), "boundary": bool(np.any((best.x < 1e-6) | (best.x > 1-1e-6)))}


def audit(gp: dict, points: np.ndarray, predicted: np.ndarray, centre: np.ndarray,
          cov: np.ndarray, evaluate, folder: Path) -> dict:
    offsets, gradients, hessians = derivative_stencil(3)
    root = np.linalg.cholesky(cov)
    stencil = centre+offsets@root.T
    if not np.all((stencil >= 0) & (stencil <= 1)):
        return {"passed": False, "reason": "stencil_crosses_prior", "calls": 0}
    mode = optimize_mode(gp, points, predicted)
    if not mode["success"]:
        return {"passed": False, "reason": "mode_optimizer_unresolved", "calls": 0}
    # Posterior-scale profiles are checked separately from infinitesimal geometry.
    profile = []
    for direction in root.T:
        for sign in [-1., 1.]:
            vector = sign*direction
            distance = np.min(np.where(vector > 0, (1-centre)/np.maximum(vector, 1e-300),
                                       -centre/np.minimum(vector, -1e-300)))
            for length in [1., 2.]:
                profile.append(centre+min(length, .99*distance)*vector)
    query = np.vstack([stencil, mode["unit"], profile])
    before = predict(gp, query)
    np.savez(folder/"audit_before.npz", unit=query, predicted=before, covariance=cov, centre=centre)
    actual = evaluate(query)
    np.savez(folder/"audit.npz", unit=query, predicted=before, actual=actual, covariance=cov, centre=centre)
    check = derivative_audit(before[:37], actual[:37], gradients, hessians)
    mode_error = float(abs((actual[37]-before[37])-(actual[0]-before[0])))
    residual = actual-before; residual -= residual[0]
    profile_scaled = float(np.max(abs(residual[37:])/(.1+.01*(actual.max()-actual[37:]))))
    ad_gradient = np.asarray(jax.grad(gp["function"])(jnp.asarray(centre)))@root
    ad_hessian = root.T@np.asarray(gp["hessian"](jnp.asarray(centre)))@root
    gp_fd_grad = gradients[-1]@before[:37]
    gp_fd_hessian = np.einsum("ijn,n->ij", hessians[-1], before[:37])
    return {**check, "passed": bool(check["passed"] and mode_error < .2), "calls": len(query),
            "profile_passed": profile_scaled <= 1, "profile_max_scaled_error": profile_scaled,
            "mode_relative_error": mode_error, "mode": mode,
            "gp_autodiff_fd_gradient_difference": float(np.linalg.norm(ad_gradient-gp_fd_grad)),
            "gp_autodiff_fd_hessian_difference": float(np.linalg.norm(ad_hessian-gp_fd_hessian, ord=2))}


def run_case(spec: tuple, folder: Path, loader, source: Path) -> dict:
    split, n, mode, case = spec
    source_folder = source/split/f"n{n}_{mode}_case{case}_bo"
    paths = sorted(source_folder.glob("checkpoint_*.npz"), key=lambda p: int(p.stem.split("_")[-1]))
    with np.load(source/split/f"catalogue_{n}_{mode}_{case}.npz") as cat:
        coeff, duration = cat["coeff"], float(cat["duration"])
    def evaluate(unit):
        native = core.LOW[core.SHAPE]+unit*core.WIDTH[core.SHAPE]
        rates = np.array([loader.fast(p[0], p[1], .012, p[2]).ravel()[loader.active] for p in native])
        return base.event_target(rates, coeff, duration)
    folder.mkdir()
    rules = {name: None for name in ["correct_counter", "gpry_inspired", "gaussian_stability",
                                   "posterior_stability", "autodiff_audit", "profile_audit"]}
    streaks = Counter(); records = []; old = None; old_count = 0; did_audit = False
    for path in paths:
        with np.load(path) as data:
            x, y = data["x"], data["lnl"]
        unit = (x-core.LOW[core.SHAPE])/core.WIDTH[core.SHAPE]
        correct = False; max_error = None
        if old is not None:
            before = predict(old, unit[old_count:])
            actual = y[old_count:]
            max_error = float(np.max(abs(actual-before)/(.1+.01*(y.max()-actual))))
            correct = max_error <= 1
        gp = fit(x, y)
        proposals = gp_proposals(gp, unit, y, 86000+n+100*case)
        predictions = [predict(gp, p) for p, q in proposals]
        logweights = [v-q for v, (p, q) in zip(predictions, proposals)]
        summaries = [posterior_summary(p, w) for (p, q), w in zip(proposals, logweights)]
        numerical, numerical_current = summaries_agree(*summaries, TOL)
        numerical_previous = None
        cov = covariance(proposals[0][0], logweights[0]); centre = np.array(summaries[0]["mean"])
        stable, gaussian = False, None; geometry = None; joint = None
        if old is not None:
            previous = [predict(old, p) for p, q in proposals]
            previous_weights = [v-q for v, (p, q) in zip(previous, proposals)]
            previous_summaries = [posterior_summary(p, w) for (p, q), w in zip(proposals, previous_weights)]
            previous_ok, numerical_previous = summaries_agree(*previous_summaries, TOL)
            numerical &= previous_ok
            joint = max(symmetric_kl(a, b) for a, b in zip(previous_weights, logweights))
            stable = numerical and joint < .001 and all(summaries_agree(a, b, TOL)[0]
                                                       for a, b in zip(previous_summaries, summaries))
            old_cov = covariance(proposals[0][0], previous_weights[0])
            gaussian = gaussian_kl(np.array(previous_summaries[0]["mean"]), old_cov, centre, cov)
            weights = np.exp(logweights[0]-logsumexp(logweights[0]))
            rng = np.random.default_rng(90000+n+case)
            ix = rng.choice(len(weights), 128, p=weights)
            probe = jnp.asarray(proposals[0][0][ix])
            geometry = geometry_change(np.asarray(old["gradient"](probe)), np.asarray(gp["gradient"](probe)),
                                       np.asarray(old["hessian"](jnp.asarray(centre))),
                                       np.asarray(gp["hessian"](jnp.asarray(centre))), cov)
        flags = {"correct_counter": correct,
                 "gpry_inspired": correct and numerical and gaussian is not None and gaussian < .02,
                 "gaussian_stability": numerical and gaussian is not None and gaussian < .02,
                 "posterior_stability": stable}
        for name, passed in flags.items():
            streaks[name] = streaks[name]+1 if passed else 0
            if streaks[name] >= 2 and rules[name] is None:
                rules[name] = {"points": len(x), "calls": len(x)}
        row = {"points": len(x), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "correct": bool(correct), "correct_max_scaled_error": max_error,
               "numerical": bool(numerical), "stable": bool(stable), "joint_kl": joint,
               "numerical_current": numerical_current, "numerical_previous": numerical_previous,
               "integration_ess": [s["ess"] for s in summaries],
               "gaussian_kl": gaussian, "geometry": geometry, "streaks": dict(streaks)}
        if streaks["gaussian_stability"] >= 2 and not did_audit:
            check = audit(gp, proposals[0][0], predictions[0], centre, cov, evaluate, folder)
            row["audit"] = check
            if check["passed"]:
                rules["autodiff_audit"] = {"points": len(x), "calls": len(x)+check["calls"]}
            if check.get("profile_passed", False):
                rules["profile_audit"] = {"points": len(x), "calls": len(x)+check["calls"]}
            did_audit = True
        records.append(row); write_json(folder/"records.json", records)
        logging.info("%s N=%s prediction=%s stable=%s numerical=%s geometry=%s audit=%s", folder.name, len(x),
                     correct, stable, numerical, geometry["passed"] if geometry else None, row.get("audit", {}).get("passed"))
        old, old_count = gp, len(x)
    result = {"spec": list(spec), "rules": rules, "source_last_points": old_count, "reference_accessed": False}
    write_json(folder/"decision.json", result)
    return result


def score(out: Path, source: Path, decisions: dict) -> list:
    references = {split: json.loads((source/split/"records.json").read_text()) for split in ["ensemble", "pilot"]}
    rows = []
    for name, decision in decisions.items():
        decision_bytes = (out/name/"decision.json").read_bytes()
        split, n, mode, case = decision["spec"]
        matches = {r["points"]: r for r in references[split] if r["exposure"] == n and r["mode"] == mode
                   and r["case"] == case and r["strategy"] == "bo"}
        for rule, selected in decision["rules"].items():
            row = {"case": name, "rule": rule, "decision": selected}
            if selected is None:
                row["outcome"] = "not_accepted_on_saved_path"
            else:
                ref = matches[selected["points"]]
                row.update(outcome="reference_unresolved" if not ref["numerical_pass"] else
                           "accurate" if ref["accuracy_pass"] else "premature",
                           reference_kl=max(max(r["kl_direct_to_gp"], r["kl_gp_to_direct"]) for r in ref["metrics"]))
            rows.append(row)
        write_json(out/name/"score_provenance.json", {"decision_sha256": hashlib.sha256(decision_bytes).hexdigest()})
        assert decision_bytes == (out/name/"decision.json").read_bytes()
    write_json(out/"scores.json", rows)
    write_json(out/"summary.json", {rule: dict(Counter(r["outcome"] for r in rows if r["rule"] == rule))
                                   for rule in decisions[next(iter(decisions))]["rules"]})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=False)
    source = UNCERTAINTY/"convergence_20260910"
    groups = [("ensemble", 100, "perfect", 0)] if args.smoke else GROUPS
    sources = [Path(__file__), HERE/"PROTOCOL.md", REPO/"src/cosmic_integration/lnl_surrogate/local_geometry.py"]
    manifest = {"groups": groups, "source": str(source), "source_sha256": {}, "created_unix": time.time()}
    for p in sources:
        relative = p.relative_to(REPO); dest = args.outdir/"source_snapshot"/relative
        dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(p.read_bytes())
        manifest["source_sha256"][str(relative)] = hashlib.sha256(p.read_bytes()).hexdigest()
    write_json(args.outdir/"manifest.json", manifest)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    loader = essential_runs.Experiment(args.outdir/"population", "COMPAS_autodiff_stopping", False)
    decisions = {}
    try:
        for group in groups:
            split, n, mode, case = group; name = f"{split}_n{n}_{mode}_case{case}"
            write_json(args.outdir/"status.json", {"state": "running", "case": name, "completed": len(decisions)})
            decisions[name] = run_case(group, args.outdir/name, loader, source)
        write_json(args.outdir/"decisions.json", decisions)
        score(args.outdir, source, decisions)
        write_json(args.outdir/"status.json", {"state": "complete", "completed": len(decisions)})
    except Exception as exc:
        write_json(args.outdir/"status.json", {"state": "failed", "completed": len(decisions), "error": repr(exc)})
        raise


if __name__ == "__main__":
    main()
