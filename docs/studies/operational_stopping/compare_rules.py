"""Compare reference-free stopping decisions and run fresh audit experiments.

References are used only by ``score_prefixes`` after durable decision files
exist. The acquisition trajectories are shared, so non-stops are censored at
the end of the recorded path rather than extrapolated to later iterations.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
from scipy.special import logsumexp
from scipy.stats import qmc

from cosmic_integration.lnl_surrogate.stopping import (
    posterior_summary, summary_change, symmetric_kl,
)

HERE = Path(__file__).resolve().parent
RULES = ["best_plateau", "prediction", "stability", "combined", "combined_audit"]
LABELS = ["Best-likelihood plateau", "New-label predictions", "Posterior stability",
          "Predictions + stability", "Combined + fresh audit"]


def load_engine() -> ModuleType:
    """Load the operational runner without shadowing the older study's name."""
    name = "operational_comparison_engine"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, HERE / "run_study.py")
        engine = importlib.util.module_from_spec(spec)
        sys.modules[name] = engine
        spec.loader.exec_module(engine)
    return sys.modules[name]


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def prior_probe_count(volume: float, miss_probability: float = .05) -> int:
    """IID prior draws needed to hit one fixed region with the given volume.

    This is conditional on a detectable region and its minimum prior mass;
    adaptive points cannot be counted as IID draws in this expression.
    """
    if not 0 < volume < 1 or not 0 < miss_probability < 1:
        raise ValueError("Volume and miss probability must lie strictly in (0, 1)")
    return int(np.ceil(np.log(miss_probability) / np.log1p(-volume)))


def importance_audit(points: np.ndarray, predicted: np.ndarray, actual: np.ndarray,
                     log_proposal: np.ndarray) -> dict:
    """Estimate posterior impact on a fresh defensive-mixture audit sample.

    Both targets use the same proposal correction. This is a finite-sample
    diagnostic, not a confidence bound or a mode-discovery guarantee. A constant
    log-likelihood offset cancels, so this does not test evidence accuracy.
    """
    arrays = [np.asarray(a, dtype=float) for a in [points, predicted, actual, log_proposal]]
    points, predicted, actual, log_proposal = arrays
    if (points.ndim != 2 or len(points) < 64 or any(a.shape != (len(points),) for a in arrays[1:])
            or not all(np.isfinite(a).all() for a in arrays)):
        return {"passed": False, "reason": "invalid_or_insufficient_audit"}
    try:
        gp_summary = posterior_summary(points, predicted-log_proposal)
        exact_summary = posterior_summary(points, actual-log_proposal)
    except ValueError:
        return {"passed": False, "reason": "degenerate_audit"}
    changes = summary_change(gp_summary, exact_summary)
    kl = symmetric_kl(predicted-log_proposal, actual-log_proposal)
    ess = min(gp_summary["ess"], exact_summary["ess"])
    passed = (ess >= 64 and kl < .005 and changes["mean_sd"] < .05
              and changes["endpoint_sd"] < .05 and changes["width_fraction"] < .03)
    return {"passed": bool(passed), "sample_kl": kl, "minimum_ess": ess, **changes}


def prefix_decisions(records: list[dict], best_values: list[float],
                     audited_decision: dict) -> dict:
    """Select first qualifying prefixes without any reference-score argument."""
    if len(records) != len(best_values) or not records:
        raise ValueError("One best training value per nonempty checkpoint is required")
    if not np.isfinite(best_values).all() or np.any(np.diff(best_values) < -1e-10):
        raise ValueError("Best observed values must be finite and nondecreasing")
    decisions = {rule: None for rule in RULES}
    streaks = Counter()
    for i, row in enumerate(records):
        flags = {
            "prediction": row["prediction"]["passed"],
            "stability": row["stable"] and row["numerical_pass"],
            "combined": (row["prediction"]["passed"] and row["stable"]
                         and row["numerical_pass"]),
        }
        for rule, passed in flags.items():
            streaks[rule] = streaks[rule] + 1 if passed else 0
        qualifies = {rule: streaks[rule] >= 2 for rule in flags}
        qualifies["best_plateau"] = i >= 3 and best_values[i] - best_values[i-3] < .01
        for rule, passed in qualifies.items():
            if passed and decisions[rule] is None:
                decisions[rule] = {"checkpoint": row["checkpoint"], "calls": row["calls"]}
    if audited_decision["state"] == "operational_stop":
        decisions["combined_audit"] = {
            key: audited_decision[key] for key in ["checkpoint", "calls"]}
    return decisions


def load_operational(folder: Path) -> tuple[list[dict], dict]:
    """Read operational records only; deliberately exclude reference files."""
    records = json.loads((folder / "records.json").read_text())
    decision = json.loads((folder / "decision.json").read_text())
    if decision["reference_accessed"]:
        raise ValueError("The source decision accessed a reference")
    return records, decision


def score_prefixes(root: Path, decisions_file: Path) -> list[dict]:
    """Reveal saved external scores only after the decisions file exists."""
    decisions_bytes = decisions_file.read_bytes()
    decisions = json.loads(decisions_bytes)
    rows = []
    for name, rules in decisions.items():
        folder = root / name
        source_bytes = (folder / "decision.json").read_bytes()
        reference = json.loads((folder / "reference_score.json").read_text())
        if hashlib.sha256(source_bytes).hexdigest() != reference["decision_sha256"]:
            raise ValueError(f"Reference decision hash mismatch: {name}")
        scores = {row["checkpoint"]: row for row in reference["scores"]}
        for rule, decision in rules.items():
            row = {"case": name, "rule": rule, "decision": decision}
            if decision is None:
                row["outcome"] = "censored"
            else:
                score = scores[decision["checkpoint"]]
                row.update(reference_kl=max(score["reference_kl"]),
                           outcome=("reference_unresolved" if not score["numerical_pass"] else
                                    "accurate" if score["accuracy_pass"] else "premature"))
            rows.append(row)
    assert decisions_bytes == decisions_file.read_bytes()
    return rows


def replay(root: Path, out: Path) -> None:
    """Replay frozen candidate rules on a complete saved campaign."""
    status = json.loads((root / "status.json").read_text())
    if status["state"] != "complete":
        raise ValueError("Campaign must be complete before replay")
    out.mkdir(parents=True, exist_ok=False)
    decisions = {}
    source_hashes = {}
    for folder in sorted(root.glob("*/")):
        if not (folder / "decision.json").exists():
            continue
        records, decision = load_operational(folder)
        best = []
        for row in records:
            with np.load(folder / f"checkpoint_{row['checkpoint']:03d}.npz") as data:
                best.append(float(data["lnl"].max()))
        decisions[folder.name] = prefix_decisions(records, best, decision)
        for filename in ["records.json", "decision.json"]:
            source_hashes[f"{folder.name}/{filename}"] = hashlib.sha256(
                (folder / filename).read_bytes()).hexdigest()
    write_json(out / "decisions.json", decisions)
    rows = score_prefixes(root, out / "decisions.json")
    summary = {}
    for rule in RULES:
        selected = [row for row in rows if row["rule"] == rule]
        calls = [row["decision"]["calls"] for row in selected if row["decision"] is not None]
        summary[rule] = {
            "cases": len(selected), "outcomes": dict(Counter(row["outcome"] for row in selected)),
            "median_calls_at_stop": float(np.median(calls)) if calls else None,
            "families": {family: dict(Counter(row["outcome"] for row in selected
                          if row["case"].rsplit("_", 1)[0] == family))
                         for family in sorted({r["case"].rsplit("_", 1)[0] for r in selected})}}
    write_json(out / "scores.json", rows)
    write_json(out / "summary.json", summary)
    write_json(out / "manifest.json", {"source": str(root.resolve()), "source_sha256": source_hashes,
               "comparison_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})


def audit_first_trigger(root: Path, out: Path, compas: bool = False) -> None:
    """Test one fresh audit at the first GP-only stability trigger in each run."""
    if json.loads((root / "status.json").read_text())["state"] != "complete":
        raise ValueError("Campaign must be complete before first-trigger screening")
    engine = load_engine()
    from toys import make_target
    out.mkdir(parents=True, exist_ok=False)
    if compas:
        import essential_runs as base
        from posterior_convergence import event_target
        loader = base.Experiment(out / "population", "first_trigger_audit", False)
    decisions, details = {}, {}
    for folder in sorted(root.glob("*/")):
        if not (folder / "decision.json").exists():
            continue
        records, _ = load_operational(folder)
        streak, selected = 0, None
        for row in records:
            streak = streak + 1 if row["stable"] and row["numerical_pass"] else 0
            if streak >= 2:
                selected = row
                break
        decisions[folder.name] = {"stability_audit": None}
        if selected is None:
            details[folder.name] = {"state": "no_trigger"}
            continue
        calls = selected["calls"] + 192
        if calls > (2200 if compas else 800):
            details[folder.name] = {"state": "audit_exceeds_budget", "required_calls": calls}
            continue
        if compas:
            n, mode = folder.name.split("_")
            seed = 10117 + int(n[1:])
            source = base.HERE / "convergence_20260910/pilot"
            with np.load(source / f"catalogue_{n[1:]}_{mode}_0.npz") as catalogue:
                coeff, duration = catalogue["coeff"], float(catalogue["duration"])
            def evaluate(unit: np.ndarray) -> np.ndarray:
                phi = engine.core.LOW[engine.core.SHAPE] + unit*engine.core.WIDTH[engine.core.SHAPE]
                rates = np.array([loader.fast(p[0], p[1], .012, p[2]).ravel()[loader.active] for p in phi])
                return event_target(rates, coeff, duration)
        else:
            family, seed = folder.name.rsplit("_", 1)
            seed = int(seed)
            evaluate = make_target(family, seed)
        with np.load(folder / f"checkpoint_{selected['checkpoint']:03d}.npz") as data:
            model, scaler = engine.fit(data["x"], data["lnl"])
            best = float(data["lnl"].max())
            dimensions = data["x"].shape[1]
        pool = qmc.Sobol(dimensions, scramble=True, seed=seed+40000).random_base2(14)
        gp = engine.predict(model, scaler, pool)
        rng = np.random.default_rng(seed+50000)
        # IID indices from an independent GP-only quadrature; no true grid labels.
        chosen = rng.choice(len(pool), size=128, replace=True, p=np.exp(gp-logsumexp(gp)))
        x = np.vstack([pool[chosen], rng.uniform(size=(64, dimensions))])
        before = engine.predict(model, scaler, x)
        np.savez(out / f"{folder.name}_before.npz", x=x, predicted=before)
        actual = evaluate(x)
        check = engine.audit_check(before, actual, 128, best, engine.TOLERANCES)
        np.savez(out / f"{folder.name}_audit.npz", x=x, predicted=before, actual=actual)
        details[folder.name] = {"state": "accepted" if check["passed"] else "rejected",
                              "checkpoint": selected["checkpoint"], "calls": calls, **check}
        if check["passed"]:
            decisions[folder.name]["stability_audit"] = {
                "checkpoint": selected["checkpoint"], "calls": calls}
        write_json(out / "audit_details.json", details)
        logging.info("first-trigger %s %s calls=%s", folder.name, details[folder.name]["state"], calls)
    write_json(out / "decisions.json", decisions)
    rows = score_prefixes(root, out / "decisions.json")
    for row in rows:
        if row["decision"] is None:
            row["outcome"] = details[row["case"]]["state"]
    write_json(out / "scores.json", rows)
    write_json(out / "summary.json", dict(Counter(row["outcome"] for row in rows)))


def audit_stress(root: Path, out: Path) -> None:
    """Conditional rejection rates on five frozen missed-mode fits."""
    engine = load_engine()
    from toys import make_target
    out.mkdir(parents=True, exist_ok=False)
    sizes = [16, 64, 256, 1024]
    results = []
    for seed in range(2000, 2005):
        folder = root / f"hidden_mode_{seed}"
        _, decision = load_operational(folder)
        with np.load(folder / f"checkpoint_{decision['checkpoint']:03d}.npz") as data:
            model, scaler = engine.fit(data["x"], data["lnl"])
            best = float(data["lnl"].max())
        pool = qmc.Sobol(2, scramble=True, seed=seed+60000).random_base2(14)
        gp = engine.predict(model, scaler, pool)
        weights = np.exp(gp-logsumexp(gp))
        rng = np.random.default_rng(seed+70000)
        blocks = []
        for _ in range(32):
            ix = rng.choice(len(pool), size=64, replace=True, p=weights)
            blocks.append(np.vstack([pool[ix], rng.uniform(size=(1024, 2))]))
        x = np.stack(blocks)
        before = engine.predict(model, scaler, x.reshape(-1, 2)).reshape(32, -1)
        np.savez(out / f"hidden_mode_{seed}_before.npz", x=x, predicted=before)
        actual = make_target("hidden_mode", seed)(x.reshape(-1, 2)).reshape(32, -1)
        np.savez(out / f"hidden_mode_{seed}_audit.npz", x=x, predicted=before, actual=actual)
        for repeat in range(32):
            for count in sizes:
                check = engine.audit_check(before[repeat, :64+count], actual[repeat, :64+count],
                                           64, best, engine.TOLERANCES)
                results.append({"case": folder.name, "repeat": repeat, "prior_calls": count,
                                "total_audit_calls": count+64, **check})
        logging.info("audit-stress %s complete", folder.name)
        write_json(out / "checks.json", results)
    summary = {str(count): {"rejected": sum(not row["passed"] for row in results
                                           if row["prior_calls"] == count),
                           "audits": 160, "frozen_fits": 5, "total_audit_calls": count+64}
               for count in sizes}
    write_json(out / "summary.json", summary)
    write_json(out / "prior_probe_requirements.json", {
        str(volume): prior_probe_count(volume) for volume in [.05, .01, .001, .0001]})


def impact_replay(root: Path, audit_root: Path, out: Path) -> None:
    """Use the already-paid fresh audit labels to estimate posterior impact."""
    engine = load_engine()
    out.mkdir(parents=True, exist_ok=False)
    details = json.loads((audit_root / "audit_details.json").read_text())
    decisions, checks = {}, {}
    for name, detail in details.items():
        decisions[name] = {"stability_impact_audit": None}
        if detail["state"] not in ["accepted", "rejected"]:
            checks[name] = {"state": detail["state"]}
            continue
        with np.load(root / name / f"checkpoint_{detail['checkpoint']:03d}.npz") as train:
            model, scaler = engine.fit(train["x"], train["lnl"])
            dimensions = train["x"].shape[1]
        seed = 10117+int(name.split("_")[0][1:]) if dimensions == 3 else int(name.rsplit("_", 1)[1])
        pool = qmc.Sobol(dimensions, scramble=True, seed=seed+40000).random_base2(14)
        gp = engine.predict(model, scaler, pool)
        log_z = logsumexp(gp)-np.log(len(pool))
        with np.load(audit_root / f"{name}_audit.npz") as audit:
            # 128 approximate GP-posterior draws plus 64 uniform-prior draws.
            log_q = np.logaddexp(np.log(2/3)+audit["predicted"]-log_z, np.log(1/3))
            check = importance_audit(audit["x"], audit["predicted"], audit["actual"], log_q)
        checks[name] = {"state": "accepted" if check["passed"] else "rejected",
                        "checkpoint": detail["checkpoint"], "calls": detail["calls"], **check}
        if check["passed"]:
            decisions[name]["stability_impact_audit"] = {
                key: detail[key] for key in ["checkpoint", "calls"]}
        logging.info("impact-audit %s %s", name, checks[name]["state"])
    write_json(out / "checks.json", checks)
    write_json(out / "decisions.json", decisions)
    rows = score_prefixes(root, out / "decisions.json")
    for row in rows:
        if row["decision"] is None:
            row["outcome"] = checks[row["case"]]["state"]
    write_json(out / "scores.json", rows)
    write_json(out / "summary.json", dict(Counter(row["outcome"] for row in rows)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["replay", "first-audit", "first-audit-compas", "audit-stress", "impact-replay"])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--audit-source", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.mode == "impact-replay":
        if args.audit_source is None:
            parser.error("impact-replay requires --audit-source")
        impact_replay(args.source, args.audit_source, args.outdir)
        return
    {"replay": replay, "first-audit": audit_first_trigger,
     "first-audit-compas": lambda root, out: audit_first_trigger(root, out, compas=True),
     "audit-stress": audit_stress}[args.mode](args.source, args.outdir)


if __name__ == "__main__":
    main()
