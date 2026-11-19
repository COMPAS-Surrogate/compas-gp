"""Repeat the frozen COMPAS audit at the stricter posterior-stability trigger."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# The historical COMPAS loader imports another module named run_study.
spec = importlib.util.spec_from_file_location("compas_geometry_study", HERE/"run_study.py")
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=False)
    source = study.UNCERTAINTY/"convergence_20260910"
    decisions = json.loads((args.input/"decisions.json").read_text())
    snapshots = [Path(__file__), HERE/"FOLLOWUP_PROTOCOL.md", HERE/"run_study.py",
                 study.REPO/"src/cosmic_integration/lnl_surrogate/local_geometry.py"]
    hashes = {}
    for path in snapshots:
        relative = path.relative_to(study.REPO)
        dest = args.outdir/"source_snapshot"/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
        hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    study.write_json(args.outdir/"manifest.json", {"source_sha256": hashes,
        "input_decisions_sha256": hashlib.sha256((args.input/"decisions.json").read_bytes()).hexdigest(),
        "interpretation": "same-catalogue development follow-up; prior scores already available"})
    loader = study.essential_runs.Experiment(args.outdir/"population", "COMPAS_later_audit", False)
    output = {}
    for name, decision in decisions.items():
        split, n, mode, case = decision["spec"]
        selected = decision["rules"]["posterior_stability"]
        folder = args.outdir/name
        folder.mkdir()
        if selected is None:
            output[name] = {"decision": None, "reason": "no_saved_trigger"}
            continue
        points = selected["points"]
        checkpoint = source/split/f"n{n}_{mode}_case{case}_bo"/f"checkpoint_{points}.npz"
        with np.load(checkpoint) as data:
            x, y = data["x"], data["lnl"]
        with np.load(source/split/f"catalogue_{n}_{mode}_{case}.npz") as cat:
            coeff, duration = cat["coeff"], float(cat["duration"])
        def evaluate(unit):
            native = study.core.LOW[study.core.SHAPE]+unit*study.core.WIDTH[study.core.SHAPE]
            rates = np.array([loader.fast(p[0], p[1], .012, p[2]).ravel()[loader.active] for p in native])
            return study.base.event_target(rates, coeff, duration)
        gp = study.fit(x, y)
        unit = (x-study.core.LOW[study.core.SHAPE])/study.core.WIDTH[study.core.SHAPE]
        query, logq = study.gp_proposals(gp, unit, y, 86000+n+100*case)[0]
        predicted = study.predict(gp, query)
        summary = study.posterior_summary(query, predicted-logq)
        cov = study.covariance(query, predicted-logq)
        check = study.audit(gp, query, predicted, np.array(summary["mean"]), cov, evaluate, folder)
        result = {"points": points, "calls": points+check["calls"], "audit": check,
                  "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
        study.write_json(folder/"decision.json", result)
        output[name] = result
    study.write_json(args.outdir/"decisions.json", output)
    frozen = (args.outdir/"decisions.json").read_bytes()
    scores = {r["case"]: r for r in json.loads((args.input/"scores.json").read_text())
              if r["rule"] == "posterior_stability"}
    rows = []
    for name, result in output.items():
        for rule, flag in [("posterior_then_profile", "profile_passed"),
                           ("posterior_then_derivatives", "passed")]:
            accepted = result.get("audit", {}).get(flag, False)
            rows.append({"case": name, "rule": rule, "accepted": accepted,
                "outcome": scores[name]["outcome"] if accepted else "audit_rejected",
                "points": result.get("points"), "calls": result.get("calls"),
                "reference_kl": scores[name].get("reference_kl")})
    study.write_json(args.outdir/"scores.json", rows)
    study.write_json(args.outdir/"summary.json", {rule: dict(Counter(r["outcome"] for r in rows if r["rule"] == rule))
                     for rule in ["posterior_then_profile", "posterior_then_derivatives"]})
    assert frozen == (args.outdir/"decisions.json").read_bytes()
    study.write_json(args.outdir/"status.json", {"state": "complete", "cases": len(output),
        "decision_sha256": hashlib.sha256(frozen).hexdigest()})


if __name__ == "__main__":
    main()
