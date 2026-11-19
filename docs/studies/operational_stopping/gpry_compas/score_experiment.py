"""Reveal independent COMPAS references only after all online decisions exist."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
import time

import numpy as np
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("gpry_compas_runner", HERE/"run_experiment.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def directed_kls(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a, b = a-logsumexp(a), b-logsumexp(b)
    return max(0., float(np.exp(a)@(a-b))), max(0., float(np.exp(b)@(b-a)))


def agreement(direct: dict, gp: dict, kl: tuple[float, float]) -> dict:
    from cosmic_integration.lnl_surrogate.stopping import summary_change
    changes = summary_change(direct, gp)
    qa, qb = np.asarray(direct["quantiles"]), np.asarray(gp["quantiles"])
    width95 = float(np.max(abs((qb[4]-qb[0])/(qa[4]-qa[0])-1)))
    return {**changes, "width95_fraction": width95, "kl_direct_to_gp": kl[0], "kl_gp_to_direct": kl[1],
            "passed": bool(changes["mean_sd"] < .2 and changes["endpoint_sd"] < .2
                           and changes["width_fraction"] < .1 and width95 < .1 and max(kl) < .01)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    status = json.loads((args.input/"status.json").read_text())
    frozen = (args.input/"decisions.json").read_bytes()
    assert status["state"] == "complete"
    assert hashlib.sha256(frozen).hexdigest() == status["decision_sha256"]
    decisions = json.loads(frozen)
    assert all(d["complete"] and not d["reference_accessed"] for d in decisions.values())
    output = args.input/"offline"
    output.mkdir(exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parity_loader = runner.data_source.Experiment(output/"population_parity", "offline_rate_parity", False)
    reference_root = HERE.parents[1]/"event_uncertainty/convergence_20260910/ensemble"
    reference_cache, hashes, rows, outcomes, parity = {}, {}, [], [], []
    start = time.perf_counter()
    for name, decision in decisions.items():
        folder = args.input/name
        assert json.loads((folder/"decision.json").read_text()) == decision
        n = decision["expected_events"]; power = 16 if n == 100 else 17
        if n not in reference_cache:
            refs = []
            for scramble in [0, 1]:
                rates_path = reference_root/f"reference_n{n}_p{power}_{scramble}.npz"
                proposal_path = reference_root/f"proposal_n{n}_p{power}_{scramble}.npz"
                for path in [rates_path, proposal_path]:
                    hashes[str(path.relative_to(runner.REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
                with np.load(rates_path) as data:
                    phi, rates = data["phi"], data["rates"]
                with np.load(proposal_path) as data:
                    logq = data["logq"]
                selected = np.unique(np.linspace(0, len(phi)-1, 3, dtype=int))
                fresh = np.array([parity_loader.fast(p[0], p[1], .012, p[2]).ravel()[parity_loader.active]
                                  for p in phi[selected]])
                np.testing.assert_allclose(fresh, rates[selected], rtol=2e-11, atol=1e-13,
                    err_msg="Cached scoring rates differ from the online COMPAS evaluator")
                parity.append({"expected_events": n, "scramble": scramble, "points_checked": len(selected),
                               "max_absolute_difference": float(np.max(abs(fresh-rates[selected])))})
                refs.append(((phi-runner.LOW)/runner.WIDTH, rates, logq))
            reference_cache[n] = refs
        references = reference_cache[n]
        with np.load(folder/"catalogue.npz") as data:
            coeff, duration = data["coeff"], float(data["duration"])
        exact = [runner.source_target.event_target(r, coeff, duration)-q for _, r, q in references]
        direct_summaries = [runner.posterior_summary(p, w) for (p, _, _), w in zip(references, exact)]
        reference_ok, reference_check = runner.summaries_agree(*direct_summaries, runner.TOL)
        checkpoints = json.loads((folder/"checkpoints.json").read_text())
        case_rows = []
        for checkpoint in checkpoints:
            points = checkpoint["points"]
            path = folder/f"checkpoint_{points}.npz"
            assert hashlib.sha256(path.read_bytes()).hexdigest() == checkpoint["sha256"]
            state, _, _ = runner.load_checkpoint(path)
            predicted = [runner.predict(state, p)-q for p, _, q in references]
            gp_summaries = [runner.posterior_summary(p, w) for (p, _, _), w in zip(references, predicted)]
            gp_ok, gp_check = runner.summaries_agree(*gp_summaries, runner.TOL)
            kls = [directed_kls(a, b) for a, b in zip(exact, predicted)]
            metrics = [agreement(a, b, kl) for a, b, kl in zip(direct_summaries, gp_summaries, kls)]
            row = {"name": name, "expected_events": n, "mode": decision["mode"], "case": decision["case"],
                "points": points, "symmetric_kl": float(np.mean(kls)),
                "symmetric_kl_replicates": [float(np.mean(k)) for k in kls],
                "maximum_directed_kl": float(np.max(kls)), "metrics": metrics,
                "reference_numerical": reference_ok, "reference_check": reference_check,
                "gp_numerical": gp_ok, "gp_check": gp_check,
                "accuracy_pass": bool(reference_ok and gp_ok and all(m["passed"] for m in metrics)),
                "reference_ess": [s["ess"] for s in direct_summaries],
                "gp_ess": [s["ess"] for s in gp_summaries]}
            case_rows.append(row)
        by_points = {r["points"]: r for r in case_rows}
        for rule, selected in decision["rules"].items():
            result = {"name": name, "rule": rule, "points": selected, "expected_events": n,
                      "mode": decision["mode"], "case": decision["case"]}
            if selected is None:
                result["outcome"] = "budget_limit"
            else:
                score = by_points[selected]
                result.update(outcome="reference_unresolved" if not (score["reference_numerical"] and score["gp_numerical"])
                              else "accurate" if score["accuracy_pass"] else "premature",
                    symmetric_kl=score["symmetric_kl"], maximum_directed_kl=score["maximum_directed_kl"],
                    final_symmetric_kl=case_rows[-1]["symmetric_kl"],
                    continuation_labels=decision["total_labels"]-selected)
            outcomes.append(result)
        rows.extend(case_rows)
        runner.write_json(output/f"{name}.json", case_rows)
        runner.write_json(output/"status.json", {"state": "running", "case": name, "scored_checkpoints": len(rows)})
        logging.info("Scored %s: %s checkpoints, reference precision=%s, final KL=%.5g",
                     name, len(case_rows), reference_ok, case_rows[-1]["symmetric_kl"])
    runner.write_json(output/"records.json", rows)
    runner.write_json(output/"outcomes.json", outcomes)
    runner.write_json(output/"summary.json", {rule: dict(Counter(r["outcome"] for r in outcomes if r["rule"] == rule))
                                            for rule in runner.RULES})
    assert frozen == (args.input/"decisions.json").read_bytes()
    runner.write_json(output/"manifest.json", {"decision_sha256": hashlib.sha256(frozen).hexdigest(),
        "reference_sha256": hashes, "score_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rate_parity": parity, "wall_seconds": time.perf_counter()-start,
        "symmetric_kl_definition": "half sum of both directed KLs"})
    runner.write_json(output/"status.json", {"state": "complete", "cases": len(decisions), "checkpoints": len(rows)})


if __name__ == "__main__":
    main()
