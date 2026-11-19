"""Verify frozen sources, training prefixes, pre-training predictions and stops."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("gpry_verification_runner", HERE/"run_experiment.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.input/"manifest.json").read_text())
    source_count = 0
    for relative, sha in manifest["source_sha256"].items():
        assert hashlib.sha256((args.input/"source_snapshot"/relative).read_bytes()).hexdigest() == sha
        assert hashlib.sha256((runner.REPO/relative).read_bytes()).hexdigest() == sha
        source_count += 1
    frozen = (args.input/"decisions.json").read_bytes()
    sha = hashlib.sha256(frozen).hexdigest()
    assert sha == json.loads((args.input/"status.json").read_text())["decision_sha256"]
    assert sha == json.loads((args.input/"offline/manifest.json").read_text())["decision_sha256"]
    labels = batches_checked = predictions_checked = checkpoints_checked = 0
    maximum_prediction_difference = 0.
    for name, decision in json.loads(frozen).items():
        folder = args.input/name
        batches = json.loads((folder/"batches.json").read_text())
        checkpoints = json.loads((folder/"checkpoints.json").read_text())
        with np.load(folder/"checkpoint_58.npz") as data:
            unit, lnl = data["unit"], data["lnl"]
        counter = runner.CorrectCounter(3)
        rules = {r: None for r in runner.RULES}
        streak_by_n = {58: 0}
        for batch in batches:
            assert batch["before_points"] == len(unit)
            assert batch["best_before"] == float(lnl.max())
            result = counter.update(np.asarray(batch["prediction"]), np.asarray(batch["actual"]), batch["best_before"])
            assert result == batch["counter"]
            new = np.asarray(batch["unit"])
            assert np.all((new >= 0) & (new <= 1))
            unit, lnl = np.vstack([unit, new]), np.r_[lnl, batch["actual"]]
            streak_by_n[len(unit)] = counter.streak
            if result["passed"] and rules["correct_counter"] is None:
                rules["correct_counter"] = len(unit)
            batches_checked += 1
        next_batch = {b["before_points"]: b for b in batches}
        gaussian_streak = joint_streak = 0
        for checkpoint in checkpoints:
            n = checkpoint["points"]; path = folder/f"checkpoint_{n}.npz"
            assert hashlib.sha256(path.read_bytes()).hexdigest() == checkpoint["sha256"]
            state, saved_unit, saved_lnl = runner.load_checkpoint(path)
            np.testing.assert_array_equal(saved_unit, unit[:n])
            np.testing.assert_array_equal(saved_lnl, lnl[:n])
            assert checkpoint["counter"] == streak_by_n[n]
            if n in next_batch:
                batch = next_batch[n]
                prediction = runner.predict(state, np.asarray(batch["unit"]))
                difference = float(np.max(abs(prediction-batch["prediction"])))
                maximum_prediction_difference = max(maximum_prediction_difference, difference)
                np.testing.assert_allclose(prediction, batch["prediction"], rtol=0, atol=1e-7)
                predictions_checked += len(prediction)
            check = checkpoint["stability"]
            if check is not None:
                gaussian_streak = gaussian_streak+1 if check["gaussian_pass"] else 0
                joint_streak = joint_streak+1 if check["joint_pass"] else 0
                if (gaussian_streak >= 2 and streak_by_n[n] >= counter.required
                        and rules["counter_and_gaussian"] is None):
                    rules["counter_and_gaussian"] = n
                if joint_streak >= 2 and rules["joint_stability"] is None:
                    rules["joint_stability"] = n
            checkpoints_checked += 1
        assert rules == decision["rules"], (name, rules, decision["rules"])
        assert len(unit) == decision["total_labels"]
        labels += len(unit)
    runner.write_json(args.input/"verification.json", {"state": "passed", "source_hashes": source_count,
        "decision_sha256": sha, "likelihood_labels": labels, "batches_replayed": batches_checked,
        "checkpoint_hashes": checkpoints_checked, "pre_training_predictions_checked": predictions_checked,
        "max_pre_training_prediction_difference": maximum_prediction_difference,
        "all_three_stopping_rules_reproduced": True})


if __name__ == "__main__":
    main()
