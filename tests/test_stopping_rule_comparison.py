"""Stopping decisions must use the available prefix, with references kept out."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / "docs/studies/operational_stopping/compare_rules.py"
SPEC = importlib.util.spec_from_file_location("stopping_rule_comparison", PATH)
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


def record(i, prediction=True, stable=True, numerical=True):
    return {"checkpoint": i, "calls": 48+32*i, "prediction": {"passed": prediction},
            "stable": stable, "numerical_pass": numerical}


def test_first_stop_does_not_see_later_records():
    records = [record(0, False, False), record(1), record(2)]
    source = {"state": "resource_limit"}
    first = comparison.prefix_decisions(records, [0, 1, 2], source)
    extended = comparison.prefix_decisions(records+[record(3, False, False)], [0, 1, 2, 3], source)
    for rule in ["prediction", "stability", "combined"]:
        assert first[rule] == extended[rule] == {"checkpoint": 2, "calls": 112}
    assert first["combined_audit"] is None


def test_numerical_failure_resets_stability_streak():
    rows = [record(0, False, False), record(1), record(2, numerical=False), record(3), record(4)]
    decisions = comparison.prefix_decisions(rows, list(range(5)), {"state": "resource_limit"})
    assert decisions["prediction"]["checkpoint"] == 2
    assert decisions["stability"]["checkpoint"] == decisions["combined"]["checkpoint"] == 4


def test_best_plateau_uses_three_updates_and_ignores_additive_offset():
    rows = [record(i, False, False) for i in range(5)]
    best = np.array([0, 1, 1.001, 1.002, 1.003])
    a = comparison.prefix_decisions(rows, best, {"state": "resource_limit"})
    b = comparison.prefix_decisions(rows, best+100, {"state": "resource_limit"})
    assert a == b
    assert a["best_plateau"] == {"checkpoint": 4, "calls": 176}


def test_audited_calls_include_accepted_audit_cost():
    source = {"state": "operational_stop", "checkpoint": 2, "calls": 192}
    decision = comparison.prefix_decisions([record(i) for i in range(3)], [0, 1, 2], source)
    assert decision["combined_audit"]["calls"] == 192


def test_scoring_requires_durable_decisions(tmp_path):
    with pytest.raises(FileNotFoundError):
        comparison.score_prefixes(tmp_path / "nonexistent_reference", tmp_path / "missing.json")


def test_operational_engine_does_not_shadow_older_study(monkeypatch):
    monkeypatch.syspath_prepend(str(PATH.parent))
    engine = comparison.load_engine()
    assert engine.CONFIG["maximum_calls"] == 800
    assert engine.core.TRUTH.shape == (4,)


def test_importance_audit_ignores_only_constant_offsets():
    rng = np.random.default_rng(27)
    x = rng.uniform(size=(192, 2))
    predicted = -np.sum((x-.5)**2, axis=1)
    logq = np.zeros(192)
    assert comparison.importance_audit(x, predicted, predicted+100, logq)["passed"]
    assert not comparison.importance_audit(x, predicted, predicted+4*x[:, 0], logq)["passed"]


def test_importance_audit_detects_observed_missed_mass_and_ignores_tiny_tail():
    rng = np.random.default_rng(28)
    x = rng.uniform(size=(192, 2))
    predicted = np.r_[np.zeros(191), -100.]
    actual = predicted.copy()
    actual[-1] += 1
    assert comparison.importance_audit(x, predicted, actual, np.zeros(192))["passed"]
    actual[-1] = 8
    assert not comparison.importance_audit(x, predicted, actual, np.zeros(192))["passed"]


def test_importance_audit_rejects_low_ess_and_nonfinite_labels():
    rng = np.random.default_rng(29)
    x = rng.uniform(size=(192, 2))
    p = np.r_[np.zeros(1), -100*np.ones(191)]
    assert not comparison.importance_audit(x, p, p, np.zeros(192))["passed"]
    p[-1] = np.nan
    assert not comparison.importance_audit(x, p, p, np.zeros(192))["passed"]


@pytest.mark.parametrize("volume", [.05, .01, .001, .0001])
def test_prior_probe_count_is_minimum_sufficient_integer(volume):
    n = comparison.prior_probe_count(volume)
    assert (1-volume)**n <= .05
    assert (1-volume)**(n-1) > .05


@pytest.mark.parametrize("volume, delta", [(0, .05), (1, .05), (.01, 0), (.01, 1), (np.nan, .05)])
def test_invalid_discovery_assumptions_rejected(volume, delta):
    with pytest.raises(ValueError):
        comparison.prior_probe_count(volume, delta)
