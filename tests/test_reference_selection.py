"""Reference checkpoint selection must not silently inherit the old gates."""
import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / 'docs/studies/paper_completion/reference_selection.py'
SPEC = importlib.util.spec_from_file_location('reference_selection_test', PATH)
RULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RULE)


def row(n, kl, **extra):
    return dict(points=n, symmetric_kl=kl, **extra)


def test_selects_second_consecutive_checkpoint_without_old_stability_gate():
    rows = [row(58, .02), row(88, .009, stable=False), row(118, .008, accuracy_pass=False)]
    assert RULE.select_checkpoint(rows)['points'] == 118


def test_excursion_or_unresolved_estimate_breaks_streak():
    rows = [row(58, .009), row(88, .01), row(118, .008),
            row(148, .007, numerical_pass=False), row(178, .006), row(208, .005)]
    assert RULE.select_checkpoint(rows)['points'] == 208
    assert RULE.select_checkpoint(rows[:-1]) is None


def test_metric_is_half_sum_and_averages_independent_designs():
    r = dict(metrics=[dict(kl_direct_to_gp=.006, kl_gp_to_direct=.01),
                      dict(kl_direct_to_gp=.004, kl_gp_to_direct=.008)])
    assert RULE.symmetric_reference_kl(r) == pytest.approx(.007)


def test_duplicate_counts_rejected_and_nonfinite_cannot_select():
    with pytest.raises(ValueError, match='Duplicate'):
        RULE.select_checkpoint([row(58, .001), row(58, .001)])
    assert RULE.select_checkpoint([row(58, float('nan')), row(88, .001)]) is None
