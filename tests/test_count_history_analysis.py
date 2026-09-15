"""Pairing must follow catalogue identities, not incidental file ordering."""
import importlib.util
from pathlib import Path

import pytest


def test_paired_cost_differences_reject_missing_or_duplicate_views(monkeypatch):
    folder = Path(__file__).resolve().parents[1]/'docs/studies/operational_stopping/gpry_compas'
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location('count_analysis_test', folder/'analyze_count_history.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [dict(expected_events=n, case=c, mode=m,
                 stop=dict(selected=dict(training=t)))
            for n, c, m, t in [(100, 201, 'uncertain', 298),
                                (100, 200, 'perfect', 298),
                                (100, 201, 'perfect', 358),
                                (100, 200, 'uncertain', 358)]]
    assert module.paired_differences(rows) == [60, -60]
    with pytest.raises(ValueError, match='Incomplete'):
        module.paired_differences(rows[:-1])
    with pytest.raises(ValueError, match='Duplicate'):
        module.paired_differences(rows + rows[:1])
