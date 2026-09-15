"""Numerical plotting contracts independent of Matplotlib appearance."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def plotting():
    path = Path(__file__).resolve().parents[1]/'docs/studies/operational_stopping/gpry_compas/plot_count_history.py'
    spec = importlib.util.spec_from_file_location('count_plot_test', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_late_self_zero_is_omitted_without_dropping_direct_endpoint(plotting):
    rows = [dict(posterior_comparisons=i, symmetric_kl_to_late_gp=v,
                 symmetric_kl=.03/(i+1), late_reference_is_self=(i == 2),
                 reference_numerical=True, gp_numerical=(i != 1), four_numerical=True)
            for i, v in enumerate([.5, .02, 0.])]
    x, y, resolved = plotting.curve_data(rows[::-1], 'symmetric_kl_to_late_gp')
    np.testing.assert_array_equal(x, [0, 1]); np.testing.assert_allclose(y, [.5, .02])
    np.testing.assert_array_equal(resolved, [True, False])
    x, y, _ = plotting.curve_data(rows, 'symmetric_kl')
    assert len(x) == 3 and y[-1] == pytest.approx(.01)
    with pytest.raises(ValueError, match='Duplicate'):
        plotting.curve_data(rows+[rows[0]], 'symmetric_kl')


def test_completed_groups_do_not_mix_counts_or_measurements(plotting):
    snapshot = dict(cases=[dict(result=dict(expected_events=n, mode=m))
                          for n in [10, 50] for m in ['perfect', 'uncertain']])
    assert len(plotting.subset(snapshot, 10, 'uncertain')) == 1
    assert plotting.subset(snapshot, 100, 'perfect') == []
