"""Protect the distinction between a plotted KL check and actual acceptance."""
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def plotter():
    path = Path(__file__).resolve().parents[1]/'docs/studies/operational_stopping/gpry_compas/plot_audit_stopping.py'
    spec = importlib.util.spec_from_file_location('audit_stopping_plot_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_passing_kl_components_do_not_invent_a_stop(plotter):
    record = dict(selected=None, audits=[dict(training=658, result=dict(
        passed=False, maximum_directed_kl=.002, bootstrap_95_kl=.004))])
    audit = plotter.plotted_audits(record)[0]
    assert not audit['observed_stop']
    assert audit['total'] == 754


def test_reference_definition_and_unresolved_points_are_preserved(plotter):
    rows = [dict(points=88, maximum_directed_kl=.03, symmetric_kl=.02,
                 reference_numerical=True, gp_numerical=False),
            dict(points=58, maximum_directed_kl=.3, symmetric_kl=.2,
                 reference_numerical=True, gp_numerical=True)]
    curve = plotter.reference_curve(rows)
    assert curve['points'] == [58, 88]
    assert curve['maximum_directed_kl'] == [.3, .03]
    assert curve['symmetric_kl'] == [.2, .02]
    assert curve['unresolved'] == [False, True]
    with pytest.raises(ValueError):
        plotter.reference_curve(rows+rows[:1])


def test_selected_checkpoint_requires_an_accepted_audit(plotter):
    with pytest.raises(ValueError):
        plotter.plotted_audits(dict(selected=dict(training=418), audits=[]))
