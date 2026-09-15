"""Incomplete and resource-limited ensembles must not look like converged groups."""
from pathlib import Path
import pytest
import numpy as np


@pytest.fixture
def reporting(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'docs/studies/event_uncertainty'))
    import publish_convergence
    return publish_convergence


def test_summary_waits_for_complete_paired_group(reporting):
    rows = [dict(exposure=30, mode='perfect', case=i, strategy=s,
                 state=reporting.SUCCESS, points=p)
            for s in ['bo', 'random'] for i, p in enumerate([610, 994])]
    assert reporting.completed_groups(rows[:-1], count=2) == []
    group = reporting.completed_groups(rows, count=2)[0]
    assert group['bo']['quantiles'][1] == 802
    with pytest.raises(ValueError, match='Duplicate'):
        reporting.completed_groups(rows+[rows[0]], count=2)


def test_censored_run_is_not_a_convergence_time(reporting):
    rows = [dict(exposure=100, mode='perfect', case=0, strategy=s,
                 state=reporting.SUCCESS, points=866) for s in ['bo', 'random']]
    rows[1].update(state='resource_limit', points=4322)
    group = reporting.completed_groups(rows, count=1)[0]
    assert group['bo']['quantiles'] == [866, 866, 866]
    assert group['random']['quantiles'] is None
    assert group['random']['unresolved'] == 1


def test_corner_kl_is_symmetric_and_uses_worst_reference(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'docs/studies/event_count_bo'))
    from posterior_evolution import largest_kl
    a=np.log([.5,.5,.1,.9]);b=np.log([.5,.5,.9,.1])
    expected=.8*np.log(9)
    assert largest_kl(a,b,[2,2]) == pytest.approx(expected)
    assert largest_kl(b,a,[2,2]) == pytest.approx(expected)
    assert largest_kl(a,a+100,[2,2]) == pytest.approx(0,abs=1e-12)
