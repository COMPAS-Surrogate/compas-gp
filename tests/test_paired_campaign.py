"""Pairing, expansion gates, and generative posterior predictive contracts."""
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'docs/studies/event_uncertainty'))
    import paired_campaign
    import paired_diagnostics
    return paired_campaign, paired_diagnostics


def test_generated_counts_and_measurements_share_latent_catalogue(modules, tmp_path):
    campaign, _ = modules
    run = campaign.PairedExperiment.__new__(campaign.PairedExperiment)
    run.out, run.catalogues, run.groups, run.pilot = tmp_path, 1, campaign.GROUPS, False
    run.active = np.arange(2)
    run.lower = np.array([[10., .1], [20., .3]])
    run.upper = run.lower + np.array([5., .1])
    run.fast = lambda *args: np.array([1., 2.])
    run.fiducial_total = 3.
    run.prepare_catalogues()
    rows = campaign.verify_pairs(tmp_path, 1)
    assert [r['expected'] for r in rows] == [30, 100, 300, 900, 1200]
    assert run.continuation_seed('perfect') == run.continuation_seed('uncertain')
    # A count/latent mismatch must be caught before any training.
    path = tmp_path / 'catalogue_30_uncertain_0.npz'
    cat = dict(np.load(path))
    cat['latent'][0,0] += 1
    np.savez(path, **cat)
    with pytest.raises(AssertionError):
        campaign.verify_pairs(tmp_path, 1)


def test_gate_rejects_numerical_failures_and_inaccurate_bo(modules):
    campaign, _ = modules
    rows = [dict(exposure=n, mode=m, strategy=s, state='reference_verified_convergence')
            for n,m in campaign.GROUPS for s in ('bo','random')]
    assert campaign.pilot_gate(rows)['passed']
    rows[1]['state'] = 'resource_limit'
    assert campaign.pilot_gate(rows)['passed']
    rows[0]['state'] = 'resource_limit'
    assert not campaign.pilot_gate(rows)['passed']
    rows[0]['state'] = 'reference_verified_convergence'
    rows[1]['state'] = 'reference_unresolved'
    assert not campaign.pilot_gate(rows)['passed']
    with pytest.raises(ValueError):
        campaign.pilot_gate(rows[:-1])


def test_predictive_counts_include_posterior_amplitude_uncertainty(modules):
    _, diagnostics = modules
    rates = np.array([[120.]])
    low, high = np.array([[10., .1]]), np.array([[11., .2]])
    perfect, counts = diagnostics.replicate(rates, np.array([0.]), 99, 1., low, high,
                                          'perfect', draws=5000, seed=91)
    uncertain, other = diagnostics.replicate(rates, np.array([0.]), 99, 1., low, high,
                                           'uncertain', draws=5000, seed=91)
    np.testing.assert_array_equal(counts, other)
    # Almost untruncated Gamma(100, rate=10000) amplitude gives E[N]=100,
    # Var[N]=200; a plug-in median parameter simulation would have Var[N]~100.
    assert abs(counts.mean()-100) < 1
    assert abs(counts.var()-200) < 15
    assert all(np.all((x>=low) & (x<=high)) for x in perfect)
    assert any(np.any((x<low) | (x>high)) for x in uncertain)
    assert all(len(x)==n for x,n in zip(uncertain, counts))


def test_reference_cache_refuses_changed_coordinates(modules, tmp_path):
    campaign, _ = modules
    run = campaign.PairedExperiment.__new__(campaign.PairedExperiment)
    run.out = tmp_path / 'pilot'
    run.out.mkdir()
    run.active = np.arange(2)
    run.cache_seconds, run.cache_calls = 0., 0
    run.fast = lambda *args: np.array([1., 2.])
    phi = np.array([[-.3,.35,4.7]])
    first = run.rates(phi, 'synthetic')
    run.fast = lambda *args: pytest.fail('Cached reference was reevaluated')
    np.testing.assert_array_equal(run.rates(phi, 'synthetic'), first)
    assert run.cache_calls == 1
    with pytest.raises(AssertionError):
        run.rates(phi+.01, 'synthetic')
