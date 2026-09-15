"""Stopping cannot silently accept unstable, inaccurate or unresolved posteriors."""
from pathlib import Path
import numpy as np
import pytest


@pytest.fixture
def study(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'docs/studies/event_uncertainty'))
    import posterior_convergence
    return posterior_convergence


def test_stop_needs_consecutive_verified_stability(study):
    assert study.stopping_streak(0, True, True, True) == 1
    assert study.stopping_streak(1, True, True, True) == 2
    assert study.stopping_streak(1, False, True, True) == 0
    assert study.stopping_streak(1, True, False, True) == 0
    assert study.stopping_streak(1, True, True, False) == 0


def test_stability_ignores_constant_offsets_but_detects_shape_changes(study):
    summary={'sd':[1.]*4,'mean':[0.]*4,
             'quantiles':np.broadcast_to(np.linspace(-2,2,7)[:,None],(7,4)).tolist()}
    old=[np.log([.2,.3,.5])]*2
    same=[x+100 for x in old]
    assert study.stable_posterior(old,same,[summary]*2,[summary]*2)
    changed=[np.log([.7,.2,.1])]*2
    assert not study.stable_posterior(old,changed,[summary]*2,[summary]*2)
    shifted={**summary,'mean':[1.]*4}
    assert not study.stable_posterior(old,same,[summary]*2,[shifted]*2)


def test_reference_mean_matches_original_gpjax_prediction(study):
    rng=np.random.default_rng(42)
    x=rng.uniform(size=(12,3));y=np.sin(x[:,0]*3)[:,None]
    core=study.core
    model=core.fit_exact_gp(core.JaxTrainingData(x,y),np.array([[0.,0.,0.],[1.,1.,1.]]),
                            config=core.JaxGPConfig(optimisation_steps=3),seed=17)
    query=rng.uniform(size=(25,3))
    np.testing.assert_allclose(study.reference_means(model,query),model.predict_f(query)[0][:,0],rtol=1e-10,atol=1e-10)


def test_streamed_event_likelihood_and_empty_catalogue(study):
    rng=np.random.default_rng(12)
    rates=rng.uniform(.1,1,(2050,4));coeff=rng.uniform(.1,1,(7,4))
    for c in [coeff,np.empty((0,4))]:
        expected=study.base.log_marginal_events(rates@c.T,rates.sum(axis=1),.2)
        np.testing.assert_allclose(study.event_target(rates,c,.2),expected,rtol=1e-12)


def test_candidate_generation_never_calls_population_model(study,monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError('Candidate generation cannot evaluate cosmic integration')
    monkeypatch.setattr(study.core.FixedPopulationRates,'__call__',forbidden)
    initial,candidates,random=study.seed_design(17,20260910,4322)
    assert initial.shape==(226,3)
    assert candidates.shape==random.shape==(4096,3)
    assert np.all(candidates>=study.core.LOW[study.core.SHAPE])
    assert np.all(candidates<=(study.core.LOW+study.core.WIDTH)[study.core.SHAPE])
