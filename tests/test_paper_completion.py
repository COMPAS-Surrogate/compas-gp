"""Scientific invariants of completion diagnostics and fresh confirmation design."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.sparse import csr_matrix
import pytest

ROOT=Path(__file__).resolve().parents[1]
def load(path,name,monkeypatch):
    monkeypatch.syspath_prepend(str(path.parent))
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def test_repeated_population_blocks_preserve_rate_normalization(monkeypatch):
    m=load(ROOT/'docs/studies/paper_completion/population_event_count.py','size_test',monkeypatch)
    original=SimpleNamespace(ci=SimpleNamespace(compas=SimpleNamespace(nSystems=256000000)),operator=csr_matrix([[5.,3.]]))
    blocks=[csr_matrix([[5.,3.]]) for _ in range(4)]
    one=m.population_model(original,blocks,1);four=m.population_model(original,blocks,4)
    np.testing.assert_allclose(one.operator.toarray()/one.ci.compas.nSystems,four.operator.toarray()/four.ci.compas.nSystems)
    assert original.ci.compas.nSystems==256000000
    four.ci.compas.nSystems=1
    assert one.ci.compas.nSystems==32000000

def test_completion_fraction_retains_censoring(monkeypatch):
    m=load(ROOT/'docs/studies/paper_completion/plot_paper.py','plots_test',monkeypatch)
    rows=[dict(case=0,state='reference_verified_convergence',points=100),dict(case=1,state='resource_limit',points=400),dict(case=2,state='reference_numerical_failure',points=0)]
    np.testing.assert_allclose(m.fraction_by_budget(rows,np.array([50,100,1000]),3),[0,1/3,1/3])
    with pytest.raises(ValueError):m.fraction_by_budget(rows[:2],np.array([100]),3)
    with pytest.raises(ValueError):m.fraction_by_budget([rows[0]]*3,np.array([100]),3)

def test_confirmation_catalogues_are_fresh_and_measurements_paired(monkeypatch):
    m=load(ROOT/'docs/studies/operational_stopping/gpry_compas/run_fresh_confirmation.py','fresh_test',monkeypatch)
    groups=m.groups();assert len(groups)==len(set(groups))==48
    assert {g[3] for g in groups}.isdisjoint({0,1,91})
    class Fake:
        active=np.arange(2)
        lower=np.array([[10.,.1],[20.,.2]])
        upper=lower+np.array([5.,.1])
        def fast(self,*args):return np.array([1.,2.])
    loader=Fake()
    new=m.train.catalogue(loader,0,100,100);same=m.train.catalogue(loader,0,100,100);old=m.train.catalogue(loader,0,100,0)
    np.testing.assert_array_equal(new['latent'],same['latent'])
    assert not np.array_equal(new['latent'],old['latent'])
    for t,n,_,c,_ in groups:
        assert len([g for g in groups if (g[0],g[1],g[3])==(t,n,c)])==4

def test_endpoint_scoring_accepts_exact_target_and_rejects_shape_bias(monkeypatch):
    folder=ROOT/'docs/studies/operational_stopping/gpry_compas'
    m=load(folder/'score_scheduled_audit.py','endpoint_scoring_test',monkeypatch)
    rng=np.random.default_rng(411)
    phi=m.base.LOW+m.base.WIDTH*rng.uniform(size=(256,3))
    rates=np.tile([3.,4.],(256,1));refs=[(phi,rates,np.zeros(256))]*2
    cat={'coeff':np.eye(2),'duration':.1}
    direct=m.score.direct_checks(refs,cat,m.base.core.TRUTH)
    monkeypatch.setattr(m.base,'load_checkpoint',lambda path: ({},None,None))
    value=m.base.source_target.event_target(rates,cat['coeff'],.1)
    monkeypatch.setattr(m.base,'predict',lambda state,unit:value)
    meta=dict(name='synthetic',truth_index=0,expected_events=2,mode='perfect',case=100,strategy='bo',points=58)
    result=m.score_checkpoint(Path('unused'),refs,cat,m.base.core.TRUTH,direct,meta)
    assert result['accuracy_pass'] and result['four_checked']
    monkeypatch.setattr(m.base,'predict',lambda state,unit:value+3*unit[:,0])
    assert not m.score_checkpoint(Path('unused'),refs,cat,m.base.core.TRUTH,direct,meta)['accuracy_pass']


def test_presentation_quantiles_keep_censored_catalogues(monkeypatch):
    m=load(ROOT/'docs/studies/paper_completion/presentation_figures.py','presentation_test',monkeypatch)
    rows=[dict(case=i,state='reference_verified_convergence',points=i+1) for i in range(100)]
    for r in rows[95:]:r['state']='resource_limit'
    # Retaining five censored cases gives ranks 10, 50 and 90, not ranks
    # conditional on the 95 successful cases.
    np.testing.assert_array_equal(m.completion_quantiles(rows),[10,50,90])
    for r in rows[89:]:r['state']='resource_limit'
    with pytest.raises(ValueError,match='not resolved'):m.completion_quantiles(rows)


def test_corner_amplitude_draws_match_independent_truncated_gamma_mean(monkeypatch):
    from scipy.stats import gamma
    m=load(ROOT/'docs/studies/paper_completion/presentation_figures.py','presentation_draw_test',monkeypatch)
    phi=np.array([[-.3,.35,4.7]])
    beta=np.array([5000.]); events=49
    samples=m.draw_posterior(phi,beta,np.array([0.]),events)
    k=events+1
    probability=lambda shape:gamma.cdf(.015,shape,scale=1/beta[0])-gamma.cdf(.005,shape,scale=1/beta[0])
    mean=k/beta[0]*probability(k+1)/probability(k)
    assert abs(samples[:,2].mean()-mean)<2e-5
    np.testing.assert_allclose(samples[:,[0,1,3]],np.broadcast_to(phi,(len(samples),3)))


def test_display_audit_score_keeps_all_audit_requirements(monkeypatch):
    m=load(ROOT/'docs/studies/paper_completion/presentation_figures.py','audit_display_test',monkeypatch)
    good=dict(maximum_directed_kl=.001,bootstrap_95_kl=.002,gp_ess=80,direct_ess=64)
    assert m.audit_ratio(good)==.5
    for key,value in [('maximum_directed_kl',.006),('bootstrap_95_kl',.010),('gp_ess',16),('direct_ess',16)]:
        assert m.audit_ratio({**good,key:value})==2


def test_corner_smoothing_preserves_probability_at_prior_boundary(monkeypatch):
    m=load(ROOT/'docs/studies/paper_completion/presentation_figures.py','corner_smoothing_test',monkeypatch)
    # A correlated cloud at a prior edge must remain nonnegative and keep its
    # mass inside the displayed prior after covariance-oriented smoothing.
    rng=np.random.default_rng(17)
    x=np.abs(rng.normal(0,2,2000));xy=np.column_stack([x,.8*x+rng.normal(0,.2,len(x))])
    h=np.zeros((32,32));h[0,0]=.8;h[1,2]=.2
    smoothed=m.smooth_density(h,xy)
    assert smoothed.shape==h.shape and np.all(smoothed>=0)
    np.testing.assert_allclose(smoothed.sum(),1,atol=1e-12)
