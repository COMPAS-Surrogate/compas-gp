"""Reference-free gates must reject observed errors and preserve offset invariance."""
import inspect
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
from cosmic_integration.lnl_surrogate import stopping as s


def test_prediction_checks_remove_only_constant_offsets():
    truth=np.linspace(-8,0,32)
    assert s.prediction_check(truth+150,truth,s.StoppingTolerances())['passed']
    wrong=truth.copy();wrong[-8:]+=1
    assert not s.prediction_check(wrong,truth,s.StoppingTolerances())['passed']
    assert not s.prediction_check(truth*np.nan,truth,s.StoppingTolerances())['passed']


def test_audit_rejects_a_missed_relevant_region():
    actual=np.r_[np.zeros(64),-30*np.ones(16)]
    pred=actual+70
    assert s.audit_check(pred,actual,64,0,s.StoppingTolerances())['passed']
    actual[-1]=0
    check=s.audit_check(pred,actual,64,0,s.StoppingTolerances())
    assert not check['passed']
    assert check['broad_max_relevant_error']==30


def test_audit_rejects_posterior_shape_error():
    actual=np.zeros(80);pred=np.r_[np.tile([-.5,.5],32),np.zeros(16)]
    assert not s.audit_check(pred,actual,64,0,s.StoppingTolerances())['passed']


def test_streak_requires_every_condition():
    assert s.next_streak(1,True,True,True)==2
    for flags in [(False,True,True),(True,False,True),(True,True,False)]:
        assert s.next_streak(1,*flags)==0


def test_kl_and_summary_ignore_offsets():
    rng=np.random.default_rng(11)
    x=rng.uniform(size=(1000,2));lw=-np.sum((x-.5)**2,axis=1)
    assert s.symmetric_kl(lw,lw+1000)==pytest.approx(0,abs=1e-12)
    a,b=[s.posterior_summary(x,w) for w in [lw,lw+1000]]
    assert s.summaries_agree(a,b,s.StoppingTolerances())[0]
    changed=s.posterior_summary(x,lw+20*x[:,0])
    assert not s.summaries_agree(a,changed,s.StoppingTolerances())[0]
    assert s.symmetric_kl(lw,lw+20*x[:,0])>.001


def test_rule_api_has_no_reference_inputs():
    for function in [s.prediction_check,s.audit_check,s.next_streak]:
        assert 'reference' not in inspect.signature(function).parameters


def test_normalization_rejects_invalid_support():
    with pytest.raises(ValueError):s.normalized_logweights(np.array([0.,np.nan]))
    with pytest.raises(ValueError):s.symmetric_kl(np.zeros(3),np.zeros(2))


def test_learning_finishes_before_reference_access(monkeypatch,tmp_path):
    folder=Path(__file__).resolve().parents[1]/'docs/studies/operational_stopping'
    monkeypatch.syspath_prepend(str(folder))
    spec=importlib.util.spec_from_file_location('blind_stop_test_engine',folder/'run_study.py')
    engine=importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules,spec.name,engine);spec.loader.exec_module(engine)
    monkeypatch.setitem(engine.CONFIG,'maximum_calls',48)
    monkeypatch.setattr(engine,'fit',lambda x,y:(None,None))
    monkeypatch.setattr(engine,'predict',lambda model,scaler,x:np.zeros(len(x)))
    evaluate=engine.Evaluations(lambda x:-np.sum((x-.5)**2,axis=1))
    decision=engine.operational_run(evaluate,42,tmp_path)
    assert evaluate.calls==48 and not decision['reference_accessed']
    assert (tmp_path/'decision.json').exists()
    assert not (tmp_path/'reference_score.json').exists()


def test_reference_scoring_requires_an_existing_decision(monkeypatch,tmp_path):
    folder=Path(__file__).resolve().parents[1]/'docs/studies/operational_stopping'
    monkeypatch.syspath_prepend(str(folder))
    spec=importlib.util.spec_from_file_location('reference_order_test_engine',folder/'run_study.py')
    engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)
    def forbidden(_):
        pytest.fail('Reference evaluator accessed before the stopping decision')
    with pytest.raises(FileNotFoundError):engine.reference_score(forbidden,tmp_path,42)
