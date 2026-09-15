"""Four-input campaign labels and compact restart checkpoints."""
import importlib.util
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture
def campaign(monkeypatch):
    path=Path(__file__).resolve().parents[1]/'docs/studies/amplitude_gp/paper_campaign.py'
    monkeypatch.syspath_prepend(str(path.parent))
    spec=importlib.util.spec_from_file_location('paper_campaign_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_campaign_retains_explicit_amplitude_and_exact_event_likelihood(campaign):
    def engine(p):
        return p[2]/.012*np.array([2+np.exp(p[0]),1+p[1],p[3]])
    cat=dict(coeff=np.array([[1.,0.,0.],[0.,.2,.8]]),duration=.7)
    unit=np.array([[.2,.3,.1,.4],[.2,.3,.8,.4],[.6,.2,.4,.9]])
    actual=campaign.evaluate(engine,cat,unit)
    expected=[]
    for p in campaign.LOW+campaign.WIDTH*unit:
        rates=engine(p)
        expected.append(np.log(.7*(cat['coeff']@rates)).sum()-.7*rates.sum())
    np.testing.assert_allclose(actual,expected,atol=1e-12)
    assert actual[0]!=actual[1]
    empty=dict(coeff=np.empty((0,3)),duration=.7)
    np.testing.assert_allclose(campaign.evaluate(engine,empty,unit),
        [-.7*engine(p).sum() for p in campaign.LOW+campaign.WIDTH*unit])


def test_compact_checkpoint_restores_prediction_and_rng(campaign,tmp_path):
    unit=np.random.default_rng(7).uniform(size=(8,4));values=-np.sum((unit-.4)**2,axis=1)
    template=dict(x=jnp.zeros((16,4)),mask=jnp.zeros(16),lengthscale=jnp.full(4,.4),
        amplitude=jnp.array(1.),constant=jnp.array(2.),noise=jnp.array(1e-5),
        jitter=jnp.array(1e-6),reference=jnp.array(0.),scale=jnp.array(1.),floor=jnp.array(-100.))
    state=campaign.study.condition_on_data(template,unit,values)
    rng=np.random.default_rng(81);rng.random(13)
    path=tmp_path/'checkpoint_8.npz';campaign.save(path,state,unit,values,rng)
    restored,x,y,recovered=campaign.restore(path,32)
    np.testing.assert_array_equal(x,unit);np.testing.assert_array_equal(y,values)
    np.testing.assert_array_equal(rng.random(8),recovered.random(8))
    query=np.random.default_rng(3).random((7,4))
    np.testing.assert_allclose(campaign.study.pilot.base.predict(state,query),
        campaign.study.pilot.base.predict(restored,query),atol=1e-9)


def test_result_summary_retains_failures_and_rejects_changed_checkpoint(tmp_path):
    import hashlib
    import json
    path=Path(__file__).resolve().parents[1]/'docs/studies/amplitude_gp/summarize_paper.py'
    spec=importlib.util.spec_from_file_location('paper_results_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for strategy,outcome,kl in [('bo','accurate',.002),('random','inaccurate',.02)]:
        folder=tmp_path/'retrieved/runs'/strategy;folder.mkdir(parents=True)
        checkpoint=folder/'checkpoint_2994.npz';checkpoint.write_bytes(b'frozen checkpoint')
        (folder/'complete.json').write_text(json.dumps(dict(labels=2994,
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())))
        score=dict(job=dict(name=strategy,catalogue='shared',strategy=strategy,group='count_history'),
            labels=2994,outcome=outcome,attempts=[dict(outcome=outcome,
            scores=[dict(accuracy_pass=outcome=='accurate',directed_kl=[kl,kl/2])],
            kl_precision=True,direct_precision=dict(passed=True),gp_precision=dict(passed=True))])
        (folder/'score.json').write_text(json.dumps(score))
    result=module.summarize(tmp_path)
    assert result['retrieved_scored']==2 and result['total_planned']==632
    assert result['outcomes']==dict(accurate=1,inaccurate=1)
    checkpoint.write_bytes(b'changed')
    with pytest.raises(ValueError,match='Checkpoint mismatch'):
        module.summarize(tmp_path)
