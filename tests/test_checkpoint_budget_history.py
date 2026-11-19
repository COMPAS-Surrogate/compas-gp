"""An isolated pass or unresolved reference is not a confirmed training budget."""
import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('budget_history',Path(__file__).resolve().parents[1]/'docs/studies/amplitude_gp/score_checkpoint_history.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_confirmation_requires_accurate_adjacent_checkpoints():
    rows=[dict(labels=414,outcome='accurate'),dict(labels=744,outcome='accurate')]
    assert module.first_confirmed(rows) is None
    rows.append(dict(labels=774,outcome='unresolved'))
    assert module.first_confirmed(rows) is None
    rows[-1]['outcome']='accurate'
    assert module.first_confirmed(rows)==744
    rows.extend([dict(labels=444,outcome='accurate'),dict(labels=1284,outcome='inaccurate')])
    assert module.first_confirmed(rows)==414


def test_fixed_design_refit_keeps_noise_and_points():
    from types import SimpleNamespace
    import numpy as np
    from cosmic_integration.lnl_surrogate import jax_active_learner as core
    from cosmic_integration.lnl_surrogate.adaptive_robust_scalar import AdaptiveRobustScaler
    from cosmic_integration.lnl_surrogate.gpry_rules import state_from_gpjax,predict_lnl
    unit=np.random.default_rng(31).random((18,4));values=-20*np.sum((unit-.5)**2,axis=1)
    scaler=AdaptiveRobustScaler(compression='sqrt',soft_clipping=False,lower_clip_value=values.min(),max_scale=1e9)
    scaler.initialize_with_data(values)
    fitted=core.fit_exact_gp(core.JaxTrainingData(unit,-scaler.transform(values)),np.array([np.zeros(4),np.ones(4)]),config=core.JaxGPConfig(optimisation_steps=5))
    state=state_from_gpjax(fitted,scaler,np.zeros(4),np.ones(4),18)
    api=SimpleNamespace(AdaptiveRobustScaler=AdaptiveRobustScaler,JaxTrainingData=core.JaxTrainingData)
    campaign=SimpleNamespace(study=SimpleNamespace(pilot=SimpleNamespace(base=SimpleNamespace(core=api))))
    candidate,info=module.refit_saved(campaign,state,state,unit,values,5)
    np.testing.assert_array_equal(candidate['x'],unit)
    np.testing.assert_allclose(candidate['noise'],state['noise'],rtol=1e-12)
    assert np.isfinite(predict_lnl(candidate,unit)).all()
    assert info['final_loss']<=info['initial_loss']


def test_summary_preserves_failure_and_checks_candidate_hash(tmp_path):
    import json
    import pytest
    folder=tmp_path/'retrieved'/'refit'/'example';folder.mkdir(parents=True)
    job=dict(name='example',catalogue='paired.npz',strategy='bo')
    (folder/'identity.json').write_text(json.dumps(dict(job=job,seed=17)))
    row=dict(labels=744,outcome='inaccurate',attempts=[dict(outcome='inaccurate')],
             maximum_directed_kl=.2,seconds=3,checkpoint_sha256='source',optimizer=None)
    history=dict(job=job,rows=[row])
    (folder/'history.json').write_text(json.dumps(history))
    assert module.summarize_retrieved(tmp_path)['rows'][0]['outcome']=='inaccurate'
    (folder/'candidate_744.npz').write_bytes(b'changed')
    row['optimizer']=dict(candidate_checkpoint_sha256='wrong')
    (folder/'history.json').write_text(json.dumps(history))
    with pytest.raises(ValueError,match='checksum'):module.summarize_retrieved(tmp_path)
