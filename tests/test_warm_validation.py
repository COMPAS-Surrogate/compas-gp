"""Bounded validation design and changed-transform warm initialization."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import itertools
import numpy as np

spec=importlib.util.spec_from_file_location('warm_validation',Path(__file__).resolve().parents[1]/'docs/studies/amplitude_gp/warm_validation.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


def test_campaign_is_bounded_and_paired():
    jobs=[dict(name=f'count_n{n}_{mode}_c{case}_s17_{strategy}',seed=17,
        catalogue=f'{case}_{n}_{mode}',strategy=strategy)
        for case,n,mode,strategy in itertools.product((200,201,202),(50,100,500),('perfect','uncertain'),('bo','random'))]
    tasks=mod.build_tasks(jobs)
    assert len(tasks)==40 and sum(t['kind']=='saved_design' for t in tasks)==8
    fresh=[t['job'] for t in tasks if t['kind']=='trajectory']
    assert len({j['name'] for j in fresh})==32
    assert sum(j['seed']==29 for j in fresh)==8
    pairs={}
    for j in fresh:pairs.setdefault((j['catalogue'],j['seed']),set()).add(j['strategy'])
    assert len(pairs)==16 and all(v=={'bo','random'} for v in pairs.values())


def test_new_transform_metadata_preserves_previous_state():
    from cosmic_integration.lnl_surrogate.adaptive_robust_scalar import AdaptiveRobustScaler
    previous=dict(reference=np.array(0.),floor=np.array(-20.),scale=np.array(3.),noise=np.array(1e-5))
    values=np.array([-100.,-10.,2.])
    current=mod.target_template(previous,values,SimpleNamespace(AdaptiveRobustScaler=AdaptiveRobustScaler))
    scaler=AdaptiveRobustScaler(compression='sqrt',soft_clipping=False,lower_clip_value=-100.,max_scale=1e9)
    scaler.initialize_with_data(values)
    np.testing.assert_allclose(np.sqrt(np.maximum(current['reference']-np.maximum(values,current['floor']),0))/current['scale'],-scaler.transform(values))
    assert float(previous['floor'])==-20. and float(current['floor'])==-100.
    assert current['noise']==previous['noise']
