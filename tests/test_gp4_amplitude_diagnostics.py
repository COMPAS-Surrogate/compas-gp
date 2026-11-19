"""Analytic and distorted amplitude controls for the post-training diagnostic."""
import importlib.util
from pathlib import Path
import sys
import numpy as np

folder=Path(__file__).resolve().parents[1]/'scripts/ozstar/gp4'
spec=importlib.util.spec_from_file_location('campaign',folder/'campaign.py')
workflow=importlib.util.module_from_spec(spec);spec.loader.exec_module(workflow)
spec=importlib.util.spec_from_file_location('gp4_diagnostics',folder/'diagnostics.py')
mod=importlib.util.module_from_spec(spec)
old=sys.modules.get('campaign');sys.modules['campaign']=workflow
try:spec.loader.exec_module(mod)
finally:
    if old is None:sys.modules.pop('campaign',None)
    else:sys.modules['campaign']=old


def test_gamma_known_answer_and_wrong_width_control():
    a=np.linspace(.005,.015,8193);n=1000;beta=100000.
    exact=n*np.log(a)-beta*a
    good,_=mod.conditional_metrics(a,n,beta,exact+12345.)
    assert abs(good['gp']['mean']-(n+1)/beta)<1e-10
    assert abs(good['gp']['fractional_sd']-1/np.sqrt(n+1))<1e-9
    assert max(abs(x) for x in good['directed_kl'])<1e-10
    bad,_=mod.conditional_metrics(a,n,beta,2*exact)
    assert bad['width68_fraction_error']>.25
    assert max(bad['directed_kl'])>.01


def test_prior_truncation_is_not_reported_as_count_scaling_failure():
    a=np.linspace(.005,.015,8193);n=10;beta=1000.
    row,_=mod.conditional_metrics(a,n,beta,n*np.log(a)-beta*a)
    assert row['analytic_cdf_max_error']<1e-5
    assert row['direct']['fractional_sd']<row['unbounded_fractional_sd']
