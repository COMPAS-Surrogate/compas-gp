"""Independent quadrature checks for conditional-amplitude reference summaries."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq


@pytest.mark.parametrize('count',[0,5])
def test_analytic_amplitude_reference_against_integration(monkeypatch,count):
    study=Path(__file__).resolve().parents[1]/'docs/studies/joint_pilot'
    monkeypatch.syspath_prepend(str(study))
    spec=importlib.util.spec_from_file_location('calibration_reference',study/'check_ensemble_quadrature.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rates=np.tile([30.,100.],(16,1)) if count else np.full((16,1),100.)
    counts=np.array([count]) if count else np.array([])
    phi=np.random.default_rng(3).uniform([-.5,.1,4.2],[-.001,.6,5.2],(16,3))
    result=module.direct_summary(rates,counts,phi,np.array([-.3,.2,.011,4.7]))
    beta=.1*100/.012
    def density(a):return (a/.01)**count*np.exp(-beta*(a-.005))
    norm=quad(density,.005,.015)[0]
    expected_mean=quad(lambda a:a*density(a),.005,.015)[0]/norm
    expected_variance=quad(lambda a:(a-expected_mean)**2*density(a),.005,.015)[0]/norm
    np.testing.assert_allclose(result['mean'][2],expected_mean,rtol=1e-10)
    np.testing.assert_allclose(result['sd'][2],np.sqrt(expected_variance),rtol=1e-10)
    for row,p in zip(result['quantiles'],[.025,.05,.16,.5,.84,.95,.975]):
        reference=brentq(lambda x:quad(density,.005,x)[0]/norm-p,.005,.015,xtol=1e-13)
        np.testing.assert_allclose(row[2],reference,atol=2e-12)

    # Surrogate shape weights must affect shape summaries, while the exact
    # conditional amplitude stays unchanged when every rate row is identical.
    log_target=np.arange(len(phi))*.1
    custom=module.direct_summary(rates,counts,phi,np.array([-.3,.2,.011,4.7]),log_target=log_target)
    weights=np.exp(log_target-log_target.max());weights/=weights.sum()
    np.testing.assert_allclose(np.array(custom['mean'])[[0,1,3]],weights@phi,rtol=1e-13)
    np.testing.assert_allclose(custom['mean'][2],expected_mean,rtol=1e-10)
    np.testing.assert_allclose(np.array(custom['quantiles'])[:,2],np.array(result['quantiles'])[:,2],atol=1e-12)

    rank_spec=importlib.util.spec_from_file_location('rank_control',study/'direct_rank_control.py')
    rank_module=importlib.util.module_from_spec(rank_spec);rank_spec.loader.exec_module(rank_module)
    ranks=rank_module.posterior_ranks(rates,counts,phi,np.array([-.3,.2,.011,4.7]))
    np.testing.assert_allclose(ranks,result['ranks'],atol=1e-12)
