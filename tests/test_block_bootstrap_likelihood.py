"""Verify the batched study likelihood against the scalar likelihood integral."""
import importlib.util
from pathlib import Path
import numpy as np
from cosmic_integration.observation.grid_likelihood import amplitude_marginal_log_likelihood


def test_compact_batched_amplitude_integral(monkeypatch):
    study=Path(__file__).resolve().parents[1]/'docs/studies/population_stability'
    monkeypatch.syspath_prepend(str(study))
    spec=importlib.util.spec_from_file_location('block_bootstrap_study',study/'bootstrap_blocks.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rates=np.array([[30.,20.,100.],[.1,.2,10.],[100.,200.,120000.]])
    counts=np.array([3.,2.])
    batched,beta=module.vector_marginal(rates,counts)
    for i,row in enumerate(rates):
        all_rates=np.r_[row[:-1],row[-1]-row[:-1].sum()][None,:]
        expected,expected_beta=amplitude_marginal_log_likelihood(all_rates,np.r_[counts,0][None,:],.1)
        np.testing.assert_allclose(batched[i],expected,atol=1e-9)
        np.testing.assert_allclose(beta[i],expected_beta,rtol=1e-14)


def test_zero_event_catalogue_retains_poisson_normalization(monkeypatch):
    """An empty catalogue still constrains total rate through exp(-expected N)."""
    study=Path(__file__).resolve().parents[1]/'docs/studies/population_stability'
    monkeypatch.syspath_prepend(str(study))
    spec=importlib.util.spec_from_file_location('empty_catalogue_study',study/'bootstrap_blocks.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rates=np.array([[1.],[1000.],[120000.]])
    actual,beta=module.vector_marginal(rates,np.array([]))
    expected=-beta*.005+np.log(-np.expm1(-beta*.01))-np.log(beta*.01)
    np.testing.assert_allclose(actual,expected,atol=1e-9)
