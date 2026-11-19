"""The predictive figure must retain the joint GP amplitude distribution."""
import importlib.util
from pathlib import Path

import numpy as np


def test_joint_intensity_draw_preserves_shape_amplitude_correlation():
    path = Path(__file__).resolve().parents[1] / 'scripts/ozstar/gp4/manuscript_figures.py'
    spec = importlib.util.spec_from_file_location('gp4_figures', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rates = np.array([[1., 0.], [0., 2.]])
    # Only low amplitude at shape zero and high amplitude at shape one have mass.
    amplitudes = np.array([.006, .012, .006, .012])
    weights = np.array([0., -np.inf, -np.inf, 0.])
    draws = module.sample_joint_intensities(rates, amplitudes, weights, 3., 10000, 12)
    allowed = ((draws == [1.5, 0.]).all(axis=1)
               | (draws == [0., 6.]).all(axis=1))
    assert allowed.all()
    np.testing.assert_allclose(draws.mean(axis=0), [.75, 3.], rtol=.03)


def test_display_smoothing_preserves_mass_and_uniform_density():
    path = Path(__file__).resolve().parents[1] / 'scripts/ozstar/gp4/manuscript_figures.py'
    spec = importlib.util.spec_from_file_location('gp4_display', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    flat = np.ones((32, 32)) / 1024
    covariance = np.array([[60., 25.], [25., 40.]])
    np.testing.assert_allclose(module.smooth_display_density(flat, covariance), flat)
    edge = np.zeros((32, 32)); edge[0, 0] = 1.
    result = module.smooth_display_density(edge, covariance)
    assert result.shape == edge.shape
    assert (result >= 0).all()
    np.testing.assert_allclose(result.sum(), 1.)


def test_symmetric_gp_kl_ignores_normalisation_and_matches_discrete_answer():
    path = Path(__file__).resolve().parents[1] / 'scripts/ozstar/gp4/manuscript_figures.py'
    spec = importlib.util.spec_from_file_location('gp4_kl', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    p = np.array([.25, .75]); q = np.array([.6, .4])
    expected = .5 * np.sum((p-q) * np.log(p/q))
    np.testing.assert_allclose(module.symmetric_gp_kl(np.log(p)+123, np.log(q)-17), expected)
    np.testing.assert_allclose(module.symmetric_gp_kl(np.log(q), np.log(p)), expected)
    np.testing.assert_allclose(module.symmetric_gp_kl(np.log(p), np.log(p)), 0.)
