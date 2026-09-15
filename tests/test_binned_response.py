import numpy as np

from cosmic_integration.observation.binned_response import correlated_bin_response, measure_counts
from cosmic_integration.observation.grid_likelihood import counts_log_likelihood


def test_response_conserves_detection_rate_and_identity():
    shape = (4, 3)
    rate = np.arange(12.) + 1
    response = correlated_bin_response(shape)
    np.testing.assert_allclose(np.asarray(response.sum(axis=0)), 1.)
    np.testing.assert_allclose((response @ rate).sum(), rate.sum())
    np.testing.assert_allclose(correlated_bin_response(shape, sigma=0).toarray(), np.eye(12))
    counts = np.arange(12).reshape(shape)
    np.testing.assert_array_equal(measure_counts(counts, correlated_bin_response(shape, sigma=0),
                                               np.random.default_rng(11)), counts)


def test_measured_poisson_likelihood_matches_explicit_event_integrals():
    response = correlated_bin_response((3, 2)).toarray()
    rate = np.array([1., 3., 2., 4., 8., 5.])
    counts = np.array([[2, 0], [1, 1], [0, 3]])
    events = np.repeat(np.arange(6), counts.ravel())
    event_integrals = np.array([sum(response[y, k]*rate[k] for k in range(6)) for y in events])
    expected = -.2*rate.sum() + np.log(.2*event_integrals).sum()
    np.testing.assert_allclose(counts_log_likelihood((response@rate).reshape(3, 2), counts, .2), expected)


def test_measurement_simulation_matches_response_including_boundaries():
    response = correlated_bin_response((7, 7))
    counts = np.zeros((7, 7), dtype=int)
    counts[0, 0] = 100000
    counts[3, 3] = 100000
    measured = measure_counts(counts, response, np.random.default_rng(77)).ravel()
    expected = response @ counts.ravel()
    assert measured.sum() == counts.sum()
    assert np.max(abs(measured-expected)/np.sqrt(expected+1)) < 5
    center = response[:, 24].toarray().ravel().reshape(7, 7)
    i, j = np.meshgrid(np.arange(7)-3, np.arange(7)-3, indexing='ij')
    assert np.sum(center*i*j) > .5  # Correlation survives the discrete response.


def test_posterior_predictive_includes_amplitude_and_poisson_variation(monkeypatch):
    from pathlib import Path
    from scipy.integrate import quad
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'docs/studies/joint_pilot'))
    from uncertain_measurements import posterior_predictive
    # Identical shape rows isolate the known conditional amplitude distribution.
    rates = np.tile([30., 70.], (8, 1))
    counts = np.array([2, 3])
    result = posterior_predictive(rates, counts, np.zeros(8), np.random.default_rng(53), (1, 2), draws=30000)
    beta = .1*100/.012
    def density(a):
        return a**5*np.exp(-beta*a)
    norm = quad(density, .005, .015, epsabs=1e-25)[0]
    mean_a = quad(lambda a: a*density(a), .005, .015, epsabs=1e-25)[0]/norm
    second_a = quad(lambda a: a*a*density(a), .005, .015, epsabs=1e-25)[0]/norm
    expected_mean = beta*mean_a
    expected_variance = expected_mean + beta**2*(second_a-mean_a**2)
    simulated = np.array(result['replicate_statistics'])[:, 0]
    assert abs(simulated.mean()-expected_mean) < 5*np.sqrt(expected_variance/len(simulated))
    np.testing.assert_allclose(simulated.var(), expected_variance, rtol=.04)
