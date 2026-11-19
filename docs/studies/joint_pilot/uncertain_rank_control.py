"""One fixed 1000-catalogue direct-only follow-up to uncertain sigma coverage.

Seed and size fixed after the 100-catalogue result. Retain this result whatever
it shows. No GP fitting, no pooling with the initial ensemble, no seed search.
"""
import hashlib
import json
import logging
import time
from pathlib import Path
import numpy as np
from scipy.stats import qmc, binomtest
from scipy.sparse import load_npz
from direct_rank_control import posterior_ranks
from confirm_482 import HERE, REPO, configurations, BinnedCosmicIntegrator, FixedPopulationRates
from cosmic_integration.observation.binned_response import correlated_bin_response


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    start = time.perf_counter()
    root = HERE/'uncertain_direct_1000_results'
    root.mkdir(exist_ok=False)
    seed, n = 20261105, 1000
    manifest = {'seed': seed, 'catalogues': n, 'validation_seeds': [seed+1, seed+2],
                'response': 'sigma1 rho0.6 radius3 categorical saturated boundaries',
                'population': 'blocks0-7 matched generation/inference',
                'scope': 'Direct-only internal calibration, no GP',
                'source_sha256': {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in [Path(__file__), HERE/'direct_rank_control.py',
                                            HERE.parent/'population_stability/bootstrap_blocks.py',
                                            *sorted((REPO/'src/cosmic_integration').rglob('*.py'))]}}
    (root/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    rng = np.random.default_rng(seed)
    truths = rng.uniform([-.5, .1, .005, 4.2], [-.001, .6, .015, 5.2], (n, 4))
    ci = BinnedCosmicIntegrator.from_compas_fpath(str(REPO/'tests/large_test_data/h5out_32M_reduced.h5'))
    mass, z, _ = configurations()['mass10']
    fast = FixedPopulationRates(ci, mass, z)
    block_dir = HERE.parent/'population_stability/block_results'
    meta = json.loads((block_dir/'population_meta.json').read_text())
    ci.compas.nSystems = 256000000
    ci.compas.zamsZmin, ci.compas.zamsZmax = meta['zmin'], meta['zmax']
    fast.p_draw = 1/np.log(meta['zmax']/meta['zmin'])
    fast.operator = sum((load_npz(block_dir/f'block_{i:02d}.npz') for i in range(1, 8)),
                        load_npz(block_dir/'block_00.npz'))
    response = correlated_bin_response(fast.shape)
    # Poisson splitting implies measured counts have Poisson means R @ rates.
    # This is equivalent to, but implemented separately from, paired event errors.
    def measured(t):
        return response @ fast(*t).ravel()
    counts = np.array([rng.poisson(.1*measured(t)) for t in truths])
    np.savez(root/'catalogues.npz', truths=truths, counts=counts)
    occupied = np.any(counts > 0, axis=0)
    obs = counts[:, occupied]
    all_ranks = []
    for scramble in manifest['validation_seeds']:
        phi = np.array([-.5, .1, 4.2])+qmc.Sobol(3, scramble=True, seed=scramble).random_base2(14)*np.array([.499, .5, 1.])
        cache = []
        for t in phi:
            r = measured([t[0], t[1], .012, t[2]])
            cache.append(np.r_[r[occupied], r.sum()])
        cache = np.array(cache)
        ranks = []
        for i, (c, truth) in enumerate(zip(obs, truths)):
            mask = c > 0
            select = np.r_[np.flatnonzero(mask), len(c)]
            ranks.append(posterior_ranks(cache[:, select], c[mask], phi, truth))
            if (i+1) % 250 == 0:
                logging.info('Scramble %d: %d/1000 catalogues', scramble, i+1)
        all_ranks.append(ranks)
    ranks = np.array(all_ranks)
    np.savez(root/'ranks.npz', ranks=ranks)
    coverage, disagreements = {}, {}
    for level in [68, 90, 95]:
        lo = (1-level/100)/2
        flags = (ranks >= lo)&(ranks <= 1-lo)
        coverage[str(level)] = flags[0].sum(0).tolist()
        disagreements[str(level)] = (flags[0] != flags[1]).sum(0).tolist()
    result = {'catalogues': n, 'coverage': coverage,
              'coverage_exact95_intervals': {k: [list(binomtest(c, n).proportion_ci()) for c in v] for k, v in coverage.items()},
              'coverage_disagreements_between_scrambles': disagreements,
              'maximum_rank_difference': np.max(abs(ranks[0]-ranks[1]), axis=0).tolist(),
              'seconds': time.perf_counter()-start, 'scope': manifest['scope']}
    (root/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    logging.info('Completed: %s', result)


if __name__ == '__main__':
    main()
