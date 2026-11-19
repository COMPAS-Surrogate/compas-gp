"""Expected likelihood loss of finite COMPAS intensities relative to 512M.

This is a Poisson-process discrepancy at a fixed physical parameter vector,
including rates and the continuous measurement distribution. It is not an
estimate of bias relative to an infinite COMPAS population or a safety bound.
"""
from __future__ import annotations
import hashlib
import logging
from pathlib import Path
import time
import numpy as np
from scipy.sparse import load_npz
import population_event_count as sizes

BASE = sizes.base
HERE = sizes.HERE
OUT = HERE/'likelihood_loss_20260912'
COUNTS = np.array([10,30,50,100,200,300,500,750,1000,1500,2250,3000])


def poisson_loss_integrand(ratio: np.ndarray) -> np.ndarray:
    """Nonnegative r - 1 - log(r), stable when competing intensities agree."""
    ratio = np.asarray(ratio,dtype=float)
    if np.any(ratio < 0) or not np.isfinite(ratio).all():
        raise ValueError('Intensity ratios must be finite and nonnegative')
    delta = ratio-1
    # The series avoids cancellation for nearly equal populations.
    small = abs(delta) < 1e-4
    with np.errstate(divide='ignore'):
        value = delta-np.log(ratio)
    value[small] = delta[small]**2*(.5-delta[small]/3+delta[small]**2/4)
    return value


def main() -> None:
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    OUT.mkdir(exist_ok=True)
    loader = BASE.Experiment(OUT/f'population_{time.time_ns()}','population_likelihood_loss',False)
    paths = [HERE.parent/f'population_stability/block_results/block_{i:02d}.npz' for i in range(16)]
    blocks = [load_npz(p) for p in paths]
    models = {32*k:sizes.population_model(loader.fast,blocks,k) for k in [1,4,8,16]}
    full = models[512]
    active = np.flatnonzero(full.operator.getnnz(axis=1))
    mass,z,_ = BASE.core.configurations()['mass10']
    mi,zi = np.unravel_index(active,full.shape); ze=z*.1
    lower = np.column_stack([mass[mi],ze[zi]])
    upper = np.column_stack([mass[mi+1],ze[zi+1]])
    rates = np.stack([model(*BASE.core.TRUTH).ravel()[active] for model in models.values()])
    np.savez(OUT/'rates.npz',sizes=list(models),rates=rates,active=active,lower=lower,upper=upper,truth=BASE.core.TRUTH)
    records = []
    # Two independent integrations quantify numerical uncertainty, not the
    # scatter between COMPAS realizations or observed catalogues.
    for replicate in [0,1]:
        path = OUT/f'integrand_{replicate}.npy'
        if path.exists():
            values = np.load(path)
        else:
            rng = np.random.default_rng(2026091209+replicate)
            pieces = []
            for start in range(0,262144,1024):
                bins = rng.choice(len(active),1024,p=rates[-1]/rates[-1].sum())
                latent = rng.uniform(lower[bins],upper[bins])
                rho = 12*rng.random(1024)**(-1/3)
                scale = np.array([.03,.3])[None,:]*12/rho[:,None]
                y = np.log(latent)+scale*rng.normal(size=(1024,2))
                _,coeff,_ = BASE.measurement(dict(latent=latent,scales=scale,y=y),1.,lower,upper)
                intensities = coeff@rates.T
                ratios = intensities[:,:-1]/intensities[:,-1,None]
                pieces.append(poisson_loss_integrand(ratios))
            values = np.concatenate(pieces)
            np.save(path,values)
        assert values.shape == (262144,3) and np.isfinite(values).all()
        for j,size in enumerate([32,128,256]):
            records.append(dict(replicate=replicate,size=size,draws=len(values),loss_per_expected_event=float(values[:,j].mean()),
                standard_error=float(values[:,j].std(ddof=1)/np.sqrt(len(values))),rate_ratio=float(rates[j].sum()/rates[-1].sum())))
        logging.info('Integration %s: %s',replicate,[r['loss_per_expected_event'] for r in records if r['replicate']==replicate])
    curves = []
    for size in [32,128,256]:
        pair = [r for r in records if r['size']==size]
        means = np.array([r['loss_per_expected_event'] for r in pair])
        errors = np.array([r['standard_error'] for r in pair])
        resolved = bool(abs(means[0]-means[1]) <= 4*np.linalg.norm(errors) and np.linalg.norm(errors)/2/means.mean() < .025)
        curves.append(dict(size=size,expected=COUNTS.tolist(),loss=(COUNTS*means.mean()).tolist(),
            numerical_se=(COUNTS*np.linalg.norm(errors)/2).tolist(),numerically_resolved=resolved,
            loss_per_expected_event=float(means.mean())))
    BASE.write_json(OUT/'results.json',dict(integrations=records,curves=curves,
        scope='Expected log-likelihood loss at the fiducial injection, nominal uncertain measurements; finite nested 512M comparison population',
        scaling='Exact linear exposure scaling; the 12 marked event counts are not independent simulations',
        numerical_uncertainty='Monte Carlo error of the expectation, not COMPAS variability or catalogue scatter',
        source_sha256={str(p.relative_to(sizes.REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),*paths]},
        state='complete' if all(r['numerically_resolved'] for r in curves) else 'unresolved'))


if __name__ == '__main__':
    main()
