"""Construct actual 2000-label GP references for the four displayed catalogues.

Extra training coordinates depend only on the saved 658-label GP and a fresh
random seed. The numerical validation grids are opened only after fitting.
Original BO stopping decisions and acquisition trajectories remain immutable.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import logging
from pathlib import Path
import time
import jax.numpy as jnp
import numpy as np
from scipy.special import logsumexp
import presentation_figures as figures

BASE = figures.BASE
OUT = figures.HERE/'dense_reference_20260912'


def augmented_design(state: dict, unit: np.ndarray, lnl: np.ndarray,
                     seed: int, count: int = 2000) -> np.ndarray:
    """Retain every original label and add a fresh GP-guided, full-prior design."""
    if len(unit) >= count:
        raise ValueError('The reference must add training coordinates')
    pool, _ = BASE.gp_designs(state, unit, lnl, seed)[0]
    rng = np.random.default_rng(seed)
    extra = pool[rng.choice(len(pool), count-len(unit), replace=False)]
    result = np.vstack([unit, extra])
    if len(np.unique(result, axis=0)) != count:
        raise ValueError('Reference design contains duplicate training points')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=['n100_perfect','n100_uncertain','n1000_perfect','n1000_uncertain'])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    OUT.mkdir(exist_ok=True)
    frozen = (figures.FRESH/'decisions.json').read_bytes()
    loader = None
    for n in [100, 1000]:
        for mode in ['perfect', 'uncertain']:
            key = f'n{n}_{mode}'
            if args.case and key != args.case:
                continue
            folder = OUT/key
            folder.mkdir(exist_ok=True)
            name = f't0_n{n}_{mode}_c100_bo'
            source = figures.FRESH/'training'/name/'checkpoint_658.npz'
            catpath = source.parent/'catalogue.npz'
            cat = dict(np.load(catpath))
            path = folder/'checkpoint_2000.npz'
            tick = time.perf_counter()
            if not path.exists():
                state, oldunit, oldlnl = BASE.load_checkpoint(source)
                seed = 2026091201+n+int(mode=='uncertain')
                unit = augmented_design(state, oldunit, oldlnl, seed)
                if loader is None:
                    loader = BASE.data_source.Experiment(OUT/f'population_{time.time_ns()}', 'dense_gp_reference', False)
                phi = BASE.LOW+BASE.WIDTH*unit[len(oldunit):]
                rates = np.array([loader.fast(p[0],p[1],.012,p[2]).ravel()[loader.active] for p in phi])
                extra = BASE.source_target.event_target(rates,cat['coeff'],float(cat['duration']))
                lnl = np.r_[oldlnl, extra]
                template = {**state, 'x':jnp.zeros((2000,3)), 'mask':jnp.zeros(2000)}
                state = BASE.condition_on_data(template,unit,lnl)
                assert np.isfinite(BASE.predict(state,unit)).all()
                sha = BASE.save_checkpoint(folder,state,unit,lnl)
                BASE.write_json(folder/'training.json',dict(points=2000,inherited_labels=len(oldunit),new_labels=len(extra),seed=seed,
                    construction='658 original labels plus 1342 fresh draws from the full-support GP-guided proposal; no validation points used',
                    hyperparameters='Frozen at the saved 658-label fit; condition on all 2000 labels',
                    prior='Original full prior; no bounds changed; all tail training points retained',
                    checkpoint_sha256=sha,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    catalogue_sha256=hashlib.sha256(catpath.read_bytes()).hexdigest(),
                    decision_sha256=hashlib.sha256(frozen).hexdigest(),seconds=time.perf_counter()-tick))
                logging.info('%s trained on 2000 labels',key)
            # The independent numerical target is accessed only for validation.
            hashes = {}
            refs = figures.validation.references(figures.ROOT/'extension_v2',0,n,hashes)
            checked = figures.validation.score.direct_checks(refs,cat,cat['truth'])
            meta = dict(name=name,truth_index=0,expected_events=n,mode=mode,case=100,strategy='dense_reference',points=2000)
            score = figures.validation.score_checkpoint(path,refs,cat,cat['truth'],checked,meta)
            score['reference_accepted'] = bool(score['accuracy_pass'] and score['maximum_directed_kl'] < .001)
            score['validation_source_sha256'] = hashes
            BASE.write_json(folder/'validation.json',score)
            logging.info('%s internal KL %.6g; reference accepted %s',key,score['maximum_directed_kl'],score['reference_accepted'])
            if not score['reference_accepted']:
                continue
            state,_,_ = BASE.load_checkpoint(path)
            weights = [BASE.predict(state,(p-BASE.LOW)/BASE.WIDTH)-q for p,_,q in refs]
            phi = np.concatenate([p for p,_,_ in refs])
            beta = np.concatenate([r.sum(1)*float(cat['duration'])/.012 for _,r,_ in refs])
            w = np.concatenate([v-logsumexp(v) for v in weights])
            samples = figures.draw_posterior(phi,beta,w,len(cat['coeff']))
            target = score['four_summaries']['gp'][0]
            assert np.max(abs(samples.mean(0)-target['mean'])/target['sd']) < .05
            np.savez_compressed(folder/'samples.npz',reference=samples)
            BASE.write_json(folder/'status.json',dict(state='complete',reference_accepted=True,points=2000))
    assert frozen == (figures.FRESH/'decisions.json').read_bytes()


if __name__ == '__main__':
    main()
