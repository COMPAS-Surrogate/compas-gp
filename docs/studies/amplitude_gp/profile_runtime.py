"""Bounded runtime benchmark; never changes a campaign or its training states.

The incremental inverse update is experimental: parity is measured against full
conditioning, and it is not used by the paper runner. Fit timings with fewer
than 250 steps are microbenchmarks, not measured production-run speedups.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from cosmic_integration.lnl_surrogate.jax_active_learner import FittedJaxGP
from cosmic_integration.lnl_surrogate.gpry_rules import (
    matern52, condition_on_data, predict_lnl, prepare_fantasies,
    acquisition_values, latent_prediction,
)


@jax.jit
def _append(state: dict, points: jax.Array, values: jax.Array) -> dict:
    """Fixed-shape Schur update with frozen kernel and target transformation."""
    indices = jnp.sum(state['mask']).astype(jnp.int32)+jnp.arange(len(points))
    cross = matern52(state['x'], points, state['lengthscale'], state['amplitude'])*state['mask'][:, None]
    beta = state['inverse']@cross
    schur = matern52(points, points, state['lengthscale'], state['amplitude'])-cross.T@beta
    schur += state['noise']*jnp.eye(len(points))
    chol = jnp.linalg.cholesky(schur)
    precision = jsp.linalg.cho_solve((chol, True), jnp.eye(len(points)))
    target = jnp.sqrt(jnp.maximum(state['reference']-jnp.maximum(values, state['floor']), 0))/state['scale']
    gamma = precision@(target-state['constant']-cross.T@state['alpha'])
    inverse = state['inverse'].at[indices, :].set(0).at[:, indices].set(0)
    inverse = inverse+beta@precision@beta.T
    cross_inverse = -beta@precision
    inverse = inverse.at[:, indices].set(cross_inverse).at[indices, :].set(cross_inverse.T)
    inverse = inverse.at[indices[:, None], indices[None, :]].set(precision)
    alpha = (state['alpha']-beta@gamma).at[indices].set(gamma)
    return {**state, 'x': state['x'].at[indices].set(points),
            'mask': state['mask'].at[indices].set(1), 'inverse': inverse, 'alpha': alpha}


def append_frozen(state: dict, points: np.ndarray, values: np.ndarray) -> dict:
    """Experimental append only; callers must rebuild whenever hyperparameters change."""
    points, values, mask = np.asarray(points), np.asarray(values), np.asarray(state['mask'])
    n = int(mask.sum())
    if (points.ndim != 2 or points.shape[1] != state['x'].shape[1]
            or values.shape != (len(points),) or not len(points)
            or n+len(points)>len(mask) or not np.array_equal(mask, np.arange(len(mask))<n)
            or not np.isfinite(points).all() or not np.isfinite(values).all()):
        raise ValueError('Finite matching points must fit a contiguous active prefix')
    return _append(state, jnp.asarray(points), jnp.asarray(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-root', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fit-steps', type=int, default=20)
    parser.add_argument('--warm-refit', action='store_true', help='Compare bounded warm starts at the final checkpoint')
    parser.add_argument('--sizes', type=int, nargs='+', default=[504, 1494, 2994])
    args = parser.parse_args()
    if args.fit_steps < 1: parser.error('fit-steps must be positive')
    sys.path.insert(0, str(args.code_root/'docs/studies/amplitude_gp'))
    import paper_campaign as campaign
    core = campaign.study.pilot.base.core
    if args.warm_refit:
        benchmark_refit(campaign, args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {'host':os.uname().nodename, 'jax':jax.__version__,
                'devices':str(jax.devices()), 'fit_steps':args.fit_steps,
                'capacity':3008, 'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'job_id':os.environ.get('SLURM_JOB_ID'), 'timings':[], 'checks':[]}
    def persist():
        (args.output/'results.json').write_text(json.dumps(metadata, indent=2)+'\n')
    def timed(n, operation, fn):
        start = time.perf_counter()
        result = fn()
        jax.block_until_ready(result) if not hasattr(result, 'posterior') else jax.block_until_ready(result.posterior)
        seconds = time.perf_counter()-start
        row = {'labels':n, 'operation':operation, 'seconds':seconds}
        metadata['timings'].append(row); persist(); print(json.dumps(row), flush=True)
        return result
    folder = args.campaign/'runs/count_n100_perfect_c200_s17_bo'
    engine = campaign.RateEngine(args.campaign/'inputs')
    cat = dict(np.load(args.campaign/'inputs/catalogues/count_n100_perfect_c200.npz'))
    for n in args.sizes:
        path = folder/f'checkpoint_{n}.npz'
        with np.load(path) as f:
            unit, values = f['unit'], f['lnl']
            params = {k[6:]:jnp.asarray(f[k]) for k in f.files if k.startswith('param_')}
        template = {**params, 'x':jnp.zeros((3008, 4)), 'mask':jnp.zeros(3008)}
        # Shape stays fixed, so only the first size incurs conditioning compilation.
        state = timed(n, 'condition_padded_first', lambda:condition_on_data(template, unit, values))
        timed(n, 'condition_padded_repeat', lambda:condition_on_data(template, unit, values))
        query = np.random.default_rng(7).random((64, 4))
        timed(n, 'direct_labels_3_first', lambda:campaign.evaluate(engine, cat, query[:3]))
        timed(n, 'direct_labels_3_repeat', lambda:campaign.evaluate(engine, cat, query[:3]))
        timed(n, 'acquisition_first', lambda:campaign.study.acquire_small_batch(state, unit, values, np.random.default_rng(17))[0])
        timed(n, 'acquisition_repeat', lambda:campaign.study.acquire_small_batch(state, unit, values, np.random.default_rng(17))[0])
        # Remove the final 3 labels, append them with the same frozen parameters.
        previous = condition_on_data(template, unit[:-3], values[:-3]); jax.block_until_ready(previous)
        appended = timed(n, 'append_3_first', lambda:append_frozen(previous, unit[-3:], values[-3:]))
        timed(n, 'append_3_repeat', lambda:append_frozen(previous, unit[-3:], values[-3:]))
        direct_prediction, updated_prediction = np.asarray(predict_lnl(state, query)), np.asarray(predict_lnl(appended, query))
        fantasies = prepare_fantasies(state, jnp.zeros((3,4)), jnp.zeros(3))
        updated_fantasies = prepare_fantasies(appended, jnp.zeros((3,4)), jnp.zeros(3))
        acq = np.asarray(acquisition_values(state, query, fantasies, 4**(-.85)))
        acq_updated = np.asarray(acquisition_values(appended, query, updated_fantasies, 4**(-.85)))
        metadata['checks'].append({'labels':n, 'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'prediction_max_abs':float(np.max(abs(direct_prediction-updated_prediction))),
            'acquisition_max_abs':float(np.max(abs(acq-acq_updated))),
            'prediction_parity':bool(np.allclose(direct_prediction, updated_prediction, rtol=1e-8, atol=1e-5)),
            'acquisition_parity':bool(np.allclose(acq, acq_updated, rtol=1e-8, atol=1e-5))})
        persist()
        scaler = core.AdaptiveRobustScaler(compression='sqrt', soft_clipping=False,
            lower_clip_value=float(values.min()), max_scale=1e9)
        scaler.initialize_with_data(values)
        data = core.JaxTrainingData(unit, -scaler.transform(values))
        def fit():
            return core.fit_exact_gp(data, np.array([np.zeros(4), np.ones(4)]),
                config=core.JaxGPConfig(optimisation_steps=args.fit_steps), seed=17)
        timed(n, 'fit_first', fit)
        timed(n, 'fit_repeat', fit)
    metadata['complete'] = True; persist()


def benchmark_refit(campaign, args) -> None:
    """Diagnostic comparison only: optimizer changes are not campaign replacements."""
    import equinox as eqx
    import gpjax as gpx
    import optax
    import paramax
    from jax.flatten_util import ravel_pytree
    from scipy.optimize import minimize
    from cosmic_integration.lnl_surrogate.gpry_rules import state_from_gpjax
    core = campaign.study.pilot.base.core
    args.output.mkdir(parents=True, exist_ok=False)
    folder = args.campaign/'runs/count_n100_perfect_c200_s17_bo'
    current, unit, values, _ = campaign.restore(folder/'checkpoint_2994.npz', 3008)
    with np.load(folder/'checkpoint_2964.npz') as f:
        previous = {k[6:]:np.asarray(f[k]) for k in f.files if k.startswith('param_')}
    # Use the exact final-checkpoint transform in every comparison.
    scaler = core.AdaptiveRobustScaler(compression='sqrt', soft_clipping=False,
        lower_clip_value=float(values.min()), max_scale=1e9)
    scaler.initialize_with_data(values)
    data = gpx.Dataset(X=jnp.asarray(unit), y=jnp.asarray(-scaler.transform(values))[:,None])
    expected_target = jnp.sqrt(jnp.maximum(current['reference']-jnp.maximum(values,current['floor']),0))/current['scale']
    np.testing.assert_allclose(data.y[:,0], expected_target, rtol=1e-12, atol=1e-12)
    def model_from(params):
        ratio = float(params['scale'])/float(current['scale'])
        kernel = gpx.kernels.Matern52(lengthscale=jnp.asarray(params['lengthscale']),
            variance=jnp.asarray(params['amplitude'])*ratio**2)
        prior = gpx.gps.Prior(kernel=kernel,
            mean_function=gpx.mean_functions.Constant(constant=jnp.asarray(params['constant'])*ratio))
        likelihood = gpx.likelihoods.Gaussian(num_datapoints=len(unit),
            obs_stddev=jnp.sqrt(current['noise']-current['jitter']))
        model = prior*likelihood
        assert float(model.jitter)==float(current['jitter'])
        return eqx.tree_at(lambda p:p.likelihood.obs_stddev, model,
            replace_fn=lambda p:paramax.NonTrainable(p.unwrap()))
    objective = lambda model, train: -gpx.objectives.conjugate_mll(model, train)
    loss = lambda model: float(objective(paramax.unwrap(model), data))
    warm = model_from(previous)
    baseline = model_from(current)
    output = dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        job_id=os.environ.get('SLURM_JOB_ID'), host=os.uname().nodename,
        labels=len(unit), previous_labels=2964, baseline_loss=loss(baseline),
        warm_initial_loss=loss(warm), transform_reference_change=float(current['reference']-previous['reference']),
        scope='One saved catalogue. Optimizer comparisons, not posterior validation or a production speedup.', runs=[])
    def persist(): (args.output/'results.json').write_text(json.dumps(output, indent=2)+'\n')
    persist()
    rng = np.random.default_rng(20260916)
    query = np.vstack([rng.random((128,4)), *[np.clip(unit[values.argmax()]+rng.normal(size=(128,4))*width,0,1) for width in (.01,.03,.1)]])
    engine = campaign.RateEngine(args.campaign/'inputs')
    cat = dict(np.load(args.campaign/'inputs/catalogues/count_n100_perfect_c200.npz'))
    exact = campaign.evaluate(engine, cat, query)
    saved = np.asarray(predict_lnl(current,query))
    predictions = dict(unit=query, exact=exact, baseline=saved)
    def assess(name, model, seconds, **extra):
        fitted = FittedJaxGP(model, core.JaxTrainingData(unit,np.asarray(data.y)), np.zeros(1))
        state = state_from_gpjax(fitted,scaler,np.zeros(4),np.ones(4),3008)
        pred = np.asarray(predict_lnl(state,query))
        peak = exact>exact.max()-10
        # A constant offset cancels out of a normalized posterior. These are
        # query-set diagnostics, not a validated posterior-accuracy gate.
        def centered_rmse(y):
            residual = y[peak]-exact[peak]
            return float(np.sqrt(np.mean((residual-residual.mean())**2)))
        row = dict(name=name,seconds=seconds,loss=loss(model),
            maximum_prediction_change=float(np.max(abs(pred-saved))),
            query_peak_points=int(peak.sum()),centered_query_peak_rmse=centered_rmse(pred),
            baseline_centered_query_peak_rmse=centered_rmse(saved),**extra)
        output['runs'].append(row);predictions[name]=pred
        persist();np.savez_compressed(args.output/'heldout_predictions.npz',**predictions)
        print(json.dumps(row),flush=True)
    start = time.perf_counter()
    model, history = gpx.fit(model=warm,objective=objective,train_data=data,
        optim=optax.adam(.03),key=jax.random.key(17),num_iters=args.fit_steps,verbose=False)
    jax.block_until_ready(model)
    assess('warm_adam',model,time.perf_counter()-start,iterations=args.fit_steps)
    # Explicitly request L-BFGS-B, and record success/termination. GPJax's local
    # fit_scipy wrapper does not expose the scipy result or select this method.
    params, static = eqx.partition(warm,eqx.is_array)
    initial, unravel = ravel_pytree(params)
    @jax.jit
    def value_gradient(vector):
        return jax.value_and_grad(lambda x:objective(paramax.unwrap(eqx.combine(unravel(x),static)),data))(vector)
    start = time.perf_counter()
    result = minimize(value_gradient,np.asarray(initial),jac=True,method='L-BFGS-B',
        options={'maxiter':args.fit_steps,'maxls':10,'ftol':1e-9,'gtol':1e-5})
    model = eqx.combine(unravel(jnp.asarray(result.x)),static)
    jax.block_until_ready(model)
    assess('warm_lbfgsb',model,time.perf_counter()-start,iterations=int(result.nit),
        evaluations=int(result.nfev),optimizer_success=bool(result.success),message=str(result.message))
    output['complete']=True;persist()


if __name__ == '__main__': main()
