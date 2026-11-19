"""First frozen joint COMPAS GP pilot, with a held-out rejection gate.

Run from the repository root. An inaccurate held-out target is rejected before
spending thousands of evaluations on a direct-reference chain. Reweighting is
diagnostic only and never labelled a converged reference posterior.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import sys
import time
import tomllib

import jax
import jax.numpy as jnp
import numpy as np

from cosmic_integration.observation.grid_likelihood import counts_log_likelihood
from cosmic_integration.ratesSampler import BinnedCosmicIntegrator
from cosmic_integration.ratesSampler.grid_diagnostics import system_grid_moments
from cosmic_integration.lnl_surrogate.adaptive_robust_scalar import AdaptiveRobustScaler
from cosmic_integration.lnl_surrogate.jax_active_learner import JaxActiveLearner, JaxGPConfig
from cosmic_integration.lnl_surrogate.nuts_sampler import make_log_likelihood, run_nuts

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "grid_resolution"))
from probe_grids import configurations

LOG = logging.getLogger(__name__)


def main() -> None:
    cfg_path = HERE.parent / "grid_resolution/joint_pilot.toml"
    cfg = tomllib.loads(cfg_path.read_text())
    g, s = cfg["gp"], cfg["sampler"]
    seed = g["training_seeds"][0]
    out = HERE / f"results_seed{seed}"
    out.mkdir(exist_ok=True)
    with np.load(REPO / cfg["data"]["catalogue"]) as f:
        counts = f[f"counts_{cfg['data']['grid']}"]
    mass, z, _ = configurations()[cfg["data"]["grid"]]
    bounds = np.array([cfg["prior"]["low"], cfg["prior"]["high"]])
    duration = cfg["data"]["duration_years"]
    h5 = REPO / cfg["data"]["compas_file"]
    metadata = {"config": cfg, "input": str(h5), "input_size": h5.stat().st_size,
                "input_mtime_ns": h5.stat().st_mtime_ns,
                "catalogue_sha256": hashlib.sha256((REPO/cfg["data"]["catalogue"]).read_bytes()).hexdigest(),
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    source = sorted((REPO / "src/cosmic_integration").rglob("*.py"))
    metadata["source_sha256"] = hashlib.sha256(b"".join(p.read_bytes() for p in source)).hexdigest()
    manifest = out / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != metadata:
        raise ValueError("Pilot provenance changed; use a new output directory")
    manifest.write_text(json.dumps(metadata, indent=2)+"\n")
    started = time.perf_counter()
    ci = None
    cache_file = out / "direct_evaluations.jsonl"
    cache = {}
    if cache_file.exists():
        for line in cache_file.read_text().splitlines():
            row = json.loads(line)
            cache[tuple(row["theta"])] = float(row["lnl"]) if row["lnl"] is not None else -np.inf
    def direct(theta):
        nonlocal ci
        key = tuple(float(x) for x in theta)
        if key in cache:
            return cache[key]
        if ci is None:
            ci = BinnedCosmicIntegrator.from_compas_fpath(str(h5))
        tic = time.perf_counter()
        rates, masses = ci.FindDetectionRate(p_BinaryFraction=.7, **dict(zip(
            ["p_Alpha", "p_Sigma", "p_SFRa", "p_SFRd"], key)))
        grid, _ = system_grid_moments(rates, masses, mass, z)
        value = counts_log_likelihood(grid, counts, duration)
        cache[key] = value
        with cache_file.open("a") as f:
            f.write(json.dumps({"theta": key, "lnl": value if np.isfinite(value) else None,
                                "seconds": time.perf_counter()-tic})+"\n")
        if len(cache) % 16 == 0:
            LOG.info("Direct calls %d; elapsed %.1fs", len(cache), time.perf_counter()-started)
        return value
    root = out / "gp_model"
    training_start = time.perf_counter()
    if not (root / "models/round_0.pkl").exists():
        rng = np.random.default_rng(seed)
        x = rng.uniform(*bounds, size=(g["initial_points"], 4))
        raw = np.array([direct(t) for t in x])
        finite = raw[np.isfinite(raw)]
        if not len(finite):
            raise ValueError("No finite initial likelihoods")
        scaler = AdaptiveRobustScaler(compression=g["compression"],
            soft_clipping=g["soft_clipping"], lower_clip_value=float(finite.min()),
            max_scale=g["max_scale"])
        scaler.initialize_with_data(finite)
        def objective(*theta):
            return -float(scaler.transform(direct(theta)))
        config = JaxGPConfig(kernel=g["kernel"], initial_noise_stddev=np.sqrt(g["noise_variance"]),
            optimise_noise=g["optimise_noise"], optimisation_steps=g["optimisation_steps"],
            candidate_count=g["candidate_count"])
        learner = JaxActiveLearner(objective, bounds, initial_data_x=x,
            initial_data_y=-scaler.transform(raw)[:, None], random_seed=seed,
            config=config, outdir=str(root), refit_every=g["refit_every"])
        learner.run(g["active_steps"], exploration_fraction=g["exploration_fraction"],
                    cycle_length=g["cycle_length"])
        root.mkdir(exist_ok=True)
        scaler.save(str(root))
        learner.save_model(round_idx=0)
        np.savez(out / "training.npz", theta=learner.data.query_points,
                 raw_lnl=np.array([direct(t) for t in learner.data.query_points]),
                 initial_reference=scaler.reference_value)
        (out / "training_timing.json").write_text(json.dumps({"seconds":time.perf_counter()-training_start})+"\n")
    LOG.info("Training complete; sampling GP target")
    sampling = out / "sampling"
    if not (sampling / "nuts_summary.json").exists():
        run_nuts(lnl_model_path=root / "models", outdir=sampling, round_idx=0,
                 target=g["target"], num_warmup=s["num_warmup"], num_samples=s["num_samples"],
                 num_chains=s["num_chains"], seed=s["seed"])
    sampler = json.loads((sampling / "nuts_summary.json").read_text())
    draws = np.load(sampling / "posterior_samples.npy")
    rng = np.random.default_rng(20260910)
    n = cfg["validation"]["heldout_points"]
    # Fresh points: 70 GP posterior, 20 expanded posterior, 10 full prior.
    npost, nexpand = int(.7*n), int(.2*n)
    post = draws[rng.choice(len(draws), npost, replace=False)]
    expanded = np.median(draws, axis=0)+2*(draws[rng.choice(len(draws), nexpand)]-np.median(draws, axis=0))
    expanded = np.clip(expanded, bounds[0]+1e-6*np.ptp(bounds, axis=0), bounds[1]-1e-6*np.ptp(bounds, axis=0))
    heldout = np.vstack([post, expanded, rng.uniform(*bounds, size=(n-npost-nexpand, 4))])
    model = JaxActiveLearner.load_model(root / "models", 0)
    scaler = AdaptiveRobustScaler.load(str(root))
    target = make_log_likelihood(model, scaler, target=g["target"])
    predicted = np.asarray(jax.jit(jax.vmap(target))(jnp.asarray(heldout)))
    truth = np.array([direct(t) for t in heldout])
    np.savez(out / "heldout.npz", theta=heldout, direct_lnl=truth, gp_lnl=predicted, posterior_count=npost)
    residual = predicted[:npost]-truth[:npost]
    if np.all(np.isfinite(residual)):
        centered = residual-np.mean(residual)
        rmse = float(np.sqrt(np.mean(centered**2)))
        weights = np.exp(-residual-np.max(-residual))
        weights /= weights.sum()
        wess = float(1/np.sum(weights**2))
        shift = ((weights[:, None]*post).sum(0)-post.mean(0))/post.std(0)
    else:
        rmse, wess, shift = None, None, np.full(4, np.nan)
    with np.load(out / "training.npz") as train:
        better = int(np.sum(train["raw_lnl"] > train["initial_reference"]+1e-8))
        ceiling_loss = float(np.max(train["raw_lnl"])-train["initial_reference"])
    gate = bool(sampler["converged"] and rmse is not None and rmse < .1)
    result = {"sampler": sampler, "posterior_centered_lnl_rmse": rmse,
              "posterior_max_centered_error": float(np.max(np.abs(centered))) if rmse is not None else None,
              "diagnostic_reweighting_ess_of_70": wess,
              "diagnostic_reweighted_mean_shift_sigma": [float(v) if np.isfinite(v) else None for v in shift],
              "training_points_above_initial_transform_ceiling": better,
              "largest_training_improvement_above_ceiling": ceiling_loss,
              "ready_for_direct_reference": gate,
              "direct_reference_completed": False,
              "decision": "Proceed to independent direct reference" if gate else
                          "Reject this GP pilot before expensive direct-reference sampling",
              "scope": "One seed; held-out rejection gate and reweighting diagnostic, not a reference posterior",
              "direct_evaluations": len(cache), "wall_seconds_this_run":time.perf_counter()-started}
    (out / "summary.json").write_text(json.dumps(result, indent=2)+"\n")
    LOG.info("%s; held-out posterior lnL RMSE=%s", result["decision"], rmse)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
