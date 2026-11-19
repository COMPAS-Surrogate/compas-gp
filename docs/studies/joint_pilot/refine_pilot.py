"""One bounded local-training refinement, retaining every baseline point.

The original held-out set becomes training data; assessment uses fresh points
from the newly sampled GP target. This is exploratory, not confirmatory SBC.
"""
from pathlib import Path
import json
import logging
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np

from cosmic_integration.lnl_surrogate.adaptive_robust_scalar import AdaptiveRobustScaler
from cosmic_integration.lnl_surrogate.jax_active_learner import JaxGPConfig, JaxTrainingData, fit_exact_gp
from cosmic_integration.lnl_surrogate.nuts_sampler import make_log_likelihood, run_nuts
from cosmic_integration.observation.grid_likelihood import counts_log_likelihood
from cosmic_integration.ratesSampler import BinnedCosmicIntegrator
from cosmic_integration.ratesSampler.grid_diagnostics import system_grid_moments
from run_pilot import HERE, REPO, configurations


def main() -> None:
    start = time.perf_counter()
    base = HERE / "results_seed17"
    manifest = json.loads((base / "manifest.json").read_text())
    cfg = manifest["config"]
    out = base / "local_refinement"
    out.mkdir(exist_ok=True)
    with np.load(base / "training.npz") as f:
        x, raw = f["theta"], f["raw_lnl"]
    with np.load(base / "heldout.npz") as f:
        x = np.vstack([x, f["theta"]])
        raw = np.r_[raw, f["direct_lnl"]]
    scaler = AdaptiveRobustScaler.load(str(base / "gp_model"))
    g = cfg["gp"]
    bounds = np.array([cfg["prior"]["low"], cfg["prior"]["high"]])
    model = fit_exact_gp(JaxTrainingData(x, -scaler.transform(raw)[:, None]), bounds,
        config=JaxGPConfig(kernel=g["kernel"], initial_noise_stddev=np.sqrt(g["noise_variance"]),
                          optimise_noise=g["optimise_noise"], optimisation_steps=g["optimisation_steps"],
                          candidate_count=g["candidate_count"]), seed=17)
    root = out / "gp_model"
    (root / "models").mkdir(parents=True, exist_ok=True)
    with (root / "models/round_0.pkl").open("wb") as f:
        pickle.dump(model, f)
    scaler.save(str(root))
    s = cfg["sampler"]
    summary = run_nuts(lnl_model_path=root / "models", outdir=out / "sampling", round_idx=0,
                       target=g["target"], num_warmup=s["num_warmup"], num_samples=s["num_samples"],
                       num_chains=s["num_chains"], seed=s["seed"])
    draws = np.load(out / "sampling/posterior_samples.npy")
    rng = np.random.default_rng(20260911)
    post = draws[rng.choice(len(draws), 70, replace=False)]
    median = np.median(draws, axis=0)
    expanded = median+2*(draws[rng.choice(len(draws), 20)]-median)
    expanded = np.clip(expanded, bounds[0]+1e-6*np.ptp(bounds, axis=0), bounds[1]-1e-6*np.ptp(bounds, axis=0))
    heldout = np.vstack([post, expanded, rng.uniform(*bounds, size=(10, 4))])
    assert not set(map(tuple, heldout)) & set(map(tuple, x)), "Validation points overlap training"
    target = make_log_likelihood(model, scaler, target=g["target"])
    predicted = np.asarray(jax.jit(jax.vmap(target))(jnp.asarray(heldout)))
    with np.load(REPO / cfg["data"]["catalogue"]) as f:
        counts = f[f"counts_{cfg['data']['grid']}"]
    mass, z, _ = configurations()[cfg["data"]["grid"]]
    ci = BinnedCosmicIntegrator.from_compas_fpath(str(REPO / cfg["data"]["compas_file"]))
    direct = []
    for i, theta in enumerate(heldout):
        rate, masses = ci.FindDetectionRate(p_BinaryFraction=.7, **dict(zip(
            ["p_Alpha", "p_Sigma", "p_SFRa", "p_SFRd"], theta)))
        matrix, _ = system_grid_moments(rate, masses, mass, z)
        value = counts_log_likelihood(matrix, counts, cfg["data"]["duration_years"])
        direct.append(value)
        with (out / "direct_evaluations.jsonl").open("a") as f:
            f.write(json.dumps({"theta": theta.tolist(), "lnl": value if np.isfinite(value) else None})+"\n")
        if (i+1) % 20 == 0:
            logging.info("Refinement fresh validation %d/100", i+1)
    direct = np.array(direct)
    residual = predicted[:70]-direct[:70]
    centered = residual-residual.mean()
    rmse = float(np.sqrt(np.mean(centered**2)))
    w = np.exp(-residual-np.max(-residual))
    w /= w.sum()
    result = {"training_points": len(x), "fresh_validation_points": len(heldout),
              "posterior_centered_lnl_rmse": rmse,
              "posterior_max_centered_error": float(np.max(np.abs(centered))),
              "diagnostic_weight_ess_of_70": float(1/np.sum(w*w)),
              "ready_for_direct_reference": bool(summary["converged"] and rmse < .1),
              "sampler": summary, "seconds": time.perf_counter()-start,
              "scope": "One local refinement; no independent direct-reference posterior",
              "original_manifest": manifest}
    np.savez(out / "heldout.npz", theta=heldout, gp_lnl=predicted, direct_lnl=direct)
    (out / "summary.json").write_text(json.dumps(result, indent=2)+"\n")
    logging.info("Refined posterior lnL RMSE %.4f; ready=%s", rmse, result["ready_for_direct_reference"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
