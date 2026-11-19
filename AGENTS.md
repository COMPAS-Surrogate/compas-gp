# AGENTS.md

This file is a short guide for humans (and coding agents) making changes to this codebase.

## What this repo is
`cosmic_integration` is a Python package for:

- **Rates / cosmic integration** utilities (see `cosmic_integration/ratesSampler/`).
- **Log-likelihood computation** for COMPAS + observations (`cosmic_integration/lnl_computer.py`, `cosmic_integration/observation/`).
- **A GP surrogate for log-likelihood** with active learning / BO-style training (`cosmic_integration/lnl_surrogate/`).
- A small set of **CLI entry scripts** (`cosmic_integration/cli_tools/`).

If you’re adding new features, prefer keeping core logic in the package modules and leaving CLIs as thin wrappers.

## Repository map
- `cosmic_integration/lnl_surrogate/`
  - `lnl_surrogate.py`: main `LnLSurrogate` Likelihood wrapper + training entrypoint
  - `jax_active_learner.py`: GPJax exact GP + BO loop (variance / expected-improvement acquisition)
  - `diagnostics/`: GP-vs-truth, posterior-KL, and trace/scatter plots
  - `adaptive_robust_scalar.py`: target transform / scaling utilities
  - `run_sampler.py`: sampling utilities using a trained surrogate
- `cosmic_integration/observation/`: observation models and plotting
- `cosmic_integration/ratesSampler/`: binned cosmic integrator + data reduction
- `cosmic_integration/cli_tools/`: command-line scripts (Click)

## Setup (local dev)
Recommended: create a dedicated virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

Then install the project dependencies.

- If you have a `pyproject.toml` / `requirements.txt` in the repo root, use that.
- If you don’t, document the install step you used in the PR/commit message.

## How to run common tasks
### Train a surrogate (example)
The project includes CLIs under `cosmic_integration/cli_tools/`.

Example (see `cli_tools/run_surrogate_workflow.py`):

```bash
run_surrogate_workflow \
  --compas-h5 path/to/COMPAS.h5 \
  --observation-file path/to/observation.h5 \
  --outdir outdir
```

(Adjust arguments to match your local data files.)

### Run other tools
Other useful scripts live in `cosmic_integration/cli_tools/`, e.g.

- `run_surrogate_workflow.py`
- `diagnose_surrogate_chain.py`
- `run_1d_lnl_check.py`

Prefer `python -m cosmic_integration.cli_tools.<script_name>` so imports resolve consistently.

## Tests (pytest)
This project uses **pytest**.

`pytest` runs the **fast unit suite only** (~20s). Heavy tests are marked and
deselected by default via `addopts` in `pyproject.toml`:

- `slow` — runs cosmic integration or trains a surrogate end to end (minutes)
- `network` — downloads the mock observation from GitHub

```bash
pytest                      # fast unit suite (default)
pytest -m slow              # heavy tests only
pytest -m ""                # everything
pytest -k <pattern>
```

### Test expectations
- New features and bug fixes should include **new tests**.
- Tests should be deterministic: set seeds where randomness is used.
- If a test needs large data files (e.g., HDF5 grids), prefer:
  - small synthetic fixtures, or
  - marking as slow/integration (e.g. `@pytest.mark.slow`) and skipping by default.

## Coding conventions
### Style
- Keep functions small and well-named; prefer readability over cleverness.
- Prefer **f-strings** for formatting.
- Add type hints for public APIs (and for any non-trivial internal functions).

### Logging
- Use the `logging` module (not `print`) for library code.
- CLIs may `click.echo(...)` for user-facing output.

### Data / I/O
- Avoid hard-coding paths.
- Treat `.h5` / cache files as potentially large; do not load whole files unless necessary.

## Making changes safely
### When touching the surrogate / scaling code
Files like `lnl_surrogate/adaptive_robust_scalar.py` affect:

- the target transformation (`LnL` → transformed space)
- stability near the optimum
- tail behaviour (soft clipping / lower clip)

If you change anything there:
- add/extend tests that cover the transform,
- include before/after plots or numeric diagnostics in the PR description when possible,
- be explicit about any changed default hyperparameters.

### Backwards compatibility
- Don’t change public function signatures without a clear reason.
- If you must change an API, add a short migration note in the PR description.

## Agent workflow checklist
Before opening a PR (or finishing an automated edit):
1. Run `pytest` locally.
2. Ensure new code is covered by tests.
3. Keep diffs focused: avoid drive-by refactors.
4. Update docstrings/comments where behaviour changed.

## Known findings / experiments
### Do not subset/downweight far-tail ("bad lnL") GP training points
See `docs/studies/subset_training/test_subset.py` (synthetic 4D correlated-Gaussian
peak, scored against a known truth). Comparing `all` (keep everything, current
default) vs `topN` (drop points far from the best lnL) vs `thinned` (informative
points + 10% random floor):

- At ~24% informative points, all three strategies are indistinguishable and good.
- At ~1.6% informative points (matching the real 3-yr run's 86-90%+ floor
  plateau), they diverge sharply: `all` stays roughly usable (corr 0.13 vs true
  0.85, width 1.5x true), while `topN`/`thinned` are catastrophic (corr ~0/0.06,
  width 2.5-5x true). Dropping the floor points destroys the correlation/degeneracy
  estimate — the far points are load-bearing, not noise.
- **Conclusion**: keep the current "train on everything, hard-clip at the floor"
  default. Do not pursue training-set subsetting/downweighting as a fix for poor
  surrogate calibration in the low-informative-fraction regime.





---
name: lnl-surrogate-validity-state
description: Where the GP LnL-surrogate validity investigation stands (Aug 2026) and what to do next
metadata: 
  node_type: memory
  type: project
  originSessionId: e7e3ec58-3211-4233-a975-9f37ed47d28c
  modified: 2026-08-19T21:39:08.719Z
---

State of the LnL-surrogate validity work as of 2026-08-19. Test harnesses live
in `docs/studies/subset_training/` (toy at COMPAS scale, ground truth known).
See [[surrogate-sample-size-criterion]] for the separate N_eff question.

## The central finding

**Dynamic range is the difficulty knob, not the number of training points.**
RMSE where the posterior lives (target << 0.5 lnL, since the 1σ contour is at
ΔlnL = 0.5), matern52 + sqrt, 250 points, 5 seeds:

| dynamic range | peak σ | 5% inf. | 15% | 30% | 60% |
| --- | --- | --- | --- | --- | --- |
| 1e2 | 6.2% of box | 1.04 | 0.77 | **0.54** | 0.96 |
| 1e3 | 2.0% | 3.05 | 0.76 | 0.60 | **0.32** ✅ |
| 1.8e4 (realistic) | 0.5% | 2.60 | 1.48 | **0.76** | 0.80 |

At realistic dynamic range the GP **plateaus at ~0.8 and never becomes usable**,
at any informative fraction. Model misspecification (stationary kernel on a
non-stationary surface), not a data problem. ~30% informative is the sweet spot;
60% is often worse (the GP loses the global structure anchoring its lengthscales).

**Implied fix: shrink the prior box iteratively** — a coarse pass to localise,
then re-run BO in a smaller box where the range is 1e2–1e3. This is trust-region
BO, and is structurally **nested importance sampling with a GP proposal**. It
inherits nested sampling's hazard: mass cut early cannot be recovered, and the
result then looks confidently wrong.

## Safety status: currently conservative, NOT overconfident

`test_coverage_gate.py` (simulation-based calibration, 24 runs, fresh truth
each): cov68 = **1.00** on all four params (nominal 0.68). Truth lands inside
the 68% interval every time. Gate validated to actually detect the dangerous
direction — shrinking posteriors 2× gives cov68 0.51, 5× gives 0.00.

Run this gate **after every narrowing stage**. `log` compression has already
produced genuinely too-narrow posteriors (3.3×), so safe→dangerous is reachable.

## ⚠️ Two scoring bugs — invalidate several earlier tables

1. Scripts divided by `WIDTH`, but true posterior σ = `WIDTH/10`. A *correct*
   posterior scored **0.10**, not the "1.00" printed everywhere.
2. `rmse_peak` used `ΔlnL < KEEP_DELTA` (=1000) ≈ **45σ** — far outside the
   posterior.

Reinterpreted: local GP is 2.7× too wide with 0.6σ bias (not "3.7× too narrow");
global sqrt is 21× too wide with 12σ bias. **The local GP is a big win.**

Also: **seed-to-seed scatter is enormous** (same config gave rmse_peak 0.52 and
230). Never trust a single-seed comparison; the harnesses now run 5 seeds.

## Settled choices

- **matern52 > matern32** (0.76 vs 2.48 at 30% informative; matern32 erratic).
- **`sqrt` compression** — most *reliable* (best median, tightest spread, 5×
  better global accuracy). `log`/`softlog` have better best-cases but 1000×
  spread. `none` is ~40× worse. Compression is essential.
- Acquisition was a hard 2/3-explore block; now cycles 1/3 explore / 2/3 exploit
  (`cycle_length=30`). Pure-exploit scored best on peak metrics, but those
  metrics are peak-focused — not yet shown to give a better posterior.
- bilby/emcee/dynesty removed; numpyro NUTS, 2 chains from separated init
  points, R-hat/ESS/divergence flagging that never raises.
- GP uncertainty handled in closed form as `μ + σ²/2` (`target="marginal"`),
  never by drawing `lnL ~ N(μ,σ)` per call (that breaks NUTS outright).

## Known-unfixed

- **`cov68` of the GP's own predictions never exceeds 0.25** (want 0.68) for any
  kernel/compression/noise setting. GP variance is badly underestimated; makes
  the `marginal` target unreliable.
- **Speed: JAX recompiles every BO step** because the training array grows one
  row (`predict_f` 14.5 ms same shape → 995 ms after +1 row; 127 XLA compiles
  per step via `JAX_LOG_COMPILES=1`). `refit_every=15` added but only buys 1.3×.
  Real fix is batch acquisition or padding arrays to fixed capacity. Applies to
  production: the real duration scan was ~6 s surrogate overhead per 0.88 s
  likelihood.

## Next steps

1. **Rerun the coverage gate at the realistic 1.8e4 dynamic range.** If coverage
   holds ≥ nominal, there is a defensible safety claim in the hard regime; if it
   degrades, that is the validity boundary and belongs in the paper.
2. Test **iterative prior narrowing** end to end, with the gate after each stage.
3. Local GP + a sparse outer ring of anchor points (or pin the local GP's
   constant mean to the floor) — local fitting gave near-exact correlation and
   bias, only the fall-off was wrong.
4. Chase the `cov68` miscalibration; check the transform's varying slope is
   propagated into σ correctly before blaming the GP.
5. Manuscript: the non-COMPAS simulation study is worth including — it is the
   real contribution ("here is the regime where this is valid, and how you know").
   Run it only with the fixed pipeline; earlier numbers are diagnostics, not
   results.
