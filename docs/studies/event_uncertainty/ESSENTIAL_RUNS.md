# Essential continuous-measurement follow-ups — frozen 2026-09-10

Purpose: distinguish information lost through measurement uncertainty from GP
approximation error, and test whether acquisition results survive different
injections and design seeds. This is an additive campaign. The original pilot,
its failures, source hashes and manuscript inputs remain unchanged.

## Frozen first stage

1. **Reference refinement:** rescore the original 500-expected-event catalogue 8,
   uncertain measurements, both BO and random, at all five saved checkpoints.
   Fit the saved training data with the original seed/configuration. Use two
   independent full-support importance references with 65536 generated points
   each. No labels enter training, and no threshold changes.
2. **Width sweep:** reuse all 20 latent catalogues and their standardized sensor
   residuals, with measurement widths multiplied by 0.5 and 2. The original
   perfect and nominal-width results provide the other two paired arms. Both
   generation and inference use each new width. Run BO and random acquisition
   with the original common initialization, candidate pool, random continuation,
   seed 17 and budgets 226/354/482/610/738: 80 new trajectories, 400 checkpoints.
   The original two 32768-point reference proposals are reused, with every new
   likelihood and posterior recomputed. Retain all numerical failures.
3. **Injection/design robustness:** three fixed injections in alpha, sigma,
   a_SFR, d_SFR order: (-0.3,0.35,0.01,4.7), (-0.46,0.15,0.006,4.35), and
   (-0.12,0.52,0.013,5.05). Use two fresh catalogues per injection at each of the
   two fixed fiducial exposures, nominal continuous errors, and seeds 17/29/43
   for GP initialization and acquisition. BO, IID random and a Sobol continuation
   share initialization within each seed. This is 12 catalogues, 108 trajectories,
   540 checkpoints. Counts have expectation T*r(truth), so 50/500 are exposure
   labels, not equal expected counts at different truths. Neither exposure nor
   prior changes with the injection. Catalogue RNG uses SeedSequence with
   [202609101, truth index, exposure index, catalogue index].

Use the existing 256M operator, full active support, fixed grid, selection
approximation, hard-tail retention and GP/noise/acquisition settings. Reference
proposals may use injection/Fisher geometry only for validation; acquisition
cannot access it. Robustness references generate 65536 points per scramble.
The tiny zero-variance numerical fallback in acquisition is uniform sampling,
not a new acquisition policy. It is not expected in ordinary trajectories.

The direct "ignore measurement uncertainty" control places each observed centre
in its bin. If any centre has no population support, report zero likelihood
everywhere and the unsupported-event count. Never discard events or clip their
measurements to manufacture a posterior. Other control cases receive the same
independent-reference precision checks.

Accuracy criteria remain unchanged: mean/interval endpoint shifts <0.2 reference
SD, 68%/95% width differences <10%, both KL directions <0.01 nats. Numerical
reference and GP-scramble criteria: means/endpoints <0.05 SD, 68% widths <3%.
An accurate GP with an unresolved reference is not a validated success. All
catalogues/seeds stay in the report; seeds within a catalogue are not independent
coverage trials. Budget crossings are retrospective, not a validated online stop.

## Execution and accounting

From the repository root, after the fast tests and a separate smoke run:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/compas-essential .venv/bin/python docs/studies/event_uncertainty/essential_runs.py refine --outdir docs/studies/event_uncertainty/essential_20260910/refine
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/compas-essential .venv/bin/python docs/studies/event_uncertainty/essential_runs.py sweep --outdir docs/studies/event_uncertainty/essential_20260910/sweep
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/compas-essential .venv/bin/python docs/studies/event_uncertainty/essential_runs.py robustness --outdir docs/studies/event_uncertainty/essential_20260910/robustness
```

Each phase refuses to overwrite its directory. It retains source snapshots,
manifests, catalogues, training checkpoints, per-checkpoint scores, atomic
status files and error records. Original cache files remain in the pilot folder.
Cache construction, fitting, acquisition, label contraction and independent
validation costs are recorded separately. Candidate rates are precomputed for
this controlled benchmark: these are not online timings or end-to-end savings.
Run phases serially to avoid competing JAX compilations and memory pressure.

## Next stage, after reviewing these results

Freeze the supported procedure and run continuous-sensor calibration on fresh
prior-drawn truths at fixed exposures, beginning with 100 catalogues per exposure.
This is not yet launched or represented by the fixed-injection robustness study.
Keep direct and GP calibration distinct and retain failures. Escalate numerical
references independently of training and do not retune on the confirmation data.
Finite PE sample convergence and genuine online timing are separate bounded
studies; analytic sensor integrals are still used here.

## Paper structure and figure plan

Lead with matched-budget posterior accuracy and adaptive parameter allocation.
Follow with continuous measurement uncertainty, then calibration and limitations.
Keep one posterior-evolution illustration in the main text; move its full corner
plot and the categorical-sensor calibration/predictive controls to the appendix.

Main figure targets: physical rate/measurement model; matched acquisition
locations; KL versus budget; direct interval widths versus error scale with
separate GP/direct discrepancies; continuous calibration; measured-coordinate
predictive checks including total count and joint mass/redshift structure.
No new numerical claims enter the paper until the relevant phase and independent
checks complete. The existing categorical calibration is not relabelled as
continuous-sensor calibration.
