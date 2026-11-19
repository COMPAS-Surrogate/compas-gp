# Operational stopping experiments

**Current result: [COMPAS stopping versus IID random sampling](gpry_compas/report_v2/RESULTS.md).**
The 48-trajectory extension varies three injections and uses paired perfect and
uncertain measurements. At 658 training labels, BO passes full posterior
accuracy in 24/24 cases versus 5/24 for random. The mass-audit rule gives 13
accurate stops, no premature stops and three non-stops in its 16-case BO
confirmation set. All final numerical checks and decision replay pass; 129
fast tests pass. Training and stopping require no parameter-space rate grid.
See the report for audit costs, the exact amplitude reconstruction, failures
of simpler rules and the finite scope of this result.

Previous [GPry-style online COMPAS experiment](gpry_compas/report_v1/RESULTS.md):
Twelve fresh COMPAS BO trajectories use JAX acquisition, three-label
Kriging-believer batches, and three stopping criteria. Independent direct
posteriors are used only afterward for the paper-style KL curves. GPry is not
installed. Counter plus Gaussian KL gives 10 accurate stops, one premature stop
under the frozen strict criterion, and one numerical non-stop.

Earlier [COMPAS stopping and JAX geometry checks](compas_geometry/RESULTS.md):
Ten saved COMPAS BO paths compare prediction, moment stability, joint-posterior
stability, and fresh derivative/profile audits. A separate follow-up tests the
unchanged audits at the later joint-stability trigger. These are checkpoint
and development screens; they do not enable a production stopping default.
The generic experiments below are retained as historical evidence.

This study tests whether a reference-free decision agrees with a subsequently
revealed reference posterior. It is separate from the earlier reference-assisted
evaluation-requirement campaign.

The [stopping-rule comparison](rule_comparison/RESULTS.md) adds 25 fresh BO
trajectories, compares five stopping criteria, tests larger broad-prior audits,
and evaluates a posterior-impact audit on fresh toy points and two COMPAS
candidate fits. It preserves v1 and distinguishes full runs from prefix/audit
screens; no production stopping default has been enabled.

Read [the frozen protocol](PROTOCOL.md) and [the generated report](report_v1/RESULTS.md).
The tested first candidate is **not ready for general use**: the 25-case toy
confirmation contains 11 accurate stops, 5 premature stops and 9 resource-limited
runs that already had accurate posteriors. All five narrow-secondary-mode cases
stopped incorrectly. Thresholds were not retuned on these confirmation cases.

## Files and evidence

- `src/cosmic_integration/lnl_surrogate/stopping.py` in the repository root:
  reusable prediction, stability and audit diagnostics.
- `run_study.py`: counted evaluation interface, reference-free learner and
  separately executed reference scoring.
- `toys.py`: five 2D target families with varied centres/orientations.
- `development_v1/`: correlated seed1000 smoke; excluded from confirmation.
- `confirmation_v1/`: five families, seeds2000--2004; complete 25-case ensemble.
- `compas_pilot.py`, `compas_v1/`: two original COMPAS catalogue pilots with
  perfect and uncertain measurements. Their exact cached references are read
  only after the learner writes its decision. Read status.json for completion.
- `report_v1/`: outcome summary, all reference-KL trajectories, operational-gate
  comparison, and a direct-versus-GP density illustration of a premature stop.
- `runtime_environment.json`: Python/package versions and thread configuration.

Each run records training checkpoints, pre-refit predictions and actual labels,
fresh audits, all true-label calls, and an immutable decision written before
reference scoring. Scoring records its SHA-256 and verifies the file is unchanged.
Reference evaluations are separate offline validation costs. Candidate and
operational-posterior quadrature coordinates require GP evaluations only.

The 2D code was generalized to accept a dimension argument for the separate 3D
COMPAS pilots. The original confirmation process retained its loaded code and
source snapshot. `dimension_equivalence_check/verification.json` confirms that
the generalized 2D default reproduces the development decision and every saved
training coordinate and label exactly. No stopping tolerances changed.

## Reproduction

From the repository root, use one BLAS/OpenMP thread and a writable Matplotlib
cache. Use a new output directory; the runner refuses to overwrite a campaign.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/compas-mpl \
  .venv/bin/python docs/studies/operational_stopping/run_study.py \
  --outdir docs/studies/operational_stopping/reproduction_v1 \
  --seeds 2000 2001 2002 2003 2004
```

`publish_results.py` rebuilds the figures/report for the archived
`confirmation_v1/` results. The protocol and source snapshots define the exact
first candidate. The small COMPAS pilot uses 128 initial Sobol points and 98
boundary points, batches of128 and a 2200-call resource guard; its thresholds
are identical to the 2D test. It is not a new generative calibration ensemble.

The software checks passed 87 fast tests (3 heavy tests deselected). Software
correctness does not imply that this statistical stopping rule passed validation.

## Next decision

Retain v1 as a rejected baseline. An improved candidate needs a more useful
posterior-impact check and explicit testing of global mode discovery. The
pre-refit tests rejected many fits after reference accuracy had been achieved;
the fresh audit still missed an isolated region carrying substantial posterior
mass. Merely tightening one tolerance cannot resolve both behaviours.

Any revised rule needs new confirmation seeds and explicit assumptions about
the target class. A finite local audit cannot guarantee discovery of an
arbitrarily narrow, unseen mode. Keep the current 100-catalogue reference-assisted
campaign as benchmark evidence; do not relabel it as operational validation.
