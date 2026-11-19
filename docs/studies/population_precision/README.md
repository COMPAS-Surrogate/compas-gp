# Population size and grid support

The approved manuscript presentation is now in Appendix D.5: Figure 9 is the
single-panel actual nested-population loss, and Figure 10 contains effective
pixel-count maps for 8M, 32M and 512M. The bootstrap and threshold figures below
remain supporting study outputs. Rebuild the manuscript figures with:

```bash
MPLCONFIGDIR=/tmp/compas-precision-mpl \
  .venv/bin/python docs/studies/population_precision/manuscript_figures.py
```

The reviewed manuscript PDF is `overleaf/output/population_precision_review/ms.pdf`.

This study uses the existing 512M archive. It runs the direct cosmic integrator
once per 8M block for system-level moments and caches a sparse rate operator for
nearby parameter checks. It neither trains a GP nor runs new stellar evolution.

The original supporting figures answer:

1. How does expected log-likelihood loss grow with the observed catalogue size?
   Show actual nested populations separately from block-bootstrap variability.
   Keep the physical parameters, mass10 grid, selection and measurement model
   fixed. Expected counts 10–1000 are exact exposure scaling, not separate runs.
2. How much of the predicted detection distribution has weak pixel support?
   Compare native, mass10 and mass20 grids; weight pixel diagnostics by the full
   reference rate, so sparse but negligible tails do not dominate the summary.

Partition by initial-system row, not selected-merger count. Each block represents
8M initial systems and retains the full archive's metallicity support and common
seed-0 selection/mass normalization. Coherent resampling preserves bin and
parameter correlations. Full-size bootstrap loss is relative to the finite
empirical 512M reference; it is not an independent validation of 512M or absolute
bias. Blocks must be exchangeable for a sampling interpretation.

The following commands regenerate the supporting study outputs. Run from the repo root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/compas-precision-mpl \
  .venv/bin/python docs/studies/population_precision/run_study.py
MPLCONFIGDIR=/tmp/compas-precision-mpl \
  .venv/bin/python docs/studies/population_precision/plot_study.py
```

The runner records source/input fingerprints and rejects stale caches. Numerical
integration and population-resampling errors are reported separately. Perfect
measurements retain infinite losses for missing support; there is no rate floor.
