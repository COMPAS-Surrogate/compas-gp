# Reference GP, physical information and population-size presentation

Revision: 2026-09-12. Current review PDF:
`overleaf/output/dense_reference_review/ms.pdf` (18 pages).

The main paper retains three figures (five overall). The four demonstration
corners remain grouped into two figures. The third figure now puts physical
posterior widths above evaluation requirements, using the same 600-catalogue
ensemble. The former Figure 5 is replaced by one likelihood-loss plot.

## Genuine 2000-label GP references

The blue contours are actual densely conditioned GP posteriors. The early BO
posterior is grey dashed, the accepted BO posterior black dashed, and every
stopping-panel curve and marker is monochrome. Original stops and their costs
are unchanged. No numerical-target posterior is overlaid.

Each reference retains all 658 saved training labels and adds 1342 fresh labels
at coordinates drawn from the full-support proposal guided by that saved GP.
All tail anchors and the original prior remain. The saved kernel hyperparameters
and target transform are retained while conditioning on all 2000 observations.
This is a separate reference fit, not a claim that BO was continued to 2000.
Numerical validation designs are first accessed after the reference fit is saved.

All four references pass the existing numerical and four-parameter accuracy
checks, plus the stricter reference requirement of maximum directed posterior
KL below 0.001. Actual maximum directed KLs are:

| Expected detections | Measurements | Maximum directed KL [nats] |
|---|---|---:|
| 100 | Perfect | 0.00031105 |
| 100 | Uncertain | 0.00007503 |
| 1000 | Perfect | 0.00007952 |
| 1000 | Uncertain | 0.00008732 |

Plotting-draw means match the independently checked summaries within 0.002
posterior SD and interval endpoints within 0.016 SD. Checkpoints, construction
metadata, source hashes, validation scores and draws are in
`dense_reference_20260912/`. The original confirmation still has 23 accurate
BO stops and one premature stop; these demonstrations do not replace that result.

## Physical information versus expected event count

The width panels use all 600 accepted BO benchmark posteriors, each with its
recorded numerical/accuracy checks and two-checkpoint stability. They include
the independently refined high-count case 49 exactly once. Curves show median
central 68% interval widths divided by the prior central 68% width; bands show
10th--90th percentiles over 100 catalogues per setting. These are GP summaries.

For perfect measurements, the median normalized widths from 30 to 1000 expected
detections change from 0.436 to 0.155 for alpha, 0.971 to 0.568 for sigma, 0.847
to 0.320 for a, and 0.996 to 0.935 for d. The weak response of d is a useful
physical result, not a failed acquisition: all these BO posteriors pass the
internal accuracy benchmark. The uncertain-measurement panel uses 50 and 500
expected detections; the two columns are not paired measurement realizations.
Saved plotting values are in `presentation_20260911/physics_widths.json`.

## Interpretable replacement for Figure 5

The new plot measures the expected log-likelihood loss at the fiducial physical
parameters, comparing measured nested 32M, 128M and 256M intensities with the
finite 512M archive. Both the total-rate and distribution-shape terms are included,
using the same nominal continuous measurement model as the manuscript.

If u is the candidate/baseline measurement-intensity ratio, the loss is
E[N] E_baseline[u - 1 - log(u)]. This nonnegative representation is equivalent
to the Poisson-process expected log-likelihood difference and avoids cancellation
of large rate and shape terms. Two independent sets of 262144 simulated event
measurements resolve the expectation. At 1000 expected detections, losses are
6.3786, 1.0117 and 0.40015 nats, respectively. The Monte Carlo standard errors
are 0.0270, 0.00381 and 0.000986 nats.

There are 12 marked event-count values from 10 to 3000. Their dependence follows
exact exposure scaling; these are not twelve independent catalogue experiments.
Bands show integration uncertainty, not scatter across independent COMPAS runs
or observed catalogues. No 512M result is extrapolated and no universal safety
threshold is drawn. Because 512M is itself finite, call this likelihood loss or
finite-population discrepancy, not bias relative to an infinite simulation.
A discrepancy at a single parameter vector does not imply a posterior shift.
The previous 36 paired posterior checks remain internal and are summarized in
the appendix, but their two-panel plot is no longer included.

Results, rates, numerical integrands and hashes are in `likelihood_loss_20260912/`.
The new formula is tested against an independently calculated expected Poisson
log likelihood, normalized measurement smearing, and nearly identical models.
The full fast suite passes 149 tests (three deselected; one existing Matplotlib
deprecation warning). Original likelihood, simulation and cosmic-integration
sections match the pre-edit archive byte for byte. The stopping-decision hash
also remains unchanged. No commits, pushes or remote synchronization were made.

The compiled PDF has no undefined references, overfull boxes or stuck floats.
All pages were rendered and inspected; final layout corrections remove an
isolated software paragraph and keep headings with their associated text.

## Reproduce

From the package root, with the existing scientific environment:

```sh
MPLCONFIGDIR=/tmp/compas-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python docs/studies/paper_completion/dense_gp_reference.py
MPLCONFIGDIR=/tmp/compas-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python docs/studies/paper_completion/population_likelihood_loss.py
MPLCONFIGDIR=/tmp/compas-mpl .venv/bin/python docs/studies/paper_completion/presentation_figures.py --plots-only
MPLCONFIGDIR=/tmp/compas-mpl .venv/bin/python docs/studies/paper_completion/plot_likelihood_loss.py
```

Existing saved reference fits and measurement integrations are reused on rerun.
The active population figure and caption are now referenced from
`overleaf/sections/bo_controls.tex`; the older `publish_completion.py` emits
historical fragments that are no longer included by the manuscript.
