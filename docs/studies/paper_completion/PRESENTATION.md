# Posterior-first presentation revision — 2026-09-11

The subsequent [corner redesign](CORNER_REDESIGN.md) supersedes the figure layout
and reference-overlay description below.

The paper now leads with one claim: GP-guided acquisition concentrates model
evaluations where they recover population posteriors, and a practical stopping
decision must combine posterior stability with independent likelihood audits.
The fresh premature stop bounds that rule's demonstrated reliability.

## Reading sequence

1. Main Figure 1: perfect-measurement posterior at 118 BO training evaluations,
   at the actual 478-evaluation stop, and from independent direct inference.
   Audit and stability insets explain why a passing audit alone is insufficient.
2. Main Figure 2: the same 979 latent mergers with uncertain measurements.
   Same prior axes and plotting conventions; actual stop is 538 training calls.
3. Main Figure 3: the full 600-catalogue acquisition benchmark summarized by
   median and 10th–90th percentile evaluation requirements in two panels.
   All catalogues remain in the empirical distributions, including three
   resource-limited random trajectories above the reported quantiles.

Five figures remain overall, down from ten: three main figures and two appendix
figures (cosmic integration and population-size sensitivity). The size result is
retained as a scope check rather than a competing main narrative. Rate heatmaps,
error-width curves, extra KL panels, stopping-cost scatter, P-P, predictive and
seed-comparison figures remain in the archive but are not manuscript figures.
Their relevant findings and failures are retained in concise text.

The abstract, result order, discussion and methods now follow this sequence.
Cosmic integration, data simulation and the principal catalogue log-likelihood
remain in the main text. The stopping appendix describes the frozen procedure;
its developmental variants are removed from the paper. The large-batch accuracy
benchmark remains explicitly distinct from the small-batch stopping experiment.

Riley & Mandel (2023), arXiv:2303.00508, Figures 5–6 informed the posterior-first
presentation. Their early stopping concerns ANN training; the displayed audit
and stability criteria are this project's independently implemented BO rule.
No ANN stopping procedure is attributed to our algorithm or vice versa.

## Figure evidence

The two corner plots use catalogue ID 100, fiducial injection, 1000 expected
events, from the existing fresh confirmation. Selection follows that identifier,
not posterior quality. They share 979 latent events. The original GP states,
training labels and decisions are unchanged; no new training or COMPAS evolution
was run for this presentation revision.

- Perfect: 478 training + 480 audit = 958 total calls; independent maximum
  directed posterior KL is 0.0025197.
- Uncertain: 538 training + 576 audit = 1114 total calls; independent maximum
  directed posterior KL is 0.0024883.
- The complete fresh experiment still reports 23 accurate and one premature BO
  stop, three accurate random stops and 21 random non-stops.

Plotting uses both independent full-support reference integrations and exact
bounded-Gamma amplitude draws. Both early and stopped plotting posteriors pass
independent numerical precision checks. Reconstructed direct/stopped means
agree with the saved summaries to under 0.003 posterior SD and interval endpoints
to under 0.016 SD. Smoothed 50%/90% contours are visual summaries; all likelihood
and KL checks use unsmoothed weights. Each view uses identical prior bounds,
64 histogram bins per dimension and Gaussian smoothing of 0.8 bins.

Reproduce from saved checkpoints:

```sh
MPLCONFIGDIR=/tmp/compas-paper-mpl .venv/bin/python docs/studies/paper_completion/presentation_figures.py
```

For layout-only regeneration, add `--plots-only`. Posterior draws, numerical
checks, audit histories, source hashes and summary figure data are under
`presentation_20260911/`. The earlier manuscript text is retained under
`overleaf/archive/presentation_before_20260911/`.

The revised PDF is `overleaf/output/presentation_review/ms.pdf`: 18 pages,
including appendices (main text and figures occupy pages 1–8). All five figures
and the complete rendered page sequence were inspected. References resolve on
the final TeX pass; no overfull boxes or duplicate labels remain. The fast test
suite passes 144 tests, with three heavy/network tests deselected and one existing
Matplotlib deprecation warning. New checks cover censoring-preserving empirical
quantiles and the amplitude-draw conditional mean. No commit, push or remote
Overleaf synchronization was performed.
