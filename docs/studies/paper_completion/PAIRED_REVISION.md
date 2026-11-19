# Paired manuscript revision — 12 September 2026

## Event weights and inferred rate maps

Figure 1 now displays `W_b = sum_i C_ib / sum_c C_ic` for the paired 30-event
pilot. Each event contributes unit total weight. The uncertain-event map is a
display of likelihood information on the native active bins, not population
posterior membership probabilities, and is never substituted for Poisson counts.
Three latent positions are selected by SNR rank for illustration. The former
scatter figure remains on disk and in the source backup.

Figure 4 now shows the posterior median expected counts per bin `T*r_b` and
the relative half-width of each bin's central 90% posterior interval. It uses
the same axes, native bins and count colour limits as Figure 1. A solid black
boundary marks the injected highest-density bin union containing at least 90%
of detected events (90.12% on this grid). This is an injected population region,
not a parameter-posterior contour. Replicated-catalogue CDF/count checks remain
available in the appendix, separately from the rate-matrix uncertainty.

Reproduce with:

```sh
MPLCONFIGDIR=/tmp/compas-paired-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python docs/studies/paper_completion/weight_rate_figures.py
```

Outputs, input hashes, numerical summaries and PNG/PDF figures are in
`docs/studies/paper_completion/weight_rate_review/`. Each measurement model uses
20000 posterior draws including the exact conditional amplitude. The maximum
directed GP-reference KL is 0.000160; splitting the draws in half changes
displayed-bin interval endpoints/medians by less than 2% of the full-sample
median. This checks reconstruction and Monte Carlo resolution, not calibration.
Manuscript figure copies were verified byte for byte. No campaign source or
training data changed; the ensemble continued advancing during this work.

The revised PDF is `overleaf/output/weight_rate_review/ms.pdf` (23 pages).
Figure 1 on page 5 and Figure 4 on page 9 were visually checked after compiling.
No unresolved references; both Git whitespace checks pass. The full fast suite
passed 156 tests, with three deselected and one existing Matplotlib warning.
Pre-edit sources are in `overleaf/archive/weight_rate_before_20260912/`.

## Completed now

- Added the normalized SNR law, Euclidean derivation, Schutz (2011) and Chen &
  Holz (2014) citations, and the Riley & Mandel (2023) width/threshold motivation.
  Distinguished conditional measurement precision from a self-consistent
  cosmological SNR prediction, and log-coordinate errors from LVK posteriors.
- Corner legends identify actual expected counts and measurement models.
  Injected values are solid black lines; posterior contours retain their styles.
- Added an illustrative 30-event paired-data figure to the manuscript. Its source
  is the new pilot catalogue with expected count 30, generated before training.
  The five likelihood contours are selected by SNR rank, not visual agreement.
- Corrected the historical ensemble description: BO uses Sobol candidates;
  random continuation draws independent uniform coordinates.
- Review PDF: `overleaf/output/paired_revision_review/ms.pdf`, 20 pages.
  Compiled with Tectonic and visually checked the revised methods/data/corners.
  Extracted PDF text contains no unresolved-reference markers.
- Tests: `MPLCONFIGDIR=/tmp/compas-paired-mpl OPENBLAS_NUM_THREADS=1
  OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q` passed 153 tests, with three
  deselected and the existing Matplotlib deprecation warning. Both Git diff
  whitespace checks pass. No commits, pushes or Overleaf synchronization.

The previous manuscript sources and corner PDFs are preserved in
`overleaf/archive/paired_revision_before_20260912/`.

## Running campaign

### Methods framing and predictive update

The abstract now leads with the population-inference motivation, adaptive GP
method, and COMPAS demonstration. It contains no raw numerical results,
citations, or implementation references. Any eventual relative efficiency claim
should describe model-evaluation savings and its comparator, not imply a
measured end-to-end runtime gain.

The pilot completed all 20 BO/random trajectories with reference-verified
convergence; its gate passed with no failures or random resource limits.
Posterior predictive plots are now included for expected counts 30 (main text)
and 1200 (appendix). Each uses 1000 replicated catalogues, full posterior
parameter draws, Poisson counts, and the matching measurement model. The blue
curves are medians of replicated empirical CDFs, not predictions evaluated at
the componentwise median parameter vector. Bands are pointwise 90% intervals.
The largest directed GP-versus-direct KL discrepancy for the reconstructed
predictive targets is below 0.0013. These fixed-injection examples provide
intuition; they do not establish generative calibration.

The manuscript copies of `posterior_predictive_30.pdf` and
`posterior_predictive_1200.pdf` come from the frozen campaign's `figures/`
directory. The JSON files there retain reconstruction checks and provenance.
The updated review PDF is
`overleaf/output/methods_predictive_review/ms.pdf` (22 pages). Its abstract and
both predictive figures were visually checked, and extracted text contains no
unresolved-reference markers. Sources before this update are backed up in
`overleaf/archive/methods_abstract_before_20260912/`.

The full ensemble is still running. The live snapshot during this update
contained 61 completed trajectories, all within the first expected-count and
measurement group. This incomplete, ordered subset must not be reported as an
ensemble-wide efficiency result. Campaign numerical sources remain unchanged.

Driver: `docs/studies/event_uncertainty/paired_campaign.py`.
Plan: `docs/studies/event_uncertainty/PAIRED_CAMPAIGN.md`.
Run: `docs/studies/event_uncertainty/paired_20260912/`.

The detached supervisor first runs 20 pilot trajectories: BO/random for
perfect/uncertain observations at 30, 100, 300, 900 and 1200 expected events.
Verified realized counts for pilot case zero are 30, 80, 331, 870 and 1174.
Perfect and uncertain latent events match exactly at each exposure.

After the pilot gate passes, the supervisor generates posterior predictive
plots at 30 and 1200 expected detections, then starts the 100-realization
ensemble. Pilot and ensemble case zero overlap and must not be double counted.
These plots and ensemble results were not complete when the initial manuscript
revision was compiled. Inspect `campaign_status.json`, phase logs and phase
`status.json` for live status; a saved running status alone does not prove that
a process remains alive after a machine restart.

Current corner figures remain the earlier 100/1000 expected-event operational
stopping demonstrations. The running campaign is the matched-count large-batch
acquisition experiment; it does not rerun the separate small-batch stopping rule.
Replace published result counts only after reviewing the new numerical outcomes.

## Rendering

`presentation_figures.py --plots-only` rebuilds the existing corners.
`paired_review_figures.py` formats the generated data figure for publication,
using the frozen campaign catalogue without changing scientific sources.
The latter suppresses crowded logarithmic minor tick labels; the campaign's
first automatically generated data preview retains the original default ticks.
