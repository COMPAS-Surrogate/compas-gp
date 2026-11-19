# Completed paired count-history analysis

Publication update: the manuscript now uses these results in Figures 2–5;
see README.md. The figure recommendation below records the initial review
before the user authorized manuscript integration.

All 70 jobs completed; saved outcomes: {'accurate': 70}.
There are 35 independent latent catalogues, each viewed with perfect and uncertain
measurements, at one fixed fiducial injection (five pairs per expected count).
All 2870 checkpoints are present. All 70 frozen shadow stops pass the direct
numerical, shape and four-parameter accuracy checks. The largest directed KL
at a stop is 0.007082444 nats (limit 0.01).
The four-parameter checks require mean/endpoint shifts below 0.2 reference SD
and relative changes in both 68% and 95% interval widths below 10%.

## Evaluation costs

Medians across five catalogues per cell; total includes fresh fixed96 audits.

| Expected events | Perfect training / total | Uncertain training / total |
|---:|---:|---:|
| 10 | 238 / 334 | 238 / 334 |
| 50 | 298 / 490 | 298 / 490 |
| 100 | 358 / 646 | 298 / 490 |
| 200 | 358 / 646 | 358 / 646 |
| 500 | 358 / 646 | 358 / 646 |
| 700 | 358 / 646 | 478 / 958 |
| 1000 | 358 / 646 | 418 / 802 |

Training costs range from 238 to 478;
total policy costs range from 334 to 958.
The stopping comparisons range from 6 to 14
and correspond to 58 + 30 times the comparison count training labels.
Uncertain measurements stop earlier in 9 paired catalogues,
at the same checkpoint in 18, and later in 8.
Thus these runs do not support a general claim that measurement uncertainty
reduces acquisition cost. Costs rise modestly with count, with considerable
catalogue scatter and no clean monotonic trend at the larger counts.
Audits contribute 28.7%–50.1%
of policy likelihood calls. Continuing every trajectory to 1258 training labels
is diagnostic work, not the policy stopping cost. Offline scoring, legacy
monitoring audits and reference construction are excluded from policy totals.
These label counts do not measure end-to-end speedup.

## Exceptions and evidence limits

- Numerically unresolved checkpoint: t0_n1000_uncertain_c204_bo, 58 training labels.
- Late reference rejected: t0_n500_perfect_c203_bo; endpoint maximum directed KL = 0.0010093169 nats (required <0.001).
- Numerically unresolved checkpoint: t0_n700_uncertain_c203_bo, 58 training labels.

69/70 continued GPs qualify as late references. The rejected endpoint still
passes the ordinary accuracy checks but misses the stricter late-reference KL
threshold. It is retained in both figures using the direct posterior target.
The two unresolved checkpoints are initialization states; none is an accepted
stop. Final self-KL is never evidence of accuracy. The direct-reference final
maximum directed KL ranges from 1.042e-05 to 0.001009 nats.
The stops were replayed and frozen before reference scoring; this is evidence
for this policy in a fixed-injection experiment, not prospective online timing,
prior-wide calibration, or a population-mismatch robustness test. Five pairs
per count cannot establish a small population-wide failure probability.
There is no random-acquisition comparator in this new campaign.

## Figure recommendation and captions

Current manuscript Figure 3 is `posterior_stop_uncertain.pdf`, paired with
Figure 2's perfect-measurement examples. Do not replace only that figure with
different catalogues while retaining its claim of matched latent events.
The new ensemble figure is a candidate addition or coordinated redesign of
the convergence presentation. Manuscript source and figures are unchanged.

**ensemble_convergence_candidate.pdf:** Symmetric KL to the direct numerical
posterior through all 40 posterior comparisons. Panels run left to right,
top to bottom through expected counts 10, 50, 100, 200, 500, 700 and 1000.
Each panel contains five paired catalogues: blue solid for perfect measurements
and orange dashed for uncertain measurements. Thin curves show every catalogue;
thick curves show medians. Open circles/squares mark the recorded shadow audit
stops; crosses mark unresolved numerical checks. Panel limits are shared.
The reference is the same direct target throughout each trajectory, including
the endpoint; comparison count excludes the 58-label initialization, and audit
costs are reported separately. Thin curves are catalogue variation, not
confidence intervals. No plotted reference value was used to select a stop.

**stopping_summary.pdf:** Each symbol is one catalogue's recorded stop, with
short horizontal strokes showing medians. Panels show comparison count,
training plus fresh audit labels, and maximum directed KL over both directions
and independent integration designs. This maximum differs from the symmetric
KL in the convergence figure. Expected counts are equally spaced categories.

Reproduce with `.venv/bin/python docs/studies/operational_stopping/gpry_compas/analyze_count_history.py
--root docs/studies/operational_stopping/gpry_compas/count_history_20260912 --output NEW_OUTPUT_DIRECTORY`.
`snapshot.json` records input membership and outcome/history hashes;
`stops.csv` contains every plotted stop. Validation reads saved metrics and
frozen decisions; it does not rerun the exact likelihood or numerical quadrature.
