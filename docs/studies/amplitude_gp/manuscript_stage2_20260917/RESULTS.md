# Completed four-input manuscript campaign

- All 44 stage-two training jobs, 12 diagnostic jobs and the pilot completed with exit `0:0`. Nothing remains queued/running in these arrays. Training took about 11 minutes for the 30-event predictive examples and 37–60 minutes for the other new fits, including scoring.
- All **140 primary trajectories** are now available, plus eight repeated-seed sensitivity runs and four predictive examples. Continued trajectories count once.
- Stage two has **474 scored checkpoints**, all computationally complete and numerically resolved. Final posterior accuracy remains separate from job completion.

| Expected events | Maximum labels | BO confirmed / tested | Random confirmed / tested |
| --- | ---: | ---: | ---: |
| 10 | 1284 | 10/10 | 10/10 |
| 50 | 1284 | 10/10 | 7/10 |
| 100 | 1284 | 10/10 | 0/10 |
| 200 | 1284 | 10/10 | 0/10 |
| 500 | 1884 | 10/10 | 0/10 |
| 700 | 1884 | 7/10 | 0/10 |
| 1000 | 1884 | 6/10 | 0/10 |

- Totals: **63/70 BO** and **17/70 random** have confirmed passes under the original strict criterion. At 50 events, one additional random fit passes only at the final checkpoint, without confirmation.
- Each ten-run count/method group consists of five latent catalogues viewed with perfect and uncertain measurements; the paired views are not independent catalogues. Repeated-seed runs are excluded from this table.
- All seven final BO misses pass the mean, endpoint and width tolerances; their largest directed KLs are 0.010023–0.014661. Across these seven, mean shifts are below 0.016 reference SD and central 68%-width errors below 1.42%. Thus their marginal summaries agree well even though the original joint-KL criterion remains unmet. The threshold has not been changed after seeing results; no extra runs were launched merely to cross it.
- Random runs that did not confirm remain budget-censored. These data establish an advantage at the tested budgets, not the eventual number of random points required in the harder cases.

## Predictive examples

| Example | Final maximum directed KL | Confirmed |
| --- | ---: | --- |
| 1200 events, perfect | 0.012632 | No |
| 1200 events, uncertain | 0.006700 | Yes |
| 30 events, perfect | 0.000497 | Yes |
| 30 events, uncertain | 0.001144 | Yes |

- The 1200-event perfect-view example narrowly misses the KL gate (0.012632), while its largest 68%-width discrepancy is 0.76% and mean shift 0.0067 reference SD. Retain this qualification when making its predictive plots.
- Both 30-event views and the 1200-event uncertain view pass with consecutive-checkpoint confirmation.

## Available artifacts and next work

- Stage-two raw training records, checkpoints, optimizer/timing records and Slurm logs are retrieved into this directory. All remote originals remain preserved.
- `results_snapshot.json` retains the source-verified scoring attempts; `combined_trajectory_summary.csv` merges the earlier warm validation, stage one and stage two without double-counting parents. `slurm_completion.txt` records final scheduler states.
- The amplitude and held-out diagnostics are complete and already supply manuscript Figures 2 and 12. They retain predictive-interval undercoverage and the small conditional-amplitude discrepancies.
- The rest of the plots can now be generated from completed campaign outputs: BO/random budget comparison, widths versus event count, selected joint posteriors, rate maps and replicated catalogues. Retrieval does not itself generate these figures or change their captions.

## Download verification

Retrieval finished on 18 September 2026: 33,583 files transferred (about 121 MB of file content). All 474 scored checkpoint hashes match; all 2,556 production training checkpoints loaded successfully with the expected four-input dimensions, finite labels and coordinates inside the prior. All 57 scheduler records (pilot, training and diagnostics) report `COMPLETED`, exit `0:0`. Exact frozen source hashes also match. Remote originals remain unchanged.
