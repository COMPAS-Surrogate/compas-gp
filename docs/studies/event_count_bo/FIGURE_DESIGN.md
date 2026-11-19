# Manuscript figure redesign

- KL convergence: two-by-two shared-axis log plots; BO blue solid / Sobol orange dashed; median and 10th--90th catalogue percentiles at five actual checkpoints. Ten catalogues per exposure; no smoothing or claim of a median confidence interval. Numerical references stay independent of acquisition.
- Contraction: one panel, four parameter colors/line styles, normalized central 68% widths; BO only because this addresses information content, not method comparison.
- Calibration: one coverage panel with four parameter curves and a per-parameter simultaneous 95% DKW envelope. Converted from posterior ranks via abs(2u-1), appropriate for central equal-tailed intervals.
- Predictive counts: keep three preselected cases; mirror spines and inward ticks.
- Remove scatter-location and crowded first-realization recovery plots from the active manuscript. Existing diagnostic assets remain available but are not evidence of efficiency by themselves.

All figures are title-free, with descriptive captions, vector PDF exports, consistent typography and mirrored spines/ticks. Compact legends identify curves. Re-render and inspect figures and compiled manuscript. No inference results or thresholds changed.

## Further revisions

The corner/surface illustration now uses 32 and 128 initial design points and the saved 482-point BO stage. The small fits contain no boundary anchors or BO acquisitions; this is an initialization diagnostic, not a new matched-budget acquisition result. Predictive catalogues are overlaid in two coordinate panels with matched marker/curve colours and a catalogue legend. The supplemental random-acquisition baseline shares the common initialization and then uses independent uniform prior draws.

Essential evidence for the main scientific claim: acquisition accuracy versus evaluation budget, one posterior-evolution illustration, calibration, and posterior predictive checks. The missing cost comparison should measure total online time (model evaluations plus GP fitting/acquisition) at matched accuracy. Surface and corner figures overlap in purpose; one can move to the appendix during final trimming. Additional decorative figures are not a priority.
