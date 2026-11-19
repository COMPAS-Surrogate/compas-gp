# Campaign snapshot

Captured 2026-09-12T15:29:40.804604+12:00. 35/70 completed jobs; outcomes: {'accurate': 35}.
The plots use the exact immutable membership and source hashes in `snapshot.json`.

- `progress`: every scheduled job, separated by measurement model. Numbers in
  completed cells are the recorded fixed96 audit stopping comparison counts.
  Grey cells are pending, not failed. Each iteration adds one paired catalogue
  at each expected count. This snapshot is not a live dashboard.
- `kl_late_reference`: symmetric KL (mean of the two directions, averaged over
  two independent numerical designs) between each GP and its continued
  1258-label GP. The final self-comparison is omitted. Each panel includes every
  completed case in its group; thin lines show cases and thick lines show their
  median. Open circles/squares mark recorded perfect/uncertain audit stops.
  These are shadow operational decisions, frozen before reference scoring.
- `kl_direct_reference`: the same trajectories scored against the independent
  direct numerical target, retaining the final checkpoint. Symmetric KL has the
  same definition. Crosses, if present, mark numerically unresolved checkpoints.
- `stopping_summary`: individual completed-case stops (open symbols) and medians
  (short horizontal strokes). Total labels include training and fixed96 audit
  calls up to acceptance, excluding diagnostic continuation, legacy monitor
  audits and offline reference costs. The right panel uses the maximum of both
  directed KLs over both designs, which differs from the symmetric trajectory
  metric. Expected-count groups are equally spaced categories in this figure.

Only completed cases are included in curves and stopping summaries. With just
two or three catalogues in most groups, the scatter is preliminary and does not
establish calibrated coverage or a final ensemble result. Catalogue pairing is
preserved; each expected-count setting has its own independent catalogues.
Late-reference acceptance flags, all outcomes, and full source hashes remain in
the snapshot. A late GP's agreement with itself is never treated as evidence.
The scientific prior, acquisition, stopping policy and manuscript are unchanged.
