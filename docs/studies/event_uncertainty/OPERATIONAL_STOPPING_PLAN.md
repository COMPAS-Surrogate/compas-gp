# Stopping without a direct posterior reference

## First validation result

The first frozen candidate has now been tested on 25 independent toy
realizations, with references revealed only after each decision. It produced
11 accurate stops, 5 premature stops (all narrow-secondary-mode cases), and
9 resource-limited non-stops despite accurate final posteriors. It is not
validated for general use. See
[the complete study](../operational_stopping/README.md) and
[plots and results](../operational_stopping/report_v1/RESULTS.md).
Two actual COMPAS catalogue pilots are recorded separately in that study.
The original proposal below remains for provenance; it must not be read as a
recommendation to deploy the tested v1 rule unchanged.

The user correctly identified that reference KL cannot be an operational stop
when direct inference is unaffordable. The current convergence campaign is a
reference-assisted benchmark: retain its records and label its evaluation
requirements accordingly. It does not validate an operational stopping rule.

## Recommended candidate rule

Combine two tests, sustained over consecutive batches:

1. **Predict new evaluations accurately before refitting.** Save the current GP
   prediction at each newly requested coordinate, then compare it with the
   acquired exact log likelihood before adding that batch to training. These
   evaluations are already paid for. Use errors in inverse-transformed log-
   likelihood units, not errors in compressed GP space. Judge errors where
   posterior mass lies and test whether the GP has dismissed a genuinely
   high-likelihood point. A data-independent additive log-likelihood offset has
   no effect on the posterior; pure offsets must not be confused with shape
   error. Retain all training points, including far tails.
2. **The inferred posterior is stable.** Compare successive GP posteriors using
   GP-only samples: means and credible intervals, with joint-distribution change
   as a supplement. Repeated sampling must resolve the tolerance before a small
   change is accepted. Do not approximate a multimodal posterior solely by its
   mean/covariance or reuse samples indefinitely without exploration checks.

When both suggest stopping, use a small fresh validation batch from the GP
posterior plus a broad-prior exploration component. Score it before training on
it. This checks beyond adaptive acquisition locations without constructing a
reference posterior. Count these evaluations in the final method budget. If
validation reveals a material change, add those evaluations and resume learning;
any later final audit uses fresh points. This audit is a finite-sample check,
not a proof that every mode has been found.

The existing stability tolerances (0.05 posterior SD for mean/endpoint changes,
3% for 68% widths) are candidate starting values, not demonstrated guarantees.
Prediction-error thresholds, validation batch size and consecutive-check count
must be frozen after a small development study and before confirmation.

## Where GP uncertainty fits

Comparing posteriors from correlated GP function draws with the mean-function
posterior would quantify the GP's own uncertainty about inference. It is useful
as a diagnostic if calibrated, but should not be the only gate: an overconfident
GP can produce little variation while being wrong. Draw functions in transformed
space and invert each function; independent Normal log-likelihood draws at each
sampler call are not the same construction. This optional diagnostic should not
become a prerequisite for testing the simpler prediction-and-stability rule.

## Bounded validation study

- Start with a small predeclared development set of saved trajectories, checking
  pre-refit errors against the next batch's labels. Replay the stopping decision
  using only information available then. No reference score, truth-informed
  proposal, reference rate or reference-derived posterior sample may enter that
  decision. Existing reference-assisted trajectories are useful retrospective
  diagnostics but may end before the operational rule would stop.
- Run a separate small pilot with GP-only posterior sampling and explicit audit
  costs. Freeze a candidate rule and evaluate it on fresh catalogues and seeds.
- Only after each operational stopping decision, reveal the independent direct
  reference to measure premature-stop frequency and posterior error. References
  are an offline validation instrument, not part of deployment or its budget.
- Report actual evaluation requirements as distributions across catalogues,
  including audit calls and unresolved cases. Do not replace resource limits
  by successes or impose 738 as a scientific stopping count.

No running inference source or threshold has been changed by this proposal.

## Sources

- El Gammal et al., GPry, section 4.3: prediction at newly acquired points is
  checked before refitting. https://arxiv.org/pdf/2211.02045
- GPry documentation recommends combining CorrectCounter with posterior
  stability: https://gpry.readthedocs.io/en/latest/module_convergence.html
- Järvenpää et al., section 4: GP uncertainty induces a distribution over
  normalized posteriors. https://arxiv.org/pdf/1905.01252

The added fresh audit and this validation design are recommendations for our
pipeline, not a claim that GPry validates our model or thresholds.
