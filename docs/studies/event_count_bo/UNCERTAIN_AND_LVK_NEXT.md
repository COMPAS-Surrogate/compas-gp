# Next study: event-level measurement uncertainty, then an optional LVK pilot

## Decision

Keep the main comparison BO versus independent random acquisition after shared
initialization. Retain the completed Sobol comparison in the appendix; similar
curves are not a demonstration of equivalence. Restore separate predictive rows.

The next scientific extension should be a Riley-inspired uncertain-measurement
simulation. The existing categorical-response calibration remains valid for its
stated model, but is not the same experiment. No LVK analysis has been executed.

## Literature and implementation findings

Riley and Mandel (2023), section 3.2.2 and equation 11, use SNR-dependent mock
mass/redshift measurement widths, stochastic shifts of posterior centres, and
an explicit mock event prior. They compare 58 and 578 events. Source:
https://arxiv.org/pdf/2303.00508 . Their real O3 illustration uses approximate
selection, reports prior-boundary sensitivity, and fails its predictive check
(section 3.2.3). These limitations motivate explicit checks here.

Current `sample_event_posterior` implements similar SNR-scaled Gaussian clouds,
but `_calculate_event_weights` just histograms them. The `mc_prior` and `z_prior`
arguments to `compute_population_weights` are not used in that event-weight
calculation. This is not a validated implementation of prior-weighted mock PE
posteriors and their subsequent likelihood recovery. Reusing a helper name does
not establish likelihood parity. Do not change this legacy behavior silently.

`posterior_grid_coefficients` explicitly supports inverse PE-prior density and
bin-area factors, but does not implement coordinate transforms or selection.
`LVKObservation` is an empty subclass. The existing operator returns detected
rates under one O3 sensitivity approximation, not the intrinsic event-rate
model needed for ordinary strain-likelihood recycling.

## Bounded proposed mock experiment (not yet run)

1. Use one fixed interior hyperparameter injection and the existing population.
   Generate Poisson catalogues at about 50 and 500 expected detections. Use the
   same latent events for each perfect/uncertain pair; retain Poisson count
   fluctuations and a declared exposure.
2. Define and normalize an event-level measurement model before making posterior
   samples. A useful Riley-inspired option uses log mass/log redshift errors
   with scales proportional to 12/SNR and an independently declared SNR law.
   Draw noisy measurements, including random centre offsets. This is a controlled
   conditional-on-detection sensor, not a claim of realistic LVK PE.
3. Evaluate the event integrals directly under that model. Generate mock event
   posterior samples using a declared PE prior, then recover the same integrals
   by dividing out that prior. Test two different PE priors: recovered population
   constraints should agree. Check event-sample effective sizes and sample-count
   convergence; integrate within coarse bins where measurement widths are small.
4. Compare BO and random acquisition at fixed budgets. Start with ten paired
   realizations per exposure as a pilot; retain existing categorical calibration
   separately. Use posterior recovery and posterior predictive checks. A broad
   P-P study follows only after this likelihood identity is verified.
5. Report the effect of measurement uncertainty and training-point allocation
   separately. Small samples can show noise fluctuations; broader uncertain
   posteriors in every single realization are not a mandatory pass criterion.

## LVK effort and scope

Public PE samples and sensitivity products exist. The GWPopulation pipeline's
GWTC-3 example documents both, but warns that its example version may be stale:
https://gwpopulation-pipe-colm-talbot-9fdcf850d287061fa1799025e6cb1a2da.docs.ligo.org/real-data.html
https://gwosc.org/GWTC-3/

A read/plot prototype is small work and needs no new strain parameter estimation.
A defensible population analysis is a separate likelihood/selection extension:

- Read a documented PE release, priors and parameter coordinates; handle source
  versus detector frame and luminosity-distance/redshift Jacobians consistently.
- Supply intrinsic population event integrals and a matched expected-detection
  count, using the same event selection and observing-time definition.
- Match exposure and selection to an observing epoch and search threshold.
  Injection reweighting needs sufficient support and effective sample sizes.
- Account for mass ratio and spin dependence consistently when reducing the
  population to chirp mass and redshift; their marginalization is an assumption.
- Retain or explicitly model unsupported/high-mass events. Arbitrary event
  removal changes selection and cannot be repaired by relabeling the sample.
- Validate prior and selection treatment with the mock experiment first, then
  compare GP and independent likelihood calculations and predictive checks.

A few events reduce compute, not these statistical requirements. For a low-cost
real-data pilot, choose one documented observing interval and an explicit event
selection. Do not cherry-pick compatible events and treat them as a complete
Poisson catalogue. An illustrative analysis under approximate selection is
possible but does not establish reliable astrophysical constraints.

Recommendation: finish the event-level uncertain mock and online-cost comparison
for the BO paper; decide on an LVK illustration after its likelihood passes the
mock prior-recycling and selection checks. This plan is not an LVK submission or
an authorization to change the scientific target silently.
