# Literature comparison only — scope fixed 2026-09-09

The user has dropped ANN implementation entirely. No rate-emulator training,
architecture reconstruction, external weights or author contact is required.

Source previously reviewed: [Riley & Mandel (2023), author preprint](https://arxiv.org/pdf/2303.00508),
DOI 10.3847/1538-4357/accf90, sections 2.1–2.5 and 3.2.

## Comparison to include

They used 1296 parameter combinations to train a reusable rate-matrix emulator,
with bootstrap realizations at each combination. We train a catalogue-specific
likelihood surrogate and analytically marginalize the amplitude parameter.
Their chosen training design is not a demonstrated minimum requirement.

Report our measured label budget at a stated posterior-accuracy threshold, and
contrast it descriptively with the published design. Do not equate bootstrap
realizations with distinct parameter locations, or different dimensionalities and
emulation targets with a controlled algorithm comparison. The linear amplitude
identity can reduce distinct shape integrations; training rows alone do not
measure irreducible computational cost.

## Proposed manuscript wording

Our surrogate targets the likelihood for a particular observed catalogue, reducing
the scope of the emulation problem. We assess training requirements directly
through agreement with the corresponding finite-population posterior. Riley &
Mandel (2023) used a larger parameter-space training design for a reusable forward
emulator. Differences in target, dimensionality, reuse and validation prevent a
direct computational-efficiency comparison based on training counts alone.

Insert the confirmed numerical budget and its observed pass/failure rates only
after the fresh-catalogue experiment finishes. Use "used" or "sufficient in the
tested experiments", not "needed" as a universal minimum.

A fiducial simulated 0.1-year scenario is already within our tested scope. A
1-year scenario, if included to align more closely with the paper's examples,
requires a new direct-reference validation; its sharper posterior cannot inherit
the small-catalogue result. Preserve the stated grid and selection differences.
No actual LVK analysis or additional COMPAS population is required.
