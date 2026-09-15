# Cosmic integration illustrations (drafts)

## Current physical draft

`plot_physical.py` builds the replacement scientific figure using actual
star-formation weights and selected binaries from the full 512M archive. It produces
four panels: formation, delay/metallicity yield, integration geometry, and
the change in observer-frame rates due to O3 selection. Mathematical definitions,
units, assumptions and a proposed caption are in `physical_caption.tex`.
The figure has only (a)--(d) panel labels, with no titles or bottom equations.
The manuscript now includes the figure and a second mass--redshift rate-grid
figure in `overleaf/sections/cosmic_integration.tex`. That appendix contains the
active captions, equation references and the binary-to-cell summation.
`physical_caption.tex` is the earlier standalone draft, retained for history.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/compas-physical-figure \
  .venv/bin/python docs/illustrations/cosmic_integration/plot_physical.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/compas-physical-figure \
  .venv/bin/python docs/illustrations/cosmic_integration/plot_rate_grid.py
```

Outputs in `output/pdf/cosmic_integration_512M/`:

- `cosmic_integration_physical.pdf`: PDF with vector text/lines and rasterized meshes.
- `cosmic_integration_physical.png`: preview.
- `physical_arrays.npz`: numerical snapshot for reproducible restyling (`--replot`).
- `provenance.json`: parameters, population selection, normalization, source hashes,
  numerical checks and total observer-frame rates.
- `rate_grid.npz` and `rate_grid_provenance.json`: the 58-by-15 fiducial grid,
  with a check that its mass marginal reproduces panel (d).

The second script publishes both figures to `overleaf/figures/bo_study/`.
It evaluates the full cached operator, divides cell rates by linear-mass and
redshift bin widths for display, and checks that the density integrates back
to the total rate. No nonzero cell is omitted from the map.

The loader streams the large system and common-envelope groups instead of
loading the whole HDF5 file. It joins metallicities by unique binary seeds.
This is a physical model illustration at one parameter choice, not a population
convergence or inference result. The illustration uses the same full 512M archive as the previous rate maps;
it does not change the population used for inference.

The yield and formation-contribution histograms retain all 1,326,573 selected DCOs.
They conserve their respective integrated weights. Panel (d) evaluates the
original operator at individual binary delays; it does not use those display
histograms. The complete-selection comparison retains volume/time conversion
and sets only detection probability to unity. The O3 rate agrees with the
sum of all sixteen independent cached block calculations after matching sampling normalization.

## Superseded conceptual draft

The original TikZ figure below was rejected as too simple and decorative. It is
retained as draft history; the physical version above is the current proposal.

Standalone TikZ artwork for discussion; not included in the manuscript.
The source follows `overleaf/sections/bayesian_model.tex`. No model evaluations
or COMPAS data are required to build it.

From the repository root:

```bash
mkdir -p output/pdf/cosmic_integration
pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory=output/pdf/cosmic_integration \
  docs/illustrations/cosmic_integration/cosmic_integration.tex
pdftoppm -scale-to 1800 -singlefile -png \
  output/pdf/cosmic_integration/cosmic_integration.pdf \
  output/pdf/cosmic_integration/preview
```

Requires a LaTeX installation with TikZ, geometry and amsmath. The fixed-size
article wrapper avoids requiring the standalone class. Output is a one-page
vector PDF suitable for later inclusion with `\includegraphics`.

## Intended reading

1. Existing binaries are weighted by metallicity-specific star formation.
2. The delay distribution connects formation times to merger epochs. Three
   example histories reach one epoch; the integration repeats over merger time.
3. Volume, observer-time conversion and detection probability produce detected
   rates, which become expected counts after multiplying by observation time.

All binary symbols, timelines, the formation curve and the analytic rate map
are illustrative. Colours distinguish example binary histories, without a
numerical metallicity scale. The selection expression is a schematic list of
factors, not the full integral or a claim that selection depends only on redshift.

## Draft caption

Cosmic integration maps a fixed simulated binary population to predicted
detected rates. Binaries are weighted by the metallicity-specific star-formation
history at formation, and their delay times determine the merger epoch.
Different formation times can contribute to the same merger epoch. Integrating
over the population and cosmic history, with comoving volume, source-to-observer
time conversion and detector selection, yields the detected rate in each
chirp-mass and merger-redshift bin. Multiplication by the observation time gives
the expected number of detections. All graphical distributions are schematic.
