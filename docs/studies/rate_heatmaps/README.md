# Rate maps for the manuscript

Question: how do the cosmic-history parameters change the predicted detected
mass-redshift distribution? Six one-parameter-at-a-time maps compare alpha,
sigma and d_SFR at two values each, holding the other parameters fixed at the
injection. The SFR amplitude is omitted because it only multiplies all rates.

Static Matplotlib heatmaps use the existing 58-by-15 inference bins and the
full sixteen COMPAS blocks (512M initial systems), with the fixed O3 selection
approximation. A common logarithmic inferno scale encodes detected rate density
per unit **linear** mass and redshift. The vertical mass axis is logarithmic;
this does not change the density's measure. Black means exactly zero rate and has a separate legend entry. The darkest
8% of inferno is reserved away from positive values to distinguish near-zero
positive rates from zero. The lower
colour extension includes all positive values below the displayed scale.
No occupied bin is cropped, interpolated or smoothed. All four spines and
mirrored inward ticks are retained; parameter values are in the paper caption.

Reproduce from repository root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/compas-grid-matplotlib .venv/bin/python docs/studies/rate_heatmaps/plot.py
```

`rate_maps.npz` contains parameters, bin edges and unnormalized per-bin rates.
`summary.json` records the total rates, colour limits and plotting-source hash.
The PDF/PNG outputs are `overleaf/figures/bo_study/rate_heatmaps.*`.

Checks performed by the script: bin-density integrals recover total rates;
omitted zero-mass/infinite guard bins have zero rate; the full fiducial map agrees
with the sum of sixteen separately evaluated blocks to rtol 1e-12; tripling the SFR
amplitude triples the rate map to the same tolerance. These are model
illustrations, not a new inference or independent population-convergence test.

The inference studies retain their original 256M population. Only these physical
illustrations use the complete 512M archive.
