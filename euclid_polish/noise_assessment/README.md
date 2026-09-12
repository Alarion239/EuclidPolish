# Four-band archive noise assessment

This independent diagnostic writes only under its output directory. It does not
activate a calibration, modify a dataset, invoke training, or change generation
or web APIs. The default output is
`data/population_comparison/noise_assessment/`.

## Reproduce the assessment

Use `EuclidPolishEnv` and run these stages **sequentially** from the repository
root. All archive requests are public, read-only IRSA Q1 requests. Network access
is needed only for acquisition. Retrying acquisition reuses checksummed products
and byte ranges and retries explicitly unavailable entries.

```sh
python scripts/measure_archive_noise.py init
python scripts/measure_archive_noise.py acquire-mer --pilot
python scripts/measure_archive_noise.py acquire-mer
python scripts/measure_archive_noise.py measure-mer
python scripts/measure_archive_noise.py select
python scripts/measure_archive_noise.py acquire-exposures --pilot
python scripts/measure_archive_noise.py acquire-exposures
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/measure_archive_noise.py measure-exposures
MPLCONFIGDIR=/private/tmp/euclid_noise_mpl python scripts/measure_archive_noise.py report
python scripts/measure_archive_noise.py status
```

`--output PATH` goes **before** the stage. `init` accepts paths to the original
44-position manifest, four-band parent manifest, and saved example provenance.
The original VIS cutouts and four example NPZs must be available for the initial
snapshot. `init` freezes the reference WCS and copies the four example arrays into
the output directory, so later measurement stages no longer depend on their
original workspace paths. The defaults match this checkout.

`manifest.json` freezes the sampling and archive identities; `exposures.json`
records native frame retrievals and exclusions; `summary.json` holds schema
version 1 measurements and recommendations. `report.html` embeds the figures and
measurement/provenance JSON and opens without a server. PNGs, NPZ arrays, sparse
operators, saved archive queries, FITS cutouts, and byte-range checksums accompany
it. Report rendering resumes from figures whose inputs, renderer and checksums
still match. A failure does not create a substitute noise value.

The survey measures one central 256×256 patch at each of the 44 saved positions
and the four exact displayed examples in all four bands. This is **192 patch-band
measurements**, not a rerun of all 4,400 original VIS subtiles. The nine exposure
validation positions use the minimum, upper-median, and maximum central official
VIS RMS in each field. Sample 31 is reserved for the separate bright-star stress
case, so these ten positions are distinct. A retrieval pilot is additional and
excluded from the validation summary unless it is one of the selected positions.

## Numerical definitions

- Official MER RMS is **total** uncertainty including source photon noise. The
  legacy statistic is the existing masked, plane-subtracted MAD; its implementation
  is called directly. Their ratio compares different estimands.
- Native product units remain in the arrays. A separate photometric comparison
  scale uses the canonical repository zeropoint conversion. RMS gets the same
  factor as science, variance its square. VIS native counts additionally use
  `EXPTIME` because `MAGZEROP` calibrates ADU/s. NIR `ZPAB` already calibrates
  integrated electrons; dividing it by exposure time again would be incorrect.
- Differences use disjoint native exposures. Supplied backgrounds are subtracted,
  full celestial WCS determines bilinear weights, and the WCS Jacobian supplies
  the pixel-area factor. Supplied effective PSFs are sampled through the actual
  bilinear operator at the patch center and cross-convolved. The default difference
  grid is 80×80 at 0.1 arcsec/pixel in VIS and 0.3 arcsec/pixel in NIR. This bounds
  sparse-operator memory and is separate from the larger MER survey patch.
- For each composed operator `L`, the covariance is represented as
  `C = L diag(native_variance) L.T`. The diagonal squares the **composed** weights.
  Aperture predictions use `sum(native_variance * (L.T @ aperture)**2)`. RMS maps
  are never interpolated as science. Native correlations absent from product
  metadata remain an explicit model limitation.
- `Z=(A-B)/sqrt(V_A+V_B)` statistics retain all valid differences, including source
  pixels and residual structure. Only documented INVALID bit 0 and incomplete
  transformation footprints are excluded. Negative fitted PSF lobes are retained,
  with their signed flux recorded; variance propagation still squares the weights.
- Reported quantities include mean, standard deviation, robust scatter, tails,
  lag covariance, aperture-sum variance, source-brightness strata, and correlations
  of residuals with brightness/gradients. Residual associations are diagnostic;
  they are not fitted away. DQ masks can leave too little coverage; this is an
  explicit failure/precision result.
- Readable pointing tables compare valid coverage, exposure count, bright/faint
  scatter, lag-one correlation and 3-pixel aperture variance. Their descriptive
  rank correlations do not establish causes or significance from nine pointings.
- Patch intervals use spatial blocks. Final intervals average pairs within a
  pointing, weight pointings equally, and bootstrap connected components of
  shared input exposures. Missing ancestry prevents an independence claim.
  The bright-star stress case is excluded from the representative mean.
- The predeclared **5% amplitude target** is met only if the full 95% interval lies
  inside `[0.95,1.05]`. Disjoint intervals imply disagreement; overlapping/wide or
  missing intervals imply insufficient precision. VIS PSF provenance limits
  prevent certification regardless of the numerical interval.
- Monte Carlo uses the actual retrieved time/zeropoint normalizations and bilinear
  weights, then takes a median in each draw. These are conditional predictions,
  separate from measured differences. Exact MER input selection/rejection is
  unavailable, so the workflow does not call them a validated MER reproduction.

## Limits established from the actual Q1 products

The public SIA association identifies each MER science/RMS/flag product by band,
archive publisher identity, and tile. The science filename must exactly match
the saved parent manifest. Release headers, dimensions, WCS, pipeline parent IDs,
and retrieved-byte checksums are checked before measurement. Full upstream source
checksums are not supplied; S3 multipart ETags are retained as identity metadata,
not mislabeled as MD5 checksums.

The tile-to-observation table supplies conservative shared-ancestry groups, but
does not identify exact per-pixel MER input layers, exposure rejection, or local
coverage counts. Native coverage and the retrieved candidate set are recorded.
No image-derived noise substitutes for absent RMS products.

The bright-star MER RMS maps contain large repeated, sentinel-like values even
where the propagated INVALID bit is clear. Their exact meaning is not established
by the queried metadata. The report retains them, records their maxima and area
fractions, and uses logarithmic RMS panels. The official median is therefore also
subject to product-quality limits and is not a background-noise ground truth.

NIR background FITS headers can retain earlier astrometry than their associated
science frames. The Q1 product definition explicitly describes matching native,
unresampled detector arrays. Background subtraction therefore uses matching
`DET_ID`, dimensions, and native pixel indices; the sky-WCS discrepancy is recorded.
Resampling the old background WCS onto the new science WCS would move the model
to the wrong detector pixels. VIS auxiliary WCS discrepancies are rejected.

The VIS PSF file contains 9×9 snapshots of 21×21 pixels but no snapshot coordinates.
The current mapping to equal-width detector cells is an explicit inference from
the documented Q1 PSF size and PSFEx grid convention. VIS differences are marked
**provisional**. Exact snapshot coordinates are needed for a certified replacement.
NIR supplies a constant, oversampled PSF per detector; its spatial/time limits are
also recorded. No source-free or temporal PSF model is invented.

Supplied background models do not themselves provide a decomposition into source
and background variance. Background-estimation covariance and native instrument
correlations are not specified by the RMS maps. Those limits remain visible.

## Focused verification

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m pytest --confcutdir=tests/noise_assessment tests/noise_assessment -q
python -m ruff check euclid_polish/noise_assessment scripts/measure_archive_noise.py tests/noise_assessment
```

Tests are isolated from web/training fixtures. They cover Gaussian and Poisson
exposures, unequal times, PSF/registration mismatch, correlated noise, exact
bilinear/convolution covariance, median Monte Carlo, units, coverage, and ancestry.
The saved four real examples are verified by `measure-mer`, with their exact
pixel and scalar checks reported under `saved_example`.

Scientific sources: [MER products](https://euclid.esac.esa.int/dr/q1/dpdd/merdpd/dpcards/mer_bksmosaic.html),
[VIS frames](https://euclid.esac.esa.int/dr/q1/dpdd/visdpd/dpcards/vis_calibratedquadframe.html),
[NIR frames](https://euclid.esac.esa.int/dr/q1/dpdd/nirdpd/dpcards/nir_calibratedframe.html),
[MER processing §3.2](https://arxiv.org/html/2503.15305v2#S3.SS2),
[IRSA guide](https://irsa.ipac.caltech.edu/data/Euclid/docs/euclid_archive_at_irsa_user_guide.pdf).
