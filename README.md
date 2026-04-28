# EuclidPolish

We use the POLISH algorithm (https://arxiv.org/abs/2111.03249) on Euclid data.

The super-resolution neural network and the code to train it comes from the WDSR implementation found here:
https://github.com/krasserm/super-resolution

## Image generation pipeline

The pipeline takes raw sky simulations through to training-ready TFRecords. Every step is designed to match the polish-pub reference implementation.

### Step 1 — Simulate clean HR sky

**Code:** `euclid_polish/sky/clean_generator.py` → `CleanGalaxySimulator.simulate_field()`

1. Create an empty GalSim image at 0.05 arcsec/pixel (Euclid VIS HR scale).
2. Sample the number of galaxies and stars from Poisson distributions based on source densities (40 gal/arcmin², 2 star/arcmin²).
3. For each galaxy: draw a random COSMOS parametric profile at a random position.
4. For each star: sample a magnitude from a 3-bin distribution (faint 22–25, mid 18–22, bright 16–18), convert to flux via `10^(-0.4 * (mag - zeropoint))`, and place as a point source (single pixel).
5. Return the image as a `SkyImage` (raw flux, float32, arbitrary range).

#### Galaxy profiles: COSMOS parametric catalog

We use `galsim.COSMOSCatalog` with `gal_type="parametric"`. The catalog contains parametric fits to ~87,000 real galaxies observed by HST in the COSMOS field. For each galaxy, GalSim fits a **bulge + disk decomposition**:

- **Bulge component:** de Vaucouleurs profile (Sersic with n = 4) — a concentrated, round light distribution describing the galaxy's central region.
- **Disk component:** exponential profile (Sersic with n = 1) — a more extended, often inclined distribution describing the galaxy's outer disk.

Each component has its own half-light radius, flux fraction, ellipticity, and position angle, all fitted from real HST imaging. When we call `catalog.makeGalaxy(gal_type="parametric")`, GalSim returns the sum of these two Sersic components with parameters drawn from the real galaxy's fit. The result preserves realistic bulge-to-disk ratios, sizes, shapes, and inclinations of observed galaxies, even though fine morphological details (e.g. spiral arms) are smoothed out by the parametric approximation.

Galaxies are drawn with `method="no_pixel"` — the profile is evaluated analytically at each pixel centre without convolving with the pixel response function. This gives the true surface brightness sampled at our HR pixel grid.

**Output:** `clean_train.tfrecord`, `clean_validate.tfrecord` — raw flux, float32, one image per Example.

*polish-pub uses a much simpler model: each galaxy is a single 2D Gaussian ellipsoid with random flux (broken power-law), size (Gamma distribution, median ~1 arcsec), ellipticity (Beta(1.7, 4.5)), and orientation (uniform). No bulge/disk decomposition or real morphology — just smooth blobs. See `scripts/polish-pub/simulation.py`.*

### Step 2 — Extract PSF from real Euclid data

**Code:** `euclid_polish/euclid/psf_extractor.py` → `PSFExtractor.build_epsf()`

1. Query the Euclid archive for bright isolated stars.
2. Download VIS cutouts centred on each star.
3. Validate cutouts (NaN, constant, WCS position match).
4. Extract a centred sub-image from each cutout, normalise to sum = 1, wrap as `EPSFStar`.
5. Stack all stars with `photutils.psf.EPSFBuilder` (iterative centroiding, oversampling = 2).
6. Save the resulting ePSF kernel to a FITS file with pixel scale, FWHM, and oversampling in the header.

The PSF kernel must be sampled at the **same pixel scale** as the HR images (0.05 arcsec/pixel). A pixel-scale mismatch raises an error at convolution time.

*polish-pub equivalent: uses a pre-computed PSF FITS file (e.g. `dsa-2000-fullband-psf.fits`) or a synthetic Gaussian kernel.*

### Step 3 — Generate dirty LR images (HR → LR convolution)

**Code:** `euclid_polish/sky/psf_convolution.py` → `PSFConvolution.process_hr_to_lr()`

For each clean HR image, the following six operations are applied in order. This matches the exact sequence of `convolvehr()` + `create_LR_image()` in polish-pub:

```
                  raw HR (float32, arbitrary flux)
                            │
              ┌─────────────┤
              │             ▼
              │   1. Add Gaussian noise
              │      std = noise_fraction × |mean(pixel values)|
              │      (default noise_fraction = 0.1)
              │             │
              │             ▼
              │   2. FFT convolve at full resolution
              │      scipy.signal.fftconvolve(noisy_hr, psf, mode='same')
              │             │
              │             ▼
              │   3. normalize_data → uint16 [0, 65535]
              │      (min-max normalise the full-res convolved image)
              │             │
              │             ▼
              │   4. Stride downsample (on uint16 data)
              │      data[N//2::N, N//2::N]  (default N = 2)
              │             │
              │             ▼
              │   5. normalize_data → uint16 [0, 65535]
              │      (re-stretches range; downsample may have
              │       dropped the brightest/faintest pixels)
              │             │
              │             ▼
              │        dirty LR norm
              │
              ▼
   6. normalize_data(noise-free HR) → uint16 [0, 65535]
              │
              ▼
         clean HR norm
```

**`normalize_data`** (matches polish-pub exactly):
```python
data = data - data.min()
data = data / data.max()      # zero-div guard if max == 0
data = data * 65535
data = data.astype(np.uint16)
```

**Why double-normalise LR?** After stride downsampling (step 4), the min/max pixels of the full-res image may not survive. Re-normalising stretches the downsampled image back to the full [0, 65535] range, matching what polish-pub does.

**Output files** (per subset, stored as TFRecords in `data/images/records/`):

| File | Contents | Used for |
|---|---|---|
| `clean_{subset}.tfrecord` | Raw float32 HR (original flux) | Inspection, re-generation |
| `clean_{subset}_norm.tfrecord` | Normalised HR, uint16-valued float32 [0, 65535] | Training target |
| `dirty_{subset}_norm.tfrecord` | Normalised LR, uint16-valued float32 [0, 65535] | Training input |

*polish-pub difference: fixed noise std = 5 instead of fraction-of-mean; default rebin = 4 (ours: 2). Saves as uint16 PNG pairs instead of TFRecords.*

### Step 4 — Train

**Code:** `euclid_polish/training/trainer.py`, `euclid_polish/training/data.py`, `euclid_polish/training/models/wdsr.py`

**Model:** WDSR-B — 32 residual blocks, 32 filters, expansion factor 6, weight normalisation (`tensorflow_probability`), pixel-shuffle upsampling.

Internal normalisation: the first layer maps `[0, 65535] → [-1, 1]` via `(x - 32768) / 32768`; the last layer inverts it. All convolutions operate in this normalised space.

**Data pipeline:** load `*_norm` TFRecords (already [0, 65535]) → cache → shuffle(200) → random crop (96×96 HR patches, 48×48 LR) → repeat → batch(16) → prefetch. Rotation and flip are disabled because the Euclid PSF is asymmetric.

**Loss:** MAE in [0, 65535] pixel space (Keras `MeanAbsoluteError`, `SUM_OVER_BATCH_SIZE` reduction).

**Optimiser:** Adam, lr = 1e-3 decaying to 5e-4 at step 200k.

**Checkpointing:** save best 3 by validation PSNR (`save_best_only=True`). PSNR is computed on float32 (clip + round, no uint16 cast) to avoid dtype-mismatch scaling in `tf.image.psnr`.

*polish-pub differences: uses random crop + random 90° rotation + random flip for augmentation; default batch size 4 (ours: 16); default scale 4 (ours: 2); `tensorflow_addons` WeightNormalization (ours: `tensorflow_probability`); pipeline order batch → repeat (ours: repeat → batch). Also has EDSR, SRGAN, and WDSR-A architectures — we only use WDSR-B.*

### Step 5 — Inference

**Code:** `euclid_polish/training/inference.py`, `euclid_polish/training/models/common.py`

1. Load model from checkpoint or `.h5` weights.
2. Input must already be in [0, 65535] (the `*_norm` TFRecords or a uint16 PNG satisfy this).
3. Feed to model → clip to [0, 65535] → round → cast to uint16.
4. No per-image normalisation at inference time.

`resolve_single` matches polish-pub's `resolve16`: clip → round → uint16.

## Key differences from polish-pub

| | polish-pub | EuclidPolish |
|---|---|---|
| Domain | Radio (DSA-2000) | Optical (Euclid VIS) |
| Galaxy profiles | Single 2D Gaussian ellipsoid per source | Bulge + disk (two Sersic components from COSMOS fits) |
| Pixel scale | 0.25 arcsec/pixel | 0.05 arcsec/pixel |
| PSF | Pre-computed / Gaussian | Empirical ePSF from real cutouts |
| Noise std | Fixed (5) | 10% of per-image mean flux |
| Default scale | 4x | 2x |
| Normalization order | normalize full-res → downsample → normalize again | Same (matching polish-pub) |
| Data format | uint16 PNG + TF cache | TFRecord (uint16-valued float32) |
| Augmentation | Crop + rotate + flip | Crop only (asymmetric PSF) |
| Pipeline order | batch → repeat | repeat → batch |
| Weight norm lib | tensorflow_addons | tensorflow_probability |
| Inference output | uint16 (rounded) | uint16 (rounded), same |
