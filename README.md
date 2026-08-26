# EuclidPolish

Super-resolution of **Euclid** VIS imaging toward Hubble-level resolution, using the
POLISH approach ([Connor+ 2022](https://arxiv.org/abs/2111.03249)) with a WDSR-A
([Yu+ 2018](https://arxiv.org/abs/1808.08718)) backbone, built on the reference
implementation at <https://github.com/krasserm/super-resolution>. The end goal is to
flag strong-lens **candidates** — especially the under-represented small-Einstein-radius
population that wide, low-resolution surveys miss.

The network is trained on **synthetic Euclid scenes** with a clean high-resolution
ground truth, optionally mixed with two real-data lanes (HST F814W and Euclid
star cutouts). The synthetic generator simulates four bands in raw electrons
(VIS + NISP Y_E / J_E / H_E), convolves each band with its own empirical PSF, applies a
per-band Poisson + read-noise model plus detector artifacts, and writes float32
TFRecords. All photometry is calibrated against the published Euclid AB zeropoints per
band; nothing is normalised to a unit interval before noise is injected.

Every field galaxy uses a TNG50 SKIRT morphology and retains its native
VIS/Y/J/H proportions. A sampled COSMOS2025 row supplies photo-z, stellar
mass, apparent half-light radius, and an HST F814W brightness anchor. A fitted
F814W→VIS transfer sets one shared four-band brightness scale.
`--galaxy-density-arcmin2` is fitted separately from Euclid cones.

**What the model learns.** The network input is the 4-channel LR stack
`(VIS, Y_E, J_E, H_E) @ 0.10″/pix`. The output is a 4-channel deconvolved HR sky
stack `(VIS, Y_E, J_E, H_E) @ 0.05″/pix`. Each training lane supervises the
corresponding output through its own instrument's forward operator (see §5).

---

## 1. Photometry: magnitudes → electrons

All sources are placed on the simulated HR plane in **electrons accumulated over the full
Wide-Survey stack** (4 dithered exposures × 565 s = 2260 s). This is the natural unit for
Poisson statistics: one electron is one detected photoelectron, and the variance of a sum
of independent counts is the sum itself.

### 1.1 Per-second AB zeropoint

The Euclid VIS instrument paper ([Cropper+ 2014](https://arxiv.org/abs/1608.08603), §2 and
Table 1; [Euclid Collaboration: Mellier+ 2024](https://arxiv.org/abs/2405.13491), §6.2) and
the Q1 release notes give the broadband VIS AB zeropoint as

```
VIS_AB_ZP_E_PER_S = 25.50          # m_AB of a source giving 1 e⁻/s
```

i.e. an AB-magnitude *m* source produces

```
F(m) = 10^(-0.4 · (m − 25.50))    [e⁻ / s]
```

at the detector after the full optical chain (mirrors, filter, QE).

### 1.2 Stack zeropoint (per-image flux scale)

Because we simulate the *coadded* Wide-Survey stack rather than a single exposure, we fold
the integration time into a stack-level zeropoint:

```
T_total = N_exp · t_exp = 4 · 565 s = 2260 s
SIM_VIS_ZEROPOINT_E = VIS_AB_ZP_E_PER_S + 2.5 · log10(T_total)
                    ≈ 25.50 + 8.385 = 33.885
```

Then, for any magnitude *m*:

```
flux_e(m) = 10^(-0.4 · (m − SIM_VIS_ZEROPOINT_E))   [e⁻ over the stack]
```

This single conversion rule is applied to every source class. It is the `sim_zeropoint_e`
property of each `BandConfig` (`euclid_polish/config.py`, the
`zp_e_per_s + 2.5·log10(t_total)` derivation) and the scalar `Config.SIM_VIS_ZEROPOINT_E`.
The mag→electron conversion itself lives in `euclid_polish/sky/multiband_generator.py`
(`10**(-0.4*(mag − band.sim_zeropoint_e))`).

### 1.3 Per-class application

Per-band electron flux for source class *X* in band *B*:
`flux_e_B = 10^(-0.4·(mag_B − band.sim_zeropoint_e))`,
where `band.sim_zeropoint_e` is each `BandConfig`'s stack-level zeropoint
(`Config.BAND_VIS.sim_zeropoint_e ≈ 33.885`, with the NISP bands at their own integration
budget — Y/J/H accumulate 4 × 112 s).

| Class | Magnitude source per band | Flux assignment |
|---|---|---|
| **Stars** (point sources) | VIS magnitude and four-band colour are sampled from the activated Q1/Gaia empirical stellar calibration: a finite-domain count law plus a magnitude-conditioned latent colour locus | Each band is deposited as a single HR-pixel electron flux (`sky_simulator.py:_deposit_star`); four-band magnitudes are persisted for fixed validate/test stars and redrawn per visit on-the-fly; the PSF is applied later by the forward model |
| **Galaxies** | The activated Euclid joint prior supplies circularized VIS Sérsic R_e and 2FWHM VIS brightness | A diversity-balanced TNG50 SKIRT donor is chosen only when at least one orientation is natively large enough, then area-downsampled to the requested R_e. Stamps are never enlarged. All four bands receive one shared brightness normalization so TNG VIS/NISP colours are unchanged. |
| **Strong lenses** | TNG subhalo masses and the lens geometry prior | SIE + external-shear mass model from lenstronomy; deflector and source light are both TNG stamps. |

The COSMOS2025 master catalog (COSMOS-Web v1.1,
[Shuntov+ 2025](https://arxiv.org/abs/2506.03243)) supplies ~784k galaxies with
photo-z, stellar-mass, HST F814W, and apparent-size measurements. UltraVISTA
Y/J/H columns are retained under their physical filter names for diagnostics,
but they do not replace the TNG Euclid-band colours.

### 1.4 TNG morphology rendering

The TNG atlas is a morphology library, not a population model. The active
Euclid joint prior independently samples each field galaxy's observed VIS
half-light radius and 2FWHM brightness. Scene generation then asks
`TNGAtlas` for a donor with a natively large-enough orientation and asks
`TNGRenderer` to produce the corresponding `RenderedTNG` electron image.

- **Field galaxies** use shrink-only observed-radius rendering: the selected
  `TNGSurfaceBrightnessImage` may be rotated and area-downsampled, but is never
  enlarged. One shared normalization matches the requested VIS brightness and
  preserves the atlas's four-band colours.
- **Strong-lens light** uses physical-redshift rendering. The intrinsic
  100 pc/pixel atlas grid is mapped to the assigned angular distance, followed
  by the configured compactness, Tolman-dimming, spectral-drift, and
  surface-brightness-cut transformations. The lens population and placement
  remain in `sky/generation`, outside the TNG domain.
- **Euclid observation** remains a later stage: TNG produces clean
  electrons/pixel on the high-resolution angular grid, while PSF convolution,
  detector sampling, sky, and noise live under `sky/observation`.

The typed TNG API lives in `euclid_polish.tng`; raw array kernels are private to
that package. The source files are still SKIRT radiative-transfer products, but
there is no separate `euclid_polish.skirt` software package.

### 1.5 Sky background

The diffuse Wide-Survey sky brightness
([Euclid Collaboration: Scaramella+ 2022](https://arxiv.org/abs/2108.01201), §3.4) is

```
SKY_MAG_AB_ARCSEC2 = 22.35   # mag AB / arcsec²
```

Converted to electrons per second per arcsec²:

```
SKY_E_PER_S_PER_ARCSEC2 = 10^(-0.4 · (22.35 − 25.50)) ≈ 18.20
```

The sky enters only at the noise stage — it is **not** added to the clean target. A 0.10″
LR pixel collects `18.20 · 0.10² · 2260 ≈ 411 e⁻` of sky over the stack.

---

## 2. PSF and HR → LR forward model

**Code:** `euclid_polish/euclid/psf_extractor.py` (extraction, band-agnostic),
`euclid_polish/euclid/psf_library.py` (per-band loaders, with Gaussian fallback),
`euclid_polish/psf/psf_set.py` (the per-band PSF *ensemble*),
`euclid_polish/sky/multiband_forward.py` (per-band convolution + rebin + noise).

### 2.1 PSF: a per-band ensemble, not a single kernel

The Euclid PSF varies across the focal plane, so each band is represented by a **`PSFSet`**
— an ordered list of K empirical PSFs, one per spatial cluster of stars. PSFs are built
with `photutils.psf.EPSFBuilder`
([Anderson & King 2000](https://ui.adsabs.harvard.edu/abs/2000PASP..112.1360A)) from bright
isolated stars in real Euclid cutouts, oversampled ×2 relative to the 0.10″ detector
(→ 0.05″/pix HR grid). Saturated cores and edge stars are rejected. Stars are clustered
**once** by sky position (K-Means++, ~400 stars per cluster—four times the former
default) on stars valid in all four bands, so cluster
index *ci* maps to the same field region in every band. The larger stacks halve
the approximate per-kernel noise amplitude; elastic warps supply continuous
exposure-to-exposure diversity despite the smaller regional set.

Each band is saved as a multi-extension FITS at `data/euclid_psf/euclid_psf_<BAND>.fits`:
`HDU[0]` is the **field-mean** PSF (so legacy single-PSF readers still work), and
`HDU[1..K]` are the per-cluster kernels with `RA/DEC/NSTARS` provenance headers. Missing
files fall back to a Gaussian PSF at each band's nominal FWHM, wrapped as a 1-element set.

At generation time, **one** PSF sample (cluster index + optional roll) is drawn per scene
and applied to all four bands (shared pointing). There is **no blending** of cluster
kernels. Roll-rotation augmentation is implemented (`PSF.rotated`) but **disabled by
default** (`MultiBandForwardConfig.psf_unrotated_prob = 1.0`): the default behaviour is a
random cluster pick with no rotation.

The forward model also samples a smooth elastic deformation of that PSF for each exposure
(`alpha ~ Uniform(0, 20)`, `sigma=3` HR pixels by default), following the original POLISH
PSF-distribution augmentation. Record-mode generation uses seeded, replayable draws for
dirty train/validate/test records. Clean-only on-the-fly training instead draws a fresh
train deformation on every visit; generated validation/test inputs remain fixed. One
deformation is shared across all four bands, and it changes only the forward PSF used to
make LR, never the clean/HR target. The probability, maximum alpha, and smoothing scale are
persistent controls on `/config`.

`differential_kernel.py` is a **separate** path used only by the HST→Euclid training lane
(§5): it solves `A ⊛ H ≈ E` (Wiener) so that convolving a real HST F814W cutout with `A`
yields a correct Euclid-PSF LR instead of double-convolving through HST's own PSF. It is
not part of the synthetic forward model.

### 2.2 Forward model

All four bands are delivered by the Euclid MER archive on a common **0.10″/pix** grid.
The deterministic optical signal remains on that grid because the empirical ePSF already
contains detector sampling and MER resampling. NISP stochastic noise is generated in four
native 0.30″ exposures and dither-resampled to reproduce the delivered covariance. From
clean HR (4-channel, 0.05″/pix) → noisy LR (4-channel, 0.10″/pix):

```
For each band B:
    hr_e_B = (psf_sample applied to psf_set[B]) ⊛ hr_4ch[..., B]      # HR plane, e⁻
    lr_e_B = sum_rebin(hr_e_B, factor=round(0.10 / 0.05) = 2)         # → 0.10″/pix
    lr_e_B = apply_archive_noise(lr_e_B, B, rng)                      # VIS native; NISP native-noise→MER
lr_4ch    = stack(lr_e_VIS, lr_e_Y, lr_e_J, lr_e_H)                   # all on the 0.10″ grid
lr_4ch    = apply_star_saturation(lr_4ch)                            # bright-star wells, on the dirty LR
hr_target = hr_4ch                                                    # 4 channels, clean, no noise
```

`sum_rebin` is **photometric**: each LR pixel is the *sum* of its (n × n) HR sub-pixels,
conserving total electron count. Only the stochastic NISP residual is Lanczos-resampled;
the optical signal is not blurred twice. Sparse CR/hot/dead residuals are injected after
that MER resampling, so they do not acquire PSF-like Lanczos wings.

---

## 3. Noise model

**Code:** `euclid_polish/sky/observation/noise.py` (`apply_archive_noise`),
`euclid_polish/sky/observation/artifacts.py` (detector-artifact injection),
`euclid_polish/sky/observation/saturation.py` (bright-star wells).

Per LR pixel, given the noise-free signal `s` (electrons of the source over the stack):

```
λ        = max(s + sky + dark, 0)
native   = Poisson(λ) − (sky + dark) + read       # per detector exposure
mer_noise = dither_resample(native − signal)      # NISP only
output   = signal + mer_noise + residual_artifacts # sparse final-grid survivors
```

This corresponds to the standard CCD noise budget — Poisson photon noise on the *total*
incident charge plus uncorrelated Gaussian read noise scaled by the number of independent
reads ([Janesick 2007, *Photon Transfer*](https://spie.org/Publications/Book/725073)). Sky
and dark are added before the Poisson draw so their shot noise is correctly captured, then
subtracted afterwards so the network's input represents the source signal alone. Detector
artifacts are injected between the Poisson draw and the read-noise draw — that ordering
matches the physical readout chain (charge integrates on-chip → ramp is read out → read
noise added). In the production forward model artifacts are **on** by default
(`MultiBandForwardConfig.add_artifacts = True`).

### 3.1 Constants (Cropper+ 2014, MSSL VIS-PP, Q1 docs)

| Symbol | Value | Source |
|---|---|---|
| `EXPOSURE_TIME_S` | 565 s | Cropper+ 2014, Table 4 (Q1 paper Romelli+ 2025 quotes 566 s nominal; we use 565 s) |
| `N_EXPOSURES` | 4 (Wide Survey dithers) | [Scaramella+ 2022](https://arxiv.org/abs/2108.01201), reference observing sequence |
| `READ_NOISE_E` | 4.5 e⁻ RMS / exposure (VIS); 7.5 e⁻ (NISP) | Cropper+ 2014, §4.2 |
| `DARK_E_PER_S_PER_PIX` | 0.001 e⁻/s/pix (VIS); 0.01 (NISP) | MSSL VIS-PP characterisation |
| `GAIN_E_PER_ADU` | 3.1 (documentation only) | Cropper+ 2014 |
| `SKY_MAG_AB_ARCSEC2` | 22.35 mag/arcsec² | [Scaramella+ 2022](https://arxiv.org/abs/2108.01201), §3.4 |
| MAGZERO (Q1 VIS stacks) | 24.57 mag (ADU s⁻¹) | [Romelli+ 2025](https://arxiv.org/abs/2503.15305); verified against the `MAGZERO` keyword in our delivered FITS (24.60) |

### 3.2 LR noise floor

For a blank-sky pixel (`s = 0`):

```
σ²_pix ≈ sky_e + dark_e + N_exp · σ²_read
       ≈ 411 + 0.001·2260 + 4·4.5²
       ≈ 411 + 2.3 + 81
       ≈ 494
σ_pix  ≈ 22.2 e⁻
```

This σ_floor ≈ 22 e⁻ is the natural calibration scale for the rest of the pipeline. Any
source flux of order 22 e⁻ per LR pixel is at S/N ≈ 1; this is why the faint end of the
catalog is clipped — galaxies fainter than ~mag 26 produce sub-electron peaks and
contribute only structured shot noise.

We cross-checked this against real Q1 VIS cutouts: sigma-clipped background RMS is
≈ 0.0025 ADU s⁻¹ in 512² stamps, which converts to ≈ 17 e⁻ RMS over the 2260 s stack via
the VIS gain (3.1 e⁻/ADU) and integration — same order of magnitude as the model's 22 e⁻.

### 3.3 Detector artifacts

**Code:** `euclid_polish/sky/observation/artifacts.py` — `inject_cosmic_rays`, `inject_hot_pixels`,
`inject_streaks`, dispatched by `inject_artifacts` and gated by
`MultiBandForwardConfig.add_artifacts`.

| Artifact | Per-frame rate | Amplitude | Source |
|---|---|---|---|
| Cosmic rays | `CR_RATE_PER_S_PER_CM2 = 5 hits/cm²/s` × delivered-MER survival factor (`0.002` VIS, `0.015` NISP). The much larger raw detector rate is rejected by image differencing, ramp fitting, masks, and dither combination. | Track length `Exp(mean=3 px)` clipped to [1, 25] px; charge per hit `Exp(scale=1500 e⁻)` (NISP residual peaks are area-scaled on the final grid). | Holmes+ SREM; calibrated to single-band compact outliers in cached real MER tiles. |
| Hot pixels | `HOT_PIXEL_FRACTION = 5e-6` (~0.33 residual pixels per 255² band plane; per-frame randomised so a fixed mask isn't memorised) | Charge `Exp(mean=10⁴ e⁻)` (area-scaled for NISP MER). | Calibrated to cached real MER tiles. |
| Long faint streaks | `STREAK_RATE_PER_KPIX2 = 4.0` per (1000×1000)-LR-pixel area (≈ 1 per 512² VIS cutout) | Length `Exp(mean=250 px)` clipped to [40, 2000] px; width 1–3 px; amplitude `Uniform(0.3, 0.8) · σ_floor` with random sign. | Calibrated against Q1 VIS cutouts. |

**Why a separate "streaks" channel?** Roughly 30–60% of Q1 VIS cutouts contain a long,
narrow, very faint oblique feature visible only at tight (≲ ±1.5 σ) stretches. These are
not bright cosmic-ray trails — the contrast is well below 1 σ and the features are smoother
than the surrounding noise. The most plausible origin is the MER pipeline masking a CR /
satellite / persistence trail and **interpolating** across it: the interpolated values
carry the local mean but suppress the local shot noise, so the masked stripe stands out as
a smooth ridge. Pure Poisson + read + bright-CR injection does not reproduce this regime, so
we add a dedicated channel of sub-σ additive ridges with random orientation. See
`tests/test_artifacts.py` for the per-amplitude / per-orientation invariants.

### 3.4 Bright-star saturation

**Code:** `euclid_polish/sky/saturation.py`. Applied to the **dirty LR** stack (not at HR
generation), gated by `MultiBandForwardConfig.add_saturation` (default True). Per-band well
depths are derived so P(saturate) = 0.5 lands at calibration magnitudes (VIS ≈ 14, NISP
≈ 17). Onset is a smooth ~1-mag transition (`Poisson(peak · 10^N(0,0.15)) ≥ well`, drawn
independently per band); the saturated footprint is the union of 1–3 overlapping
rectangles clipped to the well depth.

---

## 4. Variance stabilisation: per-band asinh stretch

The raw signal spans ~6 orders of magnitude (sky ~400 e⁻ → bright-star peak ~10⁷ e⁻ over
the stack). Linear MAE on raw electrons would be dominated by the bright tail. We apply the
inverse-hyperbolic-sine ("asinh") stretch of
[Lupton, Gunn & Szalay 1999](https://ui.adsabs.harvard.edu/abs/1999AJ....118.1406L), **with
a per-band knee**:

```
y_B = asinh(x_B / band.asinh_stretch_scale_e)
```

Each `BandConfig` carries its own `asinh_stretch_scale_e`; currently **all four bands use
100.0 e⁻** (`Config.STRETCH_SCALE_E = 100.0`). The per-channel scale broadcasts against the
loader output so the math is one constant-tensor multiply.

- Linear regime (|x| ≪ scale): `y ≈ x / 100`, preserves shape.
- Logarithmic regime (|x| ≫ scale): compresses dynamic range.
- Signed and smoothly invertible everywhere — `x = sinh(y) · scale`.
- Approximately variance-stabilising for high-count Poisson data (same family as the
  Anscombe transform; asinh extends it sensibly to the negative values from sky
  subtraction).

The stretch is applied in the **data loader** (`training/data_multiband.py`) to both inputs
and targets, so the network learns in a roughly homoscedastic, range-compressed space. The
model has no internal normalisation.

---

## 5. Training

**Code:** `euclid_polish/training/trainer.py`, `training/data_multiband.py`,
`training/forward_op.py`, `training/models/wdsr.py`.

### 5.1 Objective: one deconvolved sky, many forward operators

The model always estimates a single quantity: the **deconvolved VIS sky** `SR @ 0.05″/pix`
(asinh space). Each data lane supervises that same estimate through its own instrument's
forward operator, in a **fixed-layout** train step (`train_step_sky`) that slices the batch
into contiguous lane blocks `[n_syn | n_hst | n_anchor]` with **no per-example branching**:

| Lane | Records | Loss |
|---|---|---|
| **synthetic** (always on) | simulated `(lr, hr)` pairs; `hr` *is* the clean sky | `\|SR − scene\|` directly (the synthetic target is the deconvolved sky) |
| **HST** (optional) | real HST F814W cutouts | `\|asinh(H ⊛ SR_lin) − HST_image\|` — `SR` is un-stretched to electrons, convolved with the HST PSF (`HSTForwardOp`), re-stretched, compared to the observed image |
| **star-anchor** (optional) | real Euclid star cutouts + sparse delta-target | masked `\|SR − delta_target\|` at the star pixel only (operator-free) |

A non-negativity penalty `λ · mean(relu(−SR))` is available but **defaults to off**
(`Config.NONNEG_SR_WEIGHT = 0.0`): forcing `SR ≥ 0` forbade legitimate deconvolution
ringing and pushed the model into a blurry basin. Re-enable per run with `--nonneg-sr-weight`.

`EuclidVISForwardOp` (Fourier-domain VIS PSF convolution + sum-rebin) is retained only for
the `/inference` "forward(SR)" diagnostic panel; the operator applied **in training** is
`HSTForwardOp`. The single-source `train_step` (used by `run_pipeline.py`, the CLI, and all
validation streams) computes `loss(SR, hr)` directly, which for the synthetic lane is
identical to the sky objective.

### 5.2 Model, loss, schedule

- **Model:** WDSR-A — 32 residual blocks, 32 filters, wide-activation expansion 6,
  weight-norm convolutions, pixel-shuffle ×2 upsampling. `nchan_in = 4` (VIS, Y_E, J_E,
  H_E), `nchan_out = 1` (VIS). **No batch-norm** (per-image dynamic range varies by orders
  of magnitude; the loader-side asinh stretch + weight-norm are the preconditioning).
- **Loss:** MAE (L1) in asinh-stretched space.
- **Optimiser:** Adam with a settable LR; `PiecewiseConstantDecay([200000], [1e-3, 5e-4])`.
- **Gradient clipping:** global-L2-norm clip at `GRAD_CLIP_NORM = 5.0`
  ([Pascanu+ 2013](https://arxiv.org/abs/1211.5063)). A **divergence guard** rolls back to
  the last checkpoint when the post-warmup pre-clip grad norm exceeds 50, halving the LR
  after repeated rollbacks — this recovers from rare loss spikes that corrupt weights.
- **Augmentation:** random aligned 96×96 HR / 48×48 LR crops plus elastic observation-PSF
  warps in the forward operator. Injected stars are carried on a sparse plane for target
  bookkeeping, but share the same deformed PSF as every other source in the exposure.
  Record-mode generation applies a seeded PSF warp to every dirty train/validate/test
  exposure. Clean-only on-the-fly training draws a fresh PSF warp on every visit; generated
  validation/test records remain fixed and replayable. **No image flips/rotations** — the
  empirical PSF is asymmetric, so a flipped target is a different SR task. Clean/HR targets
  are never deformed. Above-well sources receive MER-style dark-core masks with probability
  0.2 (configurable up to 0.5), leaving most bright stellar cores intact to match the
  real-Euclid input distribution.

### 5.3 Validation metrics

Logged to `training_log.csv`. PSNR uses a physical peak pinned to a **mag-17 reference star**
(image-domain max is meaningless for unbounded electron counts):

```
PSNR_PEAK_MAG       = 17.0
PSNR_PEAK_E         = 10^(-0.4 · (17.0 − SIM_VIS_ZEROPOINT_E)) ≈ 5.68 × 10⁶ e⁻
PSNR_PEAK_STRETCHED = asinh(PSNR_PEAK_E / 100)
```

We log `psnr_stretched` / `psnr_raw` for the synthetic lane, `psnr_*_hst` for the HST lane
(scored through the forward op), and a masked `anchor_val_psnr` (clipped at 80 dB so one
nailed pixel can't spike it). Best-checkpoint selection runs **two independent tracks** — a
PSNR composite (`w_syn·PSNR_syn + w_hst·PSNR_hst + w_anchor·PSNR_anchor`, default weights
`(1, 1, 0)`) saved to the root checkpoint dir, and a combined validation-loss track saved to
`loss_best/`.

---

## 6. Output formats

Schema v2 (multi-band), under `Config.RECORDS_DIR_V2 = "./data/images/records_v2/"`:

| File | Contents | Used for |
|---|---|---|
| `clean_{train,validate}.tfrecord` | HR clean field, `(H_hr, W_hr, 1)` VIS only, raw float32 electrons | Training target (after asinh in loader) |
| `dirty_{train,validate}.tfrecord` | LR noisy field, `(H_lr, W_lr, 4)` `(VIS, Y_E, J_E, H_E)` @ 0.10″/pix, raw float32 electrons (can be negative) | Training input (after per-band asinh in loader) |

The optional real-data lanes use their own records: HST-paired records
(`fasrc_generate_hst_tfrecords.py`) and star-anchor records
(`fasrc_generate_star_anchor_tfrecords.py`).

There is no normalisation step before the TFRecord. The float32 representation preserves
negative residuals from sky/dark subtraction. Each record carries explicit `channels`,
`band_names`, and `schema_version` fields so readers can validate the channel layout.

---

## 7. Quick start

```bash
# End-to-end on the synthetic lane (needs the real COSMOS2025 master FITS when
# the configured Sérsic density is positive; there is no stub catalog):
python scripts/run_pipeline.py --image-size 252 --ntrain 64 --nvalid 16 --steps 1000

# Generate / convolve only (skip training), or train only:
python scripts/run_pipeline.py --skip-train          # data only
python scripts/run_pipeline.py --skip-generate --skip-convolve   # train on existing records

# Extract per-band empirical PSFs (clusters stars valid in all four bands):
python scripts/extract_all_band_psfs.py

# Super-resolve a real Euclid cutout at a sky position:
python scripts/infer_euclid_cutout.py --ra <deg> --dec <deg>
#   → writes original_stack.fits (4-band LR) + SR.fits (VIS @ 0.05″/pix)

# Interactive menu (Euclid ops · sky generation · training · visualization):
python -m euclid_polish.cli.main        # or: python main.py

# Web UI / FASRC job dashboard (opens an SSH ControlMaster to the cluster):
python scripts/serve.py                 # http://127.0.0.1:8765

# Test suite:
pytest tests/ -q
```

### Cluster (FASRC / Cannon) production workflow

Heavy jobs run on Harvard's FASRC cluster via SLURM. The typical order:

1. **Download** star cutouts (`download_all_bands.py`) and real-lane data
   (`fasrc_download_euclid_sky_cutouts.py`, `fasrc_download_hst_hlsp.py`,
   `fasrc_download_tng_skirt_atlas.py`).
2. **PSFs:** `extract_all_band_psfs.py` (Euclid per-band ePSFs);
   `fasrc_extract_hst_psf.py` + `fasrc_compute_differential_kernel.py` (HST→Euclid kernel).
3. **TFRecords** for the optional lanes: `fasrc_generate_hst_tfrecords.py`,
   `fasrc_generate_star_anchor_tfrecords.py`. (Synthetic records are produced inside
   `run_pipeline.py`.)
4. **Train:** `sbatch scripts/fasrc_train.sh` (full pipeline) or
   `sbatch scripts/fasrc_train_only.sh` (records already exist). Mixed-lane training is
   driven by `scripts/fasrc_train_with_hst.py` (`--n-syn / --n-hst / --n-anchor`).

Experiment tracking lives in `scripts/track.py` (campaign lab notebook — back up
models/FITS/images, log FASRC jobs, mirror to holylabs) and `scripts/timetravel.py`
(re-run a backup's exact code in an isolated worktree + second WebUI).

---

## References

- **POLISH algorithm:** Connor, Bouman, Ravi & Hallinan, [*POLISH: Deep Learning Reconstruction of Low Surface-Brightness Astronomical Sources*](https://arxiv.org/abs/2111.03249), 2022.
- **POLISH++ (single-image extension consulted for asinh / metric choices):** Wu et al., [arXiv:2603.09162](https://arxiv.org/abs/2603.09162), 2026.
- **WDSR architecture:** Yu et al., [*Wide Activation for Efficient and Accurate Image Super-Resolution*](https://arxiv.org/abs/1808.08718), 2018.
- **Euclid VIS instrument:** Cropper et al., [*VIS: the visible imager for Euclid*](https://arxiv.org/abs/1608.08603), SPIE 2014. — exposure time, read noise, gain.
- **Euclid mission overview:** Euclid Collaboration: Mellier et al., [*Euclid I. Overview of the Euclid mission*](https://arxiv.org/abs/2405.13491), 2024.
- **Euclid Wide Survey (sky background, dither pattern):** Euclid Collaboration: Scaramella et al., [*Euclid preparation. I. The Euclid Wide Survey*](https://arxiv.org/abs/2108.01201), 2022.
- **Euclid Q1 MER pipeline (CR-flag fraction, MAGZERO, masking/interpolation):** Euclid Collaboration: Romelli et al., [*Euclid Quick Data Release (Q1): the Euclid MERge Processing Function*](https://arxiv.org/abs/2503.15305), 2025.
- **Euclid Q1 release overview:** Euclid Collaboration, [*Euclid Quick Data Release (Q1)*](https://arxiv.org/abs/2503.15303), 2025.
- **COSMOS2025 / COSMOS-Web catalog:** Shuntov et al., [*COSMOS-Web: The morphological catalog*](https://arxiv.org/abs/2506.03243), 2025.
- **TNG50-SKIRT Atlas:** Baes et al., [*The TNG50-SKIRT Atlas: post-processing methodology and first data release*](https://arxiv.org/abs/2401.04224), A&A 2024 — 18 UV-to-NIR bands, 100 pc pixels, five views, and no instrument PSF/noise.
- **Euclid-band TNG50-SKIRT images:** Euclid Collaboration: Kovačić et al., [*Euclid preparation: Extracting physical parameters from galaxies with machine learning*](https://arxiv.org/abs/2501.14408), 2025 — noise-free VIS/Y/J/H images used by this project.
- **Strong-lens population:** Collett, [*The population of galaxy-galaxy strong lenses in forthcoming optical imaging surveys*](https://arxiv.org/abs/1507.02657), ApJ 2015. Ray-tracing via [lenstronomy](https://github.com/lenstronomy/lenstronomy).
- **AB magnitude system:** Oke & Gunn, [*Secondary standard stars for absolute spectrophotometry*](https://ui.adsabs.harvard.edu/abs/1983ApJ...266..713O), ApJ 1983.
- **CCD noise model:** Janesick, [*Photon Transfer*](https://spie.org/Publications/Book/725073), SPIE Press, 2007.
- **asinh stretch:** Lupton, Gunn & Szalay, [*A Modified Magnitude System*](https://ui.adsabs.harvard.edu/abs/1999AJ....118.1406L), 1999; Lupton et al., [*Preparing red-green-blue images from CCD data*](https://ui.adsabs.harvard.edu/abs/2004PASP..116..133L), 2004.
- **Anscombe (Poisson variance-stabilisation):** Anscombe, [*The transformation of Poisson, binomial and negative-binomial data*](https://www.jstor.org/stable/2332343), Biometrika 1948.
- **Gradient clipping:** Pascanu, Mikolov & Bengio, [*On the difficulty of training recurrent neural networks*](https://arxiv.org/abs/1211.5063), 2013.
- **ePSF construction:** Anderson & King, [*Toward High-Precision Astrometry with WFPC2. I.*](https://ui.adsabs.harvard.edu/abs/2000PASP..112.1360A), PASP 2000. Implementation: `photutils.psf.EPSFBuilder`.
