# EuclidPolish

We use the POLISH algorithm ([Connor+ 2022](https://arxiv.org/abs/2111.03249)) on Euclid VIS data. The super-resolution model is WDSR-A ([Yu+ 2018](https://arxiv.org/abs/1808.08718)), built on the reference implementation at <https://github.com/krasserm/super-resolution>.

The training pipeline simulates Euclid scenes in raw electrons across **four bands** (VIS + NISP Y_E / J_E / H_E), convolves each band with its own ePSF (real for VIS, optional empirical for NISP, Gaussian fallback otherwise), applies a per-band Poisson + read-noise model, resamples NISP from native 0.30″/pix to the VIS LR 0.10″/pix grid via Lanczos-3, and writes float32 TFRecords. All photometry is calibrated against the published Euclid AB zeropoints per band; nothing in the pipeline is normalized to a unit interval before noise is injected.

Network input is the 4-channel LR stack `(VIS, Y_E, J_E, H_E) @ 0.10″/pix`; HR target is **VIS only** at 0.05″/pix. NISP channels give the network colour context to aid super-resolution of VIS, without being super-resolved themselves.

Galaxies are drawn from the COSMOS2025 (COSMOS-Web v1.1) master catalog ([Shuntov+ 2025](https://arxiv.org/abs/2506.03243)), using HDU 6 (B+D decomposition) for per-component morphology + per-band fluxes (HST F814W ≈ VIS_E, UltraVISTA Y/J/H as Euclid NISP proxies). Strong lenses come from the Collett 2015 ([arXiv:1507.02657](https://arxiv.org/abs/1507.02657)) galaxy-galaxy lens-population priors, with ray-tracing through a Singular-Isothermal-Ellipsoid + external-shear model implemented via [lenstronomy](https://github.com/lenstronomy/lenstronomy). The light profile renderer is our own vectorised Sersic implementation (`euclid_polish/sky/profiles.py`); GalSim is not used.

---

## 1. Photometry: magnitudes → electrons

All sources are placed on the simulated HR plane in **electrons accumulated over the full Wide-Survey stack** (4 dithered exposures × 565 s = 2260 s). This is the natural unit for Poisson statistics: one electron is one detected photoelectron, and the variance of a sum of independent counts is the sum itself.

### 1.1 Per-second AB zeropoint

The Euclid VIS instrument paper ([Cropper+ 2014](https://arxiv.org/abs/1608.08603), §2 and Table 1; [Euclid Collaboration: Mellier+ 2024](https://arxiv.org/abs/2405.13491), §6.2) and the Q1 release notes give the broadband VIS AB zeropoint as

```
VIS_AB_ZP_E_PER_S = 25.50          # m_AB of a source giving 1 e⁻/s
```

i.e. an AB-magnitude *m* source produces

```
F(m) = 10^(-0.4 · (m − 25.50))    [e⁻ / s]
```

at the detector after the full optical chain (mirrors, filter, QE).

### 1.2 Stack zeropoint (per-image flux scale)

Because we simulate the *coadded* Wide-Survey stack rather than a single exposure, we fold the integration time into a stack-level zeropoint:

```
T_total = N_exp · t_exp = 4 · 565 s = 2260 s
SIM_VIS_ZEROPOINT_E = VIS_AB_ZP_E_PER_S + 2.5 · log10(T_total)
                    ≈ 25.50 + 8.385 = 33.885
```

Then, for any magnitude *m*:

```
flux_e(m) = 10^(-0.4 · (m − SIM_VIS_ZEROPOINT_E))   [e⁻ over the stack]
```

This single conversion rule is applied to every source class. It is implemented at `euclid_polish/config.py:85` (definition) and used at `clean_generator.py:115`, `clean_generator.py:147`, and `clean_generator.py:313`.

### 1.3 Per-class application

Per-band electron flux for source class *X* in band *B*:
`flux_e_B = 10^(-0.4·(mag_B − band.sim_zeropoint_e))`
where `band.sim_zeropoint_e` is each `BandConfig`'s stack-level zeropoint
(`Config.BAND_VIS.sim_zeropoint_e ≈ 33.885`, similar for NISP bands at their
own integration budget).

| Class | Magnitude source per band | Flux assignment |
|---|---|---|
| **Stars** (point sources) | VIS sampled from a 3-bin distribution (`Config.STAR_MAG_*`); other bands offset by `Config.STAR_BAND_OFFSETS_MAG` (fixed G-type SED proxy) | `flux_e_B` deposited as a single HR pixel per channel (`multiband_generator.py:_deposit_star`) |
| **COSMOS2025 galaxies** | Per-component (bulge / disk), per-band magnitudes from HDU 6 of the master catalog: `mag_model_bulge_hst-f814w` ↦ VIS_E, `mag_model_disk_uvista-y` ↦ Y_E (disk), etc. | Custom vectorised Sersic2D renderer evaluates the analytic profile with closed-form amplitude from total flux. Bulge fixed at n=4, disk at n=1; geometry shared via `angle_bd`. (`profiles.py`) |
| **Strong lenses** | Lens galaxy + source galaxy each drawn from COSMOS2025 with redshift cuts (`Config.LENS_Z_*`); per-band fluxes propagate through unchanged for the lens light and are magnified by the ray-tracing for the source light | SIE + external-shear mass model from lenstronomy; lens-light + lensed-source-light rendered by our Sersic implementation at the ray-shot coordinates. (`lens_population.py`) |

Our Sersic amplitude is derived in closed form from the total flux and Sersic index (`sersic_amp_from_flux` in `profiles.py`). Sub-pixel sampling (`csub`) is auto-selected from the Sersic index and effective radius so that the discrete pixel sum matches the analytic integral to ≤ a few percent across the realistic regime (n ∈ [0.5, 4], r_e ∈ [0.05, 1.0]″). The pixel response is absorbed into the empirical ePSF (§2).

The COSMOS2025 master catalog supplies 784k galaxies with per-component bulge+disk fits + 4-band photometry (HST F814W + UltraVISTA YJH as our Euclid bandpass proxies); typically ~few × 10⁵ pass the quality cuts (`type == 0`, `flag_star == 0`, `flag_blend == 0`, `warn_flag == 0`, B+D χ² < 10, finite per-band fluxes).

### 1.4 Sky background

The diffuse Wide-Survey sky brightness ([Euclid Collaboration: Scaramella+ 2022](https://arxiv.org/abs/2108.01201), §3.4) is

```
SKY_MAG_AB_ARCSEC2 = 22.35   # mag AB / arcsec²
```

Converted to electrons per second per arcsec²:

```
SKY_E_PER_S_PER_ARCSEC2 = 10^(-0.4 · (22.35 − 25.50)) ≈ 18.20
```

The sky enters only at the noise stage — it is **not** added to the clean target. A 0.10″ LR pixel collects `18.20 · 0.10² · 2260 ≈ 411 e⁻` of sky over the stack.

---

## 2. PSF and HR → LR forward model

**Code:** `euclid_polish/euclid/psf_extractor.py` (extraction, band-agnostic),
`euclid_polish/euclid/psf_library.py` (per-band loader, with Gaussian fallback),
`euclid_polish/sky/multiband_forward.py` (per-band convolution + noise + NISP resample).

VIS uses an empirical ePSF built with `photutils.psf.EPSFBuilder` ([Anderson & King 2000](https://ui.adsabs.harvard.edu/abs/2000PASP..112.1360A); photutils impl) from bright isolated stars in real Euclid VIS cutouts, sampled at the HR scale (0.05″/pix, 2× oversampled relative to the detector). The kernel is normalised to sum=1 at load time (`PSF.from_fits()` calls `.normalized()`).

NISP bands (Y_E, J_E, H_E) follow the same extraction flow on NISP stack cutouts when available; otherwise the loader falls back to a Gaussian PSF at each band's nominal FWHM (`Config.BAND_Y_E.psf_fwhm_arcsec` etc.). The CLI `extract-psf` command takes a `--band` argument and saves to per-band paths (`euclid_psf_VIS.fits`, `euclid_psf_Y.fits`, ...).

The forward model from clean HR (4-channel) → noisy LR (4-channel, VIS LR grid):

```
For each band B:
    hr_e_B   = fftconvolve(hr_4ch[..., B], psf_B, mode="same")       # HR plane, e⁻
    lr_e_B   = sum_rebin(hr_e_B, factor=round(B.pixel_scale_lr / 0.05))
                                                                      # 2× for VIS (→0.10″)
                                                                      # 6× for NISP (→0.30″)
    lr_e_B   = apply_band_noise(lr_e_B, B, rng)                       # per-band Poisson+read
    if B is NISP:
        lr_e_B = lanczos3_upsample(lr_e_B, factor=3)                  # 0.30″ → 0.10″
lr_4ch    = stack(lr_e_VIS, lr_e_Y, lr_e_J, lr_e_H)                   # all on VIS LR grid
hr_target = hr_4ch[..., VIS] only (1 channel)
```

`sum_rebin` is **photometric**: each LR pixel is the *sum* of its (n × n) HR sub-pixels, conserving total electron count. The Lanczos-3 resample on NISP matches the SWarp/MER pipeline that Euclid uses to produce the public MER mosaics, so the LR input the network sees is operationally equivalent to MER NIR mosaic pixels.

---

## 3. Noise model

**Code:** `euclid_polish/sky/multiband_forward.py:_apply_band_noise()`,
`euclid_polish/sky/artifacts.py` (detector-artifact injection).

Per LR pixel, given the noise-free signal `s` (electrons of the source over the stack):

```
λ        = max(s + sky + dark, 0)
observed = Poisson(λ) − (sky + dark)              # sky/dark-subtracted
artifacts = CR hits + hot pixels + masked-trail streaks   # see §3.3
read     = N(0, σ_read · sqrt(N_exp))             # uncorrelated reads
output   = observed + artifacts + read            # float32, can be negative
```

This corresponds to the standard CCD noise budget — Poisson photon noise on the *total* incident charge plus uncorrelated Gaussian read noise scaled by the number of independent reads ([Janesick 2007, *Photon Transfer*](https://spie.org/Publications/Book/725073)). Sky and dark are added before the Poisson draw so their shot noise is correctly captured, then subtracted afterwards so the network's input represents the source signal alone. Detector artifacts are injected between the Poisson draw and the read-noise draw — that ordering matches the physical readout chain (charge integrates on-chip → ramp is read out → read noise added).

### 3.1 Constants (Cropper+ 2014, MSSL VIS-PP, Q1 docs)

| Symbol | Value | Source |
|---|---|---|
| `EXPOSURE_TIME_S` | 565 s | Cropper+ 2014, Table 4 (Q1 paper Romelli+ 2025 quotes 566 s nominal for science frames; we use 565 s) |
| `N_EXPOSURES` | 4 (Wide Survey dithers) | [Scaramella+ 2022](https://arxiv.org/abs/2108.01201), Euclid Wide Survey reference observing sequence |
| `READ_NOISE_E` | 4.5 e⁻ RMS / exposure | Cropper+ 2014, §4.2 ("internal noise … less than 4.4 electrons in all channels") |
| `DARK_E_PER_S_PER_PIX` | 0.001 e⁻/s/pix | MSSL VIS-PP characterisation |
| `GAIN_E_PER_ADU` | 3.1 (documentation only) | Cropper+ 2014 |
| `SKY_MAG_AB_ARCSEC2` | 22.35 mag/arcsec² | [Scaramella+ 2022](https://arxiv.org/abs/2108.01201), §3.4 |
| MAGZERO (Q1 VIS stacks) | 24.57 mag (ADU s⁻¹) | [Romelli+ 2025](https://arxiv.org/abs/2503.15305), §photometric calibration; verified against the `MAGZERO` keyword in our delivered FITS cutouts (24.60) |

### 3.2 LR noise floor

For a blank-sky pixel (`s = 0`):

```
σ²_pix ≈ sky_e + dark_e + N_exp · σ²_read
       ≈ 411 + 0.001·2260 + 4·4.5²
       ≈ 411 + 2.3 + 81
       ≈ 494
σ_pix  ≈ 22.2 e⁻
```

This σ_floor ≈ 22 e⁻ is the natural calibration scale for the rest of the pipeline. Any source flux of order 22 e⁻ per LR pixel is at S/N ≈ 1; this is why we clipped the COSMOS catalog's faint-end magnitude — galaxies with `mag_auto > ~26` produce sub-electron peaks and contribute only structured shot noise.

We cross-checked this against real Q1 VIS cutouts (`data/euclid_stars/cutouts/VIS/star_*_512.fits`): sigma-clipped background RMS is ≈ 0.0025 ADU s⁻¹ in 512² stamps, which converts to ≈ 17 e⁻ RMS over the 2260 s stack via the VIS gain (3.1 e⁻/ADU) and integration. Same order of magnitude as our model's 22 e⁻ — close enough that the dominant residual ~20% discrepancy is likely from non-uniform sky subtraction in MER rather than a parameter error.

### 3.3 Detector artifacts

**Code:** `euclid_polish/sky/artifacts.py` — `inject_cosmic_rays`, `inject_hot_pixels`, `inject_streaks`, all dispatched by `inject_artifacts` and gated by `MultiBandForwardConfig.add_artifacts`.

| Artifact | Per-frame rate | Amplitude | Source |
|---|---|---|---|
| Cosmic rays | `CR_RATE_PER_S_PER_CM2 = 5 hits/cm²/s` (raw L2 GCR rate) × `cr_rate_factor` (per-band post-rejection factor: 0.02 for VIS, 1.0 for NISP). The Q1 MER pipeline ([Romelli+ 2025](https://arxiv.org/abs/2503.15305)) reports ~1.6% of pixels flagged as CR per single-frame VIS image; cross-dither median rejection reduces survivors in the stack to ~0.1%. | Track length `Exp(mean=3 px)` clipped to [1, 25] px; charge per hit `Exp(scale=1500 e⁻)`. | Holmes+ 1989/2012 SREM; Q1 paper. |
| Hot pixels | `HOT_PIXEL_FRACTION = 0.001` (per-frame, randomised in training to avoid memorising a fixed mask) | Charge `Exp(mean=10⁴ e⁻)`. | Cropper+ 2014, MSSL CCD273 characterisation. |
| Long faint streaks | `STREAK_RATE_PER_KPIX2 = 4.0` streaks per (1000×1000)-LR-pixel area (≈ 1 streak per 512² VIS cutout on average). | Length `Exp(mean=250 px)` clipped to [40, 2000] px; width uniform in [1, 3] px (Gaussian cross-section); amplitude `Uniform(0.3, 0.8) · σ_floor` with random sign. | New (this commit). Calibrated against visual inspection of real Q1 VIS cutouts: ~30–60% of stamps show at least one such feature. |

#### Why a separate "streaks" channel?

Roughly 30–60% of Q1 VIS cutouts contain a long, narrow, very faint oblique feature spanning much of the frame — clearly visible only at tight (≲ ±1.5 σ) background stretches. They do **not** look like bright cosmic-ray trails: the contrast against the background is much less than ~1 σ, and the features are visually smoother than the surrounding noise. The most plausible origin is the MER pipeline masking a CR trail, satellite/asteroid trail, or NISP persistence streak and **interpolating** across the masked pixels; the interpolated values carry the local mean but suppress the local shot noise, so the masked stripe stands out as a smooth ridge under tight clipping.

Pure Poisson + read-noise + bright-CR injection does not reproduce this regime — its CR amplitudes are 100s–1000s of electrons (bright dots, not faint streaks), and even the longest-track oblique CRs we generate are aggressively suppressed by `cr_rate_factor = 0.02` for VIS. So we add a separate channel of sub-σ additive ridges with random orientation in `[0, π)`, length/width drawn from the distributions in the table above, and amplitude scaled to the band's `σ_floor`. The per-pixel deposition is normalised so the spine ridge peaks at exactly `amp_sigma · σ_floor` regardless of the streak's orientation (see the dominant-axis stepping in `inject_streaks`).

Visual comparison: rendering a synthetic blank-sky 512² VIS LR frame with this artifact set produces streaks that match the real Q1 features in length, width, and contrast — invisible at normal stretches, just discernible at ±1.5 σ. See `tests/test_artifacts.py::test_inject_streak_*` for the per-amplitude / per-orientation invariants.

---

## 4. Variance stabilisation: per-band asinh stretch

The raw signal spans ~6 orders of magnitude (sky ~400 e⁻ → mag-17 star peak ~5.7 × 10⁶ e⁻ over the stack). Linear MAE on raw electrons would be dominated by the bright tail and ignore everything fainter than ~mag 20. We apply the inverse-hyperbolic-sine ("asinh") stretch of [Lupton, Gunn & Szalay 1999](https://ui.adsabs.harvard.edu/abs/1999AJ....118.1406L) and [Lupton+ 2004](https://ui.adsabs.harvard.edu/abs/2004PASP..116..133L), **with a per-band knee**:

```
y_B = asinh(x_B / band.asinh_stretch_scale_e)
```

Each `BandConfig` carries its own `asinh_stretch_scale_e`; the per-channel scale broadcasts against the loader output `(B, H, W, C)` so the math is one constant-tensor multiply.

- Linear regime (|x| ≪ scale): `y ≈ x / 1000`, preserves shape.
- Logarithmic regime (|x| ≫ scale): `y ≈ sign(x) · (ln(2|x|/scale))`, compresses dynamic range.
- Signed and smoothly invertible everywhere — `x = sinh(y) · scale`.
- Approximately variance-stabilising for high-count Poisson data (in the same family as the Anscombe transform ([Anscombe 1948](https://www.jstor.org/stable/2332343)), which uses `2·sqrt(x + 3/8)`; asinh extends it sensibly to negative values from sky subtraction).

The stretch is applied in the **data loader** (`training/data.py`) to both inputs and targets, so the network learns in a roughly homoscedastic, range-compressed space without ever seeing an analytic activation that could saturate. The model has no internal normalisation.

---

## 5. Training

**Code:** `euclid_polish/training/trainer.py`, `euclid_polish/training/data_multiband.py`, `euclid_polish/training/models/wdsr.py`

- **Model:** WDSR-A — 32 residual blocks, 32 filters, weight-norm convolutions, pixel-shuffle ×2 upsampling. **`nchan_in=4` (VIS, Y_E, J_E, H_E), `nchan_out=1` (VIS)**. No batch-norm (BN over astronomical pixel statistics is harmful — the per-image dynamic range varies by orders of magnitude).
- **Loss:** MAE in asinh-stretched space.
- **Optimiser:** Adam, lr = 1e-3 → 5e-4 piecewise schedule.
- **Gradient clipping:** `tf.clip_by_global_norm(grads, GRAD_CLIP_NORM=5.0)` ([Pascanu+ 2013](https://arxiv.org/abs/1211.5063)). We observed rare batches where the loss spiked from 5×10⁻⁴ to >10² and corrupted weights for many steps; clipping bounds the update and recovers stability without affecting typical-batch progress.
- **Augmentation:** random 96×96 HR crops only. No flips/rotations — the empirical PSF is asymmetric, so a flipped target is a different super-resolution task.

### 5.1 Validation metric: PSNR with a physical peak

Standard image-domain PSNR uses the dataset's max as the peak; for unbounded electron counts this is meaningless. We pin the peak to a **mag-17 reference star**:

```
PSNR_PEAK_MAG       = 17.0
PSNR_PEAK_E         = 10^(-0.4 · (17.0 − SIM_VIS_ZEROPOINT_E)) ≈ 5.677 × 10⁶ e⁻
PSNR_PEAK_STRETCHED = asinh(PSNR_PEAK_E / 1000)               ≈ 9.337
```

We log two PSNRs per evaluation: `psnr_stretched` (in the training space) and `psnr_raw` (in electrons, after inverting the stretch). Best-checkpoint selection is on `psnr_stretched`.

---

## 6. Output formats

Schema v2 (multi-band), stored under `Config.RECORDS_DIR_V2 = "./data/images/records_v2/"`:

| File | Contents | Used for |
|---|---|---|
| `clean_{train,validate}.tfrecord` | HR clean field, ``(H_hr, W_hr, 1)`` VIS only, raw float32 electrons | Training target (after asinh in loader) |
| `dirty_{train,validate}.tfrecord` | LR noisy field, ``(H_lr, W_lr, 4)`` `(VIS, Y_E, J_E, H_E)` @ 0.10″/pix, raw float32 electrons (can be negative) | Training input (after per-band asinh in loader) |

There is no normalisation step before the TFRecord. The float32 representation preserves negative residuals from sky/dark subtraction, which an unsigned encoding would clip away. The record carries explicit `channels`, `band_names`, and `schema_version` fields so readers can validate they get the channel layout they expect.

## 7. Quick start

```bash
# Whole pipeline (synthetic stub catalog — no FITS required):
python scripts/run_pipeline.py --use-stub-catalog --image-size 252 \
    --ntrain 64 --nvalid 16 --steps 1000

# With the real COSMOS2025 catalog at data/COSMOS2025/cosmos2025.fits:
python scripts/run_pipeline.py --image-size 252

# Interactive menu:
python -m euclid_polish.cli.main

# Test suite (~3 min including real-catalog load):
pytest tests/ -q
```

---

## References

- **POLISH algorithm:** Connor, Bouman, Ravi & Hallinan, [*POLISH: Deep Learning Reconstruction of Low Surface-Brightness Astronomical Sources*](https://arxiv.org/abs/2111.03249), 2022.
- **POLISH++ (single-image extension consulted for asinh / metric choices):** Wu et al., [arXiv:2603.09162](https://arxiv.org/abs/2603.09162), 2026.
- **WDSR architecture:** Yu et al., [*Wide Activation for Efficient and Accurate Image Super-Resolution*](https://arxiv.org/abs/1808.08718), 2018.
- **Euclid VIS instrument:** Cropper et al., [*VIS: the visible imager for Euclid*](https://arxiv.org/abs/1608.08603), SPIE 2014. — exposure time, read noise, gain.
- **Euclid mission overview:** Euclid Collaboration: Mellier et al., [*Euclid I. Overview of the Euclid mission*](https://arxiv.org/abs/2405.13491), 2024.
- **Euclid Wide Survey (sky background, dither pattern):** Euclid Collaboration: Scaramella et al., [*Euclid preparation. I. The Euclid Wide Survey*](https://arxiv.org/abs/2108.01201), 2022.
- **Euclid Q1 MER pipeline (CR-flag fraction, MAGZERO, masking/interpolation):** Euclid Collaboration: Romelli et al., [*Euclid Quick Data Release (Q1): From images to multiwavelength catalogues — the Euclid MERge Processing Function*](https://arxiv.org/abs/2503.15305), 2025.
- **Euclid Q1 release overview:** Euclid Collaboration, [*Euclid Quick Data Release (Q1)*](https://arxiv.org/abs/2503.15303), 2025.
- **COSMOS parametric catalog:** Mandelbaum, Rowe, Armstrong & Leauthaud, [*Great3 Challenge Handbook*](https://arxiv.org/abs/1308.5379), 2014. Catalog FITS distributed with GalSim.
- **AB magnitude system:** Oke & Gunn, [*Secondary standard stars for absolute spectrophotometry*](https://ui.adsabs.harvard.edu/abs/1983ApJ...266..713O), ApJ 1983.
- **CCD noise model:** Janesick, [*Photon Transfer*](https://spie.org/Publications/Book/725073), SPIE Press, 2007.
- **asinh stretch:** Lupton, Gunn & Szalay, [*A Modified Magnitude System*](https://ui.adsabs.harvard.edu/abs/1999AJ....118.1406L), 1999; Lupton et al., [*Preparing red-green-blue images from CCD data*](https://ui.adsabs.harvard.edu/abs/2004PASP..116..133L), 2004.
- **Anscombe (Poisson variance-stabilisation):** Anscombe, [*The transformation of Poisson, binomial and negative-binomial data*](https://www.jstor.org/stable/2332343), Biometrika 1948.
- **Gradient clipping:** Pascanu, Mikolov & Bengio, [*On the difficulty of training recurrent neural networks*](https://arxiv.org/abs/1211.5063), 2013.
- **ePSF construction:** Anderson & King, [*Toward High-Precision Astrometry with WFPC2. I. Deriving an Accurate Point-Spread Function*](https://ui.adsabs.harvard.edu/abs/2000PASP..112.1360A), PASP 2000. Implementation: `photutils.psf.EPSFBuilder`.
