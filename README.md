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

The light-profile renderer is our own vectorised Sérsic implementation
(`euclid_polish/sky/profiles.py`); **GalSim is not used.** The normal synthetic
pipeline controls the COSMOS Sérsic and TNG50 SKIRT populations independently with
`--sersic-density-arcmin2` and `--tng-density-arcmin2`. Setting the former to zero
and enabling `--tng-redshift-mode` produces a pure-TNG field without loading the
COSMOS catalog. The lens-isolation experiment uses that pure-TNG configuration by
default, with its own record and model namespace.

**What the model learns.** The network input is the 4-channel LR stack
`(VIS, Y_E, J_E, H_E) @ 0.10″/pix`. The output is a single **deconvolved VIS sky** image
`SR @ 0.05″/pix` (NISP channels give colour context but are not super-resolved). Each
training lane supervises that one estimate through its own instrument's forward
operator (see §5).

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
| **Stars** (point sources) | VIS drawn from a smooth differential star-count law dN/dm ∝ 10^(slope·m) over [bright, faint] (`Config.STAR_MAG_SLOPE=0.20`, `STAR_MAG_BRIGHT=12.0`, `STAR_MAG_FAINT=25.0`); other bands offset by `Config.STAR_BAND_OFFSETS_MAG` (fixed G-type SED proxy) | `flux_e_B` deposited as a single HR pixel per channel (`multiband_generator.py:_deposit_star`); the PSF is applied later by the forward model |
| **COSMOS2025 galaxies** (when `--sersic-density-arcmin2 > 0`) | Per-component (bulge / disk), per-band magnitudes from HDU 6 of the master catalog: `mag_model_bulge_hst-f814w` ↦ VIS_E, `mag_model_disk_uvista-y` ↦ Y_E (disk), etc. | Custom vectorised Sérsic2D renderer evaluates the analytic profile with closed-form amplitude from total flux. Bulge fixed at n=4, disk at n=1; geometry shared via `angle_bd`. (`profiles.py`) |
| **TNG50 galaxies** (when `--tng-density-arcmin2 > 0`) | Real SKIRT-rendered multi-band stamps from the TNG50 atlas | TNG stamps are injected at their configured field density and resampled to the HR grid (`sky/tng_galaxy.py`). In redshift mode (§1.5), each stamp's size and photometry follow from its own redshift draw. |
| **Strong lenses** | Catalog-backed lenses use COSMOS priors; catalog-free pure-TNG fields sample the same geometry priors while deriving σ_v from the deflector subhalo's stellar mass | SIE + external-shear mass model from lenstronomy; lens-light + lensed-source-light rendered by our Sérsic implementation or TNG stamps at the ray-shot coordinates. (`lens_population.py`) |

Our Sérsic amplitude is derived in closed form from the total flux and Sérsic index
(`sersic_amp_from_flux` in `profiles.py`). Sub-pixel sampling is auto-selected from the
Sérsic index and effective radius so the discrete pixel sum matches the analytic integral
to ≤ a few percent across the realistic regime (n ∈ [0.5, 4], r_e ∈ [0.05, 1.0]″). The
pixel response is absorbed into the empirical ePSF (§2).

The COSMOS2025 master catalog (COSMOS-Web v1.1,
[Shuntov+ 2025](https://arxiv.org/abs/2506.03243)) supplies ~784k galaxies with
per-component bulge+disk fits and 4-band photometry (HST F814W + UltraVISTA YJH as our
Euclid bandpass proxies); typically ~few × 10⁵ pass the quality cuts (`type == 0`,
`flag_star == 0`, `flag_blend == 0`, `warn_flag == 0`, B+D χ² < 10, finite per-band
fluxes). The catalog is mandatory only when the configured Sérsic density is positive;
there is no synthetic-stub fallback. Pure-TNG generation sets the Sérsic density to zero
and therefore skips the catalog entirely.

### 1.5 TNG redshift realism (`--tng-redshift-mode`)

The SKIRT atlas frames are intrinsic z = 0 images on a physical 100 pc/pixel grid.
In redshift mode (`sky/redshift_model.py`) each injected stamp draws one z from
`dN/dz ∝ dV_c/dz · exp(-(z/1.5)²)` on [0.10, 2.5] — the comoving volume element
times the declining number density of massive galaxies (Muzzin+ 2013), the right
distribution for the atlas's intrinsically luminous population (median z ≈ 1.2;
a Smail-form `n(z) ∝ z² exp(-(z/0.65)^1.5)` is available via `TNG_Z_FORM`).
That single draw sets everything:

- **Angular size** — the block-mean factor is `F(z) = θ_HR[rad] · D_A(z) / 100 pc`
  (≈3 at z = 0.5, capped at ≈4.3 by the D_A turnover near z ≈ 1.6). No separate
  "big galaxy" population is needed: nearby draws produce the resolved giants.
- **Tolman dimming** — surface brightness × (1+z)⁻³ (per-frequency intensity).
- **Compactness correction** — the atlas frames are z = 0 morphologies, but real
  galaxies at the drawn z were smaller at fixed mass (R_e ∝ (1+z)^−0.75…−1.5,
  van der Wel+ 2014): an extra flux-conserving squeeze C(z) = C₀(1+z)^β shrinks
  the stamp and boosts its surface brightness by C². C₀ = 1.3 is measured: the
  atlas runs 1.24–1.41× the observed z≈0.1 mass–size relation
  (`TNG_COMPACT_C0`/`TNG_COMPACT_BETA`).
- **Mass-function-weighted rescaling** — the atlas is a *morphology library*,
  not a population sample: each field stamp draws its target mass from the real
  Schechter mass function (α = −1.2, log M* = 10.97) over [10⁹, 10¹²], matches
  an atlas galaxy within ×30 in mass, and is rescaled down — flux × s (L ∝ M),
  size ÷ s^0.25 (the observed mass–size slope), so surface brightness falls as
  s^0.5 (Kormendy-like). The rendered population follows the observed mass
  distribution by construction, down to log M★ = 8.5 at 60/arcmin²: ~11 draws
  per 510 px field, of which ~8 render (median m_VIS ≈ 25.7) — draws predicted
  fainter than m_VIS = 28 are skipped before the stamp load
  (`TNG_FAINT_SKIP_MAG_VIS`). An R_e > 1″ giant appears in ~1 of 30 fields.
  Lens deflectors are never rescaled (`TNG_MF_*`, `TNG_MASS_WINDOW`,
  `TNG_MASS_SIZE_ALPHA`).
- **Surface-brightness truncation** — the SKIRT box carries faint light over all
  160 kpc; pixels below μ = 28 mag/arcsec² (≈ the VIS stack's 1σ per arcsec²) are
  zeroed and the stamp cropped, so apparent sizes are the detectable isophotal
  ones (`TNG_SB_TRUNCATE_MAG_ARCSEC2`).
- **Spectral drift** — observed band b samples the rest SED at λ_b/(1+z): a
  deterministic part interpolates the stamp's own 4-point SED, and a stochastic
  tilt `exp(ε·ln(λ_b/λ_H))`, `ε ~ N(0, 0.15 + 0.35·ln(1+z))`, randomizes the
  colours — red-leaning on average, sometimes bluer.
- **Physical number density** — in pure-TNG mode the field-galaxy count follows
  the real sky density of the rendered (log M★ ≥ 9) population, ≈ 33/arcmin²
  (Baldry+ 2012 φ₀ × the same weighted volume integral), not the full COSMOS
  111/arcmin² that counts undetectable dwarfs (`TNG_GAL_DENSITY_ARCMIN2`).
- **Optional Sérsic dwarf backfill** — off by default (the MF-rescaled TNG
  population covers the small end with real morphology, keeping pure-TNG
  catalog-free); `--tng-dwarf-density-arcmin2 102` mixes small COSMOS Sérsic
  rows (R_e ≤ 0.5″) back in.
- **Lens masses** — a TNG-lit deflector takes σ_v from its subhalo's stellar mass
  (Faber–Jackson on `data/_tng_infographics/tng_properties.csv`), and the system is
  rejected unless θ_E ≥ 1.2 × the lens's apparent half-light radius, so the arcs
  always clear the foreground light.

### 1.4 Sky background

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
**once** by sky position (K-Means++, ~100 stars per cluster) on stars valid in all four
bands, so cluster index *ci* maps to the same field region in every band.

Each band is saved as a multi-extension FITS at `data/euclid_psf/euclid_psf_<BAND>.fits`:
`HDU[0]` is the **field-mean** PSF (so legacy single-PSF readers still work), and
`HDU[1..K]` are the per-cluster kernels with `RA/DEC/NSTARS` provenance headers. Missing
files fall back to a Gaussian PSF at each band's nominal FWHM, wrapped as a 1-element set.

At generation time, **one** PSF sample (cluster index + optional roll) is drawn per scene
and applied to all four bands (shared pointing). There is **no blending** of cluster
kernels. Roll-rotation augmentation is implemented (`PSF.rotated`) but **disabled by
default** (`MultiBandForwardConfig.psf_unrotated_prob = 1.0`): the default behaviour is a
random cluster pick with no rotation.

`differential_kernel.py` is a **separate** path used only by the HST→Euclid training lane
(§5): it solves `A ⊛ H ≈ E` (Wiener) so that convolving a real HST F814W cutout with `A`
yields a correct Euclid-PSF LR instead of double-convolving through HST's own PSF. It is
not part of the synthetic forward model.

### 2.2 Forward model

All four bands are delivered by the Euclid MER archive on a common **0.10″/pix** grid, so
the simulator models them the same way — there is no native-0.30″ NISP stage in the
forward path. From clean HR (4-channel, 0.05″/pix) → noisy LR (4-channel, 0.10″/pix):

```
For each band B:
    hr_e_B = (psf_sample applied to psf_set[B]) ⊛ hr_4ch[..., B]      # HR plane, e⁻
    lr_e_B = sum_rebin(hr_e_B, factor=round(0.10 / 0.05) = 2)         # → 0.10″/pix
    lr_e_B = apply_band_noise(lr_e_B, B, rng)                         # per-band Poisson+read+artifacts
lr_4ch    = stack(lr_e_VIS, lr_e_Y, lr_e_J, lr_e_H)                   # all on the 0.10″ grid
lr_4ch    = apply_star_saturation(lr_4ch)                            # bright-star wells, on the dirty LR
hr_target = hr_4ch[..., VIS] only (1 channel, clean, no noise)
```

`sum_rebin` is **photometric**: each LR pixel is the *sum* of its (n × n) HR sub-pixels,
conserving total electron count. (A Lanczos-3 NISP→VIS-LR resample stage exists in the
code — `sky/resample.py`, `NISP_RESAMPLE_KERNEL="lanczos3"` — but is **dormant** under the
current uniform-0.10″ band configuration, where the resample factor evaluates to 1. It
would re-activate only if a band's LR pixel scale were restored to 0.30″.)

---

## 3. Noise model

**Code:** `euclid_polish/sky/noise.py` (`apply_band_noise`),
`euclid_polish/sky/artifacts.py` (detector-artifact injection),
`euclid_polish/sky/saturation.py` (bright-star wells).

Per LR pixel, given the noise-free signal `s` (electrons of the source over the stack):

```
λ        = max(s + sky + dark, 0)
observed = Poisson(λ) − (sky + dark)              # sky/dark-subtracted
artifacts = CR hits + hot pixels + masked-trail streaks   # see §3.3
read     = N(0, σ_read · sqrt(N_exp))             # uncorrelated reads
output   = observed + artifacts + read            # float32, can be negative
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

**Code:** `euclid_polish/sky/artifacts.py` — `inject_cosmic_rays`, `inject_hot_pixels`,
`inject_streaks`, dispatched by `inject_artifacts` and gated by
`MultiBandForwardConfig.add_artifacts`.

| Artifact | Per-frame rate | Amplitude | Source |
|---|---|---|---|
| Cosmic rays | `CR_RATE_PER_S_PER_CM2 = 5 hits/cm²/s` × `cr_rate_factor` (per-band post-rejection: 0.02 for VIS, 1.0 for NISP). Q1 MER reports ~1.6% of pixels CR-flagged per single VIS frame; cross-dither median rejection leaves ~0.1% in the stack. | Track length `Exp(mean=3 px)` clipped to [1, 25] px; charge per hit `Exp(scale=1500 e⁻)`. | Holmes+ SREM; Q1 paper. |
| Hot pixels | `HOT_PIXEL_FRACTION = 0.001` (per-frame, randomised so a fixed mask isn't memorised) | Charge `Exp(mean=10⁴ e⁻)`. | Cropper+ 2014, MSSL CCD273. |
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
- **Augmentation:** random aligned 96×96 HR / 48×48 LR crops only. **No flips/rotations** —
  the empirical PSF is asymmetric, so a flipped target is a different SR task.

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

# Pure-TNG generation (redshift-realistic stamps, no COSMOS catalog needed;
# requires the TNG50 SKIRT atlas under $DATA_DIR/tng_skirt/):
python scripts/run_pipeline.py --sersic-density-arcmin2 0 --tng-density-arcmin2 60 --tng-redshift-mode --skip-train

# Generate / convolve only (skip training), or train only:
python scripts/run_pipeline.py --skip-train          # data only
python scripts/run_pipeline.py --skip-generate --skip-convolve   # train on existing records

# Additive lens-system isolation experiment: complete lens systems are the
# clean target, while ordinary TNG galaxies and stars remain only in the input.
# For direct CLI runs, choose workers explicitly; FASRC derives them from the
# CPU allocation configured for its generation step.
python scripts/lens_isolation_generate.py --workers 16
python scripts/lens_isolation_train.py --sources member_01,member_04
python scripts/lens_isolation_evaluate.py

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
- **Strong-lens population:** Collett, [*The population of galaxy-galaxy strong lenses in forthcoming optical imaging surveys*](https://arxiv.org/abs/1507.02657), ApJ 2015. Ray-tracing via [lenstronomy](https://github.com/lenstronomy/lenstronomy).
- **AB magnitude system:** Oke & Gunn, [*Secondary standard stars for absolute spectrophotometry*](https://ui.adsabs.harvard.edu/abs/1983ApJ...266..713O), ApJ 1983.
- **CCD noise model:** Janesick, [*Photon Transfer*](https://spie.org/Publications/Book/725073), SPIE Press, 2007.
- **asinh stretch:** Lupton, Gunn & Szalay, [*A Modified Magnitude System*](https://ui.adsabs.harvard.edu/abs/1999AJ....118.1406L), 1999; Lupton et al., [*Preparing red-green-blue images from CCD data*](https://ui.adsabs.harvard.edu/abs/2004PASP..116..133L), 2004.
- **Anscombe (Poisson variance-stabilisation):** Anscombe, [*The transformation of Poisson, binomial and negative-binomial data*](https://www.jstor.org/stable/2332343), Biometrika 1948.
- **Gradient clipping:** Pascanu, Mikolov & Bengio, [*On the difficulty of training recurrent neural networks*](https://arxiv.org/abs/1211.5063), 2013.
- **ePSF construction:** Anderson & King, [*Toward High-Precision Astrometry with WFPC2. I.*](https://ui.adsabs.harvard.edu/abs/2000PASP..112.1360A), PASP 2000. Implementation: `photutils.psf.EPSFBuilder`.
